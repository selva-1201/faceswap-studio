"""FastAPI server and asynchronous background job manager for FaceSwap Studio.

Provides non-blocking job creation, streaming uploads with 95MB Cloudflare protection,
single-job execution queue, thread-safe snapshotting, pre-flight GPU/disk validation,
system resource metrics, delta log polling, and cancellation support.
Compatible with Python 3.10+.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import asyncio
import copy
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    aiofiles = None
    HAS_AIOFILES = False
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    ALLOWED_DETECTOR_MODELS,
    ALLOWED_ENHANCER_MODELS,
    ALLOWED_SWAPPER_MODELS,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_ENHANCER_BLEND,
    DEFAULT_ENHANCER_MODEL,
    DEFAULT_SWAPPER_MODEL,
    FACEFUSION_DIR,
    FACESWAP_THREADS,
    JOBS_DIR,
    MAX_UPLOAD_MB,
    MODELS_CACHE_DIR,
    OUTPUTS_DIR,
    SERVER_HOST,
    SERVER_PORT,
    UPLOADS_DIR,
    WORKSPACE_DIR,
    init_workspace,
)
from backend.parallel_runner import execute_dual_gpu_pipeline, terminate_job_processes
from backend.swap_pipeline import (
    PipelineError,
    build_facefusion_command,
    cleanup_job_chunks,
    concat_and_remux,
    probe_video,
    split_video_chunks,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("faceswap.server")

# Persistent job store file path
JOBS_STORE_PATH = JOBS_DIR / "jobs.json"


@dataclass
class JobRecord:
    """Data model representing a FaceSwap Studio job."""
    job_id: str
    status: str  # "queued", "processing", "completed", "failed", "cancelled"
    stage: str
    progress: int  # 0 to 100
    video_filename: str
    source_a_filename: str
    source_b_filename: Optional[str]
    swapper_model: str
    enhancer_model: str
    enhancer_blend: int
    detector_model: str
    pos_a: int
    pos_b: int
    ref_frame: int
    enhance_face_b: bool = False
    duration_seconds: float = 0.0
    output_filename: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    gpu_status: dict[str, Any] = field(default_factory=lambda: {
        "gpu0": {"status": "idle", "progress": 0},
        "gpu1": {"status": "idle", "progress": 0},
    })
    logs: list[str] = field(default_factory=list)


# In-memory job registry and thread-safety locks
JOBS: dict[str, JobRecord] = {}
JOBS_LOCK = threading.Lock()
_LAST_SAVE_TIME = 0.0
_JOB_QUEUE: asyncio.Queue[str] = asyncio.Queue()
_CANCELLED_JOBS: set[str] = set()


def _is_tqdm_spam(line: str) -> bool:
    """Detect and ignore progress bar lines from logs."""
    if "%|" in line or "it/s" in line or "s/it" in line:
        return True
    if re.match(r"^\s*\d{1,3}%\s*\|", line):
        return True
    return False


def _append_job_log(job: JobRecord, msg: str) -> None:
    """Append a log entry, strip progress spam, and cap logs to 300 entries."""
    if _is_tqdm_spam(msg):
        return
    job.logs.append(msg)
    if len(job.logs) > 300:
        job.logs = job.logs[-300:]


def _save_jobs_to_disk(force: bool = False) -> None:
    """Debounced persistence of in-memory job records to jobs.json."""
    global _LAST_SAVE_TIME
    now = time.time()
    if not force and (now - _LAST_SAVE_TIME < 2.0):
        return

    with JOBS_LOCK:
        _LAST_SAVE_TIME = now
        snapshot = {job_id: asdict(job) for job_id, job in JOBS.items()}

    try:
        temp_path = JOBS_STORE_PATH.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        temp_path.replace(JOBS_STORE_PATH)
    except Exception as e:
        logger.warning(f"Failed to persist jobs.json: {e}")


def _load_jobs_from_disk() -> None:
    """Load job records on server initialization. Mark interrupted jobs as failed."""
    if not JOBS_STORE_PATH.exists():
        return
    try:
        with open(JOBS_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        with JOBS_LOCK:
            for job_id, job_dict in data.items():
                record = JobRecord(**job_dict)
                # Startup recovery: any job running or queued when server died must fail
                if record.status in ("queued", "processing"):
                    record.status = "failed"
                    record.stage = "Server restarted during processing"
                    record.error = "Server restarted during processing"
                    record.completed_at = datetime.now(timezone.utc).isoformat()
                JOBS[job_id] = record
        _save_jobs_to_disk(force=True)
        logger.info(f"Loaded {len(data)} historical job records from {JOBS_STORE_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load jobs.json: {e}")


# GPU Information Inspection (Cached for 30s, strictly nvidia-smi, zero torch import)
_GPU_INFO_CACHE: Optional[dict[str, Any]] = None
_GPU_INFO_CACHE_TIME = 0.0
_GPU_INFO_LOCK = threading.Lock()


def get_system_gpu_info() -> dict[str, Any]:
    """Inspect system GPU count, driver, and VRAM via nvidia-smi with 30s caching."""
    global _GPU_INFO_CACHE, _GPU_INFO_CACHE_TIME
    now = time.time()

    with _GPU_INFO_LOCK:
        if _GPU_INFO_CACHE is not None and (now - _GPU_INFO_CACHE_TIME < 30.0):
            return _GPU_INFO_CACHE

        gpu_count = 0
        gpu_devices: list[str] = []
        driver_version = "Unknown"

        if shutil.which("nvidia-smi"):
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                lines = [line.strip() for line in proc.stdout.strip().split("\n") if line.strip()]
                gpu_count = len(lines)
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if parts:
                        dev_name = parts[0]
                        vram = parts[2] if len(parts) > 2 else ""
                        gpu_devices.append(f"{dev_name} ({vram})" if vram else dev_name)
                        if len(parts) > 1 and driver_version == "Unknown":
                            driver_version = parts[1]
            except Exception as e:
                logger.debug(f"nvidia-smi query failed: {e}")

        _GPU_INFO_CACHE = {
            "gpu_count": gpu_count,
            "gpu_devices": gpu_devices,
            "driver_version": driver_version,
            "dual_gpu_ready": gpu_count >= 2,
        }
        _GPU_INFO_CACHE_TIME = now
        return _GPU_INFO_CACHE


# CUDA Provider Check (Cached)
_CUDA_PROVIDER_OK: Optional[bool] = None
_CUDA_PROVIDER_LOCK = threading.Lock()


def check_cuda_provider() -> bool:
    """Verify CUDAExecutionProvider availability via isolated python subprocess."""
    global _CUDA_PROVIDER_OK
    with _CUDA_PROVIDER_LOCK:
        if _CUDA_PROVIDER_OK is not None:
            return _CUDA_PROVIDER_OK

        try:
            cmd = ["python3", "-c", "import onnxruntime as o; print('CUDAExecutionProvider' in o.get_available_providers())"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            _CUDA_PROVIDER_OK = res.stdout.strip() == "True"
        except Exception:
            _CUDA_PROVIDER_OK = False

        return _CUDA_PROVIDER_OK


def get_system_resources_data() -> dict[str, Any]:
    """Inspect system RAM, disk free spaces, and per-GPU VRAM usage."""
    # 1. System RAM
    ram_data = {"total_mb": 0, "available_mb": 0, "percent_used": 0.0}
    try:
        import psutil
        vmem = psutil.virtual_memory()
        ram_data = {
            "total_mb": round(vmem.total / (1024 * 1024), 1),
            "available_mb": round(vmem.available / (1024 * 1024), 1),
            "percent_used": vmem.percent,
        }
    except Exception:
        pass

    # 2. Disk Space
    def disk_free(path: Path) -> dict[str, float]:
        try:
            usage = shutil.disk_usage(str(path))
            return {
                "total_gb": round(usage.total / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
        except Exception:
            return {"total_gb": 0.0, "free_gb": 0.0}

    workspace_disk = disk_free(WORKSPACE_DIR)
    tmp_disk = disk_free(Path("/tmp"))

    # 3. GPU VRAM metrics from nvidia-smi
    gpu_metrics: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in proc.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpu_metrics.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_mb": float(parts[2]),
                        "used_mb": float(parts[3]),
                        "free_mb": float(parts[4]),
                    })
        except Exception:
            pass

    return {
        "ram": ram_data,
        "disk_workspace": workspace_disk,
        "disk_tmp": tmp_disk,
        "gpus": gpu_metrics,
    }


def _run_single_job_pipeline(job_id: str) -> None:
    """Execute dual-GPU pipeline for a single queued job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if job_id in _CANCELLED_JOBS:
            job.status = "cancelled"
            job.stage = "Job cancelled by user"
            _save_jobs_to_disk(force=True)
            return
        job.status = "processing"
        job.stage = "Probing video metadata"
        job.progress = 5
        _save_jobs_to_disk(force=True)

    job_upload_dir = UPLOADS_DIR / job_id
    video_path = job_upload_dir / job.video_filename
    source_a_path = job_upload_dir / job.source_a_filename
    source_b_path = (job_upload_dir / job.source_b_filename) if job.source_b_filename else None

    def log_and_record(msg: str) -> None:
        t = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{t}] {msg}"
        with JOBS_LOCK:
            _append_job_log(job, entry)
        logger.info(f"[{job_id}] {msg}")
        _save_jobs_to_disk()

    try:
        # Pre-flight Check 1: Required files & CUDA provider
        log_and_record("Running pre-flight system diagnostics...")
        res_start = get_system_resources_data()
        log_and_record(
            f"Pre-flight resources: RAM free={res_start['ram']['available_mb']}MB, "
            f"Disk free={res_start['disk_workspace']['free_gb']}GB"
        )

        entrypoint = FACEFUSION_DIR / "facefusion.py"
        if not entrypoint.exists():
            raise PipelineError(f"FaceFusion repository entrypoint missing at {entrypoint}. Run setup_env.sh first.")

        if not check_cuda_provider():
            log_and_record("[Warning] CUDAExecutionProvider not detected in ONNX Runtime. Processing will be slow.")

        # Pre-flight Check 2: Disk space requirement (at least 6x input video size)
        video_size_bytes = video_path.stat().st_size
        req_bytes = video_size_bytes * 6
        usage = shutil.disk_usage(str(WORKSPACE_DIR))
        if usage.free < req_bytes:
            req_mb = req_bytes / (1024 * 1024)
            free_mb = usage.free / (1024 * 1024)
            raise PipelineError(
                f"Insufficient disk space: required at least 6x video size (~{req_mb:.1f}MB), "
                f"but only {free_mb:.1f}MB available in {WORKSPACE_DIR}."
            )

        if job_id in _CANCELLED_JOBS:
            raise PipelineError("Job cancelled by user.")

        # 1. Probe video
        meta = probe_video(video_path)
        with JOBS_LOCK:
            job.duration_seconds = meta["duration"]
        log_and_record(
            f"Video probed: {meta['duration']:.2f}s, {meta['width']}x{meta['height']}, "
            f"{meta['fps']:.1f} fps, audio={meta['has_audio']}"
        )

        # 2. Split video into chunks
        with JOBS_LOCK:
            job.stage = "Splitting video into 2 chunks (libx264)"
            job.progress = 15
        _save_jobs_to_disk(force=True)

        chunk0_path, chunk1_path = split_video_chunks(
            video_path=video_path,
            job_id=job_id,
            duration=meta["duration"],
        )
        log_and_record(f"Video chunked: {chunk0_path.name} and {chunk1_path.name}")

        if job_id in _CANCELLED_JOBS:
            raise PipelineError("Job cancelled by user.")

        # 3. Dual-GPU parallel processing
        with JOBS_LOCK:
            job.stage = "Processing chunks concurrently on GPU 0 and GPU 1"
            job.progress = 30
            job.gpu_status["gpu0"] = {"status": "processing", "progress": 5}
            job.gpu_status["gpu1"] = {"status": "processing", "progress": 5}
        _save_jobs_to_disk(force=True)

        def on_gpu_progress(update: dict[str, Any]) -> None:
            with JOBS_LOCK:
                if "gpu0" in update:
                    job.gpu_status["gpu0"].update(update["gpu0"])
                if "gpu1" in update:
                    job.gpu_status["gpu1"].update(update["gpu1"])
                if "logs" in update:
                    for line in update["logs"]:
                        _append_job_log(job, line)
            _save_jobs_to_disk()

        chunk0_done, chunk1_done = execute_dual_gpu_pipeline(
            chunk0_path=chunk0_path,
            chunk1_path=chunk1_path,
            source_face_a=source_a_path,
            pos_a=job.pos_a,
            ref_frame=job.ref_frame,
            source_face_b=source_b_path,
            pos_b=job.pos_b,
            enhance_face_b=job.enhance_face_b,
            swapper_model=job.swapper_model,
            enhancer_model=job.enhancer_model,
            enhancer_blend=job.enhancer_blend,
            detector_model=job.detector_model,
            job_id=job_id,
            progress_callback=on_gpu_progress,
        )

        if job_id in _CANCELLED_JOBS:
            raise PipelineError("Job cancelled by user.")

        with JOBS_LOCK:
            job.progress = 80
            job.gpu_status["gpu0"] = {"status": "completed", "progress": 100}
            job.gpu_status["gpu1"] = {"status": "completed", "progress": 100}
        log_and_record("Both GPU workers finished chunk processing.")

        # 4. Concatenate and remux original audio
        with JOBS_LOCK:
            job.stage = "Concatenating chunks and restoring audio stream"
            job.progress = 85
        _save_jobs_to_disk(force=True)

        final_video_path = concat_and_remux(
            chunk0_processed=chunk0_done,
            chunk1_processed=chunk1_done,
            original_video=video_path,
            job_id=job_id,
        )
        log_and_record(f"Final video remuxed successfully: {final_video_path.name}")

        # 5. Clean up temporary chunks
        with JOBS_LOCK:
            job.stage = "Cleaning up temporary chunk files"
            job.progress = 95
        cleanup_job_chunks(job_id)

        # 6. Finalize job
        with JOBS_LOCK:
            job.status = "completed"
            job.stage = "Face swap completed successfully"
            job.progress = 100
            job.output_filename = final_video_path.name
            job.completed_at = datetime.now(timezone.utc).isoformat()
        log_and_record("Job finalized successfully.")
        _save_jobs_to_disk(force=True)

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        terminate_job_processes(job_id)
        cleanup_job_chunks(job_id)
        with JOBS_LOCK:
            if job_id in _CANCELLED_JOBS:
                job.status = "cancelled"
                job.stage = "Job cancelled by user"
                job.error = "Cancelled"
            else:
                job.status = "failed"
                job.stage = f"Error: {e}"
                job.error = str(e)
            job.completed_at = datetime.now(timezone.utc).isoformat()
            _append_job_log(job, f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {job.stage}")
        _save_jobs_to_disk(force=True)


async def _job_worker_loop() -> None:
    """Background task processing jobs sequentially from FIFO queue."""
    while True:
        job_id = await _JOB_QUEUE.get()
        try:
            # Run pipeline in a background thread so the async event loop stays completely responsive
            await asyncio.to_thread(_run_single_job_pipeline, job_id)
        except Exception as e:
            logger.error(f"Worker queue task {job_id} error: {e}")
        finally:
            _JOB_QUEUE.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event context manager for application startup and shutdown."""
    init_workspace()
    _load_jobs_from_disk()
    worker_task = asyncio.create_task(_job_worker_loop())
    logger.info("FaceSwap Studio backend initialized with sequential job queue.")
    yield
    worker_task.cancel()
    _save_jobs_to_disk(force=True)
    logger.info("FaceSwap Studio backend shutdown.")


app = FastAPI(
    title="FaceSwap Studio API",
    description="Dual-GPU parallel FaceFusion pipeline API for Kaggle T4 x2",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS for local and remote studio frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/system/status")
async def get_system_status() -> dict[str, Any]:
    """Return system information, GPU detection, and model cache paths."""
    gpu_info = get_system_gpu_info()
    return {
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workspace_dir": str(WORKSPACE_DIR),
        "models_cache_dir": str(MODELS_CACHE_DIR),
        "facefusion_dir": str(FACEFUSION_DIR),
        "facefusion_installed": (FACEFUSION_DIR / "facefusion.py").exists(),
        "cuda_provider_ok": check_cuda_provider(),
        "max_upload_mb": MAX_UPLOAD_MB,
        "default_threads": FACESWAP_THREADS,
        **gpu_info,
    }


@app.get("/api/system/resources")
async def get_system_resources() -> dict[str, Any]:
    """Return live system RAM, workspace/tmp disk spaces, and GPU memory metrics."""
    return get_system_resources_data()


@app.get("/api/system/models")
async def get_system_models() -> dict[str, Any]:
    """Return model allowlists and default choices."""
    return {
        "swapper_models": ALLOWED_SWAPPER_MODELS,
        "default_swapper": DEFAULT_SWAPPER_MODEL,
        "enhancer_models": ALLOWED_ENHANCER_MODELS,
        "default_enhancer": DEFAULT_ENHANCER_MODEL,
        "default_blend": DEFAULT_ENHANCER_BLEND,
        "detector_models": ALLOWED_DETECTOR_MODELS,
        "default_detector": DEFAULT_DETECTOR_MODEL,
    }


@app.post("/api/preview/command")
async def preview_cli_command(
    swapper_model: str = Form(DEFAULT_SWAPPER_MODEL),
    enhancer_model: str = Form(DEFAULT_ENHANCER_MODEL),
    enhancer_blend: int = Form(DEFAULT_ENHANCER_BLEND),
    detector_model: str = Form(DEFAULT_DETECTOR_MODEL),
    pos_a: int = Form(0),
    pos_b: int = Form(1),
    ref_frame: int = Form(0),
    has_face_b: bool = Form(False),
    enhance_face_b: bool = Form(False),
) -> dict[str, Any]:
    """Generate preview FaceFusion CLI command string based on current studio settings."""
    dummy_source_a = Path("/kaggle/working/uploads/source_a.png")
    dummy_source_b = Path("/kaggle/working/uploads/source_b.png")
    dummy_chunk0 = Path("/kaggle/working/chunks/chunk0.mp4")
    dummy_out_pass1 = Path("/kaggle/working/chunks/chunk0_pass1.mp4")
    dummy_out_pass2 = Path("/kaggle/working/chunks/chunk0_pass2.mp4")

    cmd1 = build_facefusion_command(
        target_path=dummy_chunk0,
        output_path=dummy_out_pass1,
        source_face_path=dummy_source_a,
        reference_position=pos_a,
        reference_frame=ref_frame,
        swapper_model=swapper_model,
        enhancer_model=enhancer_model,
        enhancer_blend=enhancer_blend,
        detector_model=detector_model,
        execution_threads=FACESWAP_THREADS,
        video_memory_strategy="moderate",
    )

    cmd2 = None
    if has_face_b:
        pass2_enhancer = enhancer_model if enhance_face_b else None
        cmd2 = build_facefusion_command(
            target_path=dummy_out_pass1,
            output_path=dummy_out_pass2,
            source_face_path=dummy_source_b,
            reference_position=pos_b,
            reference_frame=ref_frame,
            swapper_model=swapper_model,
            enhancer_model=pass2_enhancer,
            enhancer_blend=enhancer_blend,
            detector_model=detector_model,
            execution_threads=FACESWAP_THREADS,
            video_memory_strategy="moderate",
        )

    return {
        "pass1_command": " ".join(cmd1),
        "pass2_command": " ".join(cmd2) if cmd2 else None,
        "gpu_allocation": "GPU 0 -> chunk0.mp4 | GPU 1 -> chunk1.mp4",
    }


async def _stream_upload_to_disk(upload: UploadFile, dest_path: Path, max_bytes: int) -> int:
    """Stream an uploaded file to disk in 1MB chunks asynchronously while enforcing size cap."""
    bytes_written = 0
    if HAS_AIOFILES and aiofiles is not None:
        async with aiofiles.open(dest_path, "wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    await f.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeded maximum allowed size of {MAX_UPLOAD_MB}MB (Cloudflare Tunnel limit is 100MB).",
                    )
                await f.write(chunk)
    else:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeded maximum allowed size of {MAX_UPLOAD_MB}MB (Cloudflare Tunnel limit is 100MB).",
                )
            with open(dest_path, "ab") as f:
                f.write(chunk)
    return bytes_written


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    source_face_a: UploadFile = File(...),
    source_face_b: Optional[UploadFile] = File(None),
    swapper_model: str = Form(DEFAULT_SWAPPER_MODEL),
    enhancer_model: str = Form(DEFAULT_ENHANCER_MODEL),
    enhancer_blend: int = Form(DEFAULT_ENHANCER_BLEND),
    detector_model: str = Form(DEFAULT_DETECTOR_MODEL),
    pos_a: int = Form(0),
    pos_b: int = Form(1),
    ref_frame: int = Form(0),
    enhance_face_b: bool = Form(False),
) -> dict[str, Any]:
    """Accept video and source face uploads asynchronously with 1MB streaming, enqueue job."""
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job_upload_dir = UPLOADS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    # Validate model parameters
    if swapper_model not in ALLOWED_SWAPPER_MODELS:
        swapper_model = DEFAULT_SWAPPER_MODEL
    if enhancer_model not in ALLOWED_ENHANCER_MODELS:
        enhancer_model = DEFAULT_ENHANCER_MODEL
    if detector_model not in ALLOWED_DETECTOR_MODELS:
        detector_model = DEFAULT_DETECTOR_MODEL
    enhancer_blend = max(40, min(100, int(enhancer_blend)))

    # Save target video (chunked streaming)
    video_ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_filename = f"target{video_ext}"
    video_save_path = job_upload_dir / video_filename
    await _stream_upload_to_disk(video, video_save_path, max_bytes)

    # Save source face A (chunked streaming)
    source_a_ext = Path(source_face_a.filename or "source_a.png").suffix or ".png"
    source_a_filename = f"source_a{source_a_ext}"
    source_a_save_path = job_upload_dir / source_a_filename
    await _stream_upload_to_disk(source_face_a, source_a_save_path, max_bytes)

    # Save source face B if present (chunked streaming)
    source_b_filename = None
    if source_face_b and source_face_b.filename:
        source_b_ext = Path(source_face_b.filename).suffix or ".png"
        source_b_filename = f"source_b{source_b_ext}"
        source_b_save_path = job_upload_dir / source_b_filename
        await _stream_upload_to_disk(source_face_b, source_b_save_path, max_bytes)

    record = JobRecord(
        job_id=job_id,
        status="queued",
        stage="Queued for dual-GPU processing",
        progress=0,
        video_filename=video_filename,
        source_a_filename=source_a_filename,
        source_b_filename=source_b_filename,
        swapper_model=swapper_model,
        enhancer_model=enhancer_model,
        enhancer_blend=enhancer_blend,
        detector_model=detector_model,
        pos_a=pos_a,
        pos_b=pos_b,
        ref_frame=ref_frame,
        enhance_face_b=enhance_face_b,
    )

    with JOBS_LOCK:
        JOBS[job_id] = record
    _save_jobs_to_disk(force=True)

    # Enqueue in single sequential worker queue
    await _JOB_QUEUE.put(job_id)

    # Calculate queue position
    with JOBS_LOCK:
        queued_ids = [j.job_id for j in JOBS.values() if j.status == "queued"]
        q_pos = queued_ids.index(job_id) + 1 if job_id in queued_ids else 1

    return {
        "job_id": job_id,
        "status": "queued",
        "stage": f"Queued (position {q_pos})",
        "queue_position": q_pos,
        "created_at": record.created_at,
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel an active or queued job, killing running subprocesses and cleaning workspace."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        _CANCELLED_JOBS.add(job_id)

        if job.status in ("queued", "processing"):
            job.status = "cancelled"
            job.stage = "Job cancelled by user"
            job.error = "Cancelled"
            job.completed_at = datetime.now(timezone.utc).isoformat()
            _append_job_log(job, f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Job cancelled by user request.")

    # Terminate child processes
    terminate_job_processes(job_id)
    cleanup_job_chunks(job_id)
    _save_jobs_to_disk(force=True)

    return {"job_id": job_id, "status": "cancelled", "message": "Job cancellation dispatched"}


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """Return list of recent jobs."""
    with JOBS_LOCK:
        jobs_snapshot = [copy.deepcopy(j) for j in JOBS.values()]

    return [
        {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "has_output": bool(job.output_filename),
        }
        for job in sorted(jobs_snapshot, key=lambda j: j.created_at, reverse=True)
    ]


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, since: Optional[int] = Query(None)) -> dict[str, Any]:
    """Poll job status, progress percentage, stage, and incremental GPU worker logs."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_copy = copy.deepcopy(job)

        # Calculate live queue position if queued
        queue_position = None
        if job_copy.status == "queued":
            queued_jobs = [j.job_id for j in JOBS.values() if j.status == "queued"]
            if job_id in queued_jobs:
                queue_position = queued_jobs.index(job_id) + 1

    total_logs = len(job_copy.logs)
    delta_logs = job_copy.logs[since:] if since is not None and 0 <= since <= total_logs else job_copy.logs

    result = asdict(job_copy)
    result["logs"] = delta_logs
    result["log_count"] = total_logs
    result["queue_position"] = queue_position
    return result


@app.get("/api/jobs/{job_id}/download")
async def download_job_output(job_id: str) -> FileResponse:
    """Download completed video result."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed" or not job.output_filename:
            raise HTTPException(status_code=400, detail="Job output is not ready")
        output_name = job.output_filename

    output_path = OUTPUTS_DIR / output_name
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file missing on server disk")

    return FileResponse(
        path=str(output_path),
        filename=f"faceswap_{job_id}.mp4",
        media_type="video/mp4",
    )


# Serve frontend static assets
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
