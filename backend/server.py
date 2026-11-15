"""FastAPI server and asynchronous background job manager for FaceSwap Studio.

Provides non-blocking job creation, background dual-GPU pipeline dispatch,
status polling with JSON persistence, model allowlist inspection, and final video downloads.
"""

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import asyncio
import json
import logging
import os
import shutil
import uuid

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
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
    JOBS_DIR,
    MODELS_CACHE_DIR,
    OUTPUTS_DIR,
    SERVER_HOST,
    SERVER_PORT,
    UPLOADS_DIR,
    WORKSPACE_DIR,
    init_workspace,
)
from backend.parallel_runner import execute_dual_gpu_pipeline
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
    status: str  # "queued", "processing", "completed", "failed"
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


# In-memory job registry
JOBS: dict[str, JobRecord] = {}
JOBS_LOCK = asyncio.Lock()


def _save_jobs_to_disk() -> None:
    """Save in-memory job records to JSON store file for persistence."""
    try:
        data = {job_id: asdict(job) for job_id, job in JOBS.items()}
        temp_path = JOBS_STORE_PATH.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(JOBS_STORE_PATH)
    except Exception as e:
        logger.warning(f"Failed to persist jobs.json: {e}")


def _load_jobs_from_disk() -> None:
    """Load job records from JSON store file on server initialization."""
    if not JOBS_STORE_PATH.exists():
        return
    try:
        with open(JOBS_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for job_id, job_dict in data.items():
            JOBS[job_id] = JobRecord(**job_dict)
        logger.info(f"Loaded {len(JOBS)} historical job records from {JOBS_STORE_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load jobs.json: {e}")


def get_system_gpu_info() -> dict[str, Any]:
    """Inspect system GPU count, driver, and CUDA capability."""
    gpu_count = 0
    gpu_devices: list[str] = []
    driver_version = "Unknown"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_devices = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except ImportError:
        pass

    # Fallback to nvidia-smi if torch is not installed or detected
    if gpu_count == 0:
        try:
            import subprocess
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = [line.strip() for line in proc.stdout.strip().split("\n") if line.strip()]
            gpu_count = len(lines)
            for line in lines:
                parts = line.split(",")
                if parts:
                    gpu_devices.append(parts[0].strip())
                    if len(parts) > 1:
                        driver_version = parts[1].strip()
        except Exception:
            pass

    return {
        "gpu_count": gpu_count,
        "gpu_devices": gpu_devices,
        "driver_version": driver_version,
        "dual_gpu_ready": gpu_count >= 2,
    }


def _run_background_pipeline(job_id: str) -> None:
    """Execute full dual-GPU pipeline synchronously in a worker thread."""
    job = JOBS.get(job_id)
    if not job:
        logger.error(f"Cannot run background pipeline; job {job_id} not registered.")
        return

    job_upload_dir = UPLOADS_DIR / job_id
    video_path = job_upload_dir / job.video_filename
    source_a_path = job_upload_dir / job.source_a_filename
    source_b_path = (job_upload_dir / job.source_b_filename) if job.source_b_filename else None

    def append_log(msg: str) -> None:
        t = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{t}] {msg}"
        job.logs.append(entry)
        logger.info(f"[{job_id}] {msg}")

    try:
        append_log("Starting background pipeline execution.")
        job.status = "processing"
        job.stage = "Probing video metadata"
        job.progress = 5
        _save_jobs_to_disk()

        # 1. Probe video
        meta = probe_video(video_path)
        job.duration_seconds = meta["duration"]
        append_log(
            f"Video probed: {meta['duration']:.2f}s, {meta['width']}x{meta['height']}, "
            f"{meta['fps']:.1f} fps, audio={meta['has_audio']}"
        )

        # 2. Split video into chunks
        job.stage = "Splitting video into 2 chunks (libx264)"
        job.progress = 15
        _save_jobs_to_disk()
        chunk0_path, chunk1_path = split_video_chunks(
            video_path=video_path,
            job_id=job_id,
            duration=meta["duration"],
        )
        append_log(f"Video chunked: {chunk0_path.name} and {chunk1_path.name}")

        # 3. Dual-GPU parallel processing
        job.stage = "Processing chunks concurrently on GPU 0 and GPU 1"
        job.progress = 30
        job.gpu_status["gpu0"] = {"status": "processing", "progress": 10}
        job.gpu_status["gpu1"] = {"status": "processing", "progress": 10}
        _save_jobs_to_disk()

        def on_gpu_progress(update: dict[str, Any]) -> None:
            if "gpu0" in update:
                job.gpu_status["gpu0"].update(update["gpu0"])
            if "gpu1" in update:
                job.gpu_status["gpu1"].update(update["gpu1"])
            _save_jobs_to_disk()

        chunk0_done, chunk1_done = execute_dual_gpu_pipeline(
            chunk0_path=chunk0_path,
            chunk1_path=chunk1_path,
            source_face_a=source_a_path,
            pos_a=job.pos_a,
            ref_frame=job.ref_frame,
            source_face_b=source_b_path,
            pos_b=job.pos_b,
            swapper_model=job.swapper_model,
            enhancer_model=job.enhancer_model,
            enhancer_blend=job.enhancer_blend,
            detector_model=job.detector_model,
            job_id=job_id,
            progress_callback=on_gpu_progress,
        )

        job.progress = 80
        job.gpu_status["gpu0"] = {"status": "completed", "progress": 100}
        job.gpu_status["gpu1"] = {"status": "completed", "progress": 100}
        append_log("Both GPU workers finished chunk processing.")

        # 4. Concatenate and remux original audio
        job.stage = "Concatenating chunks and restoring audio stream"
        job.progress = 85
        _save_jobs_to_disk()

        final_video_path = concat_and_remux(
            chunk0_processed=chunk0_done,
            chunk1_processed=chunk1_done,
            original_video=video_path,
            job_id=job_id,
        )
        append_log(f"Final video remuxed successfully: {final_video_path.name}")

        # 5. Clean up temporary chunks
        job.stage = "Cleaning up temporary chunk files"
        job.progress = 95
        cleanup_job_chunks(job_id)
        append_log("Temporary chunk files cleaned up.")

        # 6. Finalize job
        job.status = "completed"
        job.stage = "Face swap completed successfully"
        job.progress = 100
        job.output_filename = final_video_path.name
        job.completed_at = datetime.now(timezone.utc).isoformat()
        append_log("Job finalized successfully.")
        _save_jobs_to_disk()

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        append_log(f"Pipeline error: {e}")
        job.status = "failed"
        job.stage = f"Error: {e}"
        job.error = str(e)
        job.completed_at = datetime.now(timezone.utc).isoformat()
        # Clean up chunks on failure as well
        cleanup_job_chunks(job_id)
        _save_jobs_to_disk()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event context manager for application startup and shutdown."""
    init_workspace()
    _load_jobs_from_disk()
    logger.info("FaceSwap Studio backend initialized.")
    yield
    _save_jobs_to_disk()
    logger.info("FaceSwap Studio backend shutdown.")


app = FastAPI(
    title="FaceSwap Studio API",
    description="Dual-GPU parallel FaceFusion pipeline API for Kaggle T4 x2",
    version="1.0.0",
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
        **gpu_info,
    }


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
    )

    cmd2 = None
    if has_face_b:
        cmd2 = build_facefusion_command(
            target_path=dummy_out_pass1,
            output_path=dummy_out_pass2,
            source_face_path=dummy_source_b,
            reference_position=pos_b,
            reference_frame=ref_frame,
            swapper_model=swapper_model,
            enhancer_model=enhancer_model,
            enhancer_blend=enhancer_blend,
            detector_model=detector_model,
        )

    return {
        "pass1_command": " ".join(cmd1),
        "pass2_command": " ".join(cmd2) if cmd2 else None,
        "gpu_allocation": "GPU 0 -> chunk0.mp4 | GPU 1 -> chunk1.mp4",
    }


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
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
) -> dict[str, Any]:
    """Accept video and source face uploads, enqueue background job, return immediately.

    Prevents HTTP timeout on Cloudflare Quick Tunnels by returning instantly with 200 OK.
    """
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job_upload_dir = UPLOADS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    # Validate model parameters
    if swapper_model not in ALLOWED_SWAPPER_MODELS:
        swapper_model = DEFAULT_SWAPPER_MODEL
    if enhancer_model not in ALLOWED_ENHANCER_MODELS:
        enhancer_model = DEFAULT_ENHANCER_MODEL
    if detector_model not in ALLOWED_DETECTOR_MODELS:
        detector_model = DEFAULT_DETECTOR_MODEL
    enhancer_blend = max(40, min(100, int(enhancer_blend)))

    # Save target video
    video_ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_filename = f"target{video_ext}"
    video_save_path = job_upload_dir / video_filename
    with open(video_save_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Save source face A
    source_a_ext = Path(source_face_a.filename or "source_a.png").suffix or ".png"
    source_a_filename = f"source_a{source_a_ext}"
    source_a_save_path = job_upload_dir / source_a_filename
    with open(source_a_save_path, "wb") as f:
        shutil.copyfileobj(source_face_a.file, f)

    # Save source face B if present
    source_b_filename = None
    if source_face_b and source_face_b.filename:
        source_b_ext = Path(source_face_b.filename).suffix or ".png"
        source_b_filename = f"source_b{source_b_ext}"
        source_b_save_path = job_upload_dir / source_b_filename
        with open(source_b_save_path, "wb") as f:
            shutil.copyfileobj(source_face_b.file, f)

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
    )

    JOBS[job_id] = record
    _save_jobs_to_disk()

    # Enqueue background execution in thread pool
    background_tasks.add_task(_run_background_pipeline, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "stage": record.stage,
        "created_at": record.created_at,
    }


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """Return list of recent jobs."""
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
        for job in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)
    ]


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll job status, progress percentage, stage, and GPU worker logs."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)


@app.get("/api/jobs/{job_id}/download")
async def download_job_output(job_id: str) -> FileResponse:
    """Download completed video result."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "completed" or not job.output_filename:
        raise HTTPException(status_code=400, detail="Job output is not ready")

    output_path = OUTPUTS_DIR / job.output_filename
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
