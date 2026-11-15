"""Dual-GPU parallel runner for FaceSwap Studio.

Spawns worker threads via concurrent.futures.ThreadPoolExecutor(max_workers=2)
binding GPU 0 to chunk0.mp4 and GPU 1 to chunk1.mp4 using CUDA_VISIBLE_DEVICES.
Executes Pass 1 (Source Face A) and optional Pass 2 (Source Face B) in sequence per GPU.
Supports live progress parsing, process lifecycle tracking, OOM auto-retry, and eager cleanup.
Compatible with Python 3.10+.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional
import logging
import os
import re
import shutil
import subprocess
import threading
import time

from backend.config import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_ENHANCER_BLEND,
    DEFAULT_ENHANCER_MODEL,
    DEFAULT_SWAPPER_MODEL,
    FACEFUSION_DIR,
    FACESWAP_THREADS,
    PASS_TIMEOUT_SECONDS,
)
from backend.swap_pipeline import build_facefusion_command, get_facefusion_help_info

logger = logging.getLogger("faceswap.runner")

# Process tracking for child termination and cancellation
_ACTIVE_PROCESSES_LOCK = threading.Lock()
_ACTIVE_PROCESSES: dict[str, list[subprocess.Popen]] = {}


def register_job_process(job_id: str, proc: subprocess.Popen) -> None:
    """Register active child process for a job."""
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES.setdefault(job_id, []).append(proc)


def unregister_job_process(job_id: str, proc: subprocess.Popen) -> None:
    """Unregister terminated child process for a job."""
    with _ACTIVE_PROCESSES_LOCK:
        if job_id in _ACTIVE_PROCESSES:
            _ACTIVE_PROCESSES[job_id] = [p for p in _ACTIVE_PROCESSES[job_id] if p != proc]
            if not _ACTIVE_PROCESSES[job_id]:
                del _ACTIVE_PROCESSES[job_id]


def terminate_job_processes(job_id: str) -> None:
    """Terminate and kill all active child processes for a job."""
    with _ACTIVE_PROCESSES_LOCK:
        procs = list(_ACTIVE_PROCESSES.get(job_id, []))

    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    time.sleep(0.5)
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


def _is_oom_failure(returncode: int, output_tail: str) -> bool:
    """Check whether process exit code or logs indicate an out-of-memory kill."""
    if returncode in (-9, 137):
        return True
    lower = output_tail.lower()
    return any(sig in lower for sig in ["out of memory", "cuda out of memory", "killed", "cuda error: out of memory"])


def _is_tqdm_spam(line: str) -> bool:
    """Detect repetitive progress bar stdout lines to avoid log explosion."""
    if "%|" in line or "it/s" in line or "s/it" in line:
        return True
    # Strip carriage returns or progress-only updates
    if re.match(r"^\s*\d{1,3}%\s*\|", line):
        return True
    return False


def _run_single_pass_command(
    cmd: list[str],
    worker_env: dict[str, str],
    job_id: str,
    gpu_id: int,
    chunk_name: str,
    pass_label: str,
    output_path: Path,
    has_face_b: bool,
    is_pass2: bool,
    log_fn: Callable[[str], None],
    progress_cb: Optional[Callable[[dict[str, Any]], None]],
    timeout_seconds: int = PASS_TIMEOUT_SECONDS,
) -> tuple[bool, str, deque[str]]:
    """Execute a single FaceFusion CLI pass with live line streaming and progress reporting."""
    tail_lines: deque[str] = deque(maxlen=300)
    proc: Optional[subprocess.Popen] = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(FACEFUSION_DIR),
            env=worker_env,
        )
        register_job_process(job_id, proc)

        last_progress_time = 0.0
        start_time = time.time()

        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    proc.kill()
                    return False, f"{pass_label} timed out after {timeout_seconds}s", tail_lines

                cleaned = line.strip()
                if not cleaned:
                    continue

                tail_lines.append(cleaned)
                now = time.time()

                # Parse percentage progress from CLI line
                m = re.search(r"(\d{1,3})%", cleaned)
                if m:
                    raw_pct = min(100, max(0, int(m.group(1))))
                    if not is_pass2:
                        overall_pct = int(raw_pct * 0.5) if has_face_b else raw_pct
                    else:
                        overall_pct = 50 + int(raw_pct * 0.5)

                    if progress_cb and (now - last_progress_time >= 0.8 or raw_pct in (0, 50, 100)):
                        last_progress_time = now
                        progress_cb({
                            f"gpu{gpu_id}": {
                                "status": "processing",
                                "progress": overall_pct,
                                "stage": f"{pass_label} ({raw_pct}%)",
                            }
                        })

                # Forward non-tqdm lines to job log
                if not _is_tqdm_spam(cleaned):
                    log_fn(f"[FaceFusion {pass_label}] {cleaned}")
                    if progress_cb and (now - last_progress_time >= 1.0):
                        progress_cb({"logs": [f"[{time.strftime('%H:%M:%S')}] [GPU {gpu_id}] [{chunk_name}] {cleaned}"]})

            proc.stdout.close()

        retcode = proc.wait(timeout=30)
        if retcode != 0:
            tail_str = "\n".join(tail_lines)
            if _is_oom_failure(retcode, tail_str):
                return False, f"OOM: {pass_label} on GPU {gpu_id} killed by OOM (code {retcode})", tail_lines
            return False, f"{pass_label} on GPU {gpu_id} failed with exit code {retcode}: {tail_str[-500:]}", tail_lines

        if not output_path.exists() or output_path.stat().st_size == 0:
            return False, f"{pass_label} finished but output file missing or empty: {output_path.name}", tail_lines

        return True, "", tail_lines

    except Exception as e:
        if proc and proc.poll() is None:
            proc.kill()
        return False, f"{pass_label} execution error: {e}", tail_lines
    finally:
        if proc:
            unregister_job_process(job_id, proc)


def _worker_gpu_task(
    gpu_id: int,
    chunk_path_str: str,
    source_face_a_str: str,
    pos_a: int,
    ref_frame: int,
    source_face_b_str: Optional[str],
    pos_b: int,
    enhance_face_b: bool,
    swapper_model: str,
    enhancer_model: str,
    enhancer_blend: int,
    detector_model: str,
    job_id: str,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Execute Pass 1 (Face A) and optional Pass 2 (Face B) inside a dedicated thread.

    Provides isolated TMPDIR, live progress callbacks, OOM single auto-retry, and eager chunk cleanup.
    """
    chunk_path = Path(chunk_path_str)
    source_a = Path(source_face_a_str)
    source_b = Path(source_face_b_str) if source_face_b_str else None
    has_face_b = bool(source_b and source_b.exists())

    chunk_dir = chunk_path.parent
    chunk_name = chunk_path.stem
    pass1_output = chunk_dir / f"{chunk_name}_pass1.mp4"
    pass2_output = chunk_dir / f"{chunk_name}_pass2.mp4"

    # Per-worker isolated temporary directory
    worker_tmp = chunk_dir / f"tmp_gpu{gpu_id}"
    worker_tmp.mkdir(parents=True, exist_ok=True)

    worker_env = os.environ.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker_env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    worker_env["PYTHONUNBUFFERED"] = "1"
    worker_env["TMPDIR"] = str(worker_tmp)
    worker_env["PYTHONPATH"] = f"{str(FACEFUSION_DIR)}:{worker_env.get('PYTHONPATH', '')}"

    logs: list[str] = []

    def log(msg: str) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] [GPU {gpu_id}] [{chunk_name}] {msg}"
        logs.append(entry)
        logger.info(entry)

    log(f"Worker initialized on GPU {gpu_id} (TMPDIR={worker_tmp.name}). Starting Pass 1.")

    try:
        # 1. Build and run Pass 1
        cmd_pass1 = build_facefusion_command(
            target_path=chunk_path,
            output_path=pass1_output,
            source_face_path=source_a,
            reference_position=pos_a,
            reference_frame=ref_frame,
            swapper_model=swapper_model,
            enhancer_model=enhancer_model,
            enhancer_blend=enhancer_blend,
            detector_model=detector_model,
            execution_threads=FACESWAP_THREADS,
            video_memory_strategy="moderate",
        )
        log(f"Pass 1 command: {' '.join(cmd_pass1)}")

        success1, err1, tail1 = _run_single_pass_command(
            cmd=cmd_pass1,
            worker_env=worker_env,
            job_id=job_id,
            gpu_id=gpu_id,
            chunk_name=chunk_name,
            pass_label="Pass 1",
            output_path=pass1_output,
            has_face_b=has_face_b,
            is_pass2=False,
            log_fn=log,
            progress_cb=progress_callback,
        )

        # OOM auto-retry for Pass 1
        if not success1 and "OOM" in err1:
            log("Pass 1 encountered OOM. Retrying ONCE with --execution-thread-count 1 and --video-memory-strategy strict...")
            cmd_pass1_retry = build_facefusion_command(
                target_path=chunk_path,
                output_path=pass1_output,
                source_face_path=source_a,
                reference_position=pos_a,
                reference_frame=ref_frame,
                swapper_model=swapper_model,
                enhancer_model=enhancer_model,
                enhancer_blend=enhancer_blend,
                detector_model=detector_model,
                execution_threads=1,
                video_memory_strategy="strict",
            )
            success1, err1, tail1 = _run_single_pass_command(
                cmd=cmd_pass1_retry,
                worker_env=worker_env,
                job_id=job_id,
                gpu_id=gpu_id,
                chunk_name=chunk_name,
                pass_label="Pass 1 (OOM Retry)",
                output_path=pass1_output,
                has_face_b=has_face_b,
                is_pass2=False,
                log_fn=log,
                progress_cb=progress_callback,
            )

        if not success1:
            log(f"Pass 1 failed: {err1}")
            return {
                "gpu_id": gpu_id,
                "success": False,
                "output_path": None,
                "error": err1,
                "logs": logs,
            }

        log(f"Pass 1 completed successfully: {pass1_output.name}")

        # Eager cleanup: Delete original raw chunk to immediately save disk
        try:
            if chunk_path.exists():
                chunk_path.unlink(missing_ok=True)
                log(f"Cleaned up input chunk file {chunk_path.name} to free disk space.")
        except Exception as e:
            logger.warning(f"Could not remove input chunk {chunk_path}: {e}")

        final_output = pass1_output

        # 2. Pass 2 (Source Face B)
        if has_face_b and source_b:
            log(f"Starting Pass 2 for Face B (position {pos_b}, enhance={enhance_face_b}).")
            pass2_enhancer = enhancer_model if enhance_face_b else None

            cmd_pass2 = build_facefusion_command(
                target_path=pass1_output,
                output_path=pass2_output,
                source_face_path=source_b,
                reference_position=pos_b,
                reference_frame=ref_frame,
                swapper_model=swapper_model,
                enhancer_model=pass2_enhancer,
                enhancer_blend=enhancer_blend,
                detector_model=detector_model,
                execution_threads=FACESWAP_THREADS,
                video_memory_strategy="moderate",
            )
            log(f"Pass 2 command: {' '.join(cmd_pass2)}")

            success2, err2, tail2 = _run_single_pass_command(
                cmd=cmd_pass2,
                worker_env=worker_env,
                job_id=job_id,
                gpu_id=gpu_id,
                chunk_name=chunk_name,
                pass_label="Pass 2",
                output_path=pass2_output,
                has_face_b=True,
                is_pass2=True,
                log_fn=log,
                progress_cb=progress_callback,
            )

            # OOM auto-retry for Pass 2
            if not success2 and "OOM" in err2:
                log("Pass 2 encountered OOM. Retrying ONCE with thread count 1 and strict memory strategy...")
                cmd_pass2_retry = build_facefusion_command(
                    target_path=pass1_output,
                    output_path=pass2_output,
                    source_face_path=source_b,
                    reference_position=pos_b,
                    reference_frame=ref_frame,
                    swapper_model=swapper_model,
                    enhancer_model=pass2_enhancer,
                    enhancer_blend=enhancer_blend,
                    detector_model=detector_model,
                    execution_threads=1,
                    video_memory_strategy="strict",
                )
                success2, err2, tail2 = _run_single_pass_command(
                    cmd=cmd_pass2_retry,
                    worker_env=worker_env,
                    job_id=job_id,
                    gpu_id=gpu_id,
                    chunk_name=chunk_name,
                    pass_label="Pass 2 (OOM Retry)",
                    output_path=pass2_output,
                    has_face_b=True,
                    is_pass2=True,
                    log_fn=log,
                    progress_cb=progress_callback,
                )

            if not success2:
                log(f"Pass 2 failed: {err2}")
                return {
                    "gpu_id": gpu_id,
                    "success": False,
                    "output_path": None,
                    "error": err2,
                    "logs": logs,
                }

            log(f"Pass 2 completed successfully: {pass2_output.name}")

            # Eager cleanup: Delete pass1 output now that pass2 is complete
            try:
                if pass1_output.exists():
                    pass1_output.unlink(missing_ok=True)
                    log(f"Cleaned up intermediate file {pass1_output.name} to free disk space.")
            except Exception as e:
                logger.warning(f"Could not remove intermediate pass1 file {pass1_output}: {e}")

            final_output = pass2_output

        return {
            "gpu_id": gpu_id,
            "success": True,
            "output_path": str(final_output),
            "error": None,
            "logs": logs,
        }

    finally:
        # Clean up worker temp directory
        try:
            shutil.rmtree(worker_tmp, ignore_errors=True)
        except Exception:
            pass


def execute_dual_gpu_pipeline(
    chunk0_path: Path,
    chunk1_path: Path,
    source_face_a: Path,
    pos_a: int,
    ref_frame: int,
    source_face_b: Optional[Path] = None,
    pos_b: int = 1,
    enhance_face_b: bool = False,
    swapper_model: str = DEFAULT_SWAPPER_MODEL,
    enhancer_model: str = DEFAULT_ENHANCER_MODEL,
    enhancer_blend: int = DEFAULT_ENHANCER_BLEND,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    job_id: str = "default_job",
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> tuple[Path, Path]:
    """Orchestrate dual-GPU parallel execution across GPU 0 and GPU 1.

    Uses ThreadPoolExecutor(max_workers=2) to avoid unsafe process forking.
    If any worker fails, immediately terminates the sibling worker's process and raises.
    """
    logger.info(
        f"Initiating dual-GPU parallel pipeline for job {job_id} on CUDA devices 0 and 1."
    )

    if progress_callback:
        progress_callback({
            "stage": "Dual-GPU processing",
            "gpu0": {"status": "processing", "progress": 5},
            "gpu1": {"status": "processing", "progress": 5},
        })

    tasks = [
        (
            0,
            str(chunk0_path),
            str(source_face_a),
            pos_a,
            ref_frame,
            str(source_face_b) if source_face_b else None,
            pos_b,
            enhance_face_b,
            swapper_model,
            enhancer_model,
            enhancer_blend,
            detector_model,
            job_id,
            progress_callback,
        ),
        (
            1,
            str(chunk1_path),
            str(source_face_a),
            pos_a,
            ref_frame,
            str(source_face_b) if source_face_b else None,
            pos_b,
            enhance_face_b,
            swapper_model,
            enhancer_model,
            enhancer_blend,
            detector_model,
            job_id,
            progress_callback,
        ),
    ]

    results: dict[int, dict[str, Any]] = {}
    failed_gpu: Optional[int] = None
    failure_error: Optional[str] = None

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_gpu = {
                executor.submit(_worker_gpu_task, *task): task[0]
                for task in tasks
            }

            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    res = future.result()
                    results[gpu_id] = res

                    if not res.get("success"):
                        failed_gpu = gpu_id
                        failure_error = res.get("error", "Unknown worker failure")
                        # Terminate sibling process immediately!
                        logger.error(f"Worker on GPU {gpu_id} failed. Aborting pipeline and terminating sibling processes.")
                        terminate_job_processes(job_id)
                        break

                    if progress_callback:
                        progress_callback({
                            f"gpu{gpu_id}": {
                                "status": "completed",
                                "progress": 100,
                                "output": res.get("output_path"),
                            },
                            "logs": res.get("logs", []),
                        })

                except Exception as e:
                    failed_gpu = gpu_id
                    failure_error = str(e)
                    logger.error(f"Worker on GPU {gpu_id} encountered exception: {e}")
                    terminate_job_processes(job_id)
                    break

    finally:
        # Guarantee no lingering processes remain
        if failed_gpu is not None:
            terminate_job_processes(job_id)

    if failed_gpu is not None or failure_error is not None:
        raise RuntimeError(f"Dual-GPU pipeline failed on GPU {failed_gpu}: {failure_error}")

    for gpu_id in (0, 1):
        worker_res = results.get(gpu_id)
        if not worker_res or not worker_res.get("success") or not worker_res.get("output_path"):
            raise RuntimeError(f"Dual-GPU pipeline missing output for GPU {gpu_id}")

    return Path(results[0]["output_path"]), Path(results[1]["output_path"])
