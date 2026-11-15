"""Dual-GPU parallel runner for FaceSwap Studio.

Spawns isolated worker processes via concurrent.futures.ProcessPoolExecutor(max_workers=2)
binding GPU 0 to chunk0.mp4 and GPU 1 to chunk1.mp4 using CUDA_VISIBLE_DEVICES.
Executes Pass 1 (Source Face A) and optional Pass 2 (Source Face B) in sequence per GPU.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional
import logging
import os
import subprocess
import time

from backend.config import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_ENHANCER_BLEND,
    DEFAULT_ENHANCER_MODEL,
    DEFAULT_SWAPPER_MODEL,
)
from backend.swap_pipeline import build_facefusion_command

logger = logging.getLogger("faceswap.runner")


def _worker_gpu_task(
    gpu_id: int,
    chunk_path_str: str,
    source_face_a_str: str,
    pos_a: int,
    ref_frame: int,
    source_face_b_str: Optional[str],
    pos_b: int,
    swapper_model: str,
    enhancer_model: str,
    enhancer_blend: int,
    detector_model: str,
    job_id: str,
) -> dict[str, Any]:
    """Isolated worker function executed inside a ProcessPoolExecutor worker process.

    Pins the process to the specified GPU device via CUDA_VISIBLE_DEVICES,
    runs Pass 1 for Source Face A, and conditionally runs Pass 2 for Source Face B.
    """
    # Enforce GPU isolation for this worker process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    chunk_path = Path(chunk_path_str)
    source_a = Path(source_face_a_str)
    source_b = Path(source_face_b_str) if source_face_b_str else None

    chunk_dir = chunk_path.parent
    chunk_name = chunk_path.stem  # e.g. "chunk0" or "chunk1"

    pass1_output = chunk_dir / f"{chunk_name}_pass1.mp4"
    logs: list[str] = []

    def log(msg: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] [GPU {gpu_id}] [{chunk_name}] {msg}"
        logs.append(entry)
        logger.info(entry)

    log(f"Worker initiated on GPU {gpu_id}. Starting Pass 1 for Source Face A.")

    # 1. Build and run Pass 1 (Face A)
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
    )

    log(f"Pass 1 command: {' '.join(cmd_pass1)}")

    proc1 = subprocess.Popen(
        cmd_pass1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if proc1.stdout:
        for line in proc1.stdout:
            cleaned = line.strip()
            if cleaned:
                log(f"[FaceFusion] {cleaned}")

    proc1.wait()
    if proc1.returncode != 0:
        error_msg = f"Pass 1 on GPU {gpu_id} failed with exit code {proc1.returncode}"
        log(error_msg)
        return {
            "gpu_id": gpu_id,
            "success": False,
            "output_path": None,
            "error": error_msg,
            "logs": logs,
        }

    if not pass1_output.exists():
        error_msg = f"Pass 1 output file missing: {pass1_output}"
        log(error_msg)
        return {
            "gpu_id": gpu_id,
            "success": False,
            "output_path": None,
            "error": error_msg,
            "logs": logs,
        }

    log(f"Pass 1 completed successfully: {pass1_output.name}")
    final_output = pass1_output

    # 2. Conditionally run Pass 2 (Face B)
    if source_b and source_b.exists():
        pass2_output = chunk_dir / f"{chunk_name}_pass2.mp4"
        log(f"Starting Pass 2 for Source Face B using position {pos_b}.")

        cmd_pass2 = build_facefusion_command(
            target_path=pass1_output,
            output_path=pass2_output,
            source_face_path=source_b,
            reference_position=pos_b,
            reference_frame=ref_frame,
            swapper_model=swapper_model,
            enhancer_model=enhancer_model,
            enhancer_blend=enhancer_blend,
            detector_model=detector_model,
        )

        log(f"Pass 2 command: {' '.join(cmd_pass2)}")

        proc2 = subprocess.Popen(
            cmd_pass2,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if proc2.stdout:
            for line in proc2.stdout:
                cleaned = line.strip()
                if cleaned:
                    log(f"[FaceFusion Pass 2] {cleaned}")

        proc2.wait()
        if proc2.returncode != 0:
            error_msg = f"Pass 2 on GPU {gpu_id} failed with exit code {proc2.returncode}"
            log(error_msg)
            return {
                "gpu_id": gpu_id,
                "success": False,
                "output_path": None,
                "error": error_msg,
                "logs": logs,
            }

        if not pass2_output.exists():
            error_msg = f"Pass 2 output file missing: {pass2_output}"
            log(error_msg)
            return {
                "gpu_id": gpu_id,
                "success": False,
                "output_path": None,
                "error": error_msg,
                "logs": logs,
            }

        log(f"Pass 2 completed successfully: {pass2_output.name}")
        final_output = pass2_output

    return {
        "gpu_id": gpu_id,
        "success": True,
        "output_path": str(final_output),
        "error": None,
        "logs": logs,
    }


def execute_dual_gpu_pipeline(
    chunk0_path: Path,
    chunk1_path: Path,
    source_face_a: Path,
    pos_a: int,
    ref_frame: int,
    source_face_b: Optional[Path] = None,
    pos_b: int = 1,
    swapper_model: str = DEFAULT_SWAPPER_MODEL,
    enhancer_model: str = DEFAULT_ENHANCER_MODEL,
    enhancer_blend: int = DEFAULT_ENHANCER_BLEND,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    job_id: str = "default_job",
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> tuple[Path, Path]:
    """Orchestrate dual-GPU parallel execution across GPU 0 and GPU 1.

    Runs chunk0.mp4 on GPU 0 and chunk1.mp4 on GPU 1 simultaneously using
    concurrent.futures.ProcessPoolExecutor(max_workers=2).
    """
    logger.info(
        f"Initiating dual-GPU parallel pipeline for job {job_id} on CUDA devices 0 and 1."
    )

    if progress_callback:
        progress_callback({
            "stage": "Dual-GPU processing",
            "gpu0": {"status": "processing", "progress": 10},
            "gpu1": {"status": "processing", "progress": 10},
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
            swapper_model,
            enhancer_model,
            enhancer_blend,
            detector_model,
            job_id,
        ),
        (
            1,
            str(chunk1_path),
            str(source_face_a),
            pos_a,
            ref_frame,
            str(source_face_b) if source_face_b else None,
            pos_b,
            swapper_model,
            enhancer_model,
            enhancer_blend,
            detector_model,
            job_id,
        ),
    ]

    results: dict[int, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=2) as executor:
        future_to_gpu = {
            executor.submit(_worker_gpu_task, *task): task[0]
            for task in tasks
        }

        for future in as_completed(future_to_gpu):
            gpu_id = future_to_gpu[future]
            try:
                res = future.result()
                results[gpu_id] = res
                logger.info(f"Worker for GPU {gpu_id} completed with success={res.get('success')}")

                if progress_callback:
                    status_str = "completed" if res.get("success") else "failed"
                    progress_callback({
                        f"gpu{gpu_id}": {
                            "status": status_str,
                            "progress": 100 if res.get("success") else 0,
                            "output": res.get("output_path"),
                            "error": res.get("error"),
                        }
                    })

            except Exception as e:
                logger.error(f"Worker on GPU {gpu_id} raised unhandled exception: {e}")
                results[gpu_id] = {
                    "gpu_id": gpu_id,
                    "success": False,
                    "output_path": None,
                    "error": str(e),
                    "logs": [f"Exception: {e}"],
                }

    # Verify both workers succeeded
    for gpu_id in (0, 1):
        worker_result = results.get(gpu_id)
        if not worker_result or not worker_result.get("success"):
            error_detail = worker_result.get("error") if worker_result else "No result returned"
            raise RuntimeError(f"Dual-GPU pipeline failed on GPU {gpu_id}: {error_detail}")

    final_chunk0 = Path(results[0]["output_path"])
    final_chunk1 = Path(results[1]["output_path"])

    return final_chunk0, final_chunk1

