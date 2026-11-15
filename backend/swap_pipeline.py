"""Face swap pipeline for video probing, frame-accurate chunking, CLI command building, and audio remuxing.

Handles FFmpeg chunk generation, FaceFusion CLI argument formatting for Pass 1 (Face A)
and Pass 2 (Face B), chunk concatenation via concat demuxer, original audio restoration,
and automatic workspace cleanup.
Compatible with Python 3.10+.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
import logging
import re
import shutil
import subprocess
import threading

from backend.config import (
    ALLOWED_DETECTOR_MODELS,
    ALLOWED_ENHANCER_MODELS,
    ALLOWED_SWAPPER_MODELS,
    CHUNKS_DIR,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_ENHANCER_BLEND,
    DEFAULT_ENHANCER_MODEL,
    DEFAULT_SWAPPER_MODEL,
    FACEFUSION_DIR,
    FACESWAP_THREADS,
    OUTPUTS_DIR,
    get_facefusion_cmd,
)

logger = logging.getLogger("faceswap.pipeline")


class PipelineError(Exception):
    """Exception raised for pipeline processing errors."""
    pass


_HELP_CACHE_LOCK = threading.Lock()
_FACEFUSION_HELP_INFO: Optional[dict[str, Any]] = None


def get_facefusion_help_info() -> dict[str, Any]:
    """Inspect and cache supported FaceFusion headless-run CLI flags and choices."""
    global _FACEFUSION_HELP_INFO
    with _HELP_CACHE_LOCK:
        if _FACEFUSION_HELP_INFO is not None:
            return _FACEFUSION_HELP_INFO

        info: dict[str, Any] = {
            "available": False,
            "has_temp_frame_format": True,
            "temp_frame_format_choice": "jpeg",
            "has_video_memory_strategy": True,
            "has_execution_thread_count": True,
            "has_face_swapper_pixel_boost": True,
            "raw_help": "",
        }

        entrypoint = FACEFUSION_DIR / "facefusion.py"
        if not entrypoint.exists():
            _FACEFUSION_HELP_INFO = info
            return info

        try:
            cmd = get_facefusion_cmd() + ["headless-run", "--help"]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(FACEFUSION_DIR),
                timeout=15,
            )
            help_text = (res.stdout or "") + (res.stderr or "")
            info["raw_help"] = help_text
            info["available"] = res.returncode == 0 or len(help_text) > 100

            info["has_temp_frame_format"] = "--temp-frame-format" in help_text
            if info["has_temp_frame_format"]:
                # Check whether choice is jpeg or jpg
                match_tff = re.search(r"--temp-frame-format\s*\{([^}]+)\}", help_text)
                if match_tff:
                    choices = [c.strip() for c in match_tff.group(1).split(",")]
                    if "jpeg" in choices:
                        info["temp_frame_format_choice"] = "jpeg"
                    elif "jpg" in choices:
                        info["temp_frame_format_choice"] = "jpg"
                    elif "png" in choices:
                        info["temp_frame_format_choice"] = "png"
                elif "jpeg" in help_text:
                    info["temp_frame_format_choice"] = "jpeg"
                elif "jpg" in help_text:
                    info["temp_frame_format_choice"] = "jpg"

            info["has_video_memory_strategy"] = "--video-memory-strategy" in help_text
            info["has_execution_thread_count"] = "--execution-thread-count" in help_text
            info["has_face_swapper_pixel_boost"] = "--face-swapper-pixel-boost" in help_text

            logger.info(
                f"FaceFusion CLI capabilities probed: temp_fmt={info['temp_frame_format_choice']} "
                f"(supported={info['has_temp_frame_format']}), "
                f"vmem_strategy={info['has_video_memory_strategy']}, "
                f"threads={info['has_execution_thread_count']}, "
                f"pixel_boost={info['has_face_swapper_pixel_boost']}"
            )
        except Exception as e:
            logger.warning(f"Could not probe FaceFusion help text: {e}")

        _FACEFUSION_HELP_INFO = info
        return _FACEFUSION_HELP_INFO


def probe_video(video_path: Path) -> dict[str, Any]:
    """Probe video metadata using ffprobe.

    Returns duration in seconds, frame count, resolution, fps, and audio track presence.
    """
    if not video_path.exists():
        raise PipelineError(f"Target video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration:stream=width,height,r_frame_rate,nb_frames,codec_type,codec_name",
        "-of", "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe execution failed: {e.stderr}")
        raise PipelineError(f"Failed to probe video: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe JSON output: {e}")
        raise PipelineError("Invalid ffprobe metadata response")

    duration = 0.0
    if "format" in data and "duration" in data["format"]:
        try:
            duration = float(data["format"]["duration"])
        except ValueError:
            duration = 0.0

    width = 0
    height = 0
    fps = 30.0
    has_audio = False
    nb_frames = 0

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and width == 0:
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            nb_frames_str = stream.get("nb_frames")
            if nb_frames_str and nb_frames_str.isdigit():
                nb_frames = int(nb_frames_str)

            r_frame_rate = stream.get("r_frame_rate", "30/1")
            if "/" in r_frame_rate:
                num, den = r_frame_rate.split("/", 1)
                try:
                    fps = float(num) / float(den) if float(den) != 0 else 30.0
                except ValueError:
                    fps = 30.0
        elif codec_type == "audio":
            has_audio = True

    if duration <= 0.0 and nb_frames > 0 and fps > 0:
        duration = nb_frames / fps

    if duration <= 0.0:
        raise PipelineError("Could not determine valid video duration.")

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "nb_frames": nb_frames,
        "has_audio": has_audio,
    }


def split_video_chunks(video_path: Path, job_id: str, duration: float) -> tuple[Path, Path]:
    """Split input video into two frame-accurate video-only chunks (chunk0.mp4, chunk1.mp4).

    Positions -ss and -to as output options (after -i) for frame-accurate seeking,
    encodes with -c:v libx264 -crf 16 -preset fast -pix_fmt yuv420p -an,
    and verifies that both generated chunks have > 0 frames.
    """
    job_chunks_dir = CHUNKS_DIR / job_id
    job_chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk0_path = job_chunks_dir / "chunk0.mp4"
    chunk1_path = job_chunks_dir / "chunk1.mp4"

    midpoint = duration / 2.0

    # Chunk 0: from 0.0 to midpoint
    cmd_chunk0 = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ss", "0.0",
        "-to", f"{midpoint:.3f}",
        "-an",
        "-c:v", "libx264",
        "-crf", "16",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(chunk0_path),
    ]

    # Chunk 1: from midpoint to end
    cmd_chunk1 = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ss", f"{midpoint:.3f}",
        "-to", f"{duration:.3f}",
        "-an",
        "-c:v", "libx264",
        "-crf", "16",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(chunk1_path),
    ]

    try:
        logger.info(f"Generating chunk0 (0.00s to {midpoint:.2f}s) for job {job_id}")
        subprocess.run(cmd_chunk0, capture_output=True, text=True, check=True)

        logger.info(f"Generating chunk1 ({midpoint:.2f}s to {duration:.2f}s) for job {job_id}")
        subprocess.run(cmd_chunk1, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg chunk split failed: {e.stderr}")
        raise PipelineError(f"Failed to split video into chunks: {e.stderr}")

    if not chunk0_path.exists() or not chunk1_path.exists():
        raise PipelineError("One or both video chunks failed to write to disk.")

    # Frame-accuracy verification: ensure both chunks have valid frames
    meta0 = probe_video(chunk0_path)
    meta1 = probe_video(chunk1_path)
    if meta0["duration"] <= 0.0 or meta1["duration"] <= 0.0:
        raise PipelineError("Generated chunk has 0 duration or unreadable frame count.")

    logger.info(
        f"Verified chunks for job {job_id}: chunk0={meta0['duration']:.2f}s, chunk1={meta1['duration']:.2f}s"
    )
    return chunk0_path, chunk1_path


def build_facefusion_command(
    target_path: Path,
    output_path: Path,
    source_face_path: Path,
    reference_position: int,
    reference_frame: int,
    swapper_model: str = DEFAULT_SWAPPER_MODEL,
    enhancer_model: Optional[str] = DEFAULT_ENHANCER_MODEL,
    enhancer_blend: int = DEFAULT_ENHANCER_BLEND,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    execution_threads: Optional[int] = None,
    video_memory_strategy: Optional[str] = "moderate",
    temp_frame_format: Optional[str] = None,
) -> list[str]:
    """Construct FaceFusion headless CLI command.

    Supports optional enhancer (omitted during Pass 2 if enhance_face_b is disabled),
    enforces canonical model allowlists, and probes supported CLI flags.
    """
    if swapper_model not in ALLOWED_SWAPPER_MODELS:
        swapper_model = DEFAULT_SWAPPER_MODEL
    if detector_model not in ALLOWED_DETECTOR_MODELS:
        detector_model = DEFAULT_DETECTOR_MODEL

    help_info = get_facefusion_help_info()

    # Determine processor list and enhancer parameters
    if enhancer_model and enhancer_model in ALLOWED_ENHANCER_MODELS:
        processors = ["face_swapper", "face_enhancer"]
        has_enhancer = True
    else:
        processors = ["face_swapper"]
        has_enhancer = False

    blend_clamped = max(40, min(100, int(enhancer_blend)))
    threads = execution_threads if execution_threads is not None else FACESWAP_THREADS

    cmd = get_facefusion_cmd() + [
        "headless-run",
        "-s", str(source_face_path),
        "-t", str(target_path),
        "-o", str(output_path),
        "--processors", *processors,
        "--face-swapper-model", swapper_model,
    ]

    if has_enhancer and enhancer_model:
        cmd.extend([
            "--face-enhancer-model", enhancer_model,
            "--face-enhancer-blend", str(blend_clamped),
        ])

    cmd.extend([
        "--face-detector-model", detector_model,
        "--face-selector-mode", "reference",
        "--face-selector-order", "left-right",
        "--reference-face-position", str(reference_position),
        "--reference-frame-number", str(reference_frame),
        "--execution-providers", "cuda",
    ])

    if help_info.get("has_execution_thread_count", True):
        cmd.extend(["--execution-thread-count", str(threads)])

    if help_info.get("has_video_memory_strategy", True) and video_memory_strategy:
        cmd.extend(["--video-memory-strategy", str(video_memory_strategy)])

    if help_info.get("has_temp_frame_format", True):
        tff = temp_frame_format or help_info.get("temp_frame_format_choice", "jpeg")
        cmd.extend(["--temp-frame-format", str(tff)])

    # Pixel boost for inswapper_128 if supported
    if swapper_model == "inswapper_128" and help_info.get("has_face_swapper_pixel_boost", True):
        cmd.extend(["--face-swapper-pixel-boost", "512x512"])

    return cmd


def concat_and_remux(
    chunk0_processed: Path,
    chunk1_processed: Path,
    original_video: Path,
    job_id: str,
) -> Path:
    """Concatenate processed video chunks using FFmpeg concat demuxer and restore audio.

    Maps processed video stream from chunks and remuxes original audio stream with
    AAC encoding: -map 0:v:0 -map 1:a:0? -c:v copy -c:a aac -shortest
    """
    job_chunks_dir = CHUNKS_DIR / job_id
    concat_list_path = job_chunks_dir / "concat_list.txt"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    final_output_path = OUTPUTS_DIR / f"{job_id}.mp4"

    # Write absolute paths into concat demuxer file
    with open(concat_list_path, "w", encoding="utf-8") as f:
        f.write(f"file '{chunk0_processed.resolve()}'\n")
        f.write(f"file '{chunk1_processed.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-i", str(original_video),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(final_output_path),
    ]

    try:
        logger.info(f"Remuxing final output for job {job_id} to {final_output_path}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg concat and remux failed: {e.stderr}")
        raise PipelineError(f"Failed to concatenate chunks and remux audio: {e.stderr}")

    if not final_output_path.exists():
        raise PipelineError(f"Final output video was not created at {final_output_path}")

    return final_output_path


def cleanup_job_chunks(job_id: str) -> None:
    """Remove temporary chunk files and directory for a job."""
    job_chunks_dir = CHUNKS_DIR / job_id
    if not job_chunks_dir.exists():
        return

    try:
        for item in job_chunks_dir.iterdir():
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        job_chunks_dir.rmdir()
        logger.info(f"Cleaned up temporary chunk directory for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to completely remove temporary chunk files for job {job_id}: {e}")
