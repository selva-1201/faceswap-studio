"""Face swap pipeline for video probing, frame-accurate chunking, CLI command building, and audio remuxing.

Handles FFmpeg chunk generation, FaceFusion CLI argument formatting for Pass 1 (Face A)
and Pass 2 (Face B), chunk concatenation via concat demuxer, original audio restoration,
and automatic workspace cleanup.
"""

from pathlib import Path
from typing import Any, Optional
import json
import logging
import subprocess

from backend.config import (
    ALLOWED_DETECTOR_MODELS,
    ALLOWED_ENHANCER_MODELS,
    ALLOWED_SWAPPER_MODELS,
    CHUNKS_DIR,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_ENHANCER_BLEND,
    DEFAULT_ENHANCER_MODEL,
    DEFAULT_SWAPPER_MODEL,
    OUTPUTS_DIR,
    get_facefusion_cmd,
)

logger = logging.getLogger("faceswap.pipeline")


class PipelineError(Exception):
    """Exception raised for pipeline processing errors."""
    pass


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

    Re-encodes with -c:v libx264 -crf 16 -preset fast without audio to ensure
    midpoint cuts fall exactly on keyframes and prevent freezing on concatenation.
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
        "-ss", "0.0",
        "-to", f"{midpoint:.3f}",
        "-i", str(video_path),
        "-an",
        "-c:v", "libx264",
        "-crf", "16",
        "-preset", "fast",
        str(chunk0_path),
    ]

    # Chunk 1: from midpoint to end
    cmd_chunk1 = [
        "ffmpeg",
        "-y",
        "-ss", f"{midpoint:.3f}",
        "-to", f"{duration:.3f}",
        "-i", str(video_path),
        "-an",
        "-c:v", "libx264",
        "-crf", "16",
        "-preset", "fast",
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

    return chunk0_path, chunk1_path


def build_facefusion_command(
    target_path: Path,
    output_path: Path,
    source_face_path: Path,
    reference_position: int,
    reference_frame: int,
    swapper_model: str = DEFAULT_SWAPPER_MODEL,
    enhancer_model: str = DEFAULT_ENHANCER_MODEL,
    enhancer_blend: int = DEFAULT_ENHANCER_BLEND,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    execution_threads: int = 4,
) -> list[str]:
    """Construct FaceFusion headless CLI command for a single pass.

    Enforces allowlists, adds reference face selector parameters, and appends
    --face-swapper-pixel-boost 512x512 when inswapper_128 is chosen.
    """
    if swapper_model not in ALLOWED_SWAPPER_MODELS:
        swapper_model = DEFAULT_SWAPPER_MODEL
    if enhancer_model not in ALLOWED_ENHANCER_MODELS:
        enhancer_model = DEFAULT_ENHANCER_MODEL
    if detector_model not in ALLOWED_DETECTOR_MODELS:
        detector_model = DEFAULT_DETECTOR_MODEL

    blend_clamped = max(40, min(100, int(enhancer_blend)))
    detector_cli = "yolo_face" if detector_model in ("yolo_face", "yoloface") else detector_model

    cmd = get_facefusion_cmd() + [
        "headless-run",
        "-s", str(source_face_path),
        "-t", str(target_path),
        "-o", str(output_path),
        "--processors", "face_swapper", "face_enhancer",
        "--face-swapper-model", swapper_model,
        "--face-enhancer-model", enhancer_model,
        "--face-enhancer-blend", str(blend_clamped),
        "--face-detector-model", detector_cli,
        "--face-selector-mode", "reference",
        "--face-selector-order", "left-right",
        "--reference-face-position", str(reference_position),
        "--reference-frame-number", str(reference_frame),
        "--execution-providers", "cuda",
        "--execution-thread-count", str(execution_threads),
    ]

    # Pixel boost requirement for inswapper_128
    if swapper_model == "inswapper_128":
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
        job_chunks_dir.rmdir()
        logger.info(f"Cleaned up temporary chunk directory for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to completely remove temporary chunk files for job {job_id}: {e}")
