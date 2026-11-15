"""Smoke test script for FaceSwap Studio pipeline.

Generates a synthetic 5-second video with audio and a dummy face image using FFmpeg,
then validates:
1. Video probing (metadata extraction, resolution, duration, audio stream detection)
2. Frame-accurate chunk splitting into chunk0.mp4 and chunk1.mp4 (libx264, yuv420p, -an)
3. FaceFusion CLI command construction for Pass 1 (swapper + enhancer) and Pass 2 (swapper only)
4. Chunk concatenation via concat demuxer and audio remuxing
5. Temporary workspace cleanup

Does NOT execute CUDA or FaceFusion model inference so it can run safely on any machine.
Compatible with Python 3.10+.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.config import CHUNKS_DIR, OUTPUTS_DIR, WORKSPACE_DIR, init_workspace
from backend.swap_pipeline import (
    build_facefusion_command,
    cleanup_job_chunks,
    concat_and_remux,
    probe_video,
    split_video_chunks,
)


def log_test(step: str, detail: str) -> None:
    print(f"[Smoke Test] [{step}] {detail}")


def main() -> int:
    print("==================================================================")
    print(" FaceSwap Studio - Local Pipeline Smoke Test")
    print("==================================================================")

    init_workspace()
    test_dir = WORKSPACE_DIR / "smoke_test_tmp"
    test_dir.mkdir(parents=True, exist_ok=True)

    job_id = "smoke_job_test"
    test_video = test_dir / "synthetic_target.mp4"
    test_face = test_dir / "synthetic_face.png"

    try:
        # Step 1: Generate synthetic 5-second test video with audio
        log_test("GENERATE", "Creating 5s synthetic test video (640x360 30fps with audio track)...")
        cmd_gen_video = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(test_video),
        ]
        subprocess.run(cmd_gen_video, capture_output=True, check=True)

        cmd_gen_image = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=blue:size=256x256",
            "-vframes", "1",
            str(test_face),
        ]
        subprocess.run(cmd_gen_image, capture_output=True, check=True)
        log_test("GENERATE", f"Generated test assets at {test_dir}")

        # Step 2: Probe video
        log_test("PROBE", "Probing video metadata using probe_video()...")
        meta = probe_video(test_video)
        log_test("PROBE", f"Duration: {meta['duration']:.2f}s, Res: {meta['width']}x{meta['height']}, Audio: {meta['has_audio']}")
        assert 4.8 <= meta["duration"] <= 5.2, f"Unexpected duration: {meta['duration']}"
        assert meta["width"] == 640 and meta["height"] == 360, "Unexpected resolution"
        assert meta["has_audio"] is True, "Audio track not detected"

        # Step 3: Split into 2 chunks
        log_test("SPLIT", "Splitting video into 2 chunks (libx264, yuv420p)...")
        chunk0, chunk1 = split_video_chunks(test_video, job_id, meta["duration"])
        assert chunk0.exists() and chunk1.exists(), "Chunk files missing on disk"
        meta0 = probe_video(chunk0)
        meta1 = probe_video(chunk1)
        log_test("SPLIT", f"Chunk 0: {meta0['duration']:.2f}s | Chunk 1: {meta1['duration']:.2f}s")
        assert meta0["duration"] > 0 and meta1["duration"] > 0, "Chunk duration must be > 0"

        # Step 4: Build CLI Commands
        log_test("BUILD", "Testing FaceFusion CLI command formatting...")
        # Pass 1: Primary Face A (swapper + enhancer)
        cmd_p1 = build_facefusion_command(
            target_path=chunk0,
            output_path=chunk0.with_name("chunk0_p1.mp4"),
            source_face_path=test_face,
            reference_position=0,
            reference_frame=0,
            swapper_model="hyperswap_1c_256",
            enhancer_model="codeformer",
            enhancer_blend=80,
            detector_model="yolo_face",
            execution_threads=2,
            video_memory_strategy="moderate",
        )
        assert "--processors" in cmd_p1 and "face_swapper" in cmd_p1 and "face_enhancer" in cmd_p1
        assert "--face-detector-model" in cmd_p1 and "yolo_face" in cmd_p1
        assert "--face-enhancer-blend" in cmd_p1 and "80" in cmd_p1
        log_test("BUILD", f"Pass 1 command verified ({len(cmd_p1)} args)")

        # Pass 2: Secondary Face B without enhancer
        cmd_p2 = build_facefusion_command(
            target_path=chunk0.with_name("chunk0_p1.mp4"),
            output_path=chunk0.with_name("chunk0_p2.mp4"),
            source_face_path=test_face,
            reference_position=1,
            reference_frame=0,
            swapper_model="hyperswap_1c_256",
            enhancer_model=None,  # No enhancer for Pass 2!
            detector_model="yolo_face",
            execution_threads=2,
            video_memory_strategy="moderate",
        )
        assert "face_enhancer" not in cmd_p2, "Pass 2 must not include face_enhancer when enhancer_model is None"
        assert "--face-enhancer-model" not in cmd_p2
        log_test("BUILD", f"Pass 2 command (swapper-only) verified ({len(cmd_p2)} args)")

        # Step 5: Concat and Audio Remux
        log_test("REMUX", "Testing concat demuxer and audio remuxing...")
        final_video = concat_and_remux(
            chunk0_processed=chunk0,
            chunk1_processed=chunk1,
            original_video=test_video,
            job_id=job_id,
        )
        assert final_video.exists(), f"Final video missing: {final_video}"
        meta_final = probe_video(final_video)
        log_test("REMUX", f"Final video generated: {meta_final['duration']:.2f}s, Audio: {meta_final['has_audio']}")
        assert meta_final["has_audio"] is True, "Final video missing audio track"

        # Step 6: Workspace Cleanup
        log_test("CLEANUP", "Testing chunk workspace cleanup...")
        cleanup_job_chunks(job_id)
        assert not (CHUNKS_DIR / job_id).exists(), "Chunk directory should be deleted"
        log_test("CLEANUP", "Chunk directory cleaned up successfully.")

        # Cleanup test directory and output
        shutil.rmtree(test_dir, ignore_errors=True)
        final_video.unlink(missing_ok=True)

        print("==================================================================")
        print(" [PASSED] ALL PIPELINE SMOKE TESTS PASSED SUCCESSFULLY!")
        print("==================================================================")
        return 0

    except Exception as e:
        print(f"\n[FAILED] Smoke test error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(test_dir, ignore_errors=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
