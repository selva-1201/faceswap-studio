"""Model weight prefetch and cache linkage script for FaceSwap Studio.

Checks for read-only Kaggle model cache in /kaggle/input/ff-models-cache and creates
symlinks into <facefusion_dir>/.assets/models (e.g. /kaggle/working/facefusion/.assets/models).
Verifies model URLs with HTTP HEAD requests, pre-downloads available models, and falls
back to FaceFusion's built-in automatic on-demand model downloader for any unresolved assets.
Compatible with Python 3.10+.
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import urllib.error
import urllib.request

# Base paths
KAGGLE_WORKING = Path("/kaggle/working")
if KAGGLE_WORKING.exists() and os.access(KAGGLE_WORKING, os.W_OK):
    WORKSPACE = KAGGLE_WORKING
else:
    WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"

FF_DIR = WORKSPACE / "facefusion"
TARGET_MODELS_DIR = FF_DIR / ".assets" / "models"
KAGGLE_CACHE_DIR = Path("/kaggle/input/ff-models-cache")

# Official model URLs for FaceFusion v3 assets (release tag models-3.0.0)
MODEL_REGISTRY: dict[str, str] = {
    # Face Swappers
    "hyperswap_1c_256.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_1c_256.onnx",
    "hyperswap_1a_256.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_1a_256.onnx",
    "inswapper_128.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
    # Face Enhancers
    "codeformer.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx",
    "gfpgan_1.4.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
    "gpen_bfr_512.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_512.onnx",
    # Face Detectors (Canonical: yolo_face.onnx)
    "yolo_face.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yolo_face.onnx",
    "retinaface.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface.onnx",
    "scrfd.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd.onnx",
    "yunet.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yunet.onnx",
}


def link_from_cache(cache_dir: Path, target_dir: Path) -> int:
    """Link pre-cached models from Kaggle dataset input into destination directory."""
    linked = 0
    print(f"[Prefetch] Scanning Kaggle model cache: {cache_dir}")
    for ext in ("*.onnx", "*.pth", "*.bin"):
        for file_path in cache_dir.rglob(ext):
            symlink_dest = target_dir / file_path.name
            if not symlink_dest.exists():
                try:
                    os.symlink(file_path, symlink_dest)
                    print(f"  [Link] {file_path.name} -> {symlink_dest}")
                    linked += 1
                except Exception as e:
                    print(f"  [Error] Failed to link {file_path.name}: {e}")
            else:
                print(f"  [Present] {file_path.name}")
                linked += 1
    return linked


def check_url_resolves(url: str) -> bool:
    """Verify that a remote model asset URL exists and responds with 200/302."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status in (200, 301, 302)
    except Exception:
        return False


def download_model(url: str, dest_path: Path) -> bool:
    """Download a model file with progress feedback."""
    print(f"[Download] Fetching {dest_path.name} from {url}")
    temp_path = dest_path.with_suffix(".tmp")
    try:
        def progress_bar(blocks_transferred: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100.0, (blocks_transferred * block_size * 100.0) / total_size)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({blocks_transferred * block_size // 1024} KB)")
                sys.stdout.flush()

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as out:
            total_size = int(resp.headers.get("content-length", 0))
            blocks = 0
            block_size = 64 * 1024
            while True:
                buf = resp.read(block_size)
                if not buf:
                    break
                out.write(buf)
                blocks += 1
                progress_bar(blocks, block_size, total_size)

        print()
        temp_path.replace(dest_path)
        print(f"  [Success] Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"\n  [Warning] Could not download {dest_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def run_facefusion_force_download() -> bool:
    """Check if facefusion force-download command exists and run it."""
    entrypoint = FF_DIR / "facefusion.py"
    if not entrypoint.exists():
        return False
    try:
        res = subprocess.run(
            ["python3", str(entrypoint), "force-download", "--help"],
            capture_output=True,
            text=True,
            cwd=str(FF_DIR),
            timeout=10,
        )
        if res.returncode == 0:
            print("[Prefetch] Running FaceFusion force-download...")
            subprocess.run(["python3", str(entrypoint), "force-download"], cwd=str(FF_DIR), check=True)
            return True
    except Exception:
        pass
    return False


def main() -> None:
    print("==================================================================")
    print(" FaceSwap Studio - Model Weight Prefetch & Linking")
    print("==================================================================")
    print(f"Target Models Directory: {TARGET_MODELS_DIR}")
    TARGET_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check if Kaggle read-only cache exists
    if KAGGLE_CACHE_DIR.exists():
        count = link_from_cache(KAGGLE_CACHE_DIR, TARGET_MODELS_DIR)
        print(f"[Prefetch] Linked {count} model files from Kaggle cache.")
        if count > 0:
            print("Model prefetch completed via Kaggle dataset.")

    # 2. Check essential models
    essential_models = [
        "hyperswap_1c_256.onnx",
        "codeformer.onnx",
        "yolo_face.onnx",
        "retinaface.onnx",
    ]

    for model_name in essential_models:
        model_path = TARGET_MODELS_DIR / model_name
        if model_path.exists() and model_path.stat().st_size > 1024:
            print(f"  [Present] {model_name} ({model_path.stat().st_size // (1024 * 1024)} MB)")
        else:
            url = MODEL_REGISTRY.get(model_name)
            if url:
                if check_url_resolves(url):
                    download_model(url, model_path)
                else:
                    print(
                        f"  [Notice] Asset URL for {model_name} did not resolve (404/network). "
                        f"FaceFusion will auto-download {model_name} on first execution."
                    )

    # Maintain backwards compatibility symlink for yoloface.onnx <-> yolo_face.onnx
    yoloface_file = TARGET_MODELS_DIR / "yoloface.onnx"
    yolo_face_file = TARGET_MODELS_DIR / "yolo_face.onnx"
    if yolo_face_file.exists() and not yoloface_file.exists():
        try:
            os.symlink("yolo_face.onnx", yoloface_file)
        except Exception:
            pass

    # Optional: check if facefusion has a force-download command
    run_facefusion_force_download()

    print("==================================================================")
    print(" Model prefetch stage completed.")
    print("==================================================================")


if __name__ == "__main__":
    main()
