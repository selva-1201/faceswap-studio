"""Model weight prefetch and cache linkage script for FaceSwap Studio.

Checks for read-only Kaggle model cache in /kaggle/input/ff-models-cache and creates
symlinks into <facefusion_dir>/.assets/models (e.g. /kaggle/working/facefusion/.assets/models).
If cache is not present, downloads essential ONNX weights directly from FaceFusion releases.
"""

from pathlib import Path
import os
import sys
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

# Official model URLs for FaceFusion v3 assets
MODEL_REGISTRY: dict[str, str] = {
    # Face Swappers
    "hyperswap_1c_256.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_1c_256.onnx",
    "hyperswap_1a_256.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/hyperswap_1a_256.onnx",
    "inswapper_128.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
    # Face Enhancers
    "codeformer.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx",
    "gfpgan_1.4.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
    # Face Detectors
    "yoloface.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface.onnx",
    "yolo_face.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yolo_face.onnx",
    "retinaface.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface.onnx",
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

        urllib.request.urlretrieve(url, temp_path, reporthook=progress_bar)
        print()
        temp_path.replace(dest_path)
        print(f"  [Success] Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"\n  [Warning] Could not download {dest_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
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
        print(f"[Prefetch] Successfully linked {count} model files from Kaggle cache.")
        if count > 0:
            print("Model prefetch completed via Kaggle dataset.")
            return

    print("[Prefetch] Kaggle read-only cache not found or empty.")
    print("[Prefetch] Checking local presence of essential models...")

    # 2. Check and download essential default models
    essential_models = [
        "hyperswap_1c_256.onnx",
        "codeformer.onnx",
        "yoloface.onnx",
    ]

    for model_name in essential_models:
        model_path = TARGET_MODELS_DIR / model_name
        if model_path.exists() and model_path.stat().st_size > 1024:
            print(f"  [Present] {model_name} ({model_path.stat().st_size // (1024 * 1024)} MB)")
        else:
            url = MODEL_REGISTRY.get(model_name)
            if url:
                download_model(url, model_path)

    # Ensure alias symlink exists between yoloface.onnx and yolo_face.onnx
    yoloface_file = TARGET_MODELS_DIR / "yoloface.onnx"
    yolo_face_file = TARGET_MODELS_DIR / "yolo_face.onnx"
    if yoloface_file.exists() and not yolo_face_file.exists():
        try:
            os.symlink("yoloface.onnx", yolo_face_file)
        except Exception:
            pass

    print("==================================================================")
    print(" Model prefetch stage completed.")
    print("==================================================================")


if __name__ == "__main__":
    main()

