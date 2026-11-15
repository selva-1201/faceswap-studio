"""Configuration and runtime constants for FaceSwap Studio.

Defines model allowlists, default parameters, directory paths, and FaceFusion CLI options.
Compatible with Python 3.10+.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil

# Model allowlists
ALLOWED_SWAPPER_MODELS: list[str] = [
    "hyperswap_1c_256",
    "hyperswap_1a_256",
    "inswapper_128",
]

ALLOWED_ENHANCER_MODELS: list[str] = [
    "codeformer",
    "gfpgan_1.4",
    "gpen_bfr_512",
]

# Canonical detector names: "yolo_face" (not "yoloface" or "many")
ALLOWED_DETECTOR_MODELS: list[str] = [
    "yolo_face",
    "retinaface",
    "scrfd",
    "yunet",
]

# Default values
DEFAULT_SWAPPER_MODEL: str = "hyperswap_1c_256"
DEFAULT_ENHANCER_MODEL: str = "codeformer"
DEFAULT_DETECTOR_MODEL: str = "yolo_face"
DEFAULT_ENHANCER_BLEND: int = 80
MIN_ENHANCER_BLEND: int = 40
MAX_ENHANCER_BLEND: int = 100

# Performance, size, and timeout limits (driven by environment variables)
# Cloudflare Quick Tunnel rejects requests > 100MB; default cap is 95MB
MAX_UPLOAD_MB: int = int(os.getenv("FACESWAP_MAX_UPLOAD_MB", "95"))
FACESWAP_THREADS: int = int(os.getenv("FACESWAP_THREADS", "2"))
PASS_TIMEOUT_SECONDS: int = int(os.getenv("FACESWAP_PASS_TIMEOUT_SECONDS", "10800"))

# Base workspace resolution
# Priority: /kaggle/working if present, otherwise local workspace directory
KAGGLE_WORKING_PATH = Path("/kaggle/working")
if KAGGLE_WORKING_PATH.exists() and os.access(KAGGLE_WORKING_PATH, os.W_OK):
    WORKSPACE_DIR = KAGGLE_WORKING_PATH
else:
    WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"

UPLOADS_DIR = WORKSPACE_DIR / "uploads"
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
CHUNKS_DIR = WORKSPACE_DIR / "chunks"
JOBS_DIR = WORKSPACE_DIR / "jobs"

# FaceFusion root directory and model cache paths
# In FaceFusion v3, model weights reside inside <facefusion_dir>/.assets/models
FACEFUSION_ROOT_CANDIDATES = [
    WORKSPACE_DIR / "facefusion",
    Path("/kaggle/working/facefusion"),
    Path(__file__).resolve().parent.parent / "facefusion",
]

FACEFUSION_DIR = next((p for p in FACEFUSION_ROOT_CANDIDATES if p.exists()), WORKSPACE_DIR / "facefusion")
DEFAULT_ASSETS_MODELS_DIR = FACEFUSION_DIR / ".assets" / "models"

# Optional read-only Kaggle model cache dataset
KAGGLE_MODEL_CACHE_DIR = Path("/kaggle/input/ff-models-cache")
if KAGGLE_MODEL_CACHE_DIR.exists():
    MODELS_CACHE_DIR = KAGGLE_MODEL_CACHE_DIR
else:
    MODELS_CACHE_DIR = DEFAULT_ASSETS_MODELS_DIR

# Server network settings
SERVER_HOST = os.getenv("FACESWAP_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("FACESWAP_PORT", "7860"))


def get_facefusion_cmd() -> list[str]:
    """Locate FaceFusion executable or python script entrypoint."""
    entrypoint = FACEFUSION_DIR / "facefusion.py"
    if entrypoint.exists():
        return ["python3", str(entrypoint)]

    system_cmd = shutil.which("facefusion")
    if system_cmd:
        return [system_cmd]

    # Fallback to python script relative path
    return ["python3", "facefusion.py"]


def init_workspace() -> None:
    """Create all required workspace directories if they do not exist."""
    for directory in [WORKSPACE_DIR, UPLOADS_DIR, OUTPUTS_DIR, CHUNKS_DIR, JOBS_DIR, DEFAULT_ASSETS_MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
