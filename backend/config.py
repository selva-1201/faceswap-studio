"""Configuration and runtime constants for FaceSwap Studio.

Defines model allowlists, default parameters, directory paths, and FaceFusion CLI options.
"""

from pathlib import Path
import os
import shutil

# Model allowlists
ALLOWED_SWAPPER_MODELS = [
    "hyperswap_1c_256",
    "hyperswap_1a_256",
    "inswapper_128",
]

ALLOWED_ENHANCER_MODELS = [
    "codeformer",
    "gfpgan_1.4",
    "gpen_bfr_512",
]

ALLOWED_DETECTOR_MODELS = [
    "yolo_face",
    "retinaface",
]

# Default values
DEFAULT_SWAPPER_MODEL = "hyperswap_1c_256"
DEFAULT_ENHANCER_MODEL = "codeformer"
DEFAULT_DETECTOR_MODEL = "yolo_face"
DEFAULT_ENHANCER_BLEND = 80
MIN_ENHANCER_BLEND = 40
MAX_ENHANCER_BLEND = 100

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

# FaceFusion CLI execution binary / script path
def get_facefusion_cmd() -> list[str]:
    """Locate FaceFusion executable or python script entrypoint."""
    entrypoint = FACEFUSION_DIR / "facefusion.py"
    if entrypoint.exists():
        return ["python3", str(entrypoint)]

    system_cmd = shutil.which("facefusion")
    if system_cmd:
        return [system_cmd]

    # Fallback to python module execution or relative script
    return ["python3", "facefusion.py"]


def init_workspace() -> None:
    """Create all required workspace directories if they do not exist."""
    for directory in [WORKSPACE_DIR, UPLOADS_DIR, OUTPUTS_DIR, CHUNKS_DIR, JOBS_DIR, DEFAULT_ASSETS_MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
