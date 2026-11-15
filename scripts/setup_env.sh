#!/usr/bin/env bash
# ==============================================================================
# FaceSwap Studio - Kaggle T4 x2 Environment Setup Script
#
# Production setup script for Kaggle dual-T4 GPU containers.
# Preserves Kaggle pre-installed packages (numpy, scipy, torch), installs CUDA-accelerated
# ONNX Runtime and FaceFusion runtime dependencies safely with --no-deps where required,
# and verifies CUDAExecutionProvider before completing.
# ==============================================================================

set -eo pipefail

log_step() {
    echo ""
    echo "=================================================================="
    echo " [$(date +'%T')] $1"
    echo "=================================================================="
}

log_info() {
    echo "  [Log] $1"
}

# Resolve script repository directory regardless of current working directory
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
log_info "Repository Directory: $REPO_DIR"

log_step "Starting FaceSwap Studio Environment Initialization"

# Detect runtime workspace
WORKSPACE="/kaggle/working"
if [ ! -d "$WORKSPACE" ] || [ ! -w "$WORKSPACE" ]; then
    WORKSPACE="$REPO_DIR/workspace"
    mkdir -p "$WORKSPACE"
fi

log_info "Target Workspace: $WORKSPACE"
cd "$WORKSPACE"

# 1. Hardware Verification
log_step "[1/6] Verifying GPU Hardware & Memory"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
    log_info "Notice: nvidia-smi not detected. Running without GPU acceleration."
fi
log_info "System RAM Available:"
free -h 2>/dev/null || true

# 2. System Packages (FFmpeg, libSM, libXext)
log_step "[2/6] Verifying System Libraries (FFmpeg, libSM, libXext)"
SUDO_CMD=""
if command -v sudo &> /dev/null; then
    SUDO_CMD="sudo"
fi

if command -v ffmpeg &> /dev/null; then
    log_info "FFmpeg is already installed: $(ffmpeg -version | head -n 1)"
else
    log_info "Installing FFmpeg via apt..."
    $SUDO_CMD apt-get update -qq || true
    $SUDO_CMD apt-get install -y -qq ffmpeg libsm6 libxext6 > /dev/null || true
fi

# 3. Clone / Update FaceFusion Repository
log_step "[3/6] Setting Up FaceFusion v3 Repository"
FF_DIR="$WORKSPACE/facefusion"
if [ ! -d "$FF_DIR" ]; then
    log_info "Cloning FaceFusion repository into $FF_DIR..."
    git clone --depth 1 https://github.com/facefusion/facefusion.git "$FF_DIR"
    log_info "FaceFusion clone completed."
else
    log_info "FaceFusion repository already exists at $FF_DIR"
fi

# Create required directory structure
log_info "Creating workspace directories..."
mkdir -p "$FF_DIR/.assets/models"
mkdir -p "$WORKSPACE/uploads"
mkdir -p "$WORKSPACE/outputs"
mkdir -p "$WORKSPACE/chunks"
mkdir -p "$WORKSPACE/jobs"

# 4. Python Package Installation
log_step "[4/6] Installing Essential Python Packages"

# Ensure setuptools compatibility
log_info "Ensuring setuptools compatibility for PyTorch..."
python3 -m pip install --no-cache-dir "setuptools<82" || true

# Clean up any conflicting ONNX Runtime packages first
log_info "Uninstalling existing onnxruntime builds to avoid conflicts..."
python3 -m pip uninstall -y onnxruntime onnxruntime-gpu || true

# Install onnxruntime-gpu preserving Kaggle pre-installed packages (numpy, scipy)
log_info "Installing onnxruntime-gpu and core server dependencies..."
python3 -m pip install --no-cache-dir \
    onnxruntime-gpu \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles \
    requests \
    psutil \
    tqdm

# Install FaceFusion runtime dependencies without modifying system numpy/scipy
log_info "Installing FaceFusion dependencies with system-package protection..."
python3 -m pip install --no-cache-dir \
    filetype \
    onnx \
    opencv-python-headless || true

python3 -m pip install --no-cache-dir --no-deps \
    gradio \
    gradio-rangeslider || true

# 5. Verify FaceFusion CLI and CUDAExecutionProvider
log_step "[5/6] Verifying FaceFusion Imports & CUDAExecutionProvider"
cd "$FF_DIR"

log_info "Testing FaceFusion general help output..."
if ! python3 facefusion.py --help > /dev/null 2>&1; then
    echo "ERROR: 'python3 facefusion.py --help' failed! Check FaceFusion dependencies." >&2
    python3 facefusion.py --help
    exit 1
fi
log_info "FaceFusion root import test passed."

log_info "Testing FaceFusion headless-run help output..."
if ! python3 facefusion.py headless-run --help > /dev/null 2>&1; then
    echo "ERROR: 'python3 facefusion.py headless-run --help' failed!" >&2
    python3 facefusion.py headless-run --help
    exit 1
fi
log_info "FaceFusion headless-run test passed."

log_info "Checking ONNX Runtime available execution providers..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available Providers:', providers)
if 'CUDAExecutionProvider' not in providers:
    print('ERROR: CUDAExecutionProvider is missing from onnxruntime-gpu!')
    exit(1)
print('CUDAExecutionProvider verified successfully.')
"

# 6. Cloudflare Quick Tunnel Binary
log_step "[6/6] Setting Up Cloudflare Tunnel Executable"
cd "$WORKSPACE"
CLOUDFLARED_BIN="$WORKSPACE/cloudflared"
if [ ! -f "$CLOUDFLARED_BIN" ]; then
    log_info "Downloading cloudflared binary from GitHub releases..."
    curl -sSL -o "$CLOUDFLARED_BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CLOUDFLARED_BIN"
    log_info "Cloudflared executable downloaded successfully to $CLOUDFLARED_BIN"
else
    log_info "Cloudflared binary already present at $CLOUDFLARED_BIN"
fi

log_info "FaceSwap Studio Environment Initialization Complete!"
echo "=================================================================="
