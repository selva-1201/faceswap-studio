#!/usr/bin/env bash
# ==============================================================================
# FaceSwap Studio - Kaggle T4 x2 Environment Setup Script
#
# Robust, lightweight initialization script optimized for Kaggle GPU containers.
# Avoids heavy dependency backtracking and memory spikes that cause OOM container halts.
# ==============================================================================

set -uo pipefail

log_step() {
    echo ""
    echo "=================================================================="
    echo " [$(date +'%T')] $1"
    echo "=================================================================="
}

log_info() {
    echo "  [Log] $1"
}

log_step "Starting FaceSwap Studio Environment Initialization"

# Detect runtime workspace
WORKSPACE="/kaggle/working"
if [ ! -d "$WORKSPACE" ] || [ ! -w "$WORKSPACE" ]; then
    WORKSPACE="$(pwd)/workspace"
    mkdir -p "$WORKSPACE"
fi

log_info "Target Workspace: $WORKSPACE"
cd "$WORKSPACE"

# 1. Hardware Verification
log_step "[1/6] Verifying GPU Hardware & Memory"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
    log_info "Notice: nvidia-smi not detected. Running in CPU mode."
fi
log_info "System RAM Available:"
free -h 2>/dev/null || true

# 2. System Packages
log_step "[2/6] Verifying System Libraries (FFmpeg, libSM, libXext)"
if command -v ffmpeg &> /dev/null; then
    log_info "FFmpeg is already installed: $(ffmpeg -version | head -n 1)"
else
    log_info "Installing FFmpeg via apt..."
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq ffmpeg libsm6 libxext6 > /dev/null || true
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
# Avoid running 'pip install -r requirements.txt' because recursive dependency
# resolution on Kaggle's pinned image consumes 12GB+ RAM and triggers OOM container shutoff.
log_step "[4/6] Installing Essential Python Packages"

log_info "Ensuring setuptools compatibility for PyTorch..."
python3 -m pip install --no-cache-dir "setuptools<82" || true

log_info "Installing FastAPI server & ONNX Runtime GPU..."
# Install with --no-deps or targeted packages to prevent dependency resolver thrashing
python3 -m pip install --no-cache-dir --no-warn-conflicts \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles \
    requests \
    tqdm \
    onnxruntime-gpu || true

# Check FaceFusion dependencies already present in Kaggle
log_info "Verifying core runtime packages in Python..."
python3 -c "
import sys
for pkg in ['torch', 'torchvision', 'fastapi', 'uvicorn', 'requests', 'tqdm']:
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', 'unknown')
        print(f'    [OK] {pkg}: {v}')
    except ImportError:
        print(f'    [Missing] {pkg}')
"

# 5. Cloudflare Quick Tunnel Binary
log_step "[5/6] Setting Up Cloudflare Tunnel Executable"
CLOUDFLARED_BIN="$WORKSPACE/cloudflared"
if [ ! -f "$CLOUDFLARED_BIN" ]; then
    log_info "Downloading cloudflared binary from GitHub releases..."
    curl -sSL -o "$CLOUDFLARED_BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CLOUDFLARED_BIN"
    log_info "Cloudflared executable downloaded successfully to $CLOUDFLARED_BIN"
else
    log_info "Cloudflared binary already present at $CLOUDFLARED_BIN"
fi

# 6. Final Summary
log_step "[6/6] Environment Initialization Complete"
log_info "Workspace: $WORKSPACE"
log_info "FaceFusion: $FF_DIR"
log_info "Model Directory: $FF_DIR/.assets/models"
log_info "System RAM After Setup:"
free -h 2>/dev/null || true
echo "=================================================================="
