#!/usr/bin/env bash
# ==============================================================================
# FaceSwap Studio - Kaggle T4 x2 Environment Setup Script
#
# Configures system packages, clones FaceFusion v3, sets up Python dependencies,
# creates model asset directories, and installs Cloudflare Quick Tunnel binary.
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo " Starting FaceSwap Studio Environment Initialization"
echo "=================================================================="

# Detect runtime workspace
WORKSPACE="/kaggle/working"
if [ ! -d "$WORKSPACE" ] || [ ! -w "$WORKSPACE" ]; then
    WORKSPACE="$(pwd)/workspace"
    mkdir -p "$WORKSPACE"
fi

echo "[1/6] Workspace target directory: $WORKSPACE"
cd "$WORKSPACE"

# 1. Verify GPU availability
echo "[2/6] Verifying NVIDIA GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
    echo "Warning: nvidia-smi not detected. Running in CPU fallback mode."
fi

# 2. System dependencies (FFmpeg and OpenCV graphics libraries)
echo "[3/6] Ensuring system packages (FFmpeg, libSM, libXext) are installed..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq ffmpeg libsm6 libxext6 curl wget git > /dev/null || true
fi

# 3. Clone or update FaceFusion repository
echo "[4/6] Setting up FaceFusion v3..."
FF_DIR="$WORKSPACE/facefusion"
if [ ! -d "$FF_DIR" ]; then
    git clone --depth 1 https://github.com/facefusion/facefusion.git "$FF_DIR"
else
    echo "FaceFusion repository already exists at $FF_DIR"
fi

# Ensure model assets directory exists for FaceFusion v3
mkdir -p "$FF_DIR/.assets/models"
mkdir -p "$WORKSPACE/uploads"
mkdir -p "$WORKSPACE/outputs"
mkdir -p "$WORKSPACE/chunks"
mkdir -p "$WORKSPACE/jobs"

# 4. Install Python dependencies
echo "[5/6] Installing Python packages (FastAPI, ONNX Runtime GPU, PyTorch compatibility)..."
# Kaggle torch 2.11.0 requires setuptools<82; ensure compatibility
python3 -m pip install --quiet --no-cache-dir "setuptools<82" || true

python3 -m pip install --quiet --no-warn-conflicts \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles \
    requests \
    tqdm || true

# Install FaceFusion requirements if present
if [ -f "$FF_DIR/requirements.txt" ]; then
    python3 -m pip install --quiet --no-warn-conflicts -r "$FF_DIR/requirements.txt" || true
fi

# 5. Download Cloudflare Quick Tunnel binary
echo "[6/6] Setting up Cloudflare tunnel (cloudflared)..."
CLOUDFLARED_BIN="$WORKSPACE/cloudflared"
if [ ! -f "$CLOUDFLARED_BIN" ]; then
    curl -sSL -o "$CLOUDFLARED_BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CLOUDFLARED_BIN"
    echo "Cloudflared binary installed at $CLOUDFLARED_BIN"
else
    echo "Cloudflared binary already present."
fi

echo "=================================================================="
echo " FaceSwap Studio Environment Setup Finished Successfully."
echo "=================================================================="

