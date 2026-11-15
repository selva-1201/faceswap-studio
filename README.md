# FaceSwap Studio

A production-ready multi-face video face swap engine running in parallel on Kaggle's dual NVIDIA T4 GPUs (`T4 x2`), controlled via a local Black Glassmorphism web studio UI connected through Cloudflare Quick Tunnel.

---

## Architecture Overview

FaceSwap Studio is architected for dual-GPU acceleration on Kaggle notebooks, distributing video rendering workloads across both GPUs to cut processing time in half while avoiding Cloudflare tunnel gateway timeouts.

```
+-----------------------------------------------------------------------------------+
|                           Target Video (Input Stream)                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
                         [ffprobe: Video Duration Probe]
                                         |
                                         v
            [ffmpeg: Frame-Accurate Split (libx264 -crf 16 -pix_fmt yuv420p)]
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
              [chunk0.mp4]                              [chunk1.mp4]
         (0.00s to midpoint)                       (midpoint to duration)
                    |                                         |
                    v                                         v
         ThreadPool Worker 0                       ThreadPool Worker 1
     (CUDA_VISIBLE_DEVICES=0)                  (CUDA_VISIBLE_DEVICES=1)
     (TMPDIR=chunks/.../tmp_gpu0)              (TMPDIR=chunks/.../tmp_gpu1)
                    |                                         |
                    v                                         v
    [Pass 1: Face A Reference Swap]           [Pass 1: Face A Reference Swap]
                    |                                         |
                    v (if Face B present)                     v (if Face B present)
    [Pass 2: Face B Reference Swap]           [Pass 2: Face B Reference Swap]
         (Swapper only by default)                 (Swapper only by default)
                    |                                         |
                    v                                         v
           [chunk0_final.mp4]                        [chunk1_final.mp4]
                    |                                         |
                    +--------------------+--------------------+
                                         |
                                         v
                     [ffmpeg Concat Demuxer + Audio Remux]
                       (-c:v copy -c:a aac -shortest)
                                         |
                                         v
                        Final Output: outputs/<job_id>.mp4
```

---

## Key Features

1. **Dual-GPU Parallel Execution:**
   - Probes target video duration with `ffprobe`.
   - Splits video into two frame-accurate video-only chunks (`chunk0.mp4`, `chunk1.mp4`) with `ffmpeg -i input.mp4 -ss <start> -to <end> -an -c:v libx264 -crf 16 -preset fast -pix_fmt yuv420p` to ensure cuts land on keyframes and prevent concatenation freezing.
   - Dispatches `chunk0` to `CUDA_VISIBLE_DEVICES=0` and `chunk1` to `CUDA_VISIBLE_DEVICES=1` concurrently using `concurrent.futures.ThreadPoolExecutor(max_workers=2)` without unsafe process forking.
   - Runs Pass 1 (`--face-selector-mode reference`, `--face-selector-order left-right`, `--reference-face-position <pos_a>`) for Source Face A.
   - If Source Face B is provided, executes Pass 2 with `<pos_b>` on the output of Pass 1. By default, Pass 2 runs only `face_swapper` (enhancer disabled) to avoid double enhancement artifacts and halve Pass 2 runtime.
   - Concatenates processed chunks and remuxes original audio stream (`-map 0:v:0 -map 1:a:0? -c:a aac`).
   - Automatically unlinks intermediate chunk files eagerly as stages complete to save disk.

2. **Asynchronous Non-Blocking Job API & Single Execution Queue:**
   - Long HTTP POST uploads return immediately with `{ "job_id": "...", "status": "queued" }`.
   - Sequential worker queue processes one job at a time to prevent multiple jobs from competing for the dual T4 GPUs and triggering container OOM halts.
   - Cloudflare Quick Tunnels never trigger the 100-second gateway timeout.
   - Job state is persisted in a thread-safe, debounced JSON store (`jobs.json`).

3. **Anti-OOM Hardening & Resource Monitoring:**
   - Server process strictly avoids importing PyTorch/CUDA; GPU detection is performed via cached `nvidia-smi` queries.
   - Pre-flight checks verify that workspace disk has at least 6x the video file size free.
   - Live system metrics endpoint (`GET /api/system/resources`) reports RAM, workspace disk free, `/tmp` disk free, and per-GPU VRAM.
   - If a worker encounters an OOM error (exit code 137 / -9 or "out of memory"), it automatically retries that pass once with `--execution-thread-count 1` and `--video-memory-strategy strict`.
   - If a worker fails, all sibling subprocesses for that job are terminated immediately.

4. **Black Glassmorphism Studio UI:**
   - Built with dark obsidian translucent cards (`backdrop-filter: blur(20px)`), delicate borders, and glowing neon accents.
   - Top-bar Remote Backend URL input field persisted in `localStorage` with a "Test Connection" button hitting `/api/system/status`.
   - Dual-GPU Hardware Pipeline HUD displaying active workers, live progress, and live system resource indicators (RAM, Disk, VRAM).
   - Real upload progress bar using `XMLHttpRequest`.
   - Incremental delta log synchronization (`?since=<index>`) without DOM re-renders.
   - Active job cancellation button (`POST /api/jobs/{job_id}/cancel`).
   - Pure inline SVGs with zero emoji characters.

5. **Model Allowlists & Defaults:**
   - **Swapper Models:** `hyperswap_1c_256` (default), `hyperswap_1a_256`, `inswapper_128`.
   - Auto-appends `--face-swapper-pixel-boost 512x512` when `inswapper_128` is selected.
   - **Enhancer Models:** `codeformer` (default), `gfpgan_1.4`, `gpen_bfr_512`.
   - **Detector Models:** `yolo_face` (canonical default), `retinaface`, `scrfd`, `yunet`.
   - **Enhancer Blend:** Default `80` (slider range `40` to `100`).

---

## Directory Structure

```text
faceswap-studio/
├── frontend/
│   ├── index.html               # Black Glassmorphism Studio UI
│   ├── styles.css               # Frosted glass styling and HUD layout
│   └── app.js                   # Client state, delta log polling, cancellation
├── backend/
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Model allowlists, directory paths, limits
│   ├── parallel_runner.py       # Dual-GPU ThreadPoolExecutor runner
│   ├── swap_pipeline.py         # FFmpeg chunking, CLI builder, remuxing
│   ├── costume_pipeline.py      # Stub module for Phase 2 outfit replacement
│   └── server.py                # FastAPI server, async queue, resource monitor
├── scripts/
│   ├── setup_env.sh             # Kaggle T4 x2 package and cloudflared installer
│   ├── prefetch_models.py       # Model cache linker (.assets/models)
│   └── smoke_test.py            # Local FFmpeg end-to-end pipeline test
├── notebooks/
│   └── kaggle_launcher.ipynb    # Kaggle one-click deployment notebook
├── .gitignore                   # Excludes outputs, chunks, and weight files
└── README.md                    # Project documentation
```

---

## Quickstart Guide

### 1. Kaggle Prerequisites & Deployment

1. Create a new notebook on [Kaggle](https://www.kaggle.com).
2. Under **Notebook Options**:
   - **Accelerator**: Select **GPU T4 x2**.
   - **Internet**: Toggle to **ON** (required to install packages and download tunnel binary).
3. Open `notebooks/kaggle_launcher.ipynb` or execute the notebook cells:
   - **Step 0**: Verifies internet connectivity and clones repository into `/kaggle/working/faceswap-studio`.
   - **Step 1**: Verifies dual NVIDIA T4 GPU visibility via `nvidia-smi -L`.
   - **Step 2**: Runs `scripts/setup_env.sh` and `scripts/prefetch_models.py`.
   - **Step 3**: Launches the FastAPI backend in a detached background session (`start_new_session=True`) on port `7860`.
   - **Step 4**: Starts Cloudflare Quick Tunnel (`--no-autoupdate`) and displays the public `https://*.trycloudflare.com` URL.
   - **Step 5**: Starts the heartbeat loop displaying live RAM, disk free, and GPU memory metrics every 10s with auto-restart recovery.

> Note: Cloudflare Quick Tunnel rejects request bodies larger than **100MB**. FaceSwap Studio enforces a maximum video upload of **95MB**.

### 2. Local Studio Web UI Usage

1. Open `frontend/index.html` directly in your local browser (or host it via any local web server).
2. In the top navigation bar:
   - Paste your Cloudflare tunnel URL (e.g., `https://random-words.trycloudflare.com`).
   - Click **Test Connection**.
   - The status badge will turn green: `Connected: 2x GPU Detected`.
3. Drag and drop your **Target Video** (up to 95MB).
4. Drag and drop **Source Face A** (Primary actor). Set face position index (0 for leftmost face) and reference frame index.
5. (Optional) Check **Enable Source Face B** to configure a second character. Toggle **Enhance Face B** if secondary face enhancement is desired.
6. Select your Swapper, Enhancer, Detector models, and Enhancer Blend factor.
7. Click **Start Parallel Face Swap**.
8. Monitor the real-time Dual-GPU HUD (GPU 0 & GPU 1 progress), live resource metrics, and terminal logs.
9. When complete, preview the result video in the player and click **Download Processed Video**.

---

## Troubleshooting & Diagnostics

### Out of Memory (OOM) Container Halts
- **Cause:** FaceFusion running multiple high-resolution models concurrently with multiple execution threads can exhaust container RAM or GPU VRAM.
- **Mitigation:**
  - FaceSwap Studio defaults to `--execution-thread-count 2` and `--video-memory-strategy moderate`.
  - Pass 2 disables `face_enhancer` by default to avoid stacking heavy enhancer models.
  - If an OOM is detected, the pipeline automatically retries once with `--execution-thread-count 1` and `--video-memory-strategy strict`.
  - In `notebooks/kaggle_launcher.ipynb`, run the **Diagnostics** cell to check `free -h` and `nvidia-smi`.
  - Inspect `/kaggle/working/server.log` or check `dmesg` for kernel kill messages.

### Disk Space Full
- **Cause:** Long videos generate thousands of temporary frames during processing.
- **Mitigation:**
  - The backend runs pre-flight disk verification requiring at least 6x the input video size free in `/kaggle/working`.
  - Original video chunks (`chunk0.mp4`, `chunk1.mp4`) are eagerly deleted immediately after Pass 1 completes.
  - Intermediate Pass 1 files are deleted immediately after Pass 2 completes.
  - Run the **Diagnostics** cell in the notebook to view `df -h /kaggle/working /tmp`.

### Missing CUDAExecutionProvider
- **Symptom:** Server reports `CPU Fallback` or `CUDA provider missing`.
- **Mitigation:**
  - `scripts/setup_env.sh` uninstalls broken onnxruntime wheels and installs CUDA-matched `onnxruntime-gpu`.
  - Run `python3 -c "import onnxruntime as o; print(o.get_available_providers())"` in Kaggle to confirm `CUDAExecutionProvider` is listed.

### Kaggle Session Limits
- Kaggle GPU sessions have a hard maximum execution time of ~9 hours and inactivity timeouts.
- `start_new_session=True` detaches background processes from the cell, but processes do NOT survive kernel restarts or session termination.
- Keep the **Step 5** heartbeat loop running to prevent inactivity timeouts.

---

## API Reference

### `GET /api/system/status`
Returns hardware detection, GPU count, model paths, upload limits, and CUDA capability.

### `GET /api/system/resources`
Returns live system RAM, workspace free disk space, `/tmp` free disk space, and per-GPU VRAM usage.

### `GET /api/system/models`
Returns model allowlists for swappers, enhancers, detectors, and default values.

### `POST /api/preview/command`
Generates preview FaceFusion CLI command string based on current settings.

### `POST /api/jobs`
Accepts video and source face files (streamed in 1MB chunks), validates parameters and disk capacity, enqueues the background job, and returns HTTP 200 immediately.

### `POST /api/jobs/{job_id}/cancel`
Cancels active or queued job, terminates running FaceFusion/FFmpeg child processes, and cleans up chunk folders.

### `GET /api/jobs/{job_id}?since=<index>`
Returns job status, progress percentage, stage, GPU HUD states, and incremental log entries since `<index>`.

### `GET /api/jobs/{job_id}/download`
Streams the final processed video file with `Content-Disposition: attachment`.
