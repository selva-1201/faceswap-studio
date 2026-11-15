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
               [ffmpeg: Frame-Accurate Split (libx264 -crf 16)]
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
              [chunk0.mp4]                              [chunk1.mp4]
         (0.00s to midpoint)                       (midpoint to duration)
                    |                                         |
                    v                                         v
         ProcessPool Worker 0                      ProcessPool Worker 1
     (CUDA_VISIBLE_DEVICES=0)                  (CUDA_VISIBLE_DEVICES=1)
                    |                                         |
                    v                                         v
    [Pass 1: Face A Reference Swap]           [Pass 1: Face A Reference Swap]
                    |                                         |
                    v (if Face B present)                     v (if Face B present)
    [Pass 2: Face B Reference Swap]           [Pass 2: Face B Reference Swap]
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
   - Splits video into two frame-accurate video-only chunks (`chunk0.mp4`, `chunk1.mp4`) with `ffmpeg -an -c:v libx264 -crf 16 -preset fast` to ensure cuts land on keyframes and prevent concatenation freezing.
   - Dispatches `chunk0` to `CUDA_VISIBLE_DEVICES=0` and `chunk1` to `CUDA_VISIBLE_DEVICES=1` concurrently using `concurrent.futures.ProcessPoolExecutor(max_workers=2)`.
   - Runs Pass 1 (`--face-selector-mode reference`, `--face-selector-order left-right`, `--reference-face-position <pos_a>`) for Source Face A.
   - If Source Face B is provided, executes Pass 2 using `<pos_b>` on the output of Pass 1.
   - Concatenates processed chunks and remuxes original audio stream (`-map 0:v:0 -map 1:a:0? -c:a aac`).
   - Automatically unlinks temporary chunk files upon job completion or failure.

2. **Asynchronous Non-Blocking Job API:**
   - Long HTTP POST uploads return immediately with `{ "job_id": "...", "status": "queued" }`.
   - Cloudflare Quick Tunnels never trigger the 100-second gateway timeout.
   - Job state is persisted in an in-memory dictionary backed by a JSON store (`jobs.json`).

3. **Black Glassmorphism Studio UI:**
   - Built with dark obsidian translucent cards (`backdrop-filter: blur(20px)`), delicate borders, and glowing neon accents.
   - Top-bar Remote Backend URL input field persisted in `localStorage` with a "Test Connection" button hitting `/api/system/status`.
   - Dual-GPU Hardware Pipeline HUD displaying active workers, progress bars, and device names.
   - Collapsible Live CLI Command Preview panel dynamically generating FaceFusion CLI arguments.
   - Drag-and-drop media zones for Video, Source Face A, and Source Face B with instant preview.
   - Pure inline SVGs with zero emoji characters.

4. **Model Allowlists & Defaults:**
   - **Swapper Models:** `hyperswap_1c_256` (default), `hyperswap_1a_256`, `inswapper_128`.
   - Auto-appends `--face-swapper-pixel-boost 512x512` when `inswapper_128` is selected.
   - **Enhancer Models:** `codeformer` (default), `gfpgan_1.4`, `gpen_bfr_512`.
   - **Detector Models:** `yolo_face` (default), `retinaface`.
   - **Enhancer Blend:** Default `80` (slider range `40` to `100`).

---

## Directory Structure

```text
faceswap-studio/
├── frontend/
│   ├── index.html               # Black Glassmorphism Studio UI
│   ├── styles.css               # Frosted glass styling and HUD layout
│   └── app.js                   # Client state, drop zones, polling loop
├── backend/
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Model allowlists, directory paths, defaults
│   ├── parallel_runner.py       # Dual-GPU ProcessPoolExecutor runner
│   ├── swap_pipeline.py         # FFmpeg chunking, CLI builder, remuxing
│   ├── costume_pipeline.py      # Stub module for Phase 2 outfit replacement
│   └── server.py                # FastAPI server and async job manager
├── scripts/
│   ├── setup_env.sh             # Kaggle T4 x2 package and cloudflared installer
│   └── prefetch_models.py       # Model cache linker (.assets/models)
├── notebooks/
│   └── kaggle_launcher.ipynb    # Kaggle one-click deployment notebook
├── .gitignore                   # Excludes outputs, chunks, and weight files
└── README.md                    # Project documentation
```

---

## Quickstart Guide

### 1. Kaggle T4 x2 Deployment

1. Create a new notebook on [Kaggle](https://www.kaggle.com).
2. Under **Notebook Options** -> **Accelerator**, select **GPU T4 x2**.
3. (Optional) Under **Add Input**, attach your pre-cached model dataset at `/kaggle/input/ff-models-cache`.
4. Upload `notebooks/kaggle_launcher.ipynb` or clone this repository:
   ```bash
   git clone https://github.com/your-username/faceswap-studio.git
   cd faceswap-studio
   ```
5. Run the notebook cells:
   - Cell 1 verifies dual NVIDIA T4 GPU visibility.
   - Cell 2 runs `scripts/setup_env.sh` and `scripts/prefetch_models.py`.
   - Cell 3 launches the FastAPI backend in the background on port `7860`.
   - Cell 4 starts Cloudflare Quick Tunnel and outputs a public `https://*.trycloudflare.com` URL.

### 2. Local Studio Web UI Usage

1. Open `frontend/index.html` directly in your local browser (or host it on any static web server).
2. In the top navigation bar:
   - Paste your Cloudflare tunnel URL (e.g., `https://random-words.trycloudflare.com`).
   - Click **Test Connection**.
   - The status badge will turn green: `Connected: 2x GPU Detected`.
3. Drag and drop your **Target Video**.
4. Drag and drop **Source Face A** (Primary actor). Set face position index (0 for leftmost face) and reference frame index.
5. (Optional) Check **Enable Source Face B** to configure a second character.
6. Select your Swapper, Enhancer, Detector models, and Enhancer Blend factor.
7. Click **Start Parallel Face Swap**.
8. Monitor the real-time Dual-GPU HUD (GPU 0 & GPU 1 progress) and terminal logs.
9. When complete, preview the result video in the player and click **Download Processed Video**.

---

## API Reference

### `GET /api/system/status`
Returns hardware detection, GPU count, and model directory paths.
- **Response:**
  ```json
  {
    "status": "online",
    "timestamp": "2026-11-15T10:30:00Z",
    "gpu_count": 2,
    "gpu_devices": ["Tesla T4", "Tesla T4"],
    "dual_gpu_ready": true,
    "workspace_dir": "/kaggle/working",
    "models_cache_dir": "/kaggle/working/facefusion/.assets/models"
  }
  ```

### `GET /api/system/models`
Returns allowlisted models and default parameter configurations.

### `POST /api/jobs`
Accepts multipart form uploads and enqueues a background job. Returns immediately to prevent tunnel timeouts.
- **Form Parameters:**
  - `video` (file, required): Target video file.
  - `source_face_a` (file, required): Portrait image for Face A.
  - `source_face_b` (file, optional): Portrait image for Face B.
  - `swapper_model` (string): `hyperswap_1c_256`, `hyperswap_1a_256`, or `inswapper_128`.
  - `enhancer_model` (string): `codeformer`, `gfpgan_1.4`, or `gpen_bfr_512`.
  - `enhancer_blend` (int): `40` to `100` (default `80`).
  - `detector_model` (string): `yolo_face` or `retinaface`.
  - `pos_a` (int): Reference position index for Face A (default `0`).
  - `pos_b` (int): Reference position index for Face B (default `1`).
  - `ref_frame` (int): Reference frame index (default `0`).
- **Response:**
  ```json
  {
    "job_id": "job_a1b2c3d4e5",
    "status": "queued",
    "stage": "Queued for dual-GPU processing",
    "created_at": "2026-11-15T10:30:00Z"
  }
  ```

### `GET /api/jobs/{job_id}`
Polls job status, overall progress, GPU 0 / GPU 1 status, and streaming execution logs.

### `GET /api/jobs/{job_id}/download`
Streams the completed output video file (`faceswap_<job_id>.mp4`).

---

## Phase 2 Roadmap: Costume & Garment Replacement
`backend/costume_pipeline.py` provides the architectural foundation and stub interfaces for Phase 2 outfit and garment replacement (Self-Correction Human Parsing / DensePose segmentation and virtual try-on inpainting) running downstream of the dual-GPU face swap stage.

---

## License
MIT License. FaceFusion and third-party ONNX models are subject to their respective upstream licenses.

