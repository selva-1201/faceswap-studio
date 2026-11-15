/**
 * FaceSwap Studio - Frontend Application Logic
 *
 * Handles remote backend connection, real-time CLI preview, file drop zones,
 * asynchronous job submission, dual-GPU status polling, and video preview.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    backendUrl: localStorage.getItem("faceswap_backend_url") || window.location.origin,
    videoFile: null,
    faceAFile: null,
    faceBFile: null,
    videoDuration: 0,
    activeJobId: null,
    pollInterval: null,
  };

  // DOM Elements
  const backendUrlInput = document.getElementById("backendUrlInput");
  const testConnectionBtn = document.getElementById("testConnectionBtn");
  const connectionStatusPill = document.getElementById("connectionStatusPill");
  const connectionStatusText = document.getElementById("connectionStatusText");

  // Inputs
  const videoDropZone = document.getElementById("videoDropZone");
  const videoFileInput = document.getElementById("videoFileInput");
  const videoPlaceholder = document.getElementById("videoPlaceholder");
  const videoPreview = document.getElementById("videoPreview");
  const videoDurationBadge = document.getElementById("videoDurationBadge");

  const faceADropZone = document.getElementById("faceADropZone");
  const faceAFileInput = document.getElementById("faceAFileInput");
  const faceAPlaceholder = document.getElementById("faceAPlaceholder");
  const faceAPreview = document.getElementById("faceAPreview");
  const posAInput = document.getElementById("posAInput");
  const refFrameInput = document.getElementById("refFrameInput");

  const enableFaceBCheckbox = document.getElementById("enableFaceBCheckbox");
  const faceBContainer = document.getElementById("faceBContainer");
  const faceBDropZone = document.getElementById("faceBDropZone");
  const faceBFileInput = document.getElementById("faceBFileInput");
  const faceBPlaceholder = document.getElementById("faceBPlaceholder");
  const faceBPreview = document.getElementById("faceBPreview");
  const posBInput = document.getElementById("posBInput");

  // Model Controls
  const swapperSelect = document.getElementById("swapperSelect");
  const enhancerSelect = document.getElementById("enhancerSelect");
  const detectorSelect = document.getElementById("detectorSelect");
  const blendSlider = document.getElementById("blendSlider");
  const blendValBadge = document.getElementById("blendValBadge");

  // CLI Preview
  const cliPreviewToggle = document.getElementById("cliPreviewToggle");
  const cliChevron = document.getElementById("cliChevron");
  const cliPreviewContent = document.getElementById("cliPreviewContent");
  const cliCommandCode = document.getElementById("cliCommandCode");

  // Dual-GPU HUD
  const gpuHudBadge = document.getElementById("gpuHudBadge");
  const gpu0Dot = document.getElementById("gpu0Dot");
  const gpu0Stage = document.getElementById("gpu0Stage");
  const gpu0Pct = document.getElementById("gpu0Pct");
  const gpu0ProgressBar = document.getElementById("gpu0ProgressBar");

  const gpu1Dot = document.getElementById("gpu1Dot");
  const gpu1Stage = document.getElementById("gpu1Stage");
  const gpu1Pct = document.getElementById("gpu1Pct");
  const gpu1ProgressBar = document.getElementById("gpu1ProgressBar");

  // Job Controls & Terminal
  const startJobBtn = document.getElementById("startJobBtn");
  const startBtnText = document.getElementById("startBtnText");
  const jobProgressContainer = document.getElementById("jobProgressContainer");
  const jobStageText = document.getElementById("jobStageText");
  const jobProgressPct = document.getElementById("jobProgressPct");
  const jobMainProgressBar = document.getElementById("jobMainProgressBar");
  const logsConsole = document.getElementById("logsConsole");

  // Result Section
  const resultSection = document.getElementById("resultSection");
  const resultVideoPlayer = document.getElementById("resultVideoPlayer");
  const downloadResultBtn = document.getElementById("downloadResultBtn");

  // Initialize Backend URL
  backendUrlInput.value = state.backendUrl;
  backendUrlInput.addEventListener("change", () => {
    let cleanUrl = backendUrlInput.value.trim().replace(/\/+$/, "");
    if (!cleanUrl) {
      cleanUrl = window.location.origin;
      backendUrlInput.value = cleanUrl;
    }
    state.backendUrl = cleanUrl;
    localStorage.setItem("faceswap_backend_url", cleanUrl);
    logToConsole(`[Config] Backend URL set to: ${cleanUrl}`);
  });

  // Logging Helper
  function logToConsole(message, isSystem = false) {
    const entry = document.createElement("div");
    entry.className = isSystem ? "log-entry system-msg" : "log-entry";
    entry.textContent = message;
    logsConsole.appendChild(entry);
    logsConsole.scrollTop = logsConsole.scrollHeight;
  }

  // Connection Test Handler
  async function testBackendConnection() {
    connectionStatusPill.className = "status-pill status-disconnected";
    connectionStatusText.textContent = "Connecting...";

    try {
      const response = await fetch(`${state.backendUrl}/api/system/status`);
      if (!response.ok) throw new Error(`HTTP error ${response.status}`);
      const data = await response.json();

      connectionStatusPill.className = "status-pill status-connected";
      if (data.gpu_count > 0) {
        connectionStatusText.textContent = `Connected: ${data.gpu_count}x GPU Detected`;
        gpuHudBadge.textContent = `${data.gpu_count}x GPU Ready`;
      } else {
        connectionStatusText.textContent = "Connected: CPU Fallback";
        gpuHudBadge.textContent = "CPU Emulation";
      }

      logToConsole(
        `[Backend Connected] GPUs: ${data.gpu_count} (${(data.gpu_devices || []).join(", ") || "None"}), Workspace: ${data.workspace_dir}`,
        true
      );
    } catch (err) {
      connectionStatusPill.className = "status-pill status-disconnected";
      connectionStatusText.textContent = "Disconnected";
      logToConsole(`[Connection Failed] Unable to reach ${state.backendUrl}: ${err.message}`, true);
    }
  }

  testConnectionBtn.addEventListener("click", testBackendConnection);

  // Auto-test connection on initial load
  testBackendConnection();

  // Model blend slider listener
  blendSlider.addEventListener("input", () => {
    blendValBadge.textContent = `${blendSlider.value}%`;
    updateCliPreview();
  });

  // CLI Command Preview Generator
  function updateCliPreview() {
    const swapper = swapperSelect.value;
    const enhancer = enhancerSelect.value;
    const detector = detectorSelect.value;
    const detectorCli = detector === "yoloface" ? "yolo_face" : detector;
    const blend = blendSlider.value;
    const posA = posAInput.value;
    const posB = posBInput.value;
    const refFrame = refFrameInput.value || "0";
    const hasFaceB = enableFaceBCheckbox.checked;

    let boostFlag = "";
    if (swapper === "inswapper_128") {
      boostFlag = " --face-swapper-pixel-boost 512x512";
    }

    const pass1Cmd =
      `# Pass 1: Face A on Chunk 0 (GPU 0) & Chunk 1 (GPU 1)\n` +
      `python3 facefusion.py headless-run \\\n` +
      `  -s /kaggle/working/uploads/source_a.png \\\n` +
      `  -t /kaggle/working/chunks/chunk0.mp4 \\\n` +
      `  -o /kaggle/working/chunks/chunk0_pass1.mp4 \\\n` +
      `  --processors face_swapper face_enhancer \\\n` +
      `  --face-swapper-model ${swapper}${boostFlag} \\\n` +
      `  --face-enhancer-model ${enhancer} \\\n` +
      `  --face-enhancer-blend ${blend} \\\n` +
      `  --face-detector-model ${detectorCli} \\\n` +
      `  --face-selector-mode reference \\\n` +
      `  --face-selector-order left-right \\\n` +
      `  --reference-face-position ${posA} \\\n` +
      `  --reference-frame-number ${refFrame} \\\n` +
      `  --execution-providers cuda`;

    let fullPreview = pass1Cmd;

    if (hasFaceB) {
      const pass2Cmd =
        `\n\n# Pass 2: Face B (Chained from Pass 1 output)\n` +
        `python3 facefusion.py headless-run \\\n` +
        `  -s /kaggle/working/uploads/source_b.png \\\n` +
        `  -t /kaggle/working/chunks/chunk0_pass1.mp4 \\\n` +
        `  -o /kaggle/working/chunks/chunk0_pass2.mp4 \\\n` +
        `  --processors face_swapper face_enhancer \\\n` +
        `  --face-swapper-model ${swapper}${boostFlag} \\\n` +
        `  --face-enhancer-model ${enhancer} \\\n` +
        `  --face-enhancer-blend ${blend} \\\n` +
        `  --face-detector-model ${detectorCli} \\\n` +
        `  --face-selector-mode reference \\\n` +
        `  --face-selector-order left-right \\\n` +
        `  --reference-face-position ${posB} \\\n` +
        `  --reference-frame-number ${refFrame} \\\n` +
        `  --execution-providers cuda`;
      fullPreview += pass2Cmd;
    }

    cliCommandCode.textContent = fullPreview;
  }

  // Bind change events to CLI preview
  [swapperSelect, enhancerSelect, detectorSelect, posAInput, posBInput, refFrameInput].forEach((el) => {
    el.addEventListener("change", updateCliPreview);
  });

  // Collapsible CLI Preview Toggle
  cliPreviewToggle.addEventListener("click", () => {
    const isCollapsed = cliPreviewContent.classList.toggle("collapsed");
    cliChevron.classList.toggle("rotated", isCollapsed);
  });

  // Toggle Source Face B Section
  enableFaceBCheckbox.addEventListener("change", () => {
    if (enableFaceBCheckbox.checked) {
      faceBContainer.classList.remove("collapsed");
    } else {
      faceBContainer.classList.add("collapsed");
    }
    updateCliPreview();
  });

  // Drag and Drop Setup Helper
  function setupDropZone(dropZone, fileInput, placeholder, previewEl, isVideo = false, onFileSelected = null) {
    dropZone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFile(files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
      }
    });

    function handleFile(file) {
      const url = URL.createObjectURL(file);
      placeholder.classList.add("hidden");
      previewEl.classList.remove("hidden");
      previewEl.src = url;

      if (isVideo) {
        previewEl.onloadedmetadata = () => {
          state.videoDuration = previewEl.duration;
          videoDurationBadge.textContent = `${previewEl.duration.toFixed(1)}s (${previewEl.videoWidth}x${previewEl.videoHeight})`;
          logToConsole(`[Media] Target video selected: ${file.name} (${previewEl.duration.toFixed(2)}s)`);
        };
      } else {
        logToConsole(`[Media] Source face selected: ${file.name}`);
      }

      if (onFileSelected) onFileSelected(file);
      updateCliPreview();
    }
  }

  // Setup input drop zones
  setupDropZone(videoDropZone, videoFileInput, videoPlaceholder, videoPreview, true, (file) => {
    state.videoFile = file;
  });

  setupDropZone(faceADropZone, faceAFileInput, faceAPlaceholder, faceAPreview, false, (file) => {
    state.faceAFile = file;
  });

  setupDropZone(faceBDropZone, faceBFileInput, faceBPlaceholder, faceBPreview, false, (file) => {
    state.faceBFile = file;
  });

  // Initial CLI preview render
  updateCliPreview();

  // Job Submission Handler
  startJobBtn.addEventListener("click", async () => {
    if (!state.videoFile) {
      alert("Please upload a target video file before starting.");
      return;
    }
    if (!state.faceAFile) {
      alert("Please upload Source Face A before starting.");
      return;
    }

    // Prepare FormData
    const formData = new FormData();
    formData.append("video", state.videoFile);
    formData.append("source_face_a", state.faceAFile);
    if (enableFaceBCheckbox.checked && state.faceBFile) {
      formData.append("source_face_b", state.faceBFile);
    }
    formData.append("swapper_model", swapperSelect.value);
    formData.append("enhancer_model", enhancerSelect.value);
    formData.append("enhancer_blend", blendSlider.value);
    formData.append("detector_model", detectorSelect.value);
    formData.append("pos_a", posAInput.value);
    formData.append("pos_b", posBInput.value);
    formData.append("ref_frame", refFrameInput.value || "0");

    // UI State: Running
    startJobBtn.disabled = true;
    startBtnText.textContent = "Pipeline Running...";
    jobProgressContainer.classList.remove("hidden");
    resultSection.classList.add("hidden");
    jobMainProgressBar.style.width = "0%";
    jobProgressPct.textContent = "0%";
    jobStageText.textContent = "Submitting job to backend...";

    // Reset GPU HUD
    gpu0Dot.className = "gpu-dot active";
    gpu1Dot.className = "gpu-dot active";
    gpu0Stage.textContent = "Preparing chunk0";
    gpu1Stage.textContent = "Preparing chunk1";
    gpu0ProgressBar.style.width = "0%";
    gpu1ProgressBar.style.width = "0%";
    gpu0Pct.textContent = "0%";
    gpu1Pct.textContent = "0%";

    try {
      logToConsole("[Job] Submitting parallel face swap job...");
      const response = await fetch(`${state.backendUrl}/api/jobs`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server returned ${response.status}`);
      }

      const jobData = await response.json();
      state.activeJobId = jobData.job_id;
      logToConsole(`[Job Created] ID: ${jobData.job_id}. Background pipeline started.`);

      // Start Polling Loop
      startPollingJob(jobData.job_id);
    } catch (err) {
      logToConsole(`[Job Failed] Submission error: ${err.message}`, true);
      jobStageText.textContent = `Error: ${err.message}`;
      startJobBtn.disabled = false;
      startBtnText.textContent = "Start Parallel Face Swap";
    }
  });

  // Polling Loop for Job Status
  function startPollingJob(jobId) {
    if (state.pollInterval) clearInterval(state.pollInterval);

    state.pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${state.backendUrl}/api/jobs/${jobId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const job = await response.json();

        // Update overall progress & stage
        jobMainProgressBar.style.width = `${job.progress}%`;
        jobProgressPct.textContent = `${job.progress}%`;
        jobStageText.textContent = job.stage || "Processing...";

        // Update Dual-GPU HUD
        if (job.gpu_status) {
          if (job.gpu_status.gpu0) {
            gpu0Stage.textContent = job.gpu_status.gpu0.status;
            gpu0ProgressBar.style.width = `${job.gpu_status.gpu0.progress}%`;
            gpu0Pct.textContent = `${job.gpu_status.gpu0.progress}%`;
          }
          if (job.gpu_status.gpu1) {
            gpu1Stage.textContent = job.gpu_status.gpu1.status;
            gpu1ProgressBar.style.width = `${job.gpu_status.gpu1.progress}%`;
            gpu1Pct.textContent = `${job.gpu_status.gpu1.progress}%`;
          }
        }

        // Sync logs
        if (job.logs && job.logs.length > 0) {
          const currentLogCount = logsConsole.querySelectorAll(".log-entry").length;
          if (job.logs.length > currentLogCount - 1) {
            for (let i = currentLogCount - 1; i < job.logs.length; i++) {
              logToConsole(job.logs[i]);
            }
          }
        }

        // Check completion
        if (job.status === "completed") {
          clearInterval(state.pollInterval);
          onJobCompleted(job);
        } else if (job.status === "failed") {
          clearInterval(state.pollInterval);
          onJobFailed(job);
        }
      } catch (err) {
        logToConsole(`[Polling Warning] Failed to fetch job status: ${err.message}`);
      }
    }, 1500);
  }

  // Job Finished Callback
  function onJobCompleted(job) {
    logToConsole("[Job Completed] Face swap completed successfully!", true);
    startJobBtn.disabled = false;
    startBtnText.textContent = "Start Parallel Face Swap";
    jobStageText.textContent = "Complete";
    jobMainProgressBar.style.width = "100%";
    jobProgressPct.textContent = "100%";

    gpu0Dot.className = "gpu-dot";
    gpu1Dot.className = "gpu-dot";
    gpu0Stage.textContent = "Complete";
    gpu1Stage.textContent = "Complete";

    // Show result video
    const downloadUrl = `${state.backendUrl}/api/jobs/${job.job_id}/download`;
    resultVideoPlayer.src = downloadUrl;
    downloadResultBtn.href = downloadUrl;
    resultSection.classList.remove("hidden");
    resultVideoPlayer.scrollIntoView({ behavior: "smooth" });
  }

  // Job Failed Callback
  function onJobFailed(job) {
    logToConsole(`[Job Failed] ${job.error || "Unknown pipeline error"}`, true);
    startJobBtn.disabled = false;
    startBtnText.textContent = "Start Parallel Face Swap";
    jobStageText.textContent = `Failed: ${job.error || "Execution error"}`;

    gpu0Dot.className = "gpu-dot";
    gpu1Dot.className = "gpu-dot";
    gpu0Stage.textContent = "Failed";
    gpu1Stage.textContent = "Failed";
  }
});

