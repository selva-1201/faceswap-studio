/**
 * FaceSwap Studio - Frontend Application Logic
 *
 * Handles remote backend connection, real-time CLI preview, file drop zones,
 * asynchronous job submission with upload progress, dual-GPU status polling,
 * delta log synchronization, cancellation support, and system resources HUD.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Determine initial backend URL
  const isFileProtocol = window.location.protocol === "file:";
  const initialBackendUrl = localStorage.getItem("faceswap_backend_url") || (isFileProtocol ? "" : window.location.origin);

  // Application State
  const state = {
    backendUrl: initialBackendUrl,
    videoFile: null,
    faceAFile: null,
    faceBFile: null,
    videoDuration: 0,
    activeJobId: null,
    pollInterval: null,
    lastLogIndex: 0,
    consecutivePollErrors: 0,
    maxUploadMb: 95,
  };

  // DOM Elements
  const backendUrlInput = document.getElementById("backendUrlInput");
  const testConnectionBtn = document.getElementById("testConnectionBtn");
  const connectionStatusPill = document.getElementById("connectionStatusPill");
  const connectionStatusText = document.getElementById("connectionStatusText");
  const cudaWarningBadge = document.getElementById("cudaWarningBadge");

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
  const enhanceFaceBCheckbox = document.getElementById("enhanceFaceBCheckbox");
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

  // Dual-GPU HUD & System Resources
  const gpuHudBadge = document.getElementById("gpuHudBadge");
  const gpu0Dot = document.getElementById("gpu0Dot");
  const gpu0Stage = document.getElementById("gpu0Stage");
  const gpu0Pct = document.getElementById("gpu0Pct");
  const gpu0ProgressBar = document.getElementById("gpu0ProgressBar");

  const gpu1Dot = document.getElementById("gpu1Dot");
  const gpu1Stage = document.getElementById("gpu1Stage");
  const gpu1Pct = document.getElementById("gpu1Pct");
  const gpu1ProgressBar = document.getElementById("gpu1ProgressBar");

  const resRam = document.getElementById("resRam");
  const resDisk = document.getElementById("resDisk");
  const resGpu = document.getElementById("resGpu");

  // Job Controls & Terminal
  const startJobBtn = document.getElementById("startJobBtn");
  const startBtnText = document.getElementById("startBtnText");
  const cancelJobBtn = document.getElementById("cancelJobBtn");
  const jobProgressContainer = document.getElementById("jobProgressContainer");
  const jobStageText = document.getElementById("jobStageText");
  const jobProgressPct = document.getElementById("jobProgressPct");
  const jobMainProgressBar = document.getElementById("jobMainProgressBar");
  const logsConsole = document.getElementById("logsConsole");

  // Result Section
  const resultSection = document.getElementById("resultSection");
  const resultVideoPlayer = document.getElementById("resultVideoPlayer");
  const downloadResultBtn = document.getElementById("downloadResultBtn");

  // Initialize Backend URL field
  backendUrlInput.value = state.backendUrl;
  backendUrlInput.addEventListener("change", () => {
    const cleanUrl = backendUrlInput.value.trim().replace(/\/+$/, "");
    state.backendUrl = cleanUrl;
    localStorage.setItem("faceswap_backend_url", cleanUrl);
    if (cleanUrl) {
      logToConsole(`[Config] Backend URL set to: ${cleanUrl}`);
      testBackendConnection();
    } else {
      connectionStatusPill.className = "status-pill status-disconnected";
      connectionStatusText.textContent = "Offline (Enter Tunnel URL)";
    }
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
    if (!state.backendUrl) {
      connectionStatusPill.className = "status-pill status-disconnected";
      connectionStatusText.textContent = "Enter Tunnel URL";
      return;
    }

    connectionStatusPill.className = "status-pill status-disconnected";
    connectionStatusText.textContent = "Connecting...";

    try {
      const response = await fetch(`${state.backendUrl}/api/system/status`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      connectionStatusPill.className = "status-pill status-connected";
      if (data.gpu_count > 0) {
        connectionStatusText.textContent = `Connected: ${data.gpu_count}x GPU Detected`;
        gpuHudBadge.textContent = `${data.gpu_count}x GPU Ready`;
      } else {
        connectionStatusText.textContent = "Connected: CPU Fallback";
        gpuHudBadge.textContent = "CPU Mode";
      }

      if (data.cuda_provider_ok === false) {
        cudaWarningBadge.classList.remove("hidden");
      } else {
        cudaWarningBadge.classList.add("hidden");
      }

      state.maxUploadMb = data.max_upload_mb || 95;

      logToConsole(
        `[Backend Connected] GPUs: ${data.gpu_count} (${(data.gpu_devices || []).join(", ") || "None"}), Workspace: ${data.workspace_dir}`,
        true
      );

      // Trigger resource inspection
      fetchSystemResources();
    } catch (err) {
      connectionStatusPill.className = "status-pill status-disconnected";
      connectionStatusText.textContent = "Disconnected";
      logToConsole(`[Connection Failed] Unable to reach ${state.backendUrl}: ${err.message}`, true);
    }
  }

  // Periodic System Resources Fetch
  async function fetchSystemResources() {
    if (!state.backendUrl) return;
    try {
      const res = await fetch(`${state.backendUrl}/api/system/resources`);
      if (!res.ok) return;
      const data = await res.json();

      if (data.ram && resRam) {
        resRam.textContent = `RAM: ${data.ram.available_mb || 0}MB free`;
      }
      if (data.disk_workspace && resDisk) {
        resDisk.textContent = `Disk: ${data.disk_workspace.free_gb || 0}GB free`;
      }
      if (data.gpus && data.gpus.length > 0 && resGpu) {
        const vramParts = data.gpus.map((g) => `GPU ${g.index}: ${Math.round(g.used_mb)}/${Math.round(g.total_mb)}MB`);
        resGpu.textContent = vramParts.join(" | ");
      }
    } catch (_) {}
  }

  testConnectionBtn.addEventListener("click", testBackendConnection);

  if (state.backendUrl) {
    testBackendConnection();
  } else {
    logToConsole("[System] Enter your remote Cloudflare Quick Tunnel URL in the top bar to connect.", true);
  }

  // Poll system resources every 10 seconds
  setInterval(fetchSystemResources, 10000);

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
    const blend = blendSlider.value;
    const posA = posAInput.value;
    const posB = posBInput.value;
    const refFrame = refFrameInput.value || "0";
    const hasFaceB = enableFaceBCheckbox.checked;
    const enhanceB = enhanceFaceBCheckbox ? enhanceFaceBCheckbox.checked : false;

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
      `  --face-detector-model ${detector} \\\n` +
      `  --face-selector-mode reference \\\n` +
      `  --face-selector-order left-right \\\n` +
      `  --reference-face-position ${posA} \\\n` +
      `  --reference-frame-number ${refFrame} \\\n` +
      `  --execution-providers cuda \\\n` +
      `  --execution-thread-count 2 \\\n` +
      `  --video-memory-strategy moderate \\\n` +
      `  --temp-frame-format jpeg`;

    let fullPreview = pass1Cmd;

    if (hasFaceB) {
      const pass2Processors = enhanceB ? "face_swapper face_enhancer" : "face_swapper";
      let pass2EnhancerFlags = "";
      if (enhanceB) {
        pass2EnhancerFlags = `  --face-enhancer-model ${enhancer} \\\n  --face-enhancer-blend ${blend} \\\n`;
      }

      const pass2Cmd =
        `\n\n# Pass 2: Face B (Chained from Pass 1 output)\n` +
        `python3 facefusion.py headless-run \\\n` +
        `  -s /kaggle/working/uploads/source_b.png \\\n` +
        `  -t /kaggle/working/chunks/chunk0_pass1.mp4 \\\n` +
        `  -o /kaggle/working/chunks/chunk0_pass2.mp4 \\\n` +
        `  --processors ${pass2Processors} \\\n` +
        `  --face-swapper-model ${swapper}${boostFlag} \\\n` +
        pass2EnhancerFlags +
        `  --face-detector-model ${detector} \\\n` +
        `  --face-selector-mode reference \\\n` +
        `  --face-selector-order left-right \\\n` +
        `  --reference-face-position ${posB} \\\n` +
        `  --reference-frame-number ${refFrame} \\\n` +
        `  --execution-providers cuda \\\n` +
        `  --execution-thread-count 2 \\\n` +
        `  --video-memory-strategy moderate \\\n` +
        `  --temp-frame-format jpeg`;
      fullPreview += pass2Cmd;
    }

    cliCommandCode.textContent = fullPreview;
  }

  // Bind change events to CLI preview
  [swapperSelect, enhancerSelect, detectorSelect, posAInput, posBInput, refFrameInput].forEach((el) => {
    el.addEventListener("change", updateCliPreview);
  });

  if (enhanceFaceBCheckbox) {
    enhanceFaceBCheckbox.addEventListener("change", updateCliPreview);
  }

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
          const mb = (file.size / (1024 * 1024)).toFixed(1);
          videoDurationBadge.textContent = `${previewEl.duration.toFixed(1)}s (${previewEl.videoWidth}x${previewEl.videoHeight}, ${mb}MB)`;
          logToConsole(`[Media] Target video selected: ${file.name} (${previewEl.duration.toFixed(2)}s, ${mb}MB)`);

          if (file.size > 95 * 1024 * 1024) {
            logToConsole(
              `[Warning] Video size (${mb}MB) is close to Cloudflare's 100MB limit. Ensure file does not exceed ${state.maxUploadMb}MB.`,
              true
            );
          }
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

  // Job Submission Handler (XMLHttpRequest for Real Upload Progress)
  startJobBtn.addEventListener("click", () => {
    if (!state.backendUrl) {
      alert("Please connect to your FaceSwap Studio backend before submitting.");
      return;
    }
    if (!state.videoFile) {
      alert("Please upload a target video file before starting.");
      return;
    }
    if (!state.faceAFile) {
      alert("Please upload Source Face A before starting.");
      return;
    }

    const videoSizeMb = state.videoFile.size / (1024 * 1024);
    if (videoSizeMb > state.maxUploadMb) {
      alert(
        `Video size (${videoSizeMb.toFixed(1)}MB) exceeds maximum allowed size of ${state.maxUploadMb}MB.\nCloudflare Quick Tunnel limits request bodies to 100MB.`
      );
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
    formData.append("enhance_face_b", enhanceFaceBCheckbox ? enhanceFaceBCheckbox.checked : false);

    // UI State: Running
    startJobBtn.disabled = true;
    startBtnText.textContent = "Uploading Media...";
    cancelJobBtn.classList.remove("hidden");
    jobProgressContainer.classList.remove("hidden");
    resultSection.classList.add("hidden");
    jobMainProgressBar.style.width = "0%";
    jobProgressPct.textContent = "0%";
    jobStageText.textContent = "Streaming upload to remote backend...";

    // Reset GPU HUD
    gpu0Dot.className = "gpu-dot active";
    gpu1Dot.className = "gpu-dot active";
    gpu0Stage.textContent = "Preparing chunk0";
    gpu1Stage.textContent = "Preparing chunk1";
    gpu0ProgressBar.style.width = "0%";
    gpu1ProgressBar.style.width = "0%";
    gpu0Pct.textContent = "0%";
    gpu1Pct.textContent = "0%";

    state.lastLogIndex = 0;
    state.consecutivePollErrors = 0;

    logToConsole("[Job] Initiating streaming upload to backend...");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${state.backendUrl}/api/jobs`);

    xhr.upload.addEventListener("progress", (evt) => {
      if (evt.lengthComputable) {
        const pct = Math.round((evt.loaded / evt.total) * 100);
        // Upload phase maps to 0% - 10% on the main progress bar
        jobMainProgressBar.style.width = `${Math.min(10, Math.round(pct * 0.1))}%`;
        jobStageText.textContent = `Uploading media files (${pct}%)...`;
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const jobData = JSON.parse(xhr.responseText);
          state.activeJobId = jobData.job_id;
          startBtnText.textContent = "Pipeline Running...";
          logToConsole(`[Job Created] ID: ${jobData.job_id}. Status: ${jobData.status}`);
          startPollingJob(jobData.job_id);
        } catch (e) {
          onJobFailed({ error: "Invalid JSON response from server" });
        }
      } else {
        let errDetail = `Server returned status ${xhr.status}`;
        try {
          const parsed = JSON.parse(xhr.responseText);
          if (parsed.detail) errDetail = parsed.detail;
        } catch (_) {}
        onJobFailed({ error: errDetail });
      }
    });

    xhr.addEventListener("error", () => {
      onJobFailed({ error: "Network error during upload to remote backend" });
    });

    xhr.send(formData);
  });

  // Cancel Job Handler
  cancelJobBtn.addEventListener("click", async () => {
    if (!state.activeJobId) return;
    if (!confirm("Are you sure you want to cancel the active face swap job?")) return;

    cancelJobBtn.disabled = true;
    jobStageText.textContent = "Cancelling job...";

    try {
      const res = await fetch(`${state.backendUrl}/api/jobs/${state.activeJobId}/cancel`, {
        method: "POST",
      });
      if (res.ok) {
        logToConsole("[Job] Job cancellation request confirmed by server.", true);
      }
    } catch (e) {
      logToConsole(`[Cancel Error] Failed to cancel job: ${e.message}`);
    }
  });

  // Polling Loop for Job Status (Delta Log Polling with ?since=)
  function startPollingJob(jobId) {
    if (state.pollInterval) clearInterval(state.pollInterval);

    let pollDelay = 1500;

    const poll = async () => {
      try {
        const response = await fetch(`${state.backendUrl}/api/jobs/${jobId}?since=${state.lastLogIndex}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const job = await response.json();

        state.consecutivePollErrors = 0;
        pollDelay = 1500;

        // Update overall progress & stage
        jobMainProgressBar.style.width = `${job.progress}%`;
        jobProgressPct.textContent = `${job.progress}%`;

        if (job.status === "queued") {
          const pos = job.queue_position ? ` (position ${job.queue_position})` : "";
          jobStageText.textContent = `Queued${pos}`;
        } else {
          jobStageText.textContent = job.stage || "Processing...";
        }

        // Update Dual-GPU HUD
        if (job.gpu_status) {
          if (job.gpu_status.gpu0) {
            gpu0Stage.textContent = job.gpu_status.gpu0.stage || job.gpu_status.gpu0.status;
            gpu0ProgressBar.style.width = `${job.gpu_status.gpu0.progress}%`;
            gpu0Pct.textContent = `${job.gpu_status.gpu0.progress}%`;
          }
          if (job.gpu_status.gpu1) {
            gpu1Stage.textContent = job.gpu_status.gpu1.stage || job.gpu_status.gpu1.status;
            gpu1ProgressBar.style.width = `${job.gpu_status.gpu1.progress}%`;
            gpu1Pct.textContent = `${job.gpu_status.gpu1.progress}%`;
          }
        }

        // Incremental log append
        if (job.logs && job.logs.length > 0) {
          for (const line of job.logs) {
            logToConsole(line);
          }
        }
        if (job.log_count !== undefined) {
          state.lastLogIndex = job.log_count;
        } else if (job.logs) {
          state.lastLogIndex += job.logs.length;
        }

        // Check completion status
        if (job.status === "completed") {
          clearInterval(state.pollInterval);
          onJobCompleted(job);
        } else if (job.status === "failed") {
          clearInterval(state.pollInterval);
          onJobFailed(job);
        } else if (job.status === "cancelled") {
          clearInterval(state.pollInterval);
          onJobCancelled(job);
        }
      } catch (err) {
        state.consecutivePollErrors += 1;
        if (state.consecutivePollErrors >= 5) {
          connectionStatusPill.className = "status-pill status-disconnected";
          connectionStatusText.textContent = "Server unreachable (possibly crashed/OOM)";
          jobStageText.textContent = "Server unreachable (possibly crashed/OOM). Retrying...";
          pollDelay = Math.min(10000, pollDelay * 1.5);
        }
      }
    };

    state.pollInterval = setInterval(poll, pollDelay);
  }

  // Job Finished Callback
  function onJobCompleted(job) {
    logToConsole("[Job Completed] Face swap completed successfully!", true);
    startJobBtn.disabled = false;
    startBtnText.textContent = "Start Parallel Face Swap";
    cancelJobBtn.classList.add("hidden");
    cancelJobBtn.disabled = false;
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
    cancelJobBtn.classList.add("hidden");
    cancelJobBtn.disabled = false;
    jobStageText.textContent = `Failed: ${job.error || "Execution error"}`;

    gpu0Dot.className = "gpu-dot";
    gpu1Dot.className = "gpu-dot";
    gpu0Stage.textContent = "Failed";
    gpu1Stage.textContent = "Failed";
  }

  // Job Cancelled Callback
  function onJobCancelled(job) {
    logToConsole(`[Job Cancelled] ${job.error || "Job was cancelled"}`, true);
    startJobBtn.disabled = false;
    startBtnText.textContent = "Start Parallel Face Swap";
    cancelJobBtn.classList.add("hidden");
    cancelJobBtn.disabled = false;
    jobStageText.textContent = "Cancelled";

    gpu0Dot.className = "gpu-dot";
    gpu1Dot.className = "gpu-dot";
    gpu0Stage.textContent = "Cancelled";
    gpu1Stage.textContent = "Cancelled";
  }
});
