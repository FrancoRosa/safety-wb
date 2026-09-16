/* app.js — YOLO26n in-browser tracking demo
 * Mirrors the Python `model.track(source, persist=True)` loop from
 * https://docs.ultralytics.com/modes/track, but running fully client-side
 * with onnxruntime-web + tracker.js instead of the ultralytics package.
 */

const COCO_CLASSES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

const IMG_SIZE = 640; // must match the imgsz used at export time
const DEFAULT_MODEL_PATH = "./yolo26n.onnx";

ort.env.wasm.numThreads = 1;

const els = {
  modelInput: document.getElementById("modelInput"),
  modelStatus: document.getElementById("modelStatus"),
  sourceSelect: document.getElementById("sourceSelect"),
  videoInput: document.getElementById("videoInput"),
  videoInputWrap: document.getElementById("videoInputWrap"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  fullscreenBtn: document.getElementById("fullscreenBtn"),
  menuBtn: document.getElementById("menuBtn"),
  closeControlsBtn: document.getElementById("closeControlsBtn"),
  controlsPanel: document.getElementById("controlsPanel"),
  scrim: document.getElementById("scrim"),
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  stage: document.getElementById("stage"),
  confSlider: document.getElementById("confSlider"),
  confVal: document.getElementById("confVal"),
  iouSlider: document.getElementById("iouSlider"),
  iouVal: document.getElementById("iouVal"),
  bufferSlider: document.getElementById("bufferSlider"),
  bufferVal: document.getElementById("bufferVal"),
  classCheckboxList: document.getElementById("classCheckboxList"),
  classAllBtn: document.getElementById("classAllBtn"),
  classNoneBtn: document.getElementById("classNoneBtn"),
  trailToggle: document.getElementById("trailToggle"),
  fpsStat: document.getElementById("fpsStat"),
  detStat: document.getElementById("detStat"),
  trackStat: document.getElementById("trackStat"),
  totalStat: document.getElementById("totalStat"),
};

// Populate source select with available video devices (cameras)
async function populateCameraOptions() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices)
    return;
  try {
    let devices = await navigator.mediaDevices.enumerateDevices();
    let cams = devices.filter((d) => d.kind === "videoinput");

    // If labels are empty (no permission yet) or no devices found, prompt for a short getUserMedia
    const labelsMissing = cams.length > 0 ? cams.every((c) => !c.label) : false;
    if (
      (labelsMissing && cams.length > 0) ||
      (cams.length === 0 && navigator.mediaDevices.getUserMedia)
    ) {
      let s = null;
      try {
        // Request a short-lived stream to trigger permission prompt and expose device labels
        s = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      } catch (err) {
        console.warn("Camera permission denied or no camera available:", err);
      }
      devices = await navigator.mediaDevices.enumerateDevices();
      cams = devices.filter((d) => d.kind === "videoinput");
      if (s) s.getTracks().forEach((t) => t.stop());
    }

    // Clear existing options and add camera entries
    els.sourceSelect.innerHTML = "";
    if (cams.length === 0) {
      const opt = document.createElement("option");
      opt.value = "webcam";
      opt.textContent = "Webcam (no cameras found)";
      els.sourceSelect.appendChild(opt);
    } else {
      cams.forEach((cam, i) => {
        const opt = document.createElement("option");
        // value encodes deviceId so we can open specific camera
        opt.value = `camera:${cam.deviceId}`;
        opt.textContent = cam.label || `Camera ${i + 1}`;
        els.sourceSelect.appendChild(opt);
      });
    }

    // Add video file option last
    const fileOpt = document.createElement("option");
    fileOpt.value = "file";
    fileOpt.textContent = "Video file";
    els.sourceSelect.appendChild(fileOpt);

    // Restore preferred source if available, otherwise pick a sensible default
    const preferred = localStorage.getItem("preferredSource");
    const values = Array.from(els.sourceSelect.options).map((o) => o.value);
    if (preferred && values.includes(preferred)) {
      els.sourceSelect.value = preferred;
    } else if (
      preferred &&
      preferred.startsWith("camera:") &&
      cams.length > 0
    ) {
      // previously selected camera not present; fall back to first camera
      els.sourceSelect.value = `camera:${cams[0].deviceId}`;
    } else if (cams.length > 0) {
      // default to first available camera
      els.sourceSelect.value = `camera:${cams[0].deviceId}`;
    } else {
      const first = els.sourceSelect.options[0];
      if (first) els.sourceSelect.value = first.value;
    }
    // Show/hide file chooser according to selection
    els.videoInputWrap.classList.toggle(
      "hidden",
      els.sourceSelect.value !== "file",
    );
  } catch (err) {
    console.warn("Failed to enumerate devices:", err);
  }
}

// Kick off camera enumeration
populateCameraOptions();

let session = null;
let tracker = null;
let running = false;
let rafId = null;
let inputName = "images";
let maxSeenId = 0;
let lastFrameTime = performance.now();
let fpsSmoothed = 0;

const ctx = els.overlay.getContext("2d");

function log(msg) {
  console.log(`[${new Date().toLocaleTimeString()}] ${msg}`);
}

function idColor(id) {
  // Deterministic, well-separated hue per track ID (golden-angle spacing).
  const hue = (id * 137.508) % 360;
  return `hsl(${hue}, 85%, 60%)`;
}

// ---------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------

async function loadModelFromBuffer(buf, label) {
  const providers = ["webgpu", "wasm"];
  let loaded = null;
  let backendUsed = "";
  for (const ep of providers) {
    try {
      loaded = await ort.InferenceSession.create(buf, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      backendUsed = ep;
      break;
    } catch (err) {
      console.warn(`EP ${ep} failed, trying next`, err);
    }
  }
  if (!loaded) throw new Error("No available execution provider");
  session = loaded;
  inputName = session.inputNames[0];
  els.modelStatus.textContent = `Loaded: ${label}`;
  els.modelStatus.className = "status ok";
  els.startBtn.disabled = false;
  log(`Model ready — input "${inputName}", backend ${backendUsed}`);
  window.setTimeout(() => maybeAutoStart(), 150);
}

async function loadModelFile(file) {
  if (!file) return;
  els.modelStatus.textContent = "Loading model…";
  els.modelStatus.className = "status pending";
  try {
    const buf = await file.arrayBuffer();
    await loadModelFromBuffer(buf, file.name);
  } catch (err) {
    console.error(err);
    els.modelStatus.textContent = "Failed to load model — see console";
    els.modelStatus.className = "status err";
    log(`Model load failed: ${err.message}`);
  }
}

async function loadDefaultModel() {
  els.modelStatus.textContent = "Loading default model…";
  els.modelStatus.className = "status pending";
  try {
    const response = await fetch(DEFAULT_MODEL_PATH, { cache: "force-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buf = await response.arrayBuffer();
    await loadModelFromBuffer(buf, "yolo26n.onnx");
  } catch (err) {
    console.error(err);
    els.modelStatus.textContent =
      "Default model could not load — choose a file manually";
    els.modelStatus.className = "status err";
    log(`Default model load failed: ${err.message}`);
  }
}

els.modelInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  await loadModelFile(file);
});

loadDefaultModel();

// ---------------------------------------------------------------------
// Source handling (webcam vs local file)
// ---------------------------------------------------------------------

els.sourceSelect.addEventListener("change", async () => {
  const val = els.sourceSelect.value;
  localStorage.setItem("preferredSource", val);
  const isFile = val === "file";
  els.videoInputWrap.classList.toggle("hidden", !isFile);

  if (running) {
    // If tracking is active, switch the video source on-the-fly
    try {
      await setupSource();
      els.overlay.width = els.video.videoWidth || 960;
      els.overlay.height = els.video.videoHeight || 720;
      log(`Switched source to ${val}`);
    } catch (err) {
      log(`Source switch error: ${err.message}`);
    }
  } else {
    maybeAutoStart();
  }
});

els.videoInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  els.video.srcObject = null;
  els.video.src = URL.createObjectURL(file);
  els.video.loop = true;
  maybeAutoStart();
});

async function setupSource() {
  const val = els.sourceSelect.value;

  // Stop any existing camera tracks before switching
  try {
    const cur = els.video.srcObject;
    if (cur && cur.getTracks) cur.getTracks().forEach((t) => t.stop());
  } catch (e) {
    // ignore
  }

  if (val && val.startsWith("camera:")) {
    const deviceId = val.split(":")[1];
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: deviceId }, width: 960, height: 720 },
      audio: false,
    });
    els.video.srcObject = stream;
    els.video.src = "";
  } else if (val === "webcam") {
    // Fallback generic webcam
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 960, height: 720 },
      audio: false,
    });
    els.video.srcObject = stream;
    els.video.src = "";
  } else if (!els.video.src) {
    throw new Error("Choose a video file first");
  }

  await els.video.play();
}

// ---------------------------------------------------------------------
// Slider wiring
// ---------------------------------------------------------------------

function wireSlider(slider, out, fmt = (v) => v) {
  const update = () => (out.textContent = fmt(slider.value));
  slider.addEventListener("input", update);
  update();
}
wireSlider(els.confSlider, els.confVal, (v) => Number(v).toFixed(2));
wireSlider(els.iouSlider, els.iouVal, (v) => Number(v).toFixed(2));
wireSlider(els.bufferSlider, els.bufferVal, (v) => `${v}f`);

// ---------------------------------------------------------------------
// Preprocessing: letterbox resize -> CHW float32 tensor
// ---------------------------------------------------------------------

const prepCanvas = document.createElement("canvas");
prepCanvas.width = IMG_SIZE;
prepCanvas.height = IMG_SIZE;
const prepCtx = prepCanvas.getContext("2d", { willReadFrequently: true });

function letterbox(video) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(IMG_SIZE / vw, IMG_SIZE / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padX = Math.floor((IMG_SIZE - nw) / 2);
  const padY = Math.floor((IMG_SIZE - nh) / 2);

  prepCtx.fillStyle = "rgb(114,114,114)"; // YOLO's standard pad color
  prepCtx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  prepCtx.drawImage(video, 0, 0, vw, vh, padX, padY, nw, nh);

  const { data } = prepCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const chw = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    const o = i * 4;
    chw[i] = data[o] / 255; // R
    chw[plane + i] = data[o + 1] / 255; // G
    chw[2 * plane + i] = data[o + 2] / 255; // B
  }
  return {
    tensor: new ort.Tensor("float32", chw, [1, 3, IMG_SIZE, IMG_SIZE]),
    scale,
    padX,
    padY,
  };
}

// ---------------------------------------------------------------------
// Postprocessing: (1, 300, 6) end-to-end output -> detections in
// original video pixel coordinates. No NMS needed (YOLO26 is NMS-free).
// ---------------------------------------------------------------------

function decode(output, scale, padX, padY, confThresh, allowedClasses) {
  const data = output.data;
  const numDet = output.dims[1];
  const stride = output.dims[2]; // 6
  const dets = [];
  for (let i = 0; i < numDet; i++) {
    const o = i * stride;
    const score = data[o + 4];
    if (score < confThresh) continue;
    const cls = Math.round(data[o + 5]);
    if (!allowedClasses.has(cls)) continue;
    let x1 = data[o + 0];
    let y1 = data[o + 1];
    let x2 = data[o + 2];
    let y2 = data[o + 3];
    // undo letterbox
    x1 = (x1 - padX) / scale;
    y1 = (y1 - padY) / scale;
    x2 = (x2 - padX) / scale;
    y2 = (y2 - padY) / scale;
    dets.push({ box: [x1, y1, x2, y2], score, cls });
  }
  return dets;
}

// ---------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------

function draw(tracks) {
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);

  for (const tr of tracks) {
    const color = idColor(tr.id);
    const [x1, y1, x2, y2] = tr.box;

    if (els.trailToggle.checked && tr.trail.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.65;
      ctx.lineWidth = 2;
      ctx.moveTo(tr.trail[0][0], tr.trail[0][1]);
      for (const [px, py] of tr.trail) ctx.lineTo(px, py);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `#${tr.id} ${COCO_CLASSES[tr.cls] ?? tr.cls} ${(tr.score * 100).toFixed(0)}%`;
    ctx.font = "600 13px 'JetBrains Mono', monospace";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1 - 1, y1 - 20, tw + 10, 20);
    ctx.fillStyle = "#0a0e0c";
    ctx.fillText(label, x1 + 4, y1 - 5);
  }
}

// ---------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------

async function frameLoop() {
  if (!running) return;
  const video = els.video;

  if (video.readyState >= 2) {
    const { tensor, scale, padX, padY } = letterbox(video);
    const outputs = await session.run({ [inputName]: tensor });
    const outName = session.outputNames[0];
    const output = outputs[outName];

    const confThresh = Number(els.confSlider.value);
    const iouThresh = Number(els.iouSlider.value);
    tracker.iouThresh = iouThresh;
    tracker.trackBuffer = Number(els.bufferSlider.value);

    const dets = decode(
      output,
      scale,
      padX,
      padY,
      confThresh,
      selectedClasses,
    );
    const tracks = tracker.update(dets);

    for (const tr of tracks) maxSeenId = Math.max(maxSeenId, tr.id);

    draw(tracks);

    els.detStat.textContent = dets.length;
    els.trackStat.textContent = tracks.length;
    els.totalStat.textContent = maxSeenId;

    const now = performance.now();
    const inst = 1000 / (now - lastFrameTime);
    fpsSmoothed = fpsSmoothed ? fpsSmoothed * 0.9 + inst * 0.1 : inst;
    lastFrameTime = now;
    els.fpsStat.textContent = fpsSmoothed.toFixed(1);
  }

  rafId = requestAnimationFrame(frameLoop);
}

// ---------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------

async function startTracking() {
  if (!session || running) return;
  try {
    await setupSource();
  } catch (err) {
    log(`Source error: ${err.message}`);
    return;
  }

  els.overlay.width = els.video.videoWidth || 960;
  els.overlay.height = els.video.videoHeight || 720;
  els.stage.classList.add("live");

  tracker = new ByteTrackLite({
    highThresh: Number(els.confSlider.value),
    lowThresh: Math.max(0.05, Number(els.confSlider.value) - 0.3),
    iouThresh: Number(els.iouSlider.value),
    trackBuffer: Number(els.bufferSlider.value),
  });
  maxSeenId = 0;
  running = true;
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  log("Tracking started (persist=True equivalent — IDs carry across frames)");
  lastFrameTime = performance.now();
  frameLoop();
}

function stopTracking() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  const stream = els.video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  els.stage.classList.remove("live");
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  log("Tracking stopped");
}

function maybeAutoStart() {
  if (!session || running) return;
  if (els.sourceSelect.value === "file" && !els.video.src) return;
  window.setTimeout(() => startTracking(), 100);
}

els.startBtn.addEventListener("click", startTracking);
els.stopBtn.addEventListener("click", stopTracking);

window.addEventListener("load", () => {
  // Populate camera list on load so user can pick a camera before starting
  populateCameraOptions().catch((e) =>
    console.warn("populateCameraOptions error:", e),
  );
  window.setTimeout(() => maybeAutoStart(), 250);
});

els.fullscreenBtn.addEventListener("click", async () => {
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  } catch (err) {
    log(`Fullscreen error: ${err.message}`);
  }
});

document.addEventListener("fullscreenchange", () => {
  const active = Boolean(document.fullscreenElement);
  els.fullscreenBtn.setAttribute("aria-pressed", String(active));
  els.fullscreenBtn.title = active ? "Exit fullscreen" : "Fullscreen";
});

// ---------------------------------------------------------------------
// Controls drawer (hamburger)
// ---------------------------------------------------------------------

function setControlsOpen(open) {
  els.controlsPanel.classList.toggle("open", open);
  els.scrim.classList.toggle("show", open);
  els.menuBtn.setAttribute("aria-expanded", String(open));
}
els.menuBtn.addEventListener("click", () =>
  setControlsOpen(!els.controlsPanel.classList.contains("open")),
);
els.closeControlsBtn.addEventListener("click", () => setControlsOpen(false));
els.scrim.addEventListener("click", () => setControlsOpen(false));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setControlsOpen(false);
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch((err) => {
      console.warn("Service worker registration failed:", err);
    });
  });
}

// ---------------------------------------------------------------------
// Class filter (checkboxes, any combination — persisted across sessions)
// ---------------------------------------------------------------------

function loadClassFilter() {
  try {
    const raw = localStorage.getItem("classFilter");
    if (raw) {
      const ids = JSON.parse(raw).filter(
        (n) => Number.isInteger(n) && n >= 0 && n < COCO_CLASSES.length,
      );
      if (ids.length) return new Set(ids);
    }
  } catch (err) {
    console.warn("Failed to read stored class filter:", err);
  }
  return new Set([0]); // default: person
}

let selectedClasses = loadClassFilter();

function persistClassFilter() {
  localStorage.setItem(
    "classFilter",
    JSON.stringify(Array.from(selectedClasses)),
  );
}

function syncClassCheckboxes() {
  els.classCheckboxList
    .querySelectorAll("input[type=checkbox]")
    .forEach((cb) => {
      cb.checked = selectedClasses.has(Number(cb.value));
    });
}

(function initClassFilters() {
  const frag = document.createDocumentFragment();
  COCO_CLASSES.forEach((name, idx) => {
    const label = document.createElement("label");
    label.className = "class-checkbox";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = idx;
    input.checked = selectedClasses.has(idx);
    input.addEventListener("change", () => {
      if (input.checked) selectedClasses.add(idx);
      else selectedClasses.delete(idx);
      persistClassFilter();
    });
    label.appendChild(input);
    label.appendChild(document.createTextNode(name));
    frag.appendChild(label);
  });
  els.classCheckboxList.appendChild(frag);
})();

els.classAllBtn.addEventListener("click", () => {
  selectedClasses = new Set(COCO_CLASSES.map((_, i) => i));
  syncClassCheckboxes();
  persistClassFilter();
});
els.classNoneBtn.addEventListener("click", () => {
  selectedClasses.clear();
  syncClassCheckboxes();
  persistClassFilter();
});
