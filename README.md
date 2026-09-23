# YOLO26 Browser Tracking Demo

A self-contained HTML/CSS/JS demo that runs [YOLO26n](https://docs.ultralytics.com/models/yolo26)
detection **and** multi-object tracking entirely in the browser — no server, no Python at
runtime. It's inspired by the `model.track(source, persist=True)` loop from the
[Ultralytics Track docs](https://docs.ultralytics.com/modes/track), reimplemented with:

- **[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)** for running the YOLO26n
  ONNX model (WebGPU with automatic fallback to WASM).
- **`tracker.js`** — a small dependency-free "ByteTrack-lite" tracker (two-stage IoU
  association + constant-velocity prediction) that assigns persistent IDs across frames,
  the same core idea as Ultralytics' `bytetrack.yaml` tracker, minus Kalman filtering,
  ReID, and camera-motion compensation.

## 1. Get a `yolo26n.onnx` file

Ultralytics doesn't publish a ready-made browser build, so export one yourself (takes
about 2 minutes on CPU):

```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=640 opset=17 simplify=True
```

This produces `yolo26n.onnx` using YOLO26's default **end-to-end, NMS-free** head: output
shape `(1, 300, 6)` = `[x1, y1, x2, y2, confidence, class_id]` in letterboxed pixel
coordinates, already deduplicated. `app.js`'s decoder assumes exactly this layout — if you
export with `end2end=False` you'll need to add NMS to `decode()` in `app.js`.

Other sizes (`yolo26s/m/l/x.pt`) work too, just export the matching `.onnx` and expect
lower FPS on CPU/WebGPU-less machines.

## 2. Run it

Any static file server works (opening `index.html` directly may block the webcam due to
`file://` permissions in some browsers):

```bash
cd yolo26-track-web
python3 -m http.server 8000
# then open http://localhost:8000
```

In the page: **01 Model** → upload `yolo26n.onnx` → **02 Source** → pick webcam or a video
file → **Start Tracking**.

## What's in here

| File | Purpose |
|---|---|
| `index.html` | Layout: video/canvas stage, model + source + parameter controls, telemetry |
| `style.css` | Visual styling only |
| `tracker.js` | `ByteTrackLite` — the two-stage IoU tracker, framework-agnostic |
| `app.js` | Model loading, letterbox preprocessing, decoding `(1,300,6)` output, drawing, main loop |

## Usage statistics (`/admin/`)

The tracker records usage in the browser's own IndexedDB (`stats-db.js`) — nothing
leaves the device:

- **Sessions** — when the page was opened and last alive (a 30 s heartbeat stands in
  for "closed", since tabs can be killed without warning).
- **Person sightings** — one sample per tracked person every 5 s, with the feet
  position (box bottom-centre) normalised 0–1 to the camera frame. In 360 modes the
  position is mapped back to the raw 360 frame, so it doesn't depend on where the
  view was pointed.

Open `/admin/` **in the same browser on the same device** to see it per day and
time window (default 06:00–18:00): stat tiles, people-per-15-min chart with app-open
bands, a map of positions, the sessions list, and a CSV export. Data older than 90
days is pruned automatically. The page has no login, so anyone who can open the app
on that device can see it.

## Tuning parameters (mirrors the Python tracker args)

- **Confidence threshold** — floor for a detection to count at all; also used as the
  high-confidence bar for stage-1 matching / spawning new tracks (like `track_high_thresh`).
- **Match IoU threshold** — minimum overlap between a track's predicted box and a detection
  to count as the same object (like `match_thresh`).
- **Track buffer** — frames a lost track is kept alive, hoping the person reappears (like
  `track_buffer`).
- **Class filter** — defaults to `person` per your use case; switch to `All classes` or any
  other COCO class.

## Honest limitations vs. the Python `ultralytics` package

This is a teaching/demo tool, not a drop-in BoT-SORT replacement:

- No Kalman filter — constant-velocity prediction only, so tracks drift more under fast,
  non-linear motion.
- No ReID / appearance matching, so look-alike people crossing paths can swap IDs more
  often than BoT-SORT with `with_reid: True`.
- No camera-motion compensation, so a moving/shaky camera will hurt ID stability more than
  it would with BoT-SORT's CMC.

For production-grade tracking (BoT-SORT, ReID, GMC), run the actual `ultralytics` Python
package server-side, or port `ultralytics/trackers/` more directly to JS/WASM.
