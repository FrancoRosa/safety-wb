/**
 * tracker.js
 * ----------------------------------------------------------------------
 * A dependency-free, browser-side re-implementation of the *idea* behind
 * Ultralytics' ByteTrack backend (see docs.ultralytics.com/modes/track).
 *
 * It is intentionally simple compared to BoT-SORT / ByteTrack in the
 * `ultralytics` Python package (no Kalman filter, no camera-motion
 * compensation, no ReID) but it follows the same two-stage association
 * idea that makes ByteTrack work well:
 *
 *   1. Match ACTIVE tracks against HIGH-confidence detections (IoU).
 *   2. Try to rescue still-unmatched ACTIVE tracks using LOW-confidence
 *      detections that were discarded by stage 1 (recovers boxes that
 *      dipped in score during partial occlusion, instead of dropping
 *      the track's identity).
 *   3. Any remaining unmatched high-confidence detections spawn new
 *      tracks. Unmatched tracks are kept "lost" for `trackBuffer` frames
 *      (in case the person reappears) before being deleted for good.
 *
 * Motion is predicted with a constant-velocity model (no Kalman gain
 * tuning needed), which is enough for webcam / video frame rates.
 * ----------------------------------------------------------------------
 */

class Track {
  constructor(id, box, score, cls, frame) {
    this.id = id;
    this.box = box; // [x1, y1, x2, y2]
    this.score = score;
    this.cls = cls;
    this.velocity = [0, 0]; // center velocity (px/frame)
    this.hits = 1; // total successful matches
    this.age = 0; // frames since creation
    this.timeSinceUpdate = 0; // frames since last matched
    this.startFrame = frame;
    this.trail = [centerOf(box)]; // for drawing motion paths
    this.maxTrail = 30;
  }

  center() {
    return centerOf(this.box);
  }

  // Constant-velocity prediction of where the box should be this frame.
  predict() {
    const [x1, y1, x2, y2] = this.box;
    const [vx, vy] = this.velocity;
    return [x1 + vx, y1 + vy, x2 + vx, y2 + vy];
  }

  update(box, score, cls) {
    const oldC = this.center();
    this.box = box;
    this.score = score;
    this.cls = cls;
    const newC = centerOf(box);
    // Light smoothing on velocity so a single noisy box doesn't whip the track.
    const alpha = 0.6;
    this.velocity = [
      alpha * (newC[0] - oldC[0]) + (1 - alpha) * this.velocity[0],
      alpha * (newC[1] - oldC[1]) + (1 - alpha) * this.velocity[1],
    ];
    this.hits += 1;
    this.timeSinceUpdate = 0;
    this.trail.push(newC);
    if (this.trail.length > this.maxTrail) this.trail.shift();
  }

  markMissed() {
    // Keep drifting with last known velocity while lost, softly decayed.
    this.box = this.predict();
    this.velocity = [this.velocity[0] * 0.9, this.velocity[1] * 0.9];
    this.timeSinceUpdate += 1;
  }
}

function centerOf(box) {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

function iou(a, b) {
  const xx1 = Math.max(a[0], b[0]);
  const yy1 = Math.max(a[1], b[1]);
  const xx2 = Math.min(a[2], b[2]);
  const yy2 = Math.min(a[3], b[3]);
  const w = Math.max(0, xx2 - xx1);
  const h = Math.max(0, yy2 - yy1);
  const inter = w * h;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

// Greedy IoU matching (highest IoU pairs first). Good enough at
// interactive frame rates and avoids pulling in a Hungarian-algorithm
// dependency for a browser demo.
function greedyIouMatch(tracks, detections, iouThresh) {
  const pairs = [];
  for (let t = 0; t < tracks.length; t++) {
    const pred = tracks[t].predict();
    for (let d = 0; d < detections.length; d++) {
      const score = iou(pred, detections[d].box);
      if (score >= iouThresh) pairs.push([score, t, d]);
    }
  }
  pairs.sort((a, b) => b[0] - a[0]);

  const usedTracks = new Set();
  const usedDets = new Set();
  const matches = []; // [trackIdx, detIdx]
  for (const [, t, d] of pairs) {
    if (usedTracks.has(t) || usedDets.has(d)) continue;
    usedTracks.add(t);
    usedDets.add(d);
    matches.push([t, d]);
  }
  return { matches, usedTracks, usedDets };
}

class ByteTrackLite {
  /**
   * @param {Object} opts
   * @param {number} opts.highThresh  confidence for stage-1 association / new tracks
   * @param {number} opts.lowThresh   confidence floor for stage-2 rescue association
   * @param {number} opts.iouThresh   min IoU to accept a match
   * @param {number} opts.trackBuffer frames a lost track survives before deletion
   */
  constructor(opts = {}) {
    this.highThresh = opts.highThresh ?? 0.5;
    this.lowThresh = opts.lowThresh ?? 0.1;
    this.iouThresh = opts.iouThresh ?? 0.3;
    this.trackBuffer = opts.trackBuffer ?? 30;
    this.tracks = [];
    this._nextId = 1;
    this.frame = 0;
  }

  reset() {
    this.tracks = [];
    this._nextId = 1;
    this.frame = 0;
  }

  /**
   * @param {Array<{box:number[], score:number, cls:number}>} detections
   * @returns {Track[]} currently active (recently updated) tracks
   */
  update(detections) {
    this.frame += 1;
    const highDets = detections.filter((d) => d.score >= this.highThresh);
    const lowDets = detections.filter(
      (d) => d.score >= this.lowThresh && d.score < this.highThresh
    );

    // ---- Stage 1: active tracks vs high-confidence detections ----
    const { matches: m1, usedTracks: ut1, usedDets: ud1 } = greedyIouMatch(
      this.tracks,
      highDets,
      this.iouThresh
    );
    for (const [t, d] of m1) {
      const det = highDets[d];
      this.tracks[t].update(det.box, det.score, det.cls);
    }

    // ---- Stage 2: still-unmatched tracks vs low-confidence detections ----
    const remainingTracks = this.tracks
      .map((tr, idx) => idx)
      .filter((idx) => !ut1.has(idx));
    const remainingTrackObjs = remainingTracks.map((idx) => this.tracks[idx]);
    const { matches: m2, usedDets: ud2 } = greedyIouMatch(
      remainingTrackObjs,
      lowDets,
      Math.max(this.iouThresh, 0.4) // stricter, low-conf boxes are noisier
    );
    const matchedStage2 = new Set();
    for (const [tLocal, d] of m2) {
      const trackIdx = remainingTracks[tLocal];
      const det = lowDets[d];
      this.tracks[trackIdx].update(det.box, det.score, det.cls);
      matchedStage2.add(trackIdx);
    }

    // ---- Mark everything else missed, then prune stale tracks ----
    for (let idx = 0; idx < this.tracks.length; idx++) {
      if (!ut1.has(idx) && !matchedStage2.has(idx)) {
        this.tracks[idx].markMissed();
      }
      this.tracks[idx].age += 1;
    }
    this.tracks = this.tracks.filter(
      (tr) => tr.timeSinceUpdate <= this.trackBuffer
    );

    // ---- Spawn new tracks for unmatched high-confidence detections ----
    for (let d = 0; d < highDets.length; d++) {
      if (ud1.has(d)) continue;
      const det = highDets[d];
      const tr = new Track(this._nextId++, det.box, det.score, det.cls, this.frame);
      this.tracks.push(tr);
    }

    // Only expose tracks that are currently visible (updated this frame or
    // within a short grace period) so drifting/occluded boxes don't render.
    return this.tracks.filter((tr) => tr.timeSinceUpdate <= 3);
  }
}

// Exposed for app.js (plain <script> tags, no bundler)
window.ByteTrackLite = ByteTrackLite;
