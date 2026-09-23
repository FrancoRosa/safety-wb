/* pano.js — YouTube-style 360° viewer
 * Treats the incoming frame as an equirectangular (2:1) panorama and renders
 * a rectilinear "virtual camera" view of it with WebGL. Every output pixel
 * casts a ray from the camera (yaw / pitch / fov), converts it to
 * longitude/latitude and samples the panorama there — exactly how 360 video
 * players flatten the sphere into a normal-looking picture.
 *
 * "planet" mode swaps the rectilinear lens for a stereographic one aimed
 * straight down, producing the classic "little planet": the ground curls
 * into a globe in the middle and the sky wraps around the edges.
 */

class PanoViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.yaw = 0; // radians, + = look right
    this.pitch = 0; // radians, + = look up
    this.fov = (90 * Math.PI) / 180; // vertical field of view
    this.minFov = (30 * Math.PI) / 180;
    this.maxFov = (120 * Math.PI) / 180;
    this.planet = false;
    // Stereographic plane radius at the screen's top/bottom edge. The
    // horizon sits at radius 2, so 4 puts it halfway to the edge.
    this.planetScale = 4;

    // preserveDrawingBuffer so the detector can drawImage() the rendered view.
    const gl = canvas.getContext("webgl", {
      preserveDrawingBuffer: true,
      antialias: false,
    });
    if (!gl) throw new Error("WebGL not available");
    this.gl = gl;

    const vs = `
      attribute vec2 aPos;
      varying vec2 vNdc;
      void main() {
        vNdc = aPos;
        gl_Position = vec4(aPos, 0.0, 1.0);
      }`;
    const fs = `
      precision highp float;
      varying vec2 vNdc;
      uniform sampler2D uTex;
      uniform float uYaw;
      uniform float uPitch;
      uniform float uTanHalfFov;
      uniform float uAspect;
      uniform bool uPlanet;
      uniform float uPlanetScale;
      const float PI = 3.141592653589793;
      void main() {
        // Ray in camera space (x right, y up, z forward).
        vec3 dir;
        if (uPlanet) {
          // Inverse stereographic: plane radius r -> angle from the view
          // axis theta = 2*atan(r/2), so the whole sphere fits on screen.
          vec2 p = vec2(vNdc.x * uAspect, vNdc.y) * uPlanetScale;
          float r = length(p);
          float theta = 2.0 * atan(r * 0.5);
          vec2 d = r > 0.0 ? p / r : vec2(0.0);
          dir = vec3(d * sin(theta), cos(theta));
        } else {
          dir = normalize(vec3(
            vNdc.x * uTanHalfFov * uAspect,
            vNdc.y * uTanHalfFov,
            1.0));
        }
        // Pitch (rotate around X), then yaw (rotate around Y).
        float cp = cos(uPitch), sp = sin(uPitch);
        dir = vec3(dir.x, dir.y * cp + dir.z * sp, -dir.y * sp + dir.z * cp);
        float cy = cos(uYaw), sy = sin(uYaw);
        dir = vec3(dir.x * cy + dir.z * sy, dir.y, -dir.x * sy + dir.z * cy);

        float lon = atan(dir.x, dir.z);            // -PI..PI
        float lat = asin(clamp(dir.y, -1.0, 1.0)); // -PI/2..PI/2
        vec2 uv = vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);
        gl_FragColor = texture2D(uTex, uv);
      }`;

    const prog = gl.createProgram();
    for (const [type, src] of [
      [gl.VERTEX_SHADER, vs],
      [gl.FRAGMENT_SHADER, fs],
    ]) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(sh));
      gl.attachShader(prog, sh);
    }
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );
    const aPos = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    // NPOT-safe params; REPEAT isn't allowed for NPOT textures in WebGL1, so
    // the longitude seam is handled by the tiny clamp at the texture edge.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    this.u = {
      yaw: gl.getUniformLocation(prog, "uYaw"),
      pitch: gl.getUniformLocation(prog, "uPitch"),
      tanHalfFov: gl.getUniformLocation(prog, "uTanHalfFov"),
      aspect: gl.getUniformLocation(prog, "uAspect"),
      planet: gl.getUniformLocation(prog, "uPlanet"),
      planetScale: gl.getUniformLocation(prog, "uPlanetScale"),
    };
  }

  // Match the drawing buffer to the on-screen aspect ratio so the view isn't
  // stretched, capped so the render + detector readback stay cheap.
  resize(cssW, cssH, maxW = 1280) {
    const scale = Math.min(1, maxW / Math.max(1, cssW));
    const w = Math.max(2, Math.round(cssW * scale));
    const h = Math.max(2, Math.round(cssH * scale));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  look(dYaw, dPitch) {
    this.yaw = (this.yaw + dYaw) % (2 * Math.PI);
    const lim = Math.PI / 2;
    this.pitch = Math.max(-lim, Math.min(lim, this.pitch + dPitch));
  }

  zoom(factor) {
    if (this.planet) {
      this.planetScale = Math.max(1, Math.min(12, this.planetScale * factor));
      return;
    }
    this.fov = Math.max(this.minFov, Math.min(this.maxFov, this.fov * factor));
  }

  setPlanet(on) {
    this.planet = on;
    this.reset();
  }

  reset() {
    this.yaw = 0;
    // Planet mode looks straight down at the ground (nadir).
    this.pitch = this.planet ? -Math.PI / 2 : 0;
    this.fov = (90 * Math.PI) / 180;
    this.planetScale = 4;
  }

  // CPU mirror of the fragment shader: a point in the rendered view
  // (nx, ny in 0..1, y down) -> where it came from in the equirectangular
  // source frame (u, v in 0..1). Keep in sync with the shader above.
  viewToUv(nx, ny) {
    const aspect = this.canvas.width / this.canvas.height;
    const ndcX = nx * 2 - 1;
    const ndcY = 1 - ny * 2;
    let x, y, z;
    if (this.planet) {
      const px = ndcX * aspect * this.planetScale;
      const py = ndcY * this.planetScale;
      const r = Math.hypot(px, py);
      const theta = 2 * Math.atan(r * 0.5);
      const s = r > 0 ? Math.sin(theta) / r : 0;
      [x, y, z] = [px * s, py * s, Math.cos(theta)];
    } else {
      const t = Math.tan(this.fov / 2);
      [x, y, z] = [ndcX * t * aspect, ndcY * t, 1];
      const len = Math.hypot(x, y, z);
      [x, y, z] = [x / len, y / len, z / len];
    }
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    [y, z] = [y * cp + z * sp, -y * sp + z * cp];
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    [x, z] = [x * cy + z * sy, -x * sy + z * cy];
    const lon = Math.atan2(x, z);
    const lat = Math.asin(Math.max(-1, Math.min(1, y)));
    return [lon / (2 * Math.PI) + 0.5, 0.5 - lat / Math.PI];
  }

  render(video) {
    const gl = this.gl;
    const w = this.canvas.width;
    const h = this.canvas.height;
    gl.viewport(0, 0, w, h);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
    gl.uniform1f(this.u.yaw, this.yaw);
    gl.uniform1f(this.u.pitch, this.pitch);
    gl.uniform1f(this.u.tanHalfFov, Math.tan(this.fov / 2));
    gl.uniform1f(this.u.aspect, w / h);
    gl.uniform1i(this.u.planet, this.planet ? 1 : 0);
    gl.uniform1f(this.u.planetScale, this.planetScale);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // Wire YouTube-like controls: drag to look around, wheel / pinch to zoom,
  // double-click to recenter. `enabled()` gates everything so the handlers
  // can stay attached while 360 mode is off.
  attachControls(target, enabled) {
    const pointers = new Map();
    let pinchDist = 0;

    // The planet shows the whole sphere, so a screen-height drag turns it
    // half a revolution rather than one field of view.
    const radPerPx = () =>
      (this.planet ? Math.PI : this.fov) / Math.max(1, target.clientHeight);

    target.addEventListener("pointerdown", (e) => {
      if (!enabled()) return;
      target.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      }
    });
    target.addEventListener("pointermove", (e) => {
      if (!enabled() || !pointers.has(e.pointerId)) return;
      const prev = pointers.get(e.pointerId);
      const cur = { x: e.clientX, y: e.clientY };
      pointers.set(e.pointerId, cur);
      if (pointers.size === 1) {
        const k = radPerPx();
        // Drag the scene with the pointer, like YouTube.
        this.look(-(cur.x - prev.x) * k, (cur.y - prev.y) * k);
      } else if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchDist > 0) this.zoom(pinchDist / d);
        pinchDist = d;
      }
    });
    const release = (e) => {
      pointers.delete(e.pointerId);
      pinchDist = 0;
    };
    target.addEventListener("pointerup", release);
    target.addEventListener("pointercancel", release);

    target.addEventListener(
      "wheel",
      (e) => {
        if (!enabled()) return;
        e.preventDefault();
        this.zoom(Math.exp(e.deltaY * 0.001));
      },
      { passive: false },
    );
    target.addEventListener("dblclick", () => {
      if (enabled()) this.reset();
    });
  }
}
