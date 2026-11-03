/**
 * The cockpit view: the selected drone's own instruments and controls.
 * Pure rendering from a DroneView snapshot — the app builds a view per
 * frame (live sim state or playback sample) and this module draws it: a
 * 3D world (sky, textured ground plane, runway with markings and a control
 * tower) filling the whole square under an MFD-style HUD — HDG tape on
 * top, ASI tape on the left edge, ALT tape on the right edge with the VSI
 * segment beside it — plus the controls square (yoke, rudder pedals,
 * throttle can with green-to-red bars and the RPM readout). Over the world
 * ride two airframe references: the flight path marker (the velocity
 * vector, glued to the world transform) and the fixed miniature airplane
 * ''''\/\/'''' (the nose). Display laws mirror the Python panel
 * (see src/app/units.ts).
 *
 * THE 3D PIPELINE is the standard one: a pinhole camera bolted to the
 * airframe — it MOVES and ROTATES with the drone — while the viewport (the
 * canvas) stays FIXED; every world point, direction and object is
 * transformed INTO it by the same perspective projection. The ground is a
 * per-pixel TEXTURE, not polygons: one ray per pixel intersects the
 * z = 0 plane and samples an infinite world-fixed checkerboard, full
 * resolution, through a MANUALLY DEFINED mip pyramid (4 levels, each
 * level the same checker at twice the cell size and brighter — the
 * fractal effect: the higher the aircraft, the brighter and bigger the
 * squares read), trilinear + anisotropically filtered, faded to
 * atmospheric haze. The ground itself is the Ground class
 * (src/app/ground.ts) — flat today, a height field with mountains and
 * valleys tomorrow.
 */
import { mphOf, vsHundredsOf, altFtOf, rpmOf } from "./units.js";
import { Ground } from "./ground.js";

/** One aircraft's instantaneous reading, for the cockpit and controls. */
export interface DroneView {
  x: number;
  y: number;
  z: number;
  fwdX: number;
  fwdY: number;
  velX: number;     // velocity components, m/s — drive the flight path marker
  velY: number;
  pitch: number;    // rad, positive nose up
  bank: number;     // rad, positive right wing down
  speed: number;    // m/s
  vs: number;       // m/s, vertical speed
  throttle: number; // commanded 0..1
  ail: number;      // physical surface positions, rad
  elev: number;
  rud: number;
  alive: boolean;
}

const FOV_H = (90 * Math.PI) / 180; // the windshield's horizontal field of view
const SKY: readonly [number, number, number] = [36, 51, 86];  // #243356
const WORLD_SCALE = 1;               // the world renders at full resolution

function fitCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// --- the camera: bolted to the airframe ----------------------------------
// The standard 3D pipeline's whole contract: the camera MOVES with the
// drone and ROTATES with its attitude; the viewport (the canvas) stays
// FIXED, and every world point and direction is transformed INTO it by
// the same perspective projection.

/** The pinhole camera: position + orientation (unit axes) + focal length. */
interface Camera {
  px: number; py: number; pz: number;    // position, world metres
  fx: number; fy: number; fz: number;   // forward, unit
  ux: number; uy: number; uz: number;   // up, unit
  rx: number; ry: number; rz: number;   // right, unit
  f: number;                             // focal length, px
  cx: number; cy: number;                // viewport centre, px
  W: number; H: number;                  // viewport size, px
}

/** The camera from one aircraft's reading: heading, deck pitch, bank. */
function cameraOf(s: DroneView, W: number, H: number): Camera {
  const heading = Math.atan2(s.fwdY, s.fwdX);
  const ch = Math.cos(heading);
  const sh = Math.sin(heading);
  const ct = Math.cos(s.pitch);
  const st = Math.sin(s.pitch);
  const cb = Math.cos(s.bank);
  const sb = Math.sin(s.bank);
  // the levelled axes, then the roll about the forward axis (positive
  // bank = right wing down: the right axis tips below the horizon)
  const rl = { x: -sh, y: ch, z: 0.0 };
  const ul = { x: -st * ch, y: -st * sh, z: ct };
  return {
    px: s.x, py: s.y, pz: s.z,
    fx: ch * ct, fy: sh * ct, fz: st,
    rx: cb * rl.x - sb * ul.x, ry: cb * rl.y - sb * ul.y, rz: cb * rl.z - sb * ul.z,
    ux: cb * ul.x + sb * rl.x, uy: cb * ul.y + sb * rl.y, uz: cb * ul.z + sb * rl.z,
    f: (W / 2) / Math.tan(FOV_H / 2),
    cx: W / 2, cy: H / 2,
    W, H,
  };
}

/** Project a world POINT into the viewport; null when behind the camera. */
function projectPoint(cam: Camera, x: number, y: number,
                      z: number): [number, number] | null {
  const dx = x - cam.px;
  const dy = y - cam.py;
  const dz = z - cam.pz;
  const depth = dx * cam.fx + dy * cam.fy + dz * cam.fz;
  if (depth <= 0.2) {
    return null;
  }
  const sx = cam.cx + cam.f * (dx * cam.rx + dy * cam.ry + dz * cam.rz) / depth;
  const sy = cam.cy - cam.f * (dx * cam.ux + dy * cam.uy + dz * cam.uz) / depth;
  return [sx, sy];
}

/** Project a world DIRECTION — a point at infinity: the horizon line, the
 *  pitch ladder's elevations, the flight path marker's velocity vector. */
function projectDir(cam: Camera, dx: number, dy: number,
                    dz: number): [number, number] | null {
  const depth = dx * cam.fx + dy * cam.fy + dz * cam.fz;
  if (depth <= 0.2) {
    return null;
  }
  const sx = cam.cx + cam.f * (dx * cam.rx + dy * cam.ry + dz * cam.rz) / depth;
  const sy = cam.cy - cam.f * (dx * cam.ux + dy * cam.uy + dz * cam.uz) / depth;
  return [sx, sy];
}

// --- the world: sky + the ground's textured squares, per pixel ------------

/** The one ground the cockpit renders (flat today; a height field later). */
const ground = new Ground();

/** The world render buffer, reused frame to frame (full resolution). */
let worldBuf: {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  img: ImageData;
  w: number;
  h: number;
  hitU: Float32Array;    // per-pixel ground hits, u (world x)
  hitV: Float32Array;    // per-pixel ground hits, v (world y)
  hitT: Float32Array;    // per-pixel ground distances; -1 = sky
} | null = null;

function worldBuffer(w: number, h: number): {
  canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D;
  img: ImageData; w: number; h: number;
  hitU: Float32Array; hitV: Float32Array; hitT: Float32Array;
} {
  if (worldBuf !== null && worldBuf.w === w && worldBuf.h === h) {
    return worldBuf;
  }
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const bufCtx = canvas.getContext("2d");
  if (bufCtx === null) {
    throw new Error("cockpit world buffer: no 2d context");
  }
  worldBuf = {
    canvas, ctx: bufCtx, img: bufCtx.createImageData(w, h), w, h,
    hitU: new Float32Array(w * h),
    hitV: new Float32Array(w * h),
    hitT: new Float32Array(w * h),
  };
  return worldBuf;
}

/**
 * Render the world — sky above, the ground's textured squares below —
 * through the camera, one ray per pixel, in two passes: the ground HITS
 * first, then the filtered texture — each pixel samples the Ground with
 * its footprint (the neighbour-stepped parallelogram), so the checker is
 * mipmapped, trilinearly filtered and anisotropically tapped: exact by
 * construction, no polygon shearing anywhere, at any attitude — the
 * spin's nose-down view included. Cells render only within the ground's
 * K distance; beyond, haze fades the far earth into the sky.
 */
function drawWorld(ctx: CanvasRenderingContext2D, cam: Camera): void {
  const w = Math.max(2, Math.ceil(cam.W / WORLD_SCALE));
  const h = Math.max(2, Math.ceil(cam.H / WORLD_SCALE));
  const buf = worldBuffer(w, h);
  const data = buf.img.data;
  const K = ground.drawDistance;
  const hazeFrom = K * 0.45;
  const hazeSpan = K * 0.5;
  const step = WORLD_SCALE;
  const { hitU, hitV, hitT } = buf;

  // PASS 1 — the ground hits: one ray per pixel, stored so pass 2 can
  // difference neighbours into footprints (the texture filter's input).
  for (let py = 0; py < h; py++) {
    const sy = (py + 0.5) * step;
    const b = (cam.cy - sy) / cam.f;        // the ray's up-axis component
    const row = py * w;
    for (let px = 0; px < w; px++) {
      const sx = (px + 0.5) * step;
      const a = (sx - cam.cx) / cam.f;      // the ray's right-axis component
      const invL = 1.0 / Math.sqrt(1.0 + a * a + b * b);
      const dx = (cam.fx + a * cam.rx + b * cam.ux) * invL;
      const dy = (cam.fy + a * cam.ry + b * cam.uy) * invL;
      const dz = (cam.fz + a * cam.rz + b * cam.uz) * invL;
      const t = ground.intersectRay(cam.px, cam.py, cam.pz, dx, dy, dz);
      const i = row + px;
      if (t !== null && t <= K) {
        hitU[i] = cam.px + dx * t;
        hitV[i] = cam.py + dy * t;
        hitT[i] = t;
      } else {
        hitT[i] = -1.0;                      // sky
      }
    }
  }

  // PASS 2 — the filtered texture: each pixel's ground footprint is the
  // parallelogram spanned by its right and down neighbour steps, and the
  // Ground sample mips/trilinear-filters/aniso-taps the checker over it.
  for (let py = 0; py < h; py++) {
    const row = py * w;
    for (let px = 0; px < w; px++) {
      const i = row + px;
      let r: number;
      let g: number;
      let bl: number;
      if (hitT[i] >= 0.0) {
        // the footprint steps: forward differences to the neighbours
        // (backward at the buffer's rim; a sky neighbour dissolves)
        const rx = px + 1 < w ? i + 1 : i - 1;
        const dy0 = py + 1 < h ? i + w : i - w;
        const sxF = hitT[rx] >= 0.0 ? (rx - i) : 0;   // 0 flags a sky step
        const syF = hitT[dy0] >= 0.0 ? (dy0 - i) : 0;
        let stepXU: number;
        let stepXV: number;
        if (sxF !== 0) {
          stepXU = hitU[i + sxF] - hitU[i];
          stepXV = hitV[i + sxF] - hitV[i];
        } else {
          stepXU = 1e6;                                // grazing the horizon
          stepXV = 1e6;
        }
        let stepYU: number;
        let stepYV: number;
        if (syF !== 0) {
          stepYU = hitU[i + syF] - hitU[i];
          stepYV = hitV[i + syF] - hitV[i];
        } else {
          stepYU = 1e6;
          stepYV = 1e6;
        }
        const c = ground.sample(hitU[i], hitV[i], stepXU, stepXV,
                                stepYU, stepYV, hitT[i],
                                cam.pz - ground.heightAt(cam.px, cam.py));
        const mix = Math.min(1.0, Math.max(0.0, (hitT[i] - hazeFrom) / hazeSpan));
        r = c[0] + (SKY[0] - c[0]) * mix;
        g = c[1] + (SKY[1] - c[1]) * mix;
        bl = c[2] + (SKY[2] - c[2]) * mix;
      } else {
        r = SKY[0];
        g = SKY[1];
        bl = SKY[2];
      }
      const o = i * 4;
      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = bl;
      data[o + 3] = 255;
    }
  }
  buf.ctx.putImageData(buf.img, 0, 0);
  ctx.drawImage(buf.canvas, 0, 0, cam.W, cam.H);
}

/**
 * The control tower: standing beside the runway, a shaft with a glass cab
 * on top — a true 3D object, projected corner by corner through the
 * camera, so it slides past on takeoff and parks on the horizon in cruise.
 */
function drawTower(ctx: CanvasRenderingContext2D, cam: Camera): void {
  const TW = { x: 260.0, y: -95.0, base: 6.0, h: 26.0 };   // world metres
  const centreDepth = (TW.x - cam.px) * cam.fx + (TW.y - cam.py) * cam.fy
    + (0.0 - cam.pz) * cam.fz;
  if (centreDepth < 30.0) {
    return;                                  // behind or beside us
  }
  const quad = (pts: ([number, number] | null)[], fill: string): void => {
    if (pts.some((p) => p === null)) {
      return;
    }
    ctx.beginPath();
    ctx.moveTo(pts[0]![0], pts[0]![1]);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i]![0], pts[i]![1]);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  };
  const p = (ox: number, oz: number): [number, number] | null =>
    projectPoint(cam, TW.x + ox, TW.y, oz);
  quad([p(-TW.base, 0.0), p(TW.base, 0.0), p(TW.base * 0.6, TW.h),
        p(-TW.base * 0.6, TW.h)], "#8b96ab");
  quad([p(-TW.base * 0.9, TW.h), p(TW.base * 0.9, TW.h),
        p(TW.base * 0.9, TW.h + 4.0), p(-TW.base * 0.9, TW.h + 4.0)], "#9fd8ff");
  ctx.strokeStyle = "#e8ecf4";
  ctx.lineWidth = 0.7;
  ctx.stroke();
}

/** The HDG tape across the top: ticks and cardinal letters, the current
 * heading boxed at the centre. */
function drawHeadingTape(ctx: CanvasRenderingContext2D, W: number,
                         heading: number): void {
  const h = 18;
  const pxPerDeg = 2.2;
  ctx.fillStyle = "rgba(10,14,22,0.55)";
  ctx.fillRect(0, 0, W, h);
  const hdg = (((heading * 180 / Math.PI) % 360) + 360) % 360;
  ctx.font = "8px ui-monospace, monospace";
  ctx.textAlign = "center";
  for (let d = 0; d < 360; d += 10) {
    let rel = d - hdg;
    while (rel > 180) rel -= 360;
    while (rel < -180) rel += 360;
    const x = W / 2 + rel * pxPerDeg;
    if (x < 4 || x > W - 4) {
      continue;
    }
    const major = d % 30 === 0;
    ctx.strokeStyle = "rgba(201,212,227,0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, h);
    ctx.lineTo(x, h - (major ? 8 : 4));
    ctx.stroke();
    if (major) {
      const cardinals: Record<number, string> = { 0: "N", 90: "E", 180: "S", 270: "W" };
      ctx.fillStyle = "rgba(201,212,227,0.9)";
      ctx.fillText(cardinals[d] ?? String(d / 10), x, h - 10);
    }
  }
  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(W / 2 - 17, h - 1, 34, 13);
  ctx.strokeStyle = "#7dff8e";
  ctx.strokeRect(W / 2 - 16.5, h - 0.5, 33, 12);
  ctx.fillStyle = "#7dff8e";
  ctx.font = "bold 10px ui-monospace, monospace";
  ctx.fillText(`${String(Math.round(hdg) % 360).padStart(3, "0")}\u00b0`, W / 2, h + 9);
  ctx.textAlign = "left";
}

/** The ASI tape on the left edge: a vertical mph ladder with the current
 * speed boxed at mid-height. */
function drawSpeedTape(ctx: CanvasRenderingContext2D, W: number, H: number,
                       mph: number): void {
  const w = 34;
  const top = 30;
  const bot = H - 6;
  ctx.fillStyle = "rgba(10,14,22,0.55)";
  ctx.fillRect(0, top, w, bot - top);
  const midY = (top + bot) / 2;
  const pxPerMph = 1.6;
  ctx.font = "8px ui-monospace, monospace";
  ctx.textAlign = "right";
  for (let v = 0; v <= 170; v += 10) {
    const y = midY - (v - mph) * pxPerMph;
    if (y < top + 2 || y > bot - 2) {
      continue;
    }
    const major = v % 20 === 0;
    ctx.strokeStyle = "rgba(201,212,227,0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w, y);
    ctx.lineTo(w - (major ? 9 : 4), y);
    ctx.stroke();
    if (major) {
      ctx.fillStyle = "rgba(201,212,227,0.9)";
      ctx.fillText(String(v), w - 11, y + 3);
    }
  }
  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(0, midY - 9, w + 16, 18);
  ctx.strokeStyle = "#7dff8e";
  ctx.strokeRect(0.5, midY - 8.5, w + 15, 17);
  ctx.fillStyle = "#7dff8e";
  ctx.font = "bold 12px ui-monospace, monospace";
  ctx.fillText(mph.toFixed(0), w + 12, midY + 4);
  ctx.textAlign = "left";
}

/** The ALT tape on the right edge, with the VSI segment beside it: a thick
 * bar that drops below the zero line for a sink and rises for a climb. */
function drawAltTape(ctx: CanvasRenderingContext2D, W: number, H: number,
                     altFt: number, vsHundreds: number): void {
  const w = 34;
  const x0 = W - w;
  const top = 30;
  const bot = H - 6;
  ctx.fillStyle = "rgba(10,14,22,0.55)";
  ctx.fillRect(x0, top, w, bot - top);
  const midY = (top + bot) / 2;
  const pxPerFt = 0.16;
  ctx.font = "8px ui-monospace, monospace";
  ctx.textAlign = "left";
  const base = Math.round(altFt / 100) * 100;
  for (let v = base - 1200; v <= base + 1200; v += 100) {
    const y = midY - (v - altFt) * pxPerFt;
    if (y < top + 2 || y > bot - 2) {
      continue;
    }
    const major = v % 200 === 0;
    ctx.strokeStyle = "rgba(201,212,227,0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x0 + (major ? 9 : 4), y);
    ctx.stroke();
    if (major) {
      ctx.fillStyle = "rgba(201,212,227,0.9)";
      ctx.fillText(String(v), x0 + 11, y + 3);
    }
  }
  // current altitude box (opens inward)
  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(x0 - 16, midY - 9, w + 16, 18);
  ctx.strokeStyle = "#7dff8e";
  ctx.strokeRect(x0 - 15.5, midY - 8.5, w + 15, 17);
  ctx.fillStyle = "#7dff8e";
  ctx.font = "bold 11px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillText(altFt.toFixed(0), x0 + w - 2, midY + 4);
  ctx.textAlign = "left";

  // VSI: the thick vertical segment beside the tape, zero at mid-height
  const vx = x0 - 22;                       // segment centre
  const vTop = midY - 66;
  const vBot = midY + 66;
  const clampV = Math.max(-16.0, Math.min(16.0, vsHundreds));
  const dy = clampV * (66.0 / 16.0);        // px, positive = climb
  ctx.strokeStyle = "rgba(139,150,171,0.5)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(vx, vTop);
  ctx.lineTo(vx, vBot);
  ctx.stroke();
  ctx.setLineDash([]);
  const y0 = Math.min(midY, midY - dy);
  const y1 = Math.max(midY, midY - dy);
  ctx.fillStyle = Math.abs(vsHundreds) < 0.05 ? "rgba(125,223,142,0.35)" : "#7dff8e";
  ctx.fillRect(vx - 4, y0, 8, Math.max(1.0, y1 - y0));
  // the zero notch: a solid bright tick across the segment at zero climb,
  // drawn over the bar so level flight reads at a glance
  ctx.strokeStyle = "#c9d4e3";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(vx - 7, midY);
  ctx.lineTo(vx + 7, midY);
  ctx.stroke();
  ctx.fillStyle = "rgba(201,212,227,0.75)";
  ctx.font = "7px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText("VSI", vx, vBot + 9);
  ctx.fillText(`${vsHundreds >= 0 ? "+" : ""}${vsHundreds.toFixed(1)}`, vx, vTop - 3);
  ctx.textAlign = "left";
}
/**
 * The flight path marker: the classic HUD velocity-vector symbol — a
 * circle with wings and a tail — drawn where the aircraft's velocity
 * vector points, projected through the camera like everything else in
 * the world, so it rides the horizon under bank and shows the gap
 * between where the machine is GOING and where the nose points: angle of
 * attack and sideslip, at a glance.
 */
function drawFlightPathMarker(ctx: CanvasRenderingContext2D, s: DroneView,
                              cam: Camera): void {
  const speed = Math.hypot(s.velX, s.velY, s.vs);
  if (speed < 1.0) {
    return;                                   // no velocity to point at
  }
  const at = projectDir(cam, s.velX / speed, s.velY / speed, s.vs / speed);
  if (at === null) {
    return;                                   // the flight path is behind us
  }
  const [fx, fy] = at;
  if (fx < -10 || fx > cam.W + 10 || fy < -10 || fy > cam.H + 10) {
    return;                                   // off the windshield
  }
  const r = 4.5;
  ctx.lineCap = "round";
  for (const [colour, width] of [["#0b0e14", 4.0], ["#7dff8e", 2.0]] as const) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.arc(fx, fy, r, 0, 2 * Math.PI);       // the circle
    ctx.moveTo(fx - r - 5.0, fy);             // the wings
    ctx.lineTo(fx - r, fy);
    ctx.moveTo(fx + r, fy);
    ctx.lineTo(fx + r + 5.0, fy);
    ctx.moveTo(fx, fy - r);                   // the tail
    ctx.lineTo(fx, fy - r - 5.0);
    ctx.stroke();
  }
  ctx.lineCap = "butt";
}

/** The cockpit square: ADI horizon card, traffic, flight + engine gauges. */
export function drawCockpitView(canvas: HTMLCanvasElement, title: string,
                                view: DroneView, others: readonly DroneView[],
                                foot: string, titleEl?: HTMLElement,
                                footEl?: HTMLElement): void {
  const ctx = fitCanvas(canvas);
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const s = view;
  // The camera: attached to this drone — it moves and rotates with the
  // airframe; the viewport (the canvas) stays fixed, and everything below
  // is transformed INTO it by the one standard projection.
  const cam = cameraOf(s, W, H);

  // the windshield: the 3D world fills the WHOLE square — sky above the
  // horizon, the ground's textured squares below, one ray per pixel
  drawWorld(ctx, cam);

  // world overlays, projected through the same camera
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  ctx.clip();
  const heading = Math.atan2(s.fwdY, s.fwdX);
  // two directions along a constant-elevation line, spread apart in
  // azimuth: the horizon (elev 0) and each pitch-ladder rung
  const horizon = (elev: number, spread: number):
      [number, number, number, number] | null => {
    const ce = Math.cos(elev);
    const azL = heading - spread;
    const azR = heading + spread;
    const pl = projectDir(cam, ce * Math.cos(azL), ce * Math.sin(azL), Math.sin(elev));
    const pr = projectDir(cam, ce * Math.cos(azR), ce * Math.sin(azR), Math.sin(elev));
    if (pl === null || pr === null) {
      return null;
    }
    return [pl[0], pl[1], pr[0], pr[1]];
  };
  // the horizon line, extended across the viewport
  const hz = horizon(0.0, 0.6);
  if (hz !== null) {
    const ex = hz[2] - hz[0];
    const ey = hz[3] - hz[1];
    const el = Math.hypot(ex, ey);
    if (el > 1e-6) {
      const k = Math.hypot(W, H) / el;
      ctx.strokeStyle = "#f2f5fa";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(hz[0] - ex * k, hz[1] - ey * k);
      ctx.lineTo(hz[0] + ex * k, hz[1] + ey * k);
      ctx.stroke();
    }
  }
  // pitch ladder: the world's elevation rungs, at ±10° and ±20°
  ctx.strokeStyle = "rgba(226,232,244,0.55)";
  ctx.fillStyle = "rgba(226,232,244,0.75)";
  ctx.font = "8px ui-monospace, monospace";
  ctx.lineWidth = 1;
  for (const d of [-20, -10, 10, 20]) {
    const rung = horizon((d * Math.PI) / 180, 0.6);
    if (rung === null) {
      continue;                               // this rung is behind us
    }
    const midX = (rung[0] + rung[2]) / 2;
    const midY = (rung[1] + rung[3]) / 2;
    const ex = rung[2] - rung[0];
    const ey = rung[3] - rung[1];
    const el = Math.hypot(ex, ey);
    if (el < 1e-6) {
      continue;
    }
    const ux = ex / el;
    const uy = ey / el;
    const half = Math.abs(d) === 20 ? 26 : 15;
    ctx.beginPath();
    ctx.moveTo(midX - ux * half, midY - uy * half);
    ctx.lineTo(midX - ux * 6, midY - uy * 6);
    ctx.moveTo(midX + ux * 6, midY + uy * 6);
    ctx.lineTo(midX + ux * half, midY + uy * half);
    ctx.stroke();
    ctx.fillText(String(d), midX + ux * (half + 4), midY + uy * (half + 4) + 3);
  }
  // the control tower, beside the runway
  drawTower(ctx, cam);
  // traffic: other drones in the forward view
  ctx.font = "8px ui-monospace, monospace";
  for (const other of others) {
    if (!other.alive) {
      continue;
    }
    const at = projectPoint(cam, other.x, other.y, other.z);
    if (at === null) {
      continue;
    }
    const [tx, ty] = at;
    if (tx < -20 || tx > W + 20 || ty < -20 || ty > H + 20) {
      continue;
    }
    const range = Math.hypot(other.x - s.x, other.y - s.y, other.z - s.z);
    if (range < 1.0) {
      continue;
    }
    const size = Math.max(3.0, Math.min(9.0, 260.0 / range));
    const dz = other.z - s.z;
    ctx.fillStyle = dz > 2 ? "#9fd8ff" : dz < -2 ? "#ffb066" : "#e8ecf4";
    ctx.beginPath();
    ctx.moveTo(tx, ty - size);
    ctx.lineTo(tx + size * 0.7, ty + size * 0.6);
    ctx.lineTo(tx - size * 0.7, ty + size * 0.6);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "rgba(200,210,228,0.7)";
    ctx.fillText(`${Math.round(range)}m`, tx + size + 2, ty + 3);
  }
  // flight path marker: rides the world, not the frame
  drawFlightPathMarker(ctx, s, cam);
  ctx.restore();                              // un-clip

  // the fixed HUD: painted on the glass, not in the world
  const cx = W / 2;
  const cy = H * 0.5;
  const R = Math.min(W, H) * 0.33;

  // fixed centre symbol: the miniature airplane ''''\/\/'''' — wing bars
  // flanking the W waterline, the airframe's own reference on the glass
  ctx.strokeStyle = "#ffb84a";
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  // left wing bar, the W (down, up, down, up), right wing bar
  ctx.moveTo(cx - 34, cy);
  ctx.lineTo(cx - 22, cy);
  ctx.lineTo(cx - 11, cy + 8);
  ctx.lineTo(cx, cy);
  ctx.lineTo(cx + 11, cy + 8);
  ctx.lineTo(cx + 22, cy);
  ctx.lineTo(cx + 34, cy);
  // wingtip fences
  ctx.moveTo(cx - 34, cy);
  ctx.lineTo(cx - 34, cy - 5);
  ctx.moveTo(cx + 34, cy);
  ctx.lineTo(cx + 34, cy - 5);
  ctx.stroke();
  ctx.lineJoin = "miter";
  ctx.lineCap = "butt";

  // bank pointer (rides the card) + fixed reference tick
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-s.bank);
  ctx.fillStyle = "#ffb84a";
  ctx.beginPath();
  ctx.moveTo(-5, -R + 2);
  ctx.lineTo(5, -R + 2);
  ctx.lineTo(0, -R + 10);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  ctx.strokeStyle = "#8b96ab";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy - R);
  ctx.lineTo(cx, cy - R + 7);
  ctx.stroke();
  ctx.restore();                             // un-clip

  // the ADI rim: a thin ring marking the instrument's reference circle
  ctx.strokeStyle = "rgba(139,150,171,0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, 2 * Math.PI);
  ctx.stroke();

  ctx.strokeStyle = "#3a4258";               // bezel
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, 2 * Math.PI);
  ctx.stroke();

  // --- MFD-style HUD: tapes over the world --------------------------------
  // ASI ladder on the left edge, ALT ladder on the right edge with the
  // VSI segment beside it, HDG tape across the top.
  drawHeadingTape(ctx, W, heading);
  drawSpeedTape(ctx, W, H, mphOf(s.speed));
  drawAltTape(ctx, W, H, altFtOf(s.z), vsHundredsOf(s.vs));

  // crashed overlay: the postmortem view
  if (!s.alive) {
    ctx.fillStyle = "rgba(255,58,48,0.18)";
    ctx.fillRect(0, cy - 15, W, 30);
    ctx.fillStyle = "#ff4a3a";
    ctx.font = "bold 13px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText("CRASHED", cx, cy + 5);
    ctx.textAlign = "left";
  }

  if (titleEl !== undefined) {
    titleEl.textContent = title;
  }
  if (footEl !== undefined) {
    footEl.textContent = foot;
  }
  void foot;
}

/** The controls square: yoke, rudder pedals, throttle lever. */
export function drawControlsView(canvas: HTMLCanvasElement, view: DroneView): void {
  const ctx = fitCanvas(canvas);
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const s = view;
  const SURF_MAX = 0.35;    // mechanical limit, rad

  // --- yoke (aileron rotates the wheel, elevator pushes/pulls it) -----------
  const yx = W * 0.3;
  const yy = H * 0.4;
  const travel = 15;                    // px of push/pull travel
  const pull = (s.elev / SURF_MAX) * travel;    // nose up = pull = yoke toward pilot
  const wheelAngle = (s.ail / SURF_MAX) * 1.15; // rad of visual wheel rotation

  // neutral cross reference
  ctx.strokeStyle = "#232b3d";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(yx - 70, yy); ctx.lineTo(yx + 70, yy);
  ctx.moveTo(yx, yy - 55); ctx.lineTo(yx, yy + 55);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.save();
  ctx.translate(yx, yy + pull);
  ctx.rotate(wheelAngle);
  // the W: wheel bar with two horns
  ctx.strokeStyle = s.alive ? "#c9d4e3" : "#5b6878";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(-46, 0);
  ctx.lineTo(-46, -26);
  ctx.moveTo(-24, 0);
  ctx.lineTo(-24, 10);
  ctx.lineTo(24, 10);
  ctx.lineTo(24, 0);
  ctx.moveTo(46, 0);
  ctx.lineTo(46, -26);
  ctx.stroke();
  ctx.fillStyle = s.alive ? "#8b96ab" : "#3d4658";
  for (const hx of [-46, 46]) {
    ctx.beginPath();
    ctx.arc(hx, -28, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
  // center hub
  ctx.fillStyle = "#ff8c3a";
  ctx.beginPath();
  ctx.arc(0, 0, 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.restore();
  ctx.lineCap = "butt";
  ctx.fillStyle = "#5b6878";
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillText("YOKE", yx - 14, yy + 74);
  ctx.fillText(s.elev >= 0 ? "PULL" : "PUSH", yx + 26, yy + pull - 4);

  // --- rudder pedals ---------------------------------------------------------
  const rx = W * 0.3;
  const ry = H * 0.78;
  const press = (s.rud / SURF_MAX) * 9;      // one pedal forward, one back
  for (const side of [-1, 1]) {
    const forward = (s.rud > 0 && side > 0) || (s.rud < 0 && side < 0);
    const disp = forward ? -press : (s.rud !== 0 ? press * 0.35 : 0);
    const engaged = (s.rud > 0 && side > 0) || (s.rud < 0 && side < 0);
    ctx.fillStyle = engaged ? "#ff8c3a" : "#3a4258";
    ctx.fillRect(rx + side * 34 - 9, ry + disp - 16, 18, 30);
  }
  ctx.strokeStyle = "#232b3d";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(rx - 62, ry); ctx.lineTo(rx + 62, ry);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#5b6878";
  ctx.fillText("L", rx - 66, ry + 3);
  ctx.fillText("R", rx + 62, ry + 3);
  ctx.fillText("RUDDER", rx - 21, ry + 26);

  // --- throttle can: horizontal bars from green (idle) to red (full) ------
  const tx = W * 0.78;
  const tyTop = H * 0.16;
  const tyBot = H * 0.62;
  const th = Math.max(0.0, Math.min(1.0, s.throttle));
  const canW = 26;
  const canH = tyBot - tyTop;
  ctx.fillStyle = "#1a2030";
  ctx.fillRect(tx - canW / 2, tyTop, canW, canH);
  const BARS = 12;
  const barH = (canH - 4) / BARS;
  for (let i = 0; i < BARS; i++) {
    const lit = (i + 0.5) / BARS <= th;      // bars up to the throttle level
    const t = i / (BARS - 1);                // green at the bottom to red at the top
    const r = Math.round(70 + (255 - 70) * t);
    const g = Math.round(210 + (70 - 210) * t);
    const b = Math.round(90 + (60 - 90) * t);
    ctx.fillStyle = lit ? `rgb(${r},${g},${b})` : "#2a3242";
    ctx.fillRect(tx - canW / 2 + 2, tyTop + canH - 2 - (i + 1) * barH + 1,
                 canW - 4, Math.max(1.0, barH - 1.5));
  }
  ctx.strokeStyle = "#3a4258";
  ctx.strokeRect(tx - canW / 2 + 0.5, tyTop + 0.5, canW - 1, canH - 1);
  ctx.fillStyle = "#5b6878";
  ctx.fillText("THROTTLE", tx - 27, tyBot + 16);
  ctx.fillStyle = "#c9d4e3";
  ctx.fillText(`${Math.round(th * 100)}%`, tx - 10, tyBot + 28);

  // --- engine: the RPM lives here, beside the throttle can ---------------
  ctx.fillStyle = "#8b96ab";
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillText("RPM", tx + 20, tyTop + 6);
  ctx.fillStyle = rpmOf(s.throttle) > 2450 ? "#ff6a5a" : "#7dff8e";
  ctx.font = "bold 13px ui-monospace, monospace";
  ctx.fillText(rpmOf(s.throttle).toFixed(0), tx + 20, tyTop + 20);

  // surface readouts, bottom right
  ctx.textAlign = "right";
  ctx.fillStyle = "#8b96ab";
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillText(`AIL ${s.ail >= 0 ? "+" : ""}${s.ail.toFixed(2)}`, W - 10, H - 34);
  ctx.fillText(`ELEV ${s.elev >= 0 ? "+" : ""}${s.elev.toFixed(2)}`, W - 10, H - 22);
  ctx.fillText(`RUD ${s.rud >= 0 ? "+" : ""}${s.rud.toFixed(2)}`, W - 10, H - 10);
  ctx.textAlign = "left";

  if (!s.alive) {
    ctx.fillStyle = "rgba(255,58,48,0.16)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#ff4a3a";
    ctx.font = "bold 11px ui-monospace, monospace";
    ctx.fillText("CONTROLS FROZEN", 12, H - 12);
  }
}
