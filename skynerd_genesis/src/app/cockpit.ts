/**
 * The cockpit view: the selected drone's own instruments and controls.
 * Pure rendering from a DroneView snapshot — the app builds a view per
 * frame (live sim state or playback sample) and this module draws it: a
 * 3D world (a gradient sky with the sun, its glare and drifting clouds;
 * textured ground plane; runway with markings and a control tower)
 * filling the whole square under an MFD-style HUD — HDG tape on
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
import { Flora, hash01, SUN_DIR,
         type TreeSpot, type HouseSpot, type CloudSpot } from "./land.js";
import type { SceneAircraft } from "./scene.js";
import { levelOf } from "../learning/instruments.js";
import type { WorldRenderer } from "../pipeline/index.js";
import { CRASH_FLOOR_EYE_M, GROUND_GRADE } from "../pipeline/index.js";
import { RENDER_STATE } from "./render-state.js";

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
  /**
   * THE PILOT'S DECISIONS — the overlay on top of the machine's state:
   *  - handChoices: the student's last per-channel decision, the
   *    commanded lever LEVEL (a stop, or -1 for hold) — null while the
   *    instructor's hand eases (no decisions happen);
   *  - handFresh: a decision window just opened — the 5 Hz brain beat.
   *
   * The CONTROLS themselves (yoke, pedals, throttle, the surfaces) are
   * the machine's own state: ail / elev / rud / throttle below, read
   * through the aircraft's sealed getters — what the selected aircraft
   * is actually holding, not what a hand asked for.
   */
  handChoices: readonly number[] | null;
  handFresh: boolean;
}

/**
 * The panel's view of one scene aircraft: its instruments (the machine's
 * physical truth) and its controls (the hand's inputs on it), as the
 * cockpit and controls canvases draw them.
 */
export function droneViewOf(entry: SceneAircraft): DroneView {
  const machine = entry.aircraft;
  const body = machine.body;
  const [fwdX, fwdY] = machine.forward;
  const choices = entry.hand.choices;
  return {
    x: body.position.x, y: body.position.y, z: body.position.z,
    fwdX, fwdY,
    pitch: machine.pitch, bank: machine.bank,
    speed: machine.asi, vs: machine.vsi,
    velX: body.velocity.x, velY: body.velocity.y,
    // The state of controls — the machine's own, via its sealed
    // getters: the yoke, the pedals, and the throttle lever as the
    // selected aircraft actually holds them.
    throttle: machine.throttle,
    ail: machine.aileron, elev: machine.elevator, rud: machine.rudder,
    handChoices: choices === null
      ? null : [choices[0], choices[1], choices[2], choices[3]],
    handFresh: entry.hand.fresh,
    alive: !entry.dead,
  };
}

const FOV_H = (90 * Math.PI) / 180; // the windshield's horizontal field of view
/** Half the view's FOV — the curved windshield's shared-focal law
 *  (main.ts derives the panes' common focal from it). */
export const FOV_HALF = FOV_H / 2;
/** The sky's gradient: pale haze at the horizon, deep blue overhead —
 *  and the ground's far haze fades INTO the same fog colour, so earth
 *  and sky melt together far away instead of meeting at a line. */
const SKY_ZENITH: readonly [number, number, number] = [30, 46, 84];
const SKY_FOG: readonly [number, number, number] = [184, 200, 214];
const SKY_GRAD_SIN = 0.35;         // sin(elevation) where the sky is fully zenith
/** THE CRASH FLOOR: the background ends 15° UNDER the horizon — below
 *  it the world is black (the crashed view), never sky. */
const SKY_FLOOR_SIN = 0.25881904510252074;   // sin(15°)
/** The sun's screen constants — its world direction (SUN_DIR) is the
 *  land's own light, shared with the terrain's relief shading. */
const SUN_ANG = 0.035;             // the disc's angular radius, rad (~2°)
const SUN_GLOW = 9.0;              // the glow's radius, in discs
const GLARE_STRENGTH = 0.42;       // the veil's peak alpha, facing the sun
const WORLD_SCALE = 1;               // the world renders at full resolution

function fitCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  // the size is only REASSIGNED when it changed — rewriting
  // canvas.width clears and reallocs the whole surface (a ~10M-pixel
  // HUD canvas per frame was the drag stall's biggest cost)
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
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

/** The camera of one aircraft's reading, yawed about its own up axis
 *  by `deg` — the side viewport panes: the pilot's head turned left
 *  (deg < 0) or right (deg > 0) while the airframe's attitude stands.
 *  Position, up axis and focal law are untouched; only the forward and
 *  right axes rotate (v' = cos·v + sin·(up × v)). */
function yawedCameraOf(s: DroneView, W: number, H: number,
                       deg: number): Camera {
  const cam = cameraOf(s, W, H);
  const a = (deg * Math.PI) / 180;
  const c = Math.cos(a);
  const sn = Math.sin(a);
  return {
    ...cam,
    fx: c * cam.fx + sn * cam.rx,
    fy: c * cam.fy + sn * cam.ry,
    fz: c * cam.fz + sn * cam.rz,
    rx: c * cam.rx - sn * cam.fx,
    ry: c * cam.ry - sn * cam.fy,
    rz: c * cam.rz - sn * cam.fz,
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
  // the mood's colours (the GPU uniform's twin): the classic look IS
  // the noon row, so the default renders exactly as before
  const mood = RENDER_STATE.mood;
  const zenith = mood.zenith;
  const fog = mood.fog;
  // THE PILOT'S IRIS: the whole composed world dims by the exposure
  // (the GPU uniform's extras.y twin)
  const ex = iris.exposure;
  // THE HOUR'S GRADE: the ground wears the mood's colour — a mean-1
  // filter (tint without darkening), noon-neutral (the GPU frame
  // uniform's ground row twin). Applied before the haze: the near
  // ground wears the hour, the far dissolves into the mood's fog
  const grade = mood.grade;
  const gradeMean = (grade[0] + grade[1] + grade[2]) / 3.0;
  const gradeR = 1.0 + (grade[0] / gradeMean - 1.0) * GROUND_GRADE;
  const gradeG = 1.0 + (grade[1] / gradeMean - 1.0) * GROUND_GRADE;
  const gradeB = 1.0 + (grade[2] / gradeMean - 1.0) * GROUND_GRADE;

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
    const b = (cam.cy - (py + 0.5) * step) / cam.f;   // the ray's up component
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
        r = (c[0] * gradeR + (fog[0] - c[0] * gradeR) * mix);
        g = (c[1] * gradeG + (fog[1] - c[1] * gradeG) * mix);
        bl = (c[2] * gradeB + (fog[2] - c[2] * gradeB) * mix);
      } else {
        // the sky's gradient: fog at the horizon, deepening with
        // elevation — the mood's own colours (the GPU uniform's twin)
        const sx = (px + 0.5) * step;
        const a = (sx - cam.cx) / cam.f;
        const invL = 1.0 / Math.sqrt(1.0 + a * a + b * b);
        const dz = (cam.fz + a * cam.rz + b * cam.uz) * invL;
        // THE CRASH FLOOR (the GPU extras.z twin): 15° under the
        // horizon the background is black ONLY for the LOW camera —
        // the crash. A flying camera keeps the haze band there: the
        // world's far edge dissolves into fog, never a black band.
        if (dz < -SKY_FLOOR_SIN && cam.pz - ground.heightAt(cam.px, cam.py)
            < CRASH_FLOOR_EYE_M) {
          // THE CRASH FLOOR: 15° under the horizon the background is
          // black — a crashed camera looks into darkness, never sky
          r = 0; g = 0; bl = 0;
        } else {
          const t = Math.min(1.0, Math.max(0.0, dz / SKY_GRAD_SIN));
          r = fog[0] + (zenith[0] - fog[0]) * t;
          g = fog[1] + (zenith[1] - fog[1]) * t;
          bl = fog[2] + (zenith[2] - fog[2]) * t;
        }
      }
      const o = i * 4;
      data[o] = r * ex;
      data[o + 1] = g * ex;
      data[o + 2] = bl * ex;
      data[o + 3] = 255;
    }
  }
  buf.ctx.putImageData(buf.img, 0, 0);
  ctx.drawImage(buf.canvas, 0, 0, cam.W, cam.H);
}

// --- the landscape's objects: trees, houses, the tower --------------------
// True 3D objects drawn as projected polygons over the ground texture
// (like the tower always was): collected each frame from the flora's
// deterministic world, depth-sorted (painter's algorithm, far to near),
// with their ground shadows painted first.

/** One drawable 3D thing, at a camera-space depth for sorting. */
interface WorldObject {
  depth: number;
  shadow?: (ctx: CanvasRenderingContext2D, cam: Camera) => void;
  draw: (ctx: CanvasRenderingContext2D, cam: Camera) => void;
}

/** The shared flora, reading the ground's own land-use plan. */
const flora = new Flora(ground.land);

/** The control tower: standing beside the runway, a shaft with a glass
 *  cab on top — a true 3D object, projected corner by corner through the
 *  camera, so it slides past on takeoff and parks on the horizon in
 *  cruise. */
function pushTower(objects: WorldObject[], cam: Camera): void {
  const TW = { x: 260.0, y: -95.0, base: 6.0, h: 26.0 };   // world metres
  const TWz = ground.heightAt(TW.x, TW.y);  // planted on the terrain
  const centreDepth = (TW.x - cam.px) * cam.fx + (TW.y - cam.py) * cam.fy
    + (TWz - cam.pz) * cam.fz;
  if (centreDepth < 30.0) {
    return;                                  // behind or beside us
  }
  objects.push({
    depth: centreDepth,
    shadow: (ctx, cam2) => {
      const sx = TW.x + TW.h * 0.7;
      const sy = TW.y + TW.base;
      const pS = projectPoint(cam2, sx, sy, ground.heightAt(sx, sy));
      if (pS === null) {
        return;
      }
      ctx.fillStyle = "rgba(16,22,14,0.30)";
      ctx.beginPath();
      ctx.ellipse(pS[0], pS[1], TW.base * 1.6 * cam2.f / centreDepth,
                  TW.base * 0.6 * cam2.f / centreDepth, 0, 0, 2 * Math.PI);
      ctx.fill();
    },
    draw: (ctx, cam2) => {
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
        projectPoint(cam2, TW.x + ox, TW.y, TWz + oz);
      quad([p(-TW.base, 0.0), p(TW.base, 0.0), p(TW.base * 0.6, TW.h),
            p(-TW.base * 0.6, TW.h)], "#35586b");
      quad([p(-TW.base * 0.9, TW.h), p(TW.base * 0.9, TW.h),
            p(TW.base * 0.9, TW.h + 4.0), p(-TW.base * 0.9, TW.h + 4.0)], "#9fd8ff");
      ctx.strokeStyle = "#dff2fc";
      ctx.lineWidth = 0.7;
      ctx.stroke();
    },
  });
}

/** One tree: a trunk stroke plus a canopy — a round deciduous blob or a
 *  conifer's triangle — projected through the camera, with its shadow on
 *  the ground. Sub-pixel trees are skipped: the forest tiles' mottled
 *  texture carries them from altitude. */
function pushTree(objects: WorldObject[], cam: Camera, t: TreeSpot): void {
  const bz = ground.heightAt(t.x, t.y);     // planted on the terrain
  const pB = projectPoint(cam, t.x, t.y, bz);
  if (pB === null) {
    return;
  }
  const depth = (t.x - cam.px) * cam.fx + (t.y - cam.py) * cam.fy
              + (bz - cam.pz) * cam.fz;
  if (depth < 3.0) {
    return;
  }
  const cr = (t.r * cam.f) / depth;          // the canopy's projected radius
  if (cr < 0.8) {
    return;                                  // sub-pixel: the texture carries
  }
  // The top can clip behind the near plane while the base does not when
  // the camera is pitched hard (the stall-recovery dive): the projection
  // returns null and the tree is simply not drawn.
  const pT = projectPoint(cam, t.x, t.y, bz + t.h);
  if (pT === null) {
    return;
  }
  const mX = 40.0 + cr * 2.0;
  if (Math.min(pB[0], pT[0]) < -mX || Math.max(pB[0], pT[0]) > cam.W + mX
      || Math.min(pB[1], pT[1]) < -mX || Math.max(pB[1], pT[1]) > cam.H + mX) {
    return;                                  // off the windshield
  }
  const trunkTopX = pB[0] + (pT[0] - pB[0]) * 0.75;
  const trunkTopY = pB[1] + (pT[1] - pB[1]) * 0.75;
  const canopyX = pB[0] + (pT[0] - pB[0]) * 0.7;
  const canopyY = pB[1] + (pT[1] - pB[1]) * 0.7;
  const baseY = pB[1] + (pT[1] - pB[1]) * 0.3;
  const green = t.shade < 0.5 ? "#2c4a22" : "#356030";
  objects.push({
    depth,
    shadow: (ctx, cam2) => {
      const sx = t.x + t.h * 0.6;
      const sy = t.y + t.h * 0.15;
      const pS = projectPoint(cam2, sx, sy, ground.heightAt(sx, sy));
      if (pS === null) {
        return;
      }
      ctx.fillStyle = "rgba(16,22,14,0.30)";
      ctx.beginPath();
      ctx.ellipse(pS[0], pS[1], cr * 0.9, cr * 0.35, 0, 0, 2 * Math.PI);
      ctx.fill();
    },
    draw: (ctx) => {
      ctx.strokeStyle = "#4a3626";            // the trunk
      ctx.lineWidth = Math.max(0.8, cr * 0.22);
      ctx.beginPath();
      ctx.moveTo(pB[0], pB[1]);
      ctx.lineTo(trunkTopX, trunkTopY);
      ctx.stroke();
      ctx.fillStyle = green;                  // the canopy
      ctx.beginPath();
      if (t.conifer) {
        ctx.moveTo(pT[0], pT[1]);
        ctx.lineTo(canopyX - cr * 1.1, baseY);
        ctx.lineTo(canopyX + cr * 1.1, baseY);
        ctx.closePath();
      } else {
        ctx.arc(canopyX, canopyY, cr, 0, 2 * Math.PI);
      }
      ctx.fill();
    },
  });
}

const WALL: readonly [number, number, number] = [178, 170, 152];
const ROOF: readonly [number, number, number] = [128, 86, 72];
const shadeOf = (base: readonly [number, number, number], f: number): string =>
  `rgb(${Math.round(base[0] * f)}, ${Math.round(base[1] * f)}, ${Math.round(base[2] * f)})`;

/** One house: an axis-aligned box with a ridge roof, every face projected
 *  and shaded by its orientation, drawn far face first (painter's
 *  algorithm within the object). */
function pushHouse(objects: WorldObject[], cam: Camera, h: HouseSpot): void {
  const x0 = h.x - h.w / 2, x1 = h.x + h.w / 2;
  const y0 = h.y - h.d / 2, y1 = h.y + h.d / 2;
  // the foundation conforms: each corner planted at its own terrain
  // height, the walls rising from the ground it stands on
  const gz = [ground.heightAt(x0, y0), ground.heightAt(x1, y0),
              ground.heightAt(x1, y1), ground.heightAt(x0, y1)];
  const bz = ground.heightAt(h.x, h.y);
  const depth = (h.x - cam.px) * cam.fx + (h.y - cam.py) * cam.fy
              + (bz - cam.pz) * cam.fz;
  if (depth < 8.0) {
    return;
  }
  const pC = projectPoint(cam, h.x, h.y, bz);
  if (pC === null) {
    return;
  }
  const extent = (Math.max(h.w, h.d) * cam.f) / depth + 40.0;
  if (pC[0] < -extent || pC[0] > cam.W + extent
      || pC[1] < -extent || pC[1] > cam.H + extent) {
    return;                                  // off the windshield
  }
  const p = (x: number, y: number, z: number): [number, number] =>
    projectPoint(cam, x, y, z) ?? [0.0, 0.0];
  const g: [number, number][] = [p(x0, y0, gz[0]), p(x1, y0, gz[1]),
                                p(x1, y1, gz[2]), p(x0, y1, gz[3])];
  const e: [number, number][] = [p(x0, y0, gz[0] + h.wallH),
                                 p(x1, y0, gz[1] + h.wallH),
                                 p(x1, y1, gz[2] + h.wallH),
                                 p(x0, y1, gz[3] + h.wallH)];
  const faceDepth = (x: number, y: number, z: number): number =>
    (x - cam.px) * cam.fx + (y - cam.py) * cam.fy + (z - cam.pz) * cam.fz;
  interface Face { depth: number; pts: [number, number][]; colour: string; }
  const faces: Face[] = [];
  // walls (up to the eaves), shaded by orientation (sun from the -x side)
  faces.push({ depth: faceDepth(x0, h.y, h.wallH / 2),
               pts: [g[0], g[3], e[3], e[0]], colour: shadeOf(WALL, 1.0) });
  faces.push({ depth: faceDepth(x1, h.y, h.wallH / 2),
               pts: [g[1], g[2], e[2], e[1]], colour: shadeOf(WALL, 0.72) });
  faces.push({ depth: faceDepth(h.x, y0, h.wallH / 2),
               pts: [g[0], g[1], e[1], e[0]], colour: shadeOf(WALL, 0.9) });
  faces.push({ depth: faceDepth(h.x, y1, h.wallH / 2),
               pts: [g[3], g[2], e[2], e[3]], colour: shadeOf(WALL, 0.8) });
  const ridgeZ = bz + h.wallH + h.roofH;   // the ridge axis, for sorting
  if (h.along === "x") {
    const r0 = p(x0, h.y, ground.heightAt(x0, h.y) + h.wallH + h.roofH);
    const r1 = p(x1, h.y, ground.heightAt(x1, h.y) + h.wallH + h.roofH);
    faces.push({ depth: faceDepth(h.x, y0, ridgeZ),
                 pts: [e[0], e[1], r1, r0], colour: shadeOf(ROOF, 1.0) });
    faces.push({ depth: faceDepth(h.x, y1, ridgeZ),
                 pts: [e[3], e[2], r1, r0], colour: shadeOf(ROOF, 0.72) });
    faces.push({ depth: faceDepth(x0, h.y, ridgeZ),
                 pts: [e[0], e[3], r0], colour: shadeOf(WALL, 1.0) });
    faces.push({ depth: faceDepth(x1, h.y, ridgeZ),
                 pts: [e[1], e[2], r1], colour: shadeOf(WALL, 0.72) });
  } else {
    const r0 = p(h.x, y0, ground.heightAt(h.x, y0) + h.wallH + h.roofH);
    const r1 = p(h.x, y1, ground.heightAt(h.x, y1) + h.wallH + h.roofH);
    faces.push({ depth: faceDepth(x0, h.y, ridgeZ),
                 pts: [e[0], e[3], r1, r0], colour: shadeOf(ROOF, 1.0) });
    faces.push({ depth: faceDepth(x1, h.y, ridgeZ),
                 pts: [e[1], e[2], r1, r0], colour: shadeOf(ROOF, 0.72) });
    faces.push({ depth: faceDepth(h.x, y0, ridgeZ),
                 pts: [e[0], e[1], r0], colour: shadeOf(WALL, 0.9) });
    faces.push({ depth: faceDepth(h.x, y1, ridgeZ),
                 pts: [e[3], e[2], r1], colour: shadeOf(WALL, 0.8) });
  }
  faces.sort((a, b) => b.depth - a.depth);
  objects.push({
    depth,
    draw: (ctx) => {
      for (const f of faces) {
        ctx.beginPath();
        ctx.moveTo(f.pts[0][0], f.pts[0][1]);
        for (let i = 1; i < f.pts.length; i++) {
          ctx.lineTo(f.pts[i][0], f.pts[i][1]);
        }
        ctx.closePath();
        ctx.fillStyle = f.colour;
        ctx.fill();
      }
    },
  });
}

/** Collect and draw the landscape's objects: the tower, the trees within
 *  700 m, the hamlet houses within 2.5 km — shadows first (ground
 *  decals), then far to near. */
/** THE PILOT'S EYE — the HDR iris: facing the sun the eye TIGHTENS
 *  and the whole world — sky, ground, sun, glare — dims by the
 *  exposure; away from it the eye stays wide open (1.0). The
 *  tightening follows the glare's own align³ bloom (the veil and the
 *  iris are one reflex), and it EASES on the eye's own time
 *  constants: a fast flinch toward the light, a slow reopening away
 *  from it — never a snap. */
class IrisAdaptation {
  private level = 1.0;                 // the exposure, 1 = wide open
  private lastMs: number | null = null;

  /** The exposure an alignment asks for: full open facing away,
   *  ~45% down dead ahead at the sun. */
  private static targetOf(align: number): number {
    return 1.0 - 0.55 * align ** 3;
  }

  /** One frame: ease the level toward the alignment's target. */
  update(align: number): void {
    const clock = (globalThis as { performance?: { now(): number } })
      .performance;
    const now = clock !== undefined ? clock.now() : 0.0;
    const dt = this.lastMs === null
      ? 0.0
      : Math.min(0.1, Math.max(0.0, (now - this.lastMs) / 1000.0));
    this.lastMs = now;
    const target = IrisAdaptation.targetOf(align);
    // tighten in a flinch (~0.2 s), reopen like night vision (~1 s)
    const tau = target < this.level ? 0.20 : 1.0;
    const k = dt > 0.0 ? 1.0 - Math.exp(-dt / tau) : 0.0;
    this.level += (target - this.level) * k;
  }

  get exposure(): number {
    return this.level;
  }
}

/** The one pilot's eye every cockpit view renders through — the main
 *  windshield and the side viewports share its exposure. */
const iris = new IrisAdaptation();

/** THE HOUR'S GRADE — the controls strip's filter: the instruments
 *  wear the mood's colour over their ink (noon's white grade tints
 *  nothing), so the panel belongs to its hour. The veil draws
 *  'source-atop' — it tints only the DRAWN ink; the strip's own CSS
 *  background stays untouched beneath the transparent canvas. */
const CONTROLS_GRADE = 0.5;
function gradeControlsVeil(ctx: CanvasRenderingContext2D,
                           W: number, H: number): void {
  const grade = RENDER_STATE.mood.grade;
  const mean = (grade[0] + grade[1] + grade[2]) / 3.0;
  if (mean <= 0.0) {
    return;
  }
  // mean-1: the grade tints without darkening; the alpha is its strength
  const r = Math.round(grade[0] / mean);
  const g = Math.round(grade[1] / mean);
  const b = Math.round(grade[2] / mean);
  ctx.globalCompositeOperation = "source-atop";
  ctx.fillStyle = `rgba(${r},${g},${b},${CONTROLS_GRADE})`;
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "source-over";
}

/** The sun's screen position when it stands in front of the camera,
 *  else null. */
function sunScreen(cam: Camera): [number, number] | null {
  const sunDir = RENDER_STATE.mood.sunDir;
  return projectDir(cam, sunDir[0], sunDir[1], sunDir[2]);
}

/** How directly the camera faces the sun, 0 (away) to 1 (dead ahead). */
function sunAlign(cam: Camera): number {
  const sunDir = RENDER_STATE.mood.sunDir;
  return Math.max(0.0, cam.fx * sunDir[0] + cam.fy * sunDir[1]
                           + cam.fz * sunDir[2]);
}

/** The sun: a blendy disc with a soft additive glare, at the mood's
 *  own direction, size and tint — behind the clouds and the objects.
 *  THE SETTING LAW (the GPU skyFs's exact twin): the bigger the disc
 *  the lower the sun, and the more the atmosphere chews its edge — a
 *  wobbled silhouette, a refraction squash, a melted rim that never
 *  turns hard — and a wider, stronger glare. */
function drawSun(ctx: CanvasRenderingContext2D, cam: Camera): void {
  const mood = RENDER_STATE.mood;
  const sunDir = mood.sunDir;
  const at = projectDir(cam, sunDir[0], sunDir[1], sunDir[2]);
  if (at === null || mood.brightness <= 0.001) {
    return;                                // no sun in this mood
  }
  const [sx, sy] = at;
  const disc = cam.f * Math.tan(mood.discSize);
  const tint = mood.disc;
  const tintCss = (a: number): string =>
    `rgba(${Math.round(tint[0])},${Math.round(tint[1])},${Math.round(tint[2])},${a.toFixed(3)})`;
  // the setting factor: 0 at noon's clean circle, 1 at the swell
  const s = Math.min(1.0, Math.max(0.0, (mood.discSize - 0.035) / 0.055));
  const squash = 1.0 + 0.35 * s;            // refraction's vertical squash
  // the pilot's iris this frame: the tightened eye dims the sun too —
  // the glare recedes first of all (the GPU extras.y twin)
  const ex = iris.exposure;
  ctx.globalCompositeOperation = "lighter";
  // the wobbled, squashed silhouette: two harmonics of atmospheric chew
  const N = 48;
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const ang = (i / N) * 2.0 * Math.PI;
    const wob = s * (0.08 * Math.sin(4.0 * ang + 1.7)
                     + 0.05 * Math.sin(9.0 * ang + 0.6));
    const r = disc * (1.0 + wob);
    const x = sx + r * Math.cos(ang);
    const y = sy + (r * Math.sin(ang)) / squash;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.closePath();
  // the disc's COLOUR (the GPU core's twin): the tint lifted toward
  // white — the sun is never darker than the sky around it; the rim
  // fades to nothing so the blendy edge melts INTO the sky's shade
  const core = [
    Math.round(tint[0] + (255 - tint[0]) * 0.65),
    Math.round(tint[1] + (255 - tint[1]) * 0.65),
    Math.round(tint[2] + (255 - tint[2]) * 0.65),
  ];
  const coreCss = (a: number): string =>
    `rgba(${core[0]},${core[1]},${core[2]},${a.toFixed(3)})`;
  // the blendy fill: solid at the core, fully melted at the rim
  const fill = ctx.createRadialGradient(sx, sy, disc * 0.55,
                                        sx, sy, disc * 1.06);
  fill.addColorStop(0.0, coreCss(0.97 * ex));
  fill.addColorStop(0.62, coreCss(0.90 * ex));
  fill.addColorStop(1.0, coreCss(0));
  ctx.fillStyle = fill;
  ctx.fill();
  // the glare: wider and far stronger as the sun sinks (the GPU law)
  const k = (0.28 + 1.1 * s) / 0.28;
  const glowR = disc * SUN_GLOW * (1.0 + 0.8 * s);
  const grad = ctx.createRadialGradient(sx, sy, disc * 0.9, sx, sy, glowR);
  grad.addColorStop(0.0, tintCss(0.55 * k * ex));
  grad.addColorStop(0.3, tintCss(0.24 * k * ex));
  grad.addColorStop(1.0, tintCss(0));
  ctx.fillStyle = grad;
  ctx.fillRect(sx - glowR, sy - glowR, glowR * 2.0, glowR * 2.0);
  ctx.globalCompositeOperation = "source-over";
}

/** The sun's glare: a wide soft veil over the whole scene, blooming as
 *  the camera swings toward the sun — the lens flare of the eye. It
 *  TINTS with the mood and nearly DOUBLES at the low sun (the setting
 *  law's veil half). */
function drawSunGlare(ctx: CanvasRenderingContext2D, cam: Camera): void {
  const mood = RENDER_STATE.mood;
  const at = sunScreen(cam);
  if (at === null) {
    return;
  }
  const align = sunAlign(cam);
  if (align <= 0.05) {
    return;                                // facing away: no glare
  }
  const s = Math.min(1.0, Math.max(0.0, (mood.discSize - 0.035) / 0.055));
  const a = GLARE_STRENGTH * (1.0 + 1.2 * s) * align ** 3
            * iris.exposure;   // the iris recedes the veil first of all
  const [sx, sy] = at;
  const R = Math.hypot(cam.W, cam.H);
  // the veil's tint: the mood's disc colour, warmed toward white
  const tint = mood.disc;
  const vr = Math.round(tint[0] * 0.5 + 255 * 0.5);
  const vg = Math.round(tint[1] * 0.5 + 246 * 0.5);
  const vb = Math.round(tint[2] * 0.5 + 220 * 0.5);
  const grad = ctx.createRadialGradient(sx, sy, 0.0, sx, sy, R);
  grad.addColorStop(0.0, `rgba(${vr},${vg},${vb},${a.toFixed(3)})`);
  grad.addColorStop(0.35, `rgba(${vr},${vg},${vb},${(a * 0.45).toFixed(3)})`);
  grad.addColorStop(1.0, `rgba(${vr},${vg},${vb},0)`);
  ctx.globalCompositeOperation = "lighter";
  ctx.fillStyle = grad;
  ctx.fillRect(0.0, 0.0, cam.W, cam.H);
  ctx.globalCompositeOperation = "source-over";
}

/** THE LENS FLARES (the toolbar's dropdown, over the world, under the
 *  instruments). Two families, both classical:
 *  - panavision: the anamorphic cinematographer's flare — one long
 *    horizontal blue streak through the sun, a soft warm halo, a
 *    whisper of a vertical streak. The look of the wide screen.
 *  - michaelbay: the blockbuster's flare — a hot white core, bright
 *    horizontal + vertical streaks, a ring, and a CHAIN OF GHOSTS
 *    marching across the frame through the centre, each a soft disc.
 *  Both live on the HUD canvas (additive ink), gate on how directly
 *  the camera faces the sun, and track the sun's mood (size + tint).
 */
function drawLensFlare(ctx: CanvasRenderingContext2D, cam: Camera): void {
  const mood = RENDER_STATE.mood;
  if (mood.brightness <= 0.001) {
    return;                                 // no sun: nothing to flare
  }
  const at = sunScreen(cam);
  if (at === null) {
    return;
  }
  const align = sunAlign(cam);
  if (align <= 0.05) {
    return;                                 // facing away
  }
  const [sx, sy] = at;
  const tint = mood.disc;
  // THE FLARE'S INK IS THE SKY'S GLARE FAMILY: the same colour the sky's
  // glow uses (the disc tint lifted toward white — skyFs's mix(tint,
  // 1.0, 0.25)), so the streaks and ghosts belong to the light that
  // made them: warm orange streaks at sunset, near-white at noon —
  // never a foreign blue line over a burning sky
  const glare = (lift: number): readonly [number, number, number] => [
    Math.round(tint[0] + (255 - tint[0]) * lift),
    Math.round(tint[1] + (255 - tint[1]) * lift),
    Math.round(tint[2] + (255 - tint[2]) * lift),
  ];
  const css = (c: readonly number[], a: number): string =>
    `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`;
  const tintCss = (a: number): string => css(tint, a);
  const W = cam.W;
  const H = cam.H;
  const cx = W / 2.0;
  const cy = H / 2.0;
  // the flare blooms with the hour: stronger at the low glaring sun
  const s = Math.min(1.0, Math.max(0.0, (mood.discSize - 0.035) / 0.055));
  const strength = align ** 2 * (0.75 + 0.6 * s)
                    * iris.exposure;   // the tightened iris dims the flare
  ctx.globalCompositeOperation = "lighter";
  if (RENDER_STATE.flare === "panavision") {
    // THE ANAMORPHIC STREAK: long, thin — the panavision stamp, worn
    // in the sky's own glare colour
    const len = W * 0.9 * strength;
    const thick = Math.max(1.5, H * 0.006);
    const edge = glare(0.25);
    const mid = glare(0.40);
    const grad = ctx.createLinearGradient(sx - len, sy, sx + len, sy);
    grad.addColorStop(0.0, css(edge, 0));
    grad.addColorStop(0.35, css(edge, 0.35 * strength));
    grad.addColorStop(0.5, css(mid, 0.55 * strength));
    grad.addColorStop(0.65, css(edge, 0.35 * strength));
    grad.addColorStop(1.0, css(edge, 0));
    ctx.fillStyle = grad;
    ctx.fillRect(sx - len, sy - thick, len * 2.0, thick * 2.0);
    // the soft halo around the source itself — the sky's glow family
    const halo = Math.max(12.0, H * 0.10 * strength);
    const g2 = ctx.createRadialGradient(sx, sy, 0.0, sx, sy, halo);
    g2.addColorStop(0.0, css(glare(0.30), 0.22 * strength));
    g2.addColorStop(1.0, css(glare(0.30), 0));
    ctx.fillStyle = g2;
    ctx.fillRect(sx - halo, sy - halo, halo * 2.0, halo * 2.0);
    // the whisper of a vertical streak (the lens's true axis)
    const vlen = H * 0.22 * strength;
    const vthick = Math.max(1.0, H * 0.003);
    const g3 = ctx.createLinearGradient(sx, sy - vlen, sx, sy + vlen);
    g3.addColorStop(0.0, css(mid, 0));
    g3.addColorStop(0.5, css(glare(0.50), 0.18 * strength));
    g3.addColorStop(1.0, css(mid, 0));
    ctx.fillStyle = g3;
    ctx.fillRect(sx - vthick, sy - vlen, vthick * 2.0, vlen * 2.0);
  } else if (RENDER_STATE.flare === "michaelbay") {
    // THE BLOCKBUSTER: hot core + cross streaks + ring + ghost chain
    const core = Math.max(6.0, H * 0.05 * strength);
    const g1 = ctx.createRadialGradient(sx, sy, 0.0, sx, sy, core);
    g1.addColorStop(0.0, `rgba(255,255,255,${(0.5 * strength).toFixed(3)})`);
    g1.addColorStop(0.5, css(glare(0.45), 0.28 * strength));
    g1.addColorStop(1.0, css(glare(0.45), 0));
    ctx.fillStyle = g1;
    ctx.fillRect(sx - core, sy - core, core * 2.0, core * 2.0);
    // the cross: two bright streaks through the source, in the glare
    // family (the same colour the sky's glow carries)
    const hlen = W * 0.55 * strength;
    const hthick = Math.max(2.0, H * 0.008);
    const streak = glare(0.25);
    const gh = ctx.createLinearGradient(sx - hlen, sy, sx + hlen, sy);
    gh.addColorStop(0.0, css(streak, 0));
    gh.addColorStop(0.5, css(streak, 0.5 * strength));
    gh.addColorStop(1.0, css(streak, 0));
    ctx.fillStyle = gh;
    ctx.fillRect(sx - hlen, sy - hthick, hlen * 2.0, hthick * 2.0);
    const vlen = H * 0.4 * strength;
    const gv = ctx.createLinearGradient(sx, sy - vlen, sx, sy + vlen);
    gv.addColorStop(0.0, css(streak, 0));
    gv.addColorStop(0.5, css(streak, 0.3 * strength));
    gv.addColorStop(1.0, css(streak, 0));
    ctx.fillStyle = gv;
    ctx.fillRect(sx - hthick, sy - vlen, hthick * 2.0, vlen * 2.0);
    // the ring, wide around the source
    const ring = Math.max(30.0, H * 0.28 * strength);
    ctx.strokeStyle = css(glare(0.30), 0.10 * strength);
    ctx.lineWidth = Math.max(2.0, H * 0.01);
    ctx.beginPath();
    ctx.arc(sx, sy, ring, 0.0, 2.0 * Math.PI);
    ctx.stroke();
    // THE GHOST CHAIN: discs marching through the frame centre, their
    // hues DERIVED FROM THE SUN'S TINT (warm lifts and cool/violet
    // mixes of it — bokeh of the same light, never fixed RGB)
    const ax = cx - sx;                      // centre relative to sun
    const ay = cy - sy;
    const mix3 = (a: readonly number[], b: readonly number[], t: number):
        readonly number[] => [
      Math.round(a[0] * (1 - t) + b[0] * t),
      Math.round(a[1] * (1 - t) + b[1] * t),
      Math.round(a[2] * (1 - t) + b[2] * t),
    ];
    const ghostCols = [
      glare(0.10),                                   // the warm echo
      mix3(tint, [120, 190, 255], 0.45),             // tinted cool
      mix3(tint, [200, 140, 255], 0.45),             // tinted violet
    ];
    for (let k = 1; k <= 5; k++) {
      for (const side of [1, -1] as const) {
        const t = side * (0.18 * k);
        const gx = sx + ax + ax * t;
        const gy = sy + ay + ay * t;
        const gr = Math.max(4.0, H * 0.030 * (0.85 + 0.2 * ((k * 3) % 4)));
        const gg = ctx.createRadialGradient(gx, gy, 0.0, gx, gy, gr);
        const col = ghostCols[(k + (side === 1 ? 0 : 1)) % 3];
        gg.addColorStop(0.0, css(col, 0.10 * strength));
        gg.addColorStop(1.0, css(col, 0));
        ctx.fillStyle = gg;
        ctx.fillRect(gx - gr, gy - gr, gr * 2.0, gr * 2.0);
      }
    }
  }
  ctx.globalCompositeOperation = "source-over";
}

/** One cloud: a cluster of soft puffs at its fixed world spot, its flat
 *  grey base shaded — 80% TRANSPARENT (a faint veil, not a whiteout) and
 *  three times the cluster radius, drawn behind anything nearer. */
function pushCloud(objects: WorldObject[], cam: Camera, c: CloudSpot): void {
  const dist = Math.hypot(c.x - cam.px, c.y - cam.py, c.alt - cam.pz);
  const cr = (c.r * cam.f) / dist;          // the cluster's projected radius
  if (cr < 2.0) {
    return;                                  // sub-pixel fluff
  }
  const p = projectPoint(cam, c.x, c.y, c.alt);
  if (p === null) {
    return;
  }
  if (p[0] < -cr * 3.0 || p[0] > cam.W + cr * 3.0
      || p[1] < -cr * 3.0 || p[1] > cam.H + cr * 3.0) {
    return;                                  // off the windshield
  }
  const fade = Math.max(0.0, 1.0 - dist / 16000.0);   // melts into the haze
  if (fade <= 0.02) {
    return;
  }
  objects.push({
    depth: dist,
    draw: (ctx) => {
      // the mood's tint: the cloud tops wear it (sunrise = pink)
      const cloud = RENDER_STATE.mood.cloud;
      const base = [Math.round(cloud[0] * 0.79), Math.round(cloud[1] * 0.83), Math.round(cloud[2] * 0.88)];
      // the puff layout: along the wind axis, deterministic per cloud
      const wind = hash01(c.seedX, c.seedY, 41) * Math.PI;
      const ax = Math.cos(wind);
      const ay = Math.sin(wind);
      for (let k = 0; k < 4; k++) {
        const off = (k - 1.5) * 0.42;                 // along the wind
        const jitterY = (hash01(c.seedX + k, c.seedY, 42) - 0.5) * 0.5;
        const jitterR = 0.45 + 0.20 * hash01(c.seedX, c.seedY + k, 43);
        const bx = p[0] + ax * off * cr;
        const by = p[1] + ay * off * cr * 0.4 + jitterY * cr * 0.3;
        const br = cr * jitterR;
        const grad = ctx.createRadialGradient(bx, by - br * 0.15, 0.0,
                                              bx, by, br);
        grad.addColorStop(0.0, `rgba(${cloud[0]},${cloud[1]},${cloud[2]},${(0.15 * fade).toFixed(3)})`);
        grad.addColorStop(1.0, `rgba(${cloud[0]},${cloud[1]},${cloud[2]},0)`);
        ctx.fillStyle = grad;
        ctx.fillRect(bx - br, by - br, br * 2.0, br * 2.0);
      }
      // the flat grey base — the cloud's own shadow on itself
      const grad = ctx.createRadialGradient(p[0], p[1] + cr * 0.22, 0.0,
                                            p[0], p[1] + cr * 0.22, cr * 0.9);
      grad.addColorStop(0.0, `rgba(${base[0]},${base[1]},${base[2]},${(0.06 * fade).toFixed(3)})`);
      grad.addColorStop(1.0, `rgba(${base[0]},${base[1]},${base[2]},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(p[0] - cr, p[1] - cr * 0.7, cr * 2.0, cr * 1.8);
    },
  });
}

/** One draw of the landscape's objects: the tower, the trees, the
 * houses and the clouds — collected from the flora's deterministic
 * world and depth-sorted (painter's algorithm, far to near), ground
 * shadows painted first. `withTrees` false when the GPU pipeline
 * already instanced the forest (its clump owns the trees then). */
function drawWorldObjects(ctx: CanvasRenderingContext2D, cam: Camera,
                          withTrees: boolean): void {
  const objects: WorldObject[] = [];
  pushTower(objects, cam);
  if (withTrees) {
    const trees: TreeSpot[] = [];
    flora.treesNear(cam.px - 700.0, cam.py - 700.0,
                    cam.px + 700.0, cam.py + 700.0, trees);
    for (const t of trees) {
      pushTree(objects, cam, t);
    }
  }
  const houses: HouseSpot[] = [];
  flora.housesNear(cam.px - 2500.0, cam.py - 2500.0,
                   cam.px + 2500.0, cam.py + 2500.0, houses);
  for (const h of houses) {
    pushHouse(objects, cam, h);
  }
  const clouds: CloudSpot[] = [];
  flora.cloudsNear(cam.px - 8000.0, cam.py - 8000.0,
                   cam.px + 8000.0, cam.py + 8000.0, clouds);
  for (const c of clouds) {
    pushCloud(objects, cam, c);
  }
  for (const o of objects) {
    if (o.shadow !== undefined) {
      o.shadow(ctx, cam);
    }
  }
  objects.sort((a, b) => b.depth - a.depth);
  for (const o of objects) {
    o.draw(ctx, cam);
  }
}

/** The HDG tape across the top: ticks and cardinal letters, the current
 * heading boxed at the centre. */
function drawHeadingTape(ctx: CanvasRenderingContext2D, W: number,
                         heading: number): void {
  const h = 18;
  const pxPerDeg = 2.2;
  ctx.fillStyle = "rgba(2,6,12,0.62)";
  ctx.fillRect(0, 0, W, h);
  const hdg = (((heading * 180 / Math.PI) % 360) + 360) % 360;
  ctx.font = "10px Rajdhani, 'SF Mono', monospace";
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
    ctx.strokeStyle = "rgba(160,220,245,0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, h);
    ctx.lineTo(x, h - (major ? 8 : 4));
    ctx.stroke();
    if (major) {
      const cardinals: Record<number, string> = { 0: "N", 90: "E", 180: "S", 270: "W" };
      ctx.fillStyle = "rgba(160,220,245,0.9)";
      ctx.fillText(cardinals[d] ?? String(d / 10), x, h - 10);
    }
  }
  ctx.fillStyle = "#040a12";
  ctx.fillRect(W / 2 - 17, h - 1, 34, 13);
  ctx.strokeStyle = "#35d0ff";
  ctx.strokeRect(W / 2 - 16.5, h - 0.5, 33, 12);
  ctx.fillStyle = "#35d0ff";
  ctx.font = "bold 12px Rajdhani, 'SF Mono', monospace";
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
  ctx.fillStyle = "rgba(2,6,12,0.62)";
  ctx.fillRect(0, top, w, bot - top);
  const midY = (top + bot) / 2;
  const pxPerMph = 1.6;
  ctx.font = "10px Rajdhani, 'SF Mono', monospace";
  ctx.textAlign = "right";
  for (let v = 0; v <= 170; v += 10) {
    const y = midY - (v - mph) * pxPerMph;
    if (y < top + 2 || y > bot - 2) {
      continue;
    }
    const major = v % 20 === 0;
    ctx.strokeStyle = "rgba(160,220,245,0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w, y);
    ctx.lineTo(w - (major ? 9 : 4), y);
    ctx.stroke();
    if (major) {
      ctx.fillStyle = "rgba(160,220,245,0.9)";
      ctx.fillText(String(v), w - 11, y + 3);
    }
  }
  ctx.fillStyle = "#040a12";
  ctx.fillRect(0, midY - 9, w + 16, 18);
  ctx.strokeStyle = "#35d0ff";
  ctx.strokeRect(0.5, midY - 8.5, w + 15, 17);
  ctx.fillStyle = "#35d0ff";
  ctx.font = "bold 14px Rajdhani, 'SF Mono', monospace";
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
  ctx.fillStyle = "rgba(2,6,12,0.62)";
  ctx.fillRect(x0, top, w, bot - top);
  const midY = (top + bot) / 2;
  const pxPerFt = 0.16;
  ctx.font = "10px Rajdhani, 'SF Mono', monospace";
  ctx.textAlign = "left";
  const base = Math.round(altFt / 100) * 100;
  for (let v = base - 1200; v <= base + 1200; v += 100) {
    const y = midY - (v - altFt) * pxPerFt;
    if (y < top + 2 || y > bot - 2) {
      continue;
    }
    const major = v % 200 === 0;
    ctx.strokeStyle = "rgba(160,220,245,0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x0 + (major ? 9 : 4), y);
    ctx.stroke();
    if (major) {
      ctx.fillStyle = "rgba(160,220,245,0.9)";
      ctx.fillText(String(v), x0 + 11, y + 3);
    }
  }
  // current altitude box (opens inward)
  ctx.fillStyle = "#040a12";
  ctx.fillRect(x0 - 16, midY - 9, w + 16, 18);
  ctx.strokeStyle = "#35d0ff";
  ctx.strokeRect(x0 - 15.5, midY - 8.5, w + 15, 17);
  ctx.fillStyle = "#35d0ff";
  ctx.font = "bold 13px Rajdhani, 'SF Mono', monospace";
  ctx.textAlign = "right";
  ctx.fillText(altFt.toFixed(0), x0 + w - 2, midY + 4);
  ctx.textAlign = "left";

  // VSI: the thick vertical segment beside the tape, zero at mid-height
  const vx = x0 - 22;                       // segment centre
  const vTop = midY - 66;
  const vBot = midY + 66;
  const clampV = Math.max(-16.0, Math.min(16.0, vsHundreds));
  const dy = clampV * (66.0 / 16.0);        // px, positive = climb
  ctx.strokeStyle = "rgba(70,115,140,0.5)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(vx, vTop);
  ctx.lineTo(vx, vBot);
  ctx.stroke();
  ctx.setLineDash([]);
  const y0 = Math.min(midY, midY - dy);
  const y1 = Math.max(midY, midY - dy);
  ctx.fillStyle = Math.abs(vsHundreds) < 0.05 ? "rgba(53,208,255,0.35)" : "#35d0ff";
  ctx.fillRect(vx - 4, y0, 8, Math.max(1.0, y1 - y0));
  // the zero notch: a solid bright tick across the segment at zero climb,
  // drawn over the bar so level flight reads at a glance
  ctx.strokeStyle = "#d0ecff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(vx - 7, midY);
  ctx.lineTo(vx + 7, midY);
  ctx.stroke();
  ctx.fillStyle = "rgba(160,220,245,0.75)";
  ctx.font = "9px Rajdhani, 'SF Mono', monospace";
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
  for (const [colour, width] of [["#040a12", 4.0], ["#35d0ff", 2.0]] as const) {
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

/**
 * The cockpit square: ADI horizon card, traffic, flight + engine gauges.
 *
 * THE EXECUTOR SEAM: when a GPU WorldRenderer is handed in and ready,
 * it renders the world (sky, terrain quads, the instanced forest) into
 * the canvas stack's world layer and this canvas carries only the HUD
 * ink — the pitch ladder, the tapes, the marker, the 2D overlays; the
 * CPU path (the pixel-exact reference, and the only path the headless
 * harness ever sees) renders the world right here, as it always has.
 */
export function drawCockpitView(canvas: HTMLCanvasElement, title: string,
                                view: DroneView, others: readonly DroneView[],
                                foot: string, titleEl?: HTMLElement,
                                footEl?: HTMLElement,
                                world?: WorldRenderer | null,
                                lightCanvas?: HTMLCanvasElement | null,
                                focal?: number): void {
  const ctx = fitCanvas(canvas);
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const s = view;
  // The camera: attached to this drone — it moves and rotates with the
  // airframe; the viewport (the canvas) stays fixed, and everything below
  // is transformed INTO it by the one standard projection.
  const cam = cameraOf(s, W, H);
  // THE WINDSHIELD'S SHARED FOCAL: the curved view's three panes share
  // ONE projection scale — the whole view spans the 90° law, so each
  // pane carries its own slice at the same px-per-degree (a pane alone
  // would zoom; the shared focal keeps the seams continuous)
  if (focal !== undefined && focal > 0) {
    cam.f = focal;
  }

  // THE PILOT'S IRIS (HDR adaptation): one reflex per frame — the eye
  // tightens toward the sun and the whole world dims by its exposure
  // (the GPU uniform's extras.y; the CPU compose and the light ink
  // read the same number)
  iris.update(sunAlign(cam));

  // THE LIGHT LAYER: sun, flare and glare ink — drawn on the light
  // canvas (composited ADDITIVE/plus-lighter between the world and
  // the HUD) when the world rendered on its own canvas below, or
  // drawn right here with the lighter composite when the CPU path
  // painted the world into THIS canvas (its sun/glare already add).
  // The layer takes ALL light ink, both paths — one owner.
  const lightCtx = lightCanvas !== null && lightCanvas !== undefined
    ? fitCanvas(lightCanvas)
    : null;
  if (lightCtx !== null) {
    lightCtx.clearRect(0, 0, lightCanvas!.clientWidth,
                       lightCanvas!.clientHeight);
  }

  // the windshield: the GPU pipeline renders the world layer when it
  // is handed in and ready; otherwise the CPU path — the per-pixel
  // march, sky, sun and all — fills the WHOLE square right here
  const gpuWorld = world !== null && world !== undefined
                   && world.ready;
  // THE HUD GLASS: additive only when the world lives on its own
  // canvas below (GPU path) — then the HUD ink ADDS to the world.
  // On the CPU path THIS canvas holds the opaque world itself, so it
  // must composite normally (adding it would double-brighten).
  canvas.style.mixBlendMode = gpuWorld ? "plus-lighter" : "normal";
  if (gpuWorld) {
    const eyeHeight = cam.pz - ground.heightAt(cam.px, cam.py);
    world!.setExposure(iris.exposure);
    world!.renderFrame(cam, eyeHeight);    // the sun, flare and glare veil: light ink on the additive layer
    // — the sky's own glow family, ADDED to the world's own glare
    const ink = lightCtx ?? ctx;
    if (ink === ctx) {
      ctx.globalCompositeOperation = "lighter";
    }
    drawSun(ink, cam);
    drawLensFlare(ink, cam);
    if (world!.featuresOf.glare) {
      drawSunGlare(ink, cam);
    }
    if (ink === ctx) {
      ctx.globalCompositeOperation = "source-over";
    }
  } else {
    drawWorld(ctx, cam);
    // the CPU reference path: the sun, flare and glare add onto the
    // world painted in THIS canvas (their own lighter composite)
    drawSun(ctx, cam);
    drawLensFlare(ctx, cam);
    drawSunGlare(ctx, cam);
  }

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
  // no horizon line is drawn: the ground's far haze and the sky's fog
  // band are the SAME pale colour, so the world melts at the horizon —
  // the pitch ladder below is the attitude reference
  // pitch ladder: the world's elevation rungs, every 10° from −90° up to
  // +90° — the full range. Zero is the horizon itself: the world melts
  // into haze there, so the ladder is the only attitude reference.
  ctx.strokeStyle = "rgba(170,232,255,0.55)";
  ctx.fillStyle = "rgba(170,232,255,0.75)";
  ctx.font = "10px Rajdhani, 'SF Mono', monospace";
  ctx.lineWidth = 1;
  for (let d = -90; d <= 90; d += 10) {
    if (d === 0) {
      continue;                               // the haze line is the 0° rung
    }
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
  // the landscape's objects: tower, trees, houses, clouds, depth-sorted
  // (the GPU pipeline instanced the forest itself — no painter trees)
  drawWorldObjects(ctx, cam, !gpuWorld);
  // traffic: other drones in the forward view
  ctx.font = "10px Rajdhani, 'SF Mono', monospace";
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
    ctx.fillStyle = dz > 2 ? "#9fd8ff" : dz < -2 ? "#ffb066" : "#dff2fc";
    ctx.beginPath();
    ctx.moveTo(tx, ty - size);
    ctx.lineTo(tx + size * 0.7, ty + size * 0.6);
    ctx.lineTo(tx - size * 0.7, ty + size * 0.6);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "rgba(170,225,250,0.7)";
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

  // the ADI: a bright reference circle carrying the BANK SCALE — a notch
  // every 5°, longer notches at 15/30/60°, and the CARDINAL marks 0/45/90°
  // BIGGER and THICKER — all 65% transparent, drawn in the HUD's ink (the
  // same pale blue as the rest of its text); the amber bank pointer rides
  // the card against it
  ctx.strokeStyle = "rgba(170,232,255,0.35)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, 2 * Math.PI);
  ctx.stroke();
  const BANK_MAJOR = new Set([15, 30, 60]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let a = -90; a <= 90; a += 5) {
    const aa = Math.abs(a);
    if (aa === 0 || aa === 45 || aa === 90) {
      continue;                        // the cardinals get their own pass
    }
    const len = BANK_MAJOR.has(aa) ? 10 : 5;
    const th = (a * Math.PI) / 180 - Math.PI / 2;   // angles from the top
    ctx.moveTo(cx + Math.cos(th) * R, cy + Math.sin(th) * R);
    ctx.lineTo(cx + Math.cos(th) * (R - len), cy + Math.sin(th) * (R - len));
  }
  ctx.stroke();
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (const a of [-90, -45, 0, 45, 90]) {
    const th = (a * Math.PI) / 180 - Math.PI / 2;
    ctx.moveTo(cx + Math.cos(th) * R, cy + Math.sin(th) * R);
    ctx.lineTo(cx + Math.cos(th) * (R - 15), cy + Math.sin(th) * (R - 15));
  }
  ctx.stroke();

  // bank pointer (rides the card): the amber triangle against the scale
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
    ctx.font = "bold 15px Rajdhani, 'SF Mono', monospace";
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

/** The controls square: one NARROW HORIZONTAL STRIP —
 *        LEFT RUDDER | YOKE | RIGHT RUDDER | A E R T | the RPM bar —
 * with every label riding the strip's BOTTOM EDGE.
 *
 * THE HAND'S INPUTS drive this panel — what the pilot at the controls is
 * commanding right now (the student's own inputs while it flies), not
 * merely where the airframe's surfaces happen to be. The A E R T strip
 * carries the four channels' decisions (A aileron, E elevator, R rudder,
 * T throttle — ▲ driving the lever up a level, ▼ down a level, · hold;
 * the beat dot pulses when a decision window just opened; while the
 * instructor demonstrates there are no discrete decisions — its hand
 * eases smoothly — the strip rests). The RPM bar reads the engine:
 * idle at the RIGHT edge, INCREASING LEFTWARDS to full — green at
 * idle, red at the redline.
 */

/** THE SIDE VIEWPORT: one third of the curved windshield — the same
 *  aircraft, the same attitude, the pilot's head yawed `deg` left or
 *  right, at the SAME shared focal (the windshield's own
 *  px-per-degree, so the pane continues the centre view's projection
 *  with no zoom jump at the seam). The GPU world renders through the
 *  yawed camera (the panes are GPU-only: the CPU per-pixel path is
 *  the reference view, the centre pane keeps it). One canvas, one
 *  context: the pane canvas IS the renderer's webgpu context, so no
 *  2D ink rides here — the sky shader's own sun disc and glow are
 *  the pane's light. No HUD instruments either: the panes are
 *  peripheral vision. */
export function drawSideViewport(canvas: HTMLCanvasElement | null,
                                 view: DroneView,
                                 world: WorldRenderer | null,
                                 deg: number,
                                 focal: number): void {
  if (canvas === null || world === null || !world.ready) {
    return;
  }
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (W < 8 || H < 8) {
    return;                          // the pane is hidden: nothing to do
  }
  const cam = yawedCameraOf(view, W, H, deg);
  cam.f = focal;                     // the shared windshield focal
  // the world, the iris, the eye height — the windshield's own laws
  world.setExposure(iris.exposure);
  const eyeHeight = cam.pz - ground.heightAt(cam.px, cam.py);
  world.renderFrame(cam, eyeHeight);
}

export function drawControlsView(canvas: HTMLCanvasElement, view: DroneView): void {
  const ctx = fitCanvas(canvas);
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const s = view;
  const SURF_MAX = 0.35;    // mechanical limit, rad

  // the controls as the SELECTED AIRCRAFT actually holds them — the
  // machine's own state, read through its sealed getters
  const inAil = s.ail;
  const inElev = s.elev;
  const inRud = s.rud;
  const inThr = Math.max(0.0, Math.min(1.0, s.throttle));

  // THE STRIP: left rudder pedal, yoke, right rudder pedal, the A E R T
  // decision boxes, the RPM bar — all centred on one row, labels below
  const textY = H - 8;              // the bottom text band's baseline
  const midY = (H - 26) / 2 + 4;    // the content row's centre
  const pl = W * 0.08;              // the left pedal's centre
  const yk = W * 0.21;              // the yoke's centre
  const pr = W * 0.355;             // the right pedal's centre
  const aert0 = W * 0.44;           // the A E R T boxes' left edge
  const rpmL = W * 0.66;            // the RPM bar's LEFT edge (full)
  const rpmR = W - 16;              // the RPM bar's RIGHT edge (idle)

  ctx.font = "11px Rajdhani, 'SF Mono', monospace";

  // --- the rudder pedals: left of the yoke, right of the yoke --------------
  const press = (inRud / SURF_MAX) * H * 0.10;   // one forward, one back
  const pedalH = H * 0.34;
  for (const side of [-1, 1]) {
    const cx = side < 0 ? pl : pr;
    const forward = (inRud > 0 && side > 0) || (inRud < 0 && side < 0);
    const disp = inRud === 0 ? 0.0 : forward ? -press : press * 0.35;
    const engaged = inRud !== 0 && forward;
    // the rest reference: a dashed tick at the pedal's neutral height
    ctx.strokeStyle = "#232b3d";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx - 11, midY);
    ctx.lineTo(cx + 11, midY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = engaged ? "#ff8c3a" : "#14283a";
    ctx.fillRect(cx - 7, midY + disp - pedalH / 2, 14, pedalH);
  }

  // --- the yoke: aileron rotates the wheel, elevator pushes/pulls it ------
  const pull = (inElev / SURF_MAX) * H * 0.08;    // nose up = pull = toward pilot
  const wheelAngle = (inAil / SURF_MAX) * 1.15;   // rad of visual rotation
  const hw = Math.min(30.0, H * 0.26);            // the horn half-spacing
  const hornUp = Math.min(18.0, H * 0.16);
  const dip = Math.min(9.0, H * 0.08);           // the lower bar's depth

  // the neutral cross reference
  ctx.strokeStyle = "#232b3d";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(yk - hw - 10, midY);
  ctx.lineTo(yk + hw + 10, midY);
  ctx.moveTo(yk, midY - hornUp - 12);
  ctx.lineTo(yk, midY + dip + 8);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.save();
  ctx.translate(yk, midY + pull);
  ctx.rotate(wheelAngle);
  // the W: two horns, the shallow bar between them
  ctx.strokeStyle = s.alive ? "#d0ecff" : "#3d5a70";
  ctx.lineWidth = Math.max(3.0, H * 0.045);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(-hw, 0);
  ctx.lineTo(-hw, -hornUp);
  ctx.moveTo(-hw * 0.55, 0);
  ctx.lineTo(-hw * 0.55, dip);
  ctx.lineTo(hw * 0.55, dip);
  ctx.lineTo(hw * 0.55, 0);
  ctx.moveTo(hw, 0);
  ctx.lineTo(hw, -hornUp);
  ctx.stroke();
  ctx.fillStyle = s.alive ? "#35586b" : "#3d4658";   // the horn grips
  for (const hx of [-hw, hw]) {
    ctx.beginPath();
    ctx.arc(hx, -hornUp - 2, Math.max(3.5, H * 0.04), 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.fillStyle = "#ff8c3a";                         // the centre hub
  ctx.beginPath();
  ctx.arc(0, 0, 3, 0, 2 * Math.PI);
  ctx.fill();
  ctx.restore();
  ctx.lineCap = "butt";

  // --- A E R T: the four channels' decision boxes --------------------------
  // ▲ driving the lever up a level, ▼ down a level, · hold/already there;
  // the beat dot pulses when a decision window just opened. While the
  // instructor demonstrates (no discrete decisions) the boxes rest.
  const BOX = Math.max(10.0, Math.min(18.0, H * 0.16));
  const CHANNEL_LABELS = ["A", "E", "R", "T"];
  ctx.fillStyle = s.handFresh ? "#ff8c3a" : "#0c1a26";
  ctx.beginPath();
  ctx.arc(aert0 - 9, midY, 2.5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.textAlign = "center";
  for (let i = 0; i < 4; i++) {
    const bx = aert0 + i * (BOX + 7);
    ctx.strokeStyle = "#14283a";
    ctx.strokeRect(bx + 0.5, midY - BOX / 2 + 0.5, BOX, BOX);
    ctx.fillStyle = "#3d5a70";
    ctx.fillText(CHANNEL_LABELS[i], bx + BOX / 2, textY);
    const choice = s.handChoices;
    if (choice === null) {
      ctx.fillStyle = "#14283a";            // the instructor eases: rest
      ctx.fillText("\u2014", bx + BOX / 2, midY + 3.5);
      continue;
    }
    const level = choice[i];
    if (level < 0) {                        // hold: the hand rests
      ctx.fillStyle = "#3d5a70";
      ctx.fillText("\u00b7", bx + BOX / 2, midY + 3.5);
      continue;
    }
    const now = [inAil, inElev, inRud, inThr][i];
    const nowLevel = levelOf(i, now);       // the stop the lever is at
    if (level > nowLevel) {                 // driving the lever up
      ctx.fillStyle = "#ff8c3a";
      ctx.fillText("\u25b2", bx + BOX / 2, midY + 3.5);
    } else if (level < nowLevel) {          // driving the lever down
      ctx.fillStyle = "#9fd8ff";
      ctx.fillText("\u25bc", bx + BOX / 2, midY + 4.5);
    } else {                                // already at the level
      ctx.fillStyle = "#3d5a70";
      ctx.fillText("\u00b7", bx + BOX / 2, midY + 3.5);
    }
  }

  // --- the engine: a horizontal RPM bar, INCREASING LEFTWARDS ---------------
  // Idle at the right edge, full at the left; green at idle, red at the
  // redline — the segment ramp runs right (idle) to left (full), lit
  // from the right edge as the revs rise.
  const rpm = rpmOf(inThr);
  const maxRpm = rpmOf(1.0);
  const revs = Math.max(0.0, Math.min(1.0, rpm / maxRpm));
  const barH = Math.max(10.0, Math.min(22.0, H * 0.30));
  const barY = midY - barH / 2;
  const SEGS = 14;
  const segW = (rpmR - rpmL) / SEGS;
  for (let i = 0; i < SEGS; i++) {
    const xR = rpmR - i * segW;             // i counts leftwards from idle
    const t = i / (SEGS - 1);               // 0 at idle, 1 at full
    const r = Math.round(70 + (255 - 70) * t);
    const g = Math.round(210 + (70 - 210) * t);
    const b = Math.round(90 + (60 - 90) * t);
    const lit = (i + 0.5) / SEGS <= revs;
    ctx.fillStyle = lit ? `rgb(${r},${g},${b})` : "#0c1a26";
    ctx.fillRect(xR - segW + 1.5, barY, segW - 3, barH);
  }
  ctx.strokeStyle = "#14283a";
  ctx.strokeRect(rpmL + 0.5, barY + 0.5, rpmR - rpmL - 1, barH - 1);
  // ticks every 500 rpm along the scale
  ctx.beginPath();
  for (let tick = 500; tick < maxRpm; tick += 500) {
    const x = rpmR - (tick / maxRpm) * (rpmR - rpmL);
    ctx.moveTo(x, barY + barH);
    ctx.lineTo(x, barY + barH + 3);
  }
  ctx.stroke();
  // the number, bottom edge, under the bar's right end
  ctx.font = "bold 12px Rajdhani, 'SF Mono', monospace";
  ctx.fillStyle = rpm > 2450 ? "#ff6a5a" : "#35d0ff";
  ctx.fillText(`RPM ${rpm.toFixed(0)} \u00b7 THR ${Math.round(inThr * 100)}%`,
              (rpmL + rpmR) / 2, textY);
  ctx.font = "11px Rajdhani, 'SF Mono', monospace";

  // --- the labels: every text rides the BOTTOM EDGE ------------------------
  ctx.textAlign = "center";
  ctx.fillStyle = "#3d5a70";
  ctx.fillText("LEFT RUDDER", pl, textY);
  ctx.fillText("YOKE", yk, textY);
  ctx.fillText("RIGHT RUDDER", pr, textY);

  // THE HOUR'S GRADE: the instruments wear the mood's colour — under
  // the frozen alarm, which stays its own red whatever the hour
  gradeControlsVeil(ctx, W, H);

  if (!s.alive) {
    ctx.fillStyle = "rgba(255,58,48,0.16)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#ff4a3a";
    ctx.font = "bold 13px Rajdhani, 'SF Mono', monospace";
    ctx.fillText("CONTROLS FROZEN", W / 2, midY + H * 0.32);
  }
  ctx.textAlign = "left";
}
