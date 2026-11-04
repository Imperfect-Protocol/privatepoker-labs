/**
 * Skynerd Genesis web app — LIVE.
 *
 * ALL of the simulation happens as you watch it: every physics tick is
 * computed in real time. Nothing is recorded, nothing is played back,
 * nothing waits.
 *
 * ONE Aircraft class (physics/aircraft.ts) is the machine for every
 * pilot — the instructor's deterministic hand, the student's AI, the
 * swarm's drones. The SCENE holds every aircraft in the world right
 * now; exactly one is SELECTED and the panel shows its instruments and
 * its controls.
 *
 * The scenario dropdown picks the world:
 *  - every flight school lesson: the endless classroom — instructor
 *    demonstrates, the student repeats, graded against the maneuver's
 *    known goals; fail and the instructor shows it again, forever; ONE
 *    student (same brain, same tensors, every weight carried over)
 *    for the whole page life;
 *  - drone attack: the swarm as a conveyor of PAIRS — two drones lift
 *    off from the runway abreast, the next pair five seconds behind,
 *    pair after pair; each flies the attack mission and then its own
 *    will decides; the flock maintains one shared SingularityTensor
 *    — updated from every neighbour and every neighbour's neighbour —
 *    whose super-goal keeps the carpet in formation; every drone
 *    connects to every nearby drone, and the dying scream their memory
 *    to whoever hears.
 *
 * The aircraft dropdown lists the scene's aircraft. Only a page reload
 * or the restart button ever starts anything over.
 *
 * The clock is a fixed-dt accumulator: wall time x speed becomes sim
 * ticks at exactly 1/50 s each — physics never sees a variable dt.
 */
import { mphOf, vsHundredsOf, altFtOf, rpmOf } from "./units.js";
import {
  drawCockpitView, drawControlsView, drawSideViewport,
  droneViewOf, type DroneView, FOV_HALF,
} from "./cockpit.js";
import { FlightSchoolShow } from "./school-show.js";
import { WavesShow } from "./waves-show.js";
import { BootTrainer } from "./boot-train.js";
import { FlightSchool, elementNameOf } from "../learning/school.js";
import type { TensorStateView } from "../learning/servo-tensor.js";
import type { HudDelta } from "./hud-delta.js";
import type { ControlMode } from "../learning/guardian.js";
import { controlModeLabel } from "../learning/guardian.js";
import { FlightInstruments } from "../learning/instruments.js";
import { TRIM_SPEED } from "../learning/flight.js";
import { drawBrainView, tensorCounts } from "./boot-brain.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import type { Scene, SceneAircraft } from "./scene.js";
import { LandUse } from "./land.js";
import {
  DEFAULT_FEATURES, GpuWorldRenderer,
  type RenderFeatures, type TimeOfDay,
} from "../pipeline/index.js";
import {
  RENDER_STATE, setTimeOfDay,
  type LensFlare,
} from "./render-state.js";
import { groundRender } from "./ground-render.js";

interface GoldenLesson {
  lesson: string;
  attempts: number;
  score: number;
  goals: { label: string; fraction: number }[];
}

const DT = 0.02;

const els = {
  app: document.getElementById("app")!,
  bar: document.getElementById("bar")!,
  sideGrip: document.getElementById("sideGrip")!,
  bottomGrip: document.getElementById("bottomGrip")!,
  sky: document.getElementById("sky") as HTMLCanvasElement,
  status: document.getElementById("status")!,
  narration: document.getElementById("narration")!,
  play: document.getElementById("play") as HTMLButtonElement,
  speed: document.getElementById("speed") as HTMLSelectElement,
  scenario: document.getElementById("scenario") as HTMLSelectElement,
  aircraft: document.getElementById("aircraft") as HTMLSelectElement,
  controls: document.getElementById("controls") as HTMLSelectElement,
  next: document.getElementById("next") as HTMLButtonElement,
  restart: document.getElementById("restart") as HTMLButtonElement,
  zoomIn: document.getElementById("zoomIn") as HTMLButtonElement,
  zoomOut: document.getElementById("zoomOut") as HTMLButtonElement,
  zoomReset: document.getElementById("zoomReset") as HTMLButtonElement,
  timeOfDay: document.getElementById("timeOfDay") as HTMLSelectElement | null,
  lensFlare: document.getElementById("lensFlare") as HTMLSelectElement | null,
  badge: document.getElementById("badge")!,
  tabWiki: document.getElementById("tabWiki")!,
  tabAi: document.getElementById("tabAi")!,
  results: document.getElementById("results") as HTMLElement,
  cockpit: document.getElementById("cockpit") as HTMLElement,
  cockpitTitle: document.getElementById("cockpit-title")!,
  cockpitFoot: document.getElementById("cockpit-foot")!,
  cockpitView: document.getElementById("cockpit-view") as HTMLElement | null,
  worldCanvas: document.getElementById("worldCanvas") as HTMLCanvasElement | null,
  lightCanvas: document.getElementById("lightCanvas") as HTMLCanvasElement | null,
  worldCanvasL: document.getElementById("worldCanvasL") as HTMLCanvasElement | null,
  worldCanvasR: document.getElementById("worldCanvasR") as HTMLCanvasElement | null,
  cockpitCanvas: document.getElementById("cockpitCanvas") as HTMLCanvasElement,
  controlsCanvas: document.getElementById("controlsCanvas") as HTMLCanvasElement,
  boot: document.getElementById("boot") as HTMLElement,
  bootFill: document.getElementById("bootFill") as HTMLElement,
  bootLog: document.getElementById("bootLog") as HTMLElement,
  bootBrain: document.getElementById("bootBrain") as HTMLCanvasElement,
  bootExpand: document.getElementById("bootExpand") as HTMLButtonElement,
  bootDone: document.getElementById("bootDone") as HTMLButtonElement,
};

/** The scenario ids: every lesson name, then the drone attack waves. */
const WAVES_SCENARIO = "drone-attack-waves";

// --- the world renderer: the GPU pipeline when the browser grants one
// (the cockpit falls back to its CPU path per frame otherwise) --------
const renderFeatures: RenderFeatures = { ...DEFAULT_FEATURES };
let gpuWorld: GpuWorldRenderer | null = null;
/** The side viewport panes' own renderers — created on the MAIN
 *  renderer's device once it exists (one device, three canvases);
 *  empty until then (and forever on the CPU-path machines — the
 *  strips stay hidden without a GPU world). */
const sideWorlds: GpuWorldRenderer[] = [];
{
  const world = new LandUse();   // the plan's own arrays, shared
  const textures = {
    heights: world.heightsOf(),
    biomes: world.biomesOf(),
  };
  void GpuWorldRenderer.create(els.worldCanvas, textures)
    .then(async (renderer) => {
      gpuWorld = renderer;
      if (gpuWorld === null) {
        return;
      }
      gpuWorld.setFeatures(renderFeatures);
      gpuWorld.setTimeOfDay(RENDER_STATE.tod);
      // the side panes: the same world on the same device, one
      // renderer per strip canvas
      const sides = await Promise.all([
        GpuWorldRenderer.create(els.worldCanvasL, textures,
                                gpuWorld.gpuDevice),
        GpuWorldRenderer.create(els.worldCanvasR, textures,
                                gpuWorld.gpuDevice),
      ]);
      for (const side of sides) {
        if (side !== null) {
          side.setFeatures(renderFeatures);
          side.setTimeOfDay(RENDER_STATE.tod);
          sideWorlds.push(side);
        }
      }
    });
}

// the toolbar's render toggles (plan §5), next to RESET; absent
// headlessly — the harness stubs have no checkbox elements
const FEATURE_CHECKS: readonly (readonly [string,
                                        keyof RenderFeatures])[] = [
  ["cbRelief", "relief"],
  ["cbGlare", "glare"],
  ["cbHaze", "haze"],
  ["cbSunShadows", "sunShadows"],
  ["cbWaterRefl", "waterReflections"],
];
/** The toolbar's choices reach EVERY pane: the windshield and the two
 *  side strips — one features object, fanned out. */
function applyFeatures(): void {
  if (gpuWorld !== null) {
    gpuWorld.setFeatures(renderFeatures);
  }
  for (const side of sideWorlds) {
    side.setFeatures(renderFeatures);
  }
}

/** The hour reaches every pane too. */
function applyTimeOfDay(tod: TimeOfDay): void {
  if (gpuWorld !== null) {
    gpuWorld.setTimeOfDay(tod);
  }
  for (const side of sideWorlds) {
    side.setTimeOfDay(tod);
  }
}

for (const [id, key] of FEATURE_CHECKS) {
  const el = document.getElementById(id) as HTMLInputElement | null;
  if (el === null) {
    continue;
  }
  renderFeatures[key] = el.checked;
  el.addEventListener("change", () => {
    renderFeatures[key] = el.checked;
    applyFeatures();
  });
}

// the checkerboard ground (plan §5): a feature toggle like the rest —
// both the uniform (extras.x) and the CPU ground read it
{
  const el = document.getElementById("cbChecker") as HTMLInputElement | null;
  if (el !== null) {
    renderFeatures.checker = el.checked;
    RENDER_STATE.checker = el.checked;
    groundRender.checker = el.checked;
    el.addEventListener("change", () => {
      renderFeatures.checker = el.checked;
      RENDER_STATE.checker = el.checked;
      groundRender.checker = el.checked;
      applyFeatures();
    });
  }
}

// the sky's moods (plan §5): the uniform carries them on GPU, the
// CPU path reads the same table — one source, both renderers
{
  const el = els.timeOfDay;
  if (el !== null) {
    el.addEventListener("change", () => {
      const tod = el.value as TimeOfDay;
      setTimeOfDay(tod);                          // the CPU twin
      groundRender.setMood(RENDER_STATE.mood);
      applyTimeOfDay(tod);
    });
  }
}
{
  const el = els.lensFlare;
  if (el !== null) {
    el.addEventListener("change", () => {
      RENDER_STATE.flare = el.value as LensFlare;
    });
  }
}

const state = {
  /** The classroom — ONE student for the whole page life (or null). */
  school: null as FlightSchoolShow | null,
  /** The drone attack waves (or null). */
  waves: null as WavesShow | null,
  /** The boot-trained brain — transplanted into the student and every
   *  newborn drone. */
  brain: null as ServoPolicy | null,
  /** Which scenario is running. */
  scenario: "" as string,
  /** The frozen classroom reference (per-maneuver golden scores). */
  golden: null as GoldenLesson[] | null,
  /** The waves master dice (from reference/rng-ref.json). */
  seed: null as { state: string; inc: string } | null,
  acc: 0.0,
  playing: true,
  reportKey: "",
  /** The user pinned an aircraft in the dropdown (stop auto-following). */
  pinned: false,
  /** The camera: where the sky looks, world metres. While tracking is
   *  on it follows the action; a two-finger drag (the trackpad's wheel)
   *  pans it and turns tracking OFF until the view is reset. */
  camera: { x: 0.0, y: 0.0, tracking: true },
  /** The school top-view zoom: metres spanned across the canvas. */
  span: 260.0,
};

// --- the active world ------------------------------------------------------------

/** The scene of the running scenario. */
function scene(): Scene | null {
  if (state.scenario === WAVES_SCENARIO) {
    return state.waves?.scene ?? null;
  }
  return state.school?.scene ?? null;
}

/** The aircraft to follow: the pinned one, else the action. */
function followTarget(): SceneAircraft | null {
  const world = scene();
  if (world === null) {
    return null;
  }
  if (state.pinned && world.current !== null) {
    return world.current;
  }
  if (state.scenario === WAVES_SCENARIO && state.waves !== null) {
    return world.current;
  }
  const school = state.school;
  if (school === null) {
    return null;
  }
  // ONE aircraft: it is the classroom's whole sky — on the ground or
  // in the air, whoever holds the levers. A manual dropdown pick pins
  // whichever aircraft the person chose (here, always this one).
  return school.flyingEntry;
}/** Keep the scene's selection honest: follow the action unless pinned. */
function autoSelect(): void {
  const world = scene();
  if (world === null || state.pinned) {
    return;
  }
  if (state.scenario === WAVES_SCENARIO) {
    return;                 // the waves keep whatever the dropdown holds
  }
  const target = followTarget();
  if (target !== null && world.current !== target) {
    world.select(target);
  }
}

// --- canvas helpers ---------------------------------------------------------------

/** Size both sidebar panels from the sidebar's own height: each canvas
 *  exactly 3:2, both the same size, the two of them filling the whole
 *  vertical area — so the sidebar (an auto grid column) is exactly as
 *  wide as the panels need and no space is left under the controls. */
/** The sidebar's explicit width once the grip has dragged it (null: the
 *  column sizes itself to its panels). */
let sideWidth: number | null = null;

function layoutSidebar(): void {
  const aside = els.cockpit;
  const style = getComputedStyle(aside);
  const padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
  const padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const gap = parseFloat(style.rowGap) || 8.0;
  // The flow children that are not the view: the title head and the
  // foot readout. The controls strip FLOATS on the view now — it is
  // not a row, so the view takes every metre the column has.
  const others = [...aside.children].filter(
    (child): child is HTMLElement =>
      !(child instanceof HTMLCanvasElement) && child !== els.cockpitView);
  const used = others.reduce((sum, child) => sum + child.offsetHeight, 0.0);
  const gaps = gap * others.length;   // head|view and view|foot
  const avail = aside.clientHeight - padY;
  // THE VIEW = THE CURVED WINDSHIELD: the whole content area, split
  // into three seamless panes when the GPU world renders (left, centre,
  // right — the eye at the centre, the side cameras yawed for a
  // continuous projection). One pane only on the CPU path — the
  // reference view keeps the whole width.
  // THE WIDTH LAW (the original anchor, kept): the sidebar is an
  // AUTO grid track — its width follows its content — so the view's
  // width must come from the HEIGHT (the 16:9 law), never from
  // aside.clientWidth (that feeds the track back into itself and
  // collapses the column to a stub). When the GRIP has pinned the
  // column, the pinned width wins and the panes fill it exactly.
  let viewH = Math.max(120.0, avail - used - gaps);
  let viewW = Math.round(viewH * 16 / 9);
  if (sideWidth !== null) {
    const pinned = Math.max(120.0, Math.round(sideWidth - padX));
    if (pinned < viewW) {
      viewW = pinned;
      viewH = Math.round(viewW * 9 / 16);
    } else {
      viewW = pinned;                    // wide: the panes fill the void
    }
  }
  const thirds = gpuWorld !== null && gpuWorld.ready
    && sideWorlds.length === 2 && sideWorlds.every((w) => w.ready);
  let centreX = 0.0;
  let centreW = viewW;
  if (els.cockpitView !== null) {
    els.cockpitView.style.width = `${Math.round(viewW)}px`;
    els.cockpitView.style.height = `${Math.round(viewH)}px`;
  }
  if (thirds) {
    // the three panes tile the view exactly — widths rounded so they
    // SUM to the view (no subpixel seams), the centre keeps the rest
    const wL = Math.floor(viewW / 3.0);
    const wR = Math.floor((viewW - wL) / 2.0);
    const wC = Math.round(viewW - wL - wR);
    centreX = wL;
    centreW = wC;
    placeCanvas(els.worldCanvasL, 0, wL, viewH);
    placeCanvas(els.worldCanvas, wL, wC, viewH);
    placeCanvas(els.lightCanvas, wL, wC, viewH);
    placeCanvas(els.cockpitCanvas, wL, wC, viewH);
    placeCanvas(els.worldCanvasR, wL + wC, wR, viewH);
  } else {
    hideCanvas(els.worldCanvasL);
    hideCanvas(els.worldCanvasR);
    placeCanvas(els.worldCanvas, 0, viewW, viewH);
    placeCanvas(els.lightCanvas, 0, viewW, viewH);
    placeCanvas(els.cockpitCanvas, 0, viewW, viewH);
  }
  // THE FLOATING CONTROLS: the console slab over the view's bottom,
  // as wide as the centre pane — its own size law, unchanged
  const stripH = Math.max(48.0,
    Math.min(140.0, Math.round(viewH * 0.22)));
  els.controlsCanvas.style.left = `${Math.round(centreX)}px`;
  els.controlsCanvas.style.width = `${Math.round(centreW)}px`;
  els.controlsCanvas.style.height = `${Math.round(stripH)}px`;
}

/** Place one pane canvas inside the view (guarded: an unchanged
 *  placement writes nothing — the drag's layout stays cheap). */
function placeCanvas(canvas: HTMLCanvasElement | null, x: number,
                     w: number, h: number): void {
  if (canvas === null) {
    return;
  }
  if (canvas.style.display === "none") {
    canvas.style.display = "block";
  }
  const left = `${Math.round(x)}px`;
  const width = `${Math.round(w)}px`;
  const height = `${Math.round(h)}px`;
  if (canvas.style.left !== left || canvas.style.width !== width
      || canvas.style.height !== height) {
    canvas.style.left = left;
    canvas.style.width = width;
    canvas.style.height = height;
  }
}

/** Rest one pane: out of the layout entirely. */
function hideCanvas(canvas: HTMLCanvasElement | null): void {
  if (canvas !== null && canvas.style.display !== "none") {
    canvas.style.display = "none";
  }
}

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

function altitudeTint(z: number): string {
  const t = Math.max(0.0, Math.min(1.0, z / 180.0));
  const r = Math.round(30 + (140 - 30) * t);
  const g = Math.round(110 + (235 - 110) * t);
  const b = Math.round(150 + (255 - 150) * t);
  return `rgb(${r},${g},${b})`;
}

function drawDroneTriangle(ctx: CanvasRenderingContext2D, px: number, py: number,
                           heading: number, size: number, fill: string): void {
  ctx.save();
  ctx.translate(px, py);
  // From-sky top view: heading 0 (+X, north) points UP, so the triangle
  // (drawn pointing screen-right) rotates by heading − 90°.
  ctx.rotate(heading - Math.PI / 2);
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.moveTo(size * 1.6, 0);
  ctx.lineTo(-size, size * 0.8);
  ctx.lineTo(-size * 0.55, 0);
  ctx.lineTo(-size, -size * 0.8);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// --- the sky -----------------------------------------------------------------------

function drawSky(): void {
  const ctx = fitCanvas(els.sky);
  const W = els.sky.clientWidth;
  const H = els.sky.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const world = scene();
  const focus = followTarget() ?? world?.current ?? null;
  if (world === null) {
    return;
  }
  // follow camera: the selected aircraft stays centred. FROM-SKY top view:
  // north (+X) points up the screen, east (+Y) points right.
  // While tracking, the camera rides the action; once a two-finger drag
  // pans the view, it stays where you put it until the view is reset.
  if (state.camera.tracking && focus !== null) {
    state.camera.x = focus.aircraft.body.position.x;
    state.camera.y = focus.aircraft.body.position.y;
  }
  const cx = state.camera.x;
  const cy = state.camera.y;
  const span = state.span;                    // metres across the canvas (zoom)
  const scale = W / span;
  const halfY = H / 2 / scale;
  const sx = (y: number) => W / 2 + (y - cy) * scale;
  const sy = (x: number) => H / 2 - (x - cx) * scale;

  // world grid, fixed to the ground
  ctx.strokeStyle = "#0c2334";
  ctx.lineWidth = 1;
  for (let gy = Math.floor((cy - span / 2) / 50) * 50;
       gy <= cy + span / 2; gy += 50) {
    ctx.beginPath();
    ctx.moveTo(sx(gy), 0);
    ctx.lineTo(sx(gy), H);
    ctx.stroke();
  }
  for (let gx = Math.floor((cx - halfY) / 50) * 50;
       gx <= cx + halfY; gx += 50) {
    ctx.beginPath();
    ctx.moveTo(0, sy(gx));
    ctx.lineTo(W, sy(gx));
    ctx.stroke();
  }

  if (state.scenario === WAVES_SCENARIO && state.waves !== null) {
    drawWavesSky(ctx, sx, sy, state.waves);
    return;
  }
  drawSchoolSky(ctx, sx, sy);
}

/** The classroom's sky: the ONE aircraft's flight path. */
function drawSchoolSky(ctx: CanvasRenderingContext2D,
                       sx: (y: number) => number,
                       sy: (x: number) => number): void {
  const school = state.school!;
  const entry = school.aircraftEntry;
  drawTrail(ctx, entry, sx, sy);
  drawAircraftTop(ctx, entry, sx, sy, 10);
}

/** The waves' sky: every drone's trail, the swarm's connections, and
 *  every machine in the air. */
function drawWavesSky(ctx: CanvasRenderingContext2D,
                     sx: (y: number) => number,
                     sy: (x: number) => number,
                     waves: WavesShow): void {
  // the swarm's mesh: every nearby pair, across waves
  ctx.strokeStyle = "rgba(255,184,74,0.22)";
  ctx.lineWidth = 1;
  for (const link of waves.links()) {
    const a = link.a.aircraft.body.position;
    const b = link.b.aircraft.body.position;
    ctx.beginPath();
    ctx.moveTo(sx(a.y), sy(a.x));
    ctx.lineTo(sx(b.y), sy(b.x));
    ctx.stroke();
  }
  for (const entry of waves.scene.aircraft) {
    drawTrail(ctx, entry, sx, sy);
    drawAircraftTop(ctx, entry, sx, sy, 7);
  }
}

/** One aircraft's flight path, altitude-tinted. */
function drawTrail(ctx: CanvasRenderingContext2D, entry: SceneAircraft,
                   sx: (y: number) => number, sy: (x: number) => number): void {
  if (entry.trail.length < 2) {
    return;
  }
  ctx.lineWidth = 1.8;
  for (let i = 1; i < entry.trail.length; i++) {
    const a = entry.trail[i - 1];
    const b = entry.trail[i];
    ctx.globalAlpha = 0.15 + 0.55 * (i / entry.trail.length);
    ctx.strokeStyle = altitudeTint(b.z);
    ctx.beginPath();
    ctx.moveTo(sx(a.y), sy(a.x));
    ctx.lineTo(sx(b.y), sy(b.x));
    ctx.stroke();
  }
  ctx.globalAlpha = 1.0;
}

/** One aircraft from the sky, or its crash site. */
function drawAircraftTop(ctx: CanvasRenderingContext2D, entry: SceneAircraft,
                         sx: (y: number) => number, sy: (x: number) => number,
                         size: number): void {
  const body = entry.aircraft.body;
  const px = sx(body.position.y);
  const py = sy(body.position.x);
  const [fwdX, fwdY] = entry.aircraft.forward;
  const heading = Math.atan2(fwdY, fwdX);
  drawDroneTriangle(ctx, px, py, heading, size,
                    entry.dead ? "#ff4a3a" : altitudeTint(body.position.z));
  if (entry.dead) {
    ctx.strokeStyle = "#ff4a3a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px - 6, py - 6);
    ctx.lineTo(px + 6, py + 6);
    ctx.stroke();
  }
  // the selected one wears the ring
  if (scene()?.current === entry) {
    ctx.strokeStyle = "#ffb84a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, size + 6, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

// --- the panel: the selected aircraft's instruments and controls -------------------

function drawPanel(): void {
  const world = scene();
  const entry = world?.current ?? null;
  if (world === null || entry === null) {
    return;
  }
  const view = droneViewOf(entry);
  const title = `${entry.label.toUpperCase()} \u00b7 ` +
    `${view.alive ? "AIRBORNE" : "DOWN"}`;
  const foot = view.alive
    ? `${altFtOf(view.z).toFixed(0)} ft \u00b7 ` +
      `${view.vs >= 0 ? "+" : ""}${vsHundredsOf(view.vs).toFixed(1)}\u00d7100fpm \u00b7 ` +
      `${mphOf(view.speed).toFixed(0)} mph \u00b7 ${rpmOf(view.throttle).toFixed(0)} rpm`
    : "crashed \u2014 the flight ended on the ground";
  // THE CURVED WINDSHIELD: the view is three seamless panes — left,
  // centre, right — one eye, one shared focal (the whole view spans
  // the 90° law, so every pane renders at the same px-per-degree and
  // the seams continue the projection), the side cameras yawed by
  // exactly the continuity angle: half the centre pane's span plus
  // half the side pane's span. What the centre pane's edge shows, the
  // side pane's edge continues — a curved projection in three facets.
  const viewW = els.cockpitView !== null
    ? els.cockpitView.clientWidth : 0;
  const focal = viewW / 2.0 / Math.tan(FOV_HALF);
  drawCockpitView(els.cockpitCanvas, title, view, [], foot,
                  els.cockpitTitle, els.cockpitFoot, gpuWorld,
                  els.lightCanvas, focal);
  drawControlsView(els.controlsCanvas, view);
  const centreW = els.cockpitCanvas.clientWidth;
  const sideW = els.worldCanvasL !== null
    ? els.worldCanvasL.clientWidth : 0;
  const yawDeg = (Math.atan(centreW / (2.0 * focal))
                  + Math.atan(sideW / (2.0 * focal))) * 180.0 / Math.PI;
  drawSideViewport(els.worldCanvasL, view, sideWorlds[0] ?? null,
                   -yawDeg, focal);
  drawSideViewport(els.worldCanvasR, view, sideWorlds[1] ?? null,
                   yawDeg, focal);
}

// --- narration + status ------------------------------------------------------------

function renderNarration(): void {
  if (state.scenario === WAVES_SCENARIO && state.waves !== null) {
    const waves = state.waves;
    els.narration.textContent =
      `drone attack \u00b7 pair ${waves.pairCount} lifted off \u00b7 ` +
      `${waves.aliveCount} flying \u00b7 the will decides after the mission`;
    els.narration.className = "narration stage-waves";
    els.status.textContent =
      `t ${waves.simSeconds.toFixed(1)}s \u00b7 pair ${waves.pairCount} \u00b7 ` +
      `${waves.aliveCount} flying \u00b7 ${waves.births} born \u00b7 ` +
      `${waves.obituaries} screams heard \u00b7 flock ${waves.flockSize}` +
      ` \u00b7 links ${waves.links().length}` +
      ` \u00b7 singularity ${waves.singularityGoal}` +
      ` (${waves.singularityRounds} rounds)`;
    els.results.hidden = false;
    els.badge.textContent = "";
    els.badge.className = "";
    return;
  }
  const school = state.school;
  if (school === null) {
    return;
  }
  const moment = school.moment;
  els.narration.textContent = moment.narration;
  els.narration.className = `narration stage-${moment.stage}`;
  const attempt = moment.attempt > 1 ? ` \u00b7 attempt ${moment.attempt}` : "";
  const score = school.scores.get(moment.lessonName);
  const scoreText = score === undefined ? "" : ` \u00b7 last flight ${score.toFixed(2)}/6`;
  const intent = moment.flying
    ? ` \u00b7 goal: ${moment.goalName} \u00b7 element: ${moment.elementName}` : "";
  const finishedText = moment.finished ? " \u00b7 flight finished" : "";
  const byText = moment.scoreByInstructor ? " (the instructor's hands)" : "";
  const whoText = moment.instructorAtControls
    ? `\u2691 INSTRUCTOR AT THE CONTROLS (${controlModeLabel(moment.controlsMode)})`
    : `${controlModeLabel(moment.controlsMode)} at the controls`;
  els.status.textContent =
    `maneuver ${moment.lessonIndex}/${moment.lessonCount} \u00b7 ${moment.lessonName}` +
    `${attempt}${scoreText}${byText}${intent}${finishedText} \u00b7 t ${moment.seconds.toFixed(1)}s \u00b7 ` +
    whoText;
  els.results.hidden = false;
  updateBadge();
}

// --- the bottom panel's TAB GROUP --------------------------------------------
// Tab 1: the scenario's wiki-style description. Tab 2: the live AI state
// log — one row per decision window of the selected AI, newest on top.

const tabButtons =
  [...document.querySelectorAll<HTMLButtonElement>(".tab")];

/** How many decision rows the live log keeps (newest first). */
const AI_LOG_ROWS = 20;

const tabs = {
  active: "wiki" as "wiki" | "ai",
  wikiKey: "",
  /** Which AI the log belongs to (a new subject clears the log). */
  aiKey: "",
  aiLastDecision: -1,
  rows: [] as AiLogRow[],
};

/** One row of the live log. The same state held over many decision
 *  windows collapses into ONE row: firstTick is where it began,
 *  lastTick is where it is now (the tick cell shows both). */
interface AiLogRow {
  firstTick: number;
  lastTick: number;
  /** What makes this row THIS row — labels only, the drifting
 *      shares and the live delta never take part in the identity. */
  key: string;
  /** The row's <td> cells after the tick cell. */
  cells: string;
}

for (const button of tabButtons) {
  button.addEventListener("click", () => {
    tabs.active = button.dataset.tab as "wiki" | "ai";
    for (const other of tabButtons) {
      other.classList.toggle("active", other === button);
    }
    els.tabWiki.hidden = tabs.active !== "wiki";
    els.tabAi.hidden = tabs.active !== "ai";
  });
}

/** The selected AI's state, per scenario: the classroom's student, or
 *  the drone the aircraft dropdown holds. */
function aiSubject(): {
  view: TensorStateView | null;
  goalName: string;
  delta: HudDelta | null;
  key: string;
  note: string;
} {
  if (state.scenario === WAVES_SCENARIO && state.waves !== null) {
    const waves = state.waves;
    const label = waves.scene.current?.label ?? "";
    return {
      view: waves.selectedStateView,
      goalName: "free flight",
      delta: waves.carpetDelta,
      key: `waves|${label}`,
      note: "select a flying drone in the aircraft dropdown",
    };
  }
  const school = state.school;
  if (school === null) {
    return { view: null, goalName: "", delta: null, key: "", note: "" };
  }
  const moment = school.moment;
  return {
    view: school.aircraftStateView,
    goalName: moment.goalName,
    delta: school.instructorDelta,
    key: `school|${moment.lessonName}|${moment.attempt}`,
    note: moment.finished
      ? `the result holds \u2014 the next attempt begins in a moment`
      : "the aircraft is waiting to fly",
  };
}

/** A class label split for display: the text, and the parenthetical
 *  detail that belongs on the grey row below it ("accelerating
 *  descent" / "(gaining speed, losing altitude)"). */
function splitLabel(label: string): { main: string; detail: string } {
  const at = label.indexOf(" (");
  if (at < 0) {
    return { main: label, detail: "" };
  }
  return { main: label.slice(0, at), detail: label.slice(at + 1) };
}

/** The grey detail line (empty when the label has no parenthetical). */
function detailLine(detail: string): string {
  return detail === "" ? "" : `<div class="dom">${detail}</div>`;
}

/** One tensor cell: the fired classification, then the 5 dominant ones
 *  with their % belonging. */
function tensorCell(className: string, share: number,
                    dominant: readonly { label: string; share: number }[]): string {
  const fired = splitLabel(className);
  const firedShare = share > 0.0
    ? ` <span class="dom">${Math.round(share * 100)}%</span>` : "";
  const domLines = dominant.slice(1).map((d) => {
    const entry = splitLabel(d.label);
    return `<div>${entry.main} ${Math.round(d.share * 100)}%</div>` +
           detailLine(entry.detail);
  }).join("");
  return `<td><div class="cls">${fired.main}${firedShare}</div>` +
         detailLine(fired.detail) +
         `<div class="dom">${domLines}</div></td>`;
}

/** The log's header (the requested columns, one per tensor). */
const AI_TABLE_HEAD =
  "<thead><tr><th>tick</th><th>goal</th><th>element</th><th>situation</th>" +
  "<th>LocationTensor</th><th>FlightTensor</th><th>EnergyTensor</th>" +
  "<th>ControlsTensor</th><th>&Delta;</th><th>ServoTensor</th></tr></thead>";

/** One log row for one decision window of the AI: every cell but the
 *  tick (the tick cell is rendered from the row's tick span). */
function aiRowCells(view: TensorStateView, goalName: string,
                    delta: HudDelta | null): string {
  const sensors = view.sensors;
  const servoDom = view.servo.dominant.slice(1).map((d) => {
    const entry = splitLabel(d.label);
    return `<div>${entry.main} ${Math.round(d.share * 100)}%</div>` +
           detailLine(entry.detail);
  }).join("");
  return `<td>${goalName}</td>` +
    `<td>${elementNameOf(view.elementId)}</td>` +
    `<td>${view.situation}</td>` +
    tensorCell(sensors[0].className, sensors[0].share, sensors[0].dominant) +
    tensorCell(sensors[1].className, sensors[1].share, sensors[1].dominant) +
    tensorCell(sensors[2].className, sensors[2].share, sensors[2].dominant) +
    tensorCell(sensors[3].className, sensors[3].share, sensors[3].dominant) +
    `<td>${delta === null ? "" :
      `<div>${delta.currentLine}</div>` +
      `<div class="dom">(${delta.deltaLine})</div>`}</td>` +
    `<td><div class="cls">${view.servo.command}${view.acted ? "" :
      " <span class=\"dom\">(stunned)</span>"}</div>` +
    `<div class="dom">${servoDom}</div></td>`;
}

/** The row's identity: the qualitative state — goal, element, situation,
 *  the fired and dominant LABELS, the servo command. Shares and the
 *  live delta drift every window and never make a new row. */
function rowKeyOf(view: TensorStateView, goalName: string): string {
  const sensorKeys = view.sensors.map((s) =>
    `${s.className}:${s.dominant.map((d) => d.label).join(",")}`).join("|");
  const servoKey =
    `${view.servo.command}:${view.servo.dominant.map((d) => d.label).join(",")}`;
  return `${goalName}#${view.elementId}#${view.situation}` +
         `#${view.acted}#${sensorKeys}#${servoKey}`;
}

/** The tick cell: the current tick, and — once the row has spanned more
 *  than one window — the first tick it held this state, in braces. */
function tickCellOf(row: AiLogRow): string {
  const origin = row.lastTick === row.firstTick ? "" :
    `<br><span class="dom">(${row.firstTick})</span>`;
  return `<td>${row.lastTick}${origin}</td>`;
}

/** Render both tabs (cheap guards: only redraw on real change). */
function renderTabs(): void {
  // TAB 1 — the wiki.
  let wiki: { title: string; goal: string; elements: readonly string[];
              goals: readonly string[] } | null = null;
  if (state.scenario === WAVES_SCENARIO && state.waves !== null) {
    wiki = state.waves.wiki;
  } else if (state.school !== null) {
    wiki = state.school.wiki;
  }
  if (wiki !== null) {
    const key = `${wiki.title}|${wiki.elements.join(">")}`;
    if (key !== tabs.wikiKey) {
      tabs.wikiKey = key;
      els.tabWiki.innerHTML =
        `<h3>${wiki.title}</h3>` +
        `<p>${wiki.goal}</p>` +
        `<p class="chain">elements: ${wiki.elements.join(" \u2192 ")}</p>` +
        `<p>known goals:</p>` +
        `<ul>${wiki.goals.map((g) => {
          const entry = splitLabel(g);
          return `<li>${entry.main}${detailLine(entry.detail)}</li>`;
        }).join("")}</ul>`;
    }
  }
  // TAB 2 — the live AI state log.
  const subject = aiSubject();
  if (subject.key !== tabs.aiKey) {
    tabs.aiKey = subject.key;
    tabs.rows.length = 0;
    tabs.aiLastDecision = -1;
  }
  const view = subject.view;
  if (view !== null && view.decisionCount !== tabs.aiLastDecision) {
    tabs.aiLastDecision = view.decisionCount;
    const key = rowKeyOf(view, subject.goalName);
    const head = tabs.rows[0];
    if (head !== undefined && head.key === key) {
      // the SAME state, held another window: no new row, the span grows
      head.lastTick = view.decisionCount;
      head.cells = aiRowCells(view, subject.goalName, subject.delta);
    } else {
      tabs.rows.unshift({
        firstTick: view.decisionCount,
        lastTick: view.decisionCount,
        key,
        cells: aiRowCells(view, subject.goalName, subject.delta),
      });
      if (tabs.rows.length > AI_LOG_ROWS) {
        tabs.rows.length = AI_LOG_ROWS;
      }
    }
  }
  const bodyHtml = tabs.rows.length > 0
    ? `<table>${AI_TABLE_HEAD}<tbody>` +
      `${tabs.rows.map((r) => `<tr>${tickCellOf(r)}${r.cells}</tr>`).join("")}` +
      `</tbody></table>`
    : `<p class="chain">${subject.note}</p>`;
  const bodyKey = `${tabs.rows.length}|${tabs.aiLastDecision}|${subject.note}|` +
                  `${tabs.rows[0]?.lastTick ?? -1}`;
  if (els.tabAi.dataset.key !== bodyKey) {
    els.tabAi.dataset.key = bodyKey;
    els.tabAi.innerHTML = bodyHtml;
  }
}

function updateBadge(): void {
  const school = state.school;
  if (school === null || state.scenario === WAVES_SCENARIO) {
    els.badge.textContent = "";
    els.badge.className = "";
    return;
  }
  // Live against the frozen reference: every maneuver the student has
  // already repeated is compared as it happens.
  const graded = school.reportRows(state.golden ?? [])
    .filter((r) => r.exact !== null);
  if (graded.length === 0) {
    els.badge.textContent = "";
    els.badge.className = "";
    return;
  }
  const exact = graded.every((r) => r.exact === true);
  els.badge.textContent = exact
    ? `golden: \u2713 bit-exact so far (${graded.length})`
    : "golden: \u2717 mismatch";
  els.badge.className = exact ? "ok" : "bad";
}

// --- the aircraft dropdown ----------------------------------------------------------

/** Refresh the aircraft dropdown from the scene (labels are stable). */
function refreshAircraftDropdown(): void {
  const world = scene();
  if (world === null) {
    els.aircraft.innerHTML = "";
    return;
  }
  const labels = world.aircraft.map((e) => e.label);
  const key = labels.join("|");
  if (els.aircraft.dataset.key !== key) {
    els.aircraft.dataset.key = key;
    els.aircraft.innerHTML = labels
      .map((label) => `<option value="${label}">${label}</option>`)
      .join("");
  }
  const current = world.current;
  if (current !== null && els.aircraft.value !== current.label) {
    els.aircraft.value = current.label;
  }
}

// --- the clock -----------------------------------------------------------------------

let lastTs = 0;
function frame(ts: number): void {
  const wall = Math.min(0.05, (ts - lastTs) / 1000);
  lastTs = ts;
  const speed = Number(els.speed.value);
  if (state.playing) {
    state.acc += wall * speed;
    while (state.acc >= DT) {
      if (state.scenario === WAVES_SCENARIO) {
        state.waves?.step();
      } else {
        state.school?.step();
      }
      state.acc -= DT;
    }
  }
  autoSelect();
  layoutSidebar();
  layoutBottom();
  drawSky();
  drawPanel();
  renderNarration();
  renderTabs();
  refreshAircraftDropdown();
  refreshControlsDropdown();
  requestAnimationFrame(frame);
}

// --- scenario plumbing ----------------------------------------------------------------

/** Switch the running scenario. The classroom's ONE student and the
 *  waves each keep living across switches — resuming is not restarting.
 *  The chosen maneuver begins: the instructor demonstrates and the
 *  student launches 10 s behind it. */
function setScenario(id: string): void {
  state.scenario = id;
  els.scenario.value = id;
  state.pinned = false;
  state.reportKey = "";
  state.camera.tracking = true;   // a fresh world: the view follows the action
  if (id === WAVES_SCENARIO) {
    if (state.waves === null && state.seed !== null) {
      state.waves = new WavesShow(state.seed, state.brain ?? undefined);
    }
  } else {
    state.school?.switchToManeuver(id);     // the SAME student, into the chosen lesson
  }
  state.acc = 0.0;
  state.playing = true;
  els.play.textContent = "\u23f8 pause";
}

/** The developer's fast path: ?boot=skip — no offline pre-learning,
 *  a fresh untrained brain, straight into the live world (the classroom's
 *  student and every newborn drone fly on beginner's luck). */
const BOOT_SKIP =
  new URLSearchParams(location.search).get("boot") === "skip";

/** Skip mode's world: a new classroom around a beginner's brain — the
 *  same world shape bootTraining lands, minus the training. */
function startUntrainedWorld(): void {
  const curriculum = new FlightSchool(new FlightInstruments(TRIM_SPEED));
  state.brain = null;
  state.golden = [];
  state.school = new FlightSchoolShow(curriculum, new ServoPolicy());
  populateScenarioDropdown([...curriculum.lessons.keys()]);
  els.boot.classList.add("hidden");
  setScenario(els.scenario.value || "takeoff");
  if (!frameRunning) {
    frameRunning = true;
    requestAnimationFrame((ts) => {
      lastTs = ts;
      requestAnimationFrame(frame);
    });
  }
}

/** The restart button: start the CURRENT SCENARIO over — and only
 *  it. The student's trained brain, the golden reference, and the
 *  waves all stay alive; the chosen maneuver (or the drone waves)
 *  simply begins again from its start. A full world reset (new brain,
 *  new boot) is the scenario dropdown's job, not this button's. */
function restart(): void {
  state.reportKey = "";
  state.camera.tracking = true;
  if (state.scenario === WAVES_SCENARIO) {
    if (state.seed !== null) {
      state.waves = new WavesShow(state.seed, state.brain ?? undefined);
    }
    return;
  }
  if (state.school !== null && state.scenario !== "") {
    state.school.switchToManeuver(state.scenario);
  }
}

function populateScenarioDropdown(lessons: readonly string[]): void {
  const options = lessons.map((name) =>
    `<option value="${name}">lesson: ${name.replace(/_/g, " ")}</option>`);
  options.push(`<option value="${WAVES_SCENARIO}">drone attack waves</option>`);
  els.scenario.innerHTML = options.join("");
}

// --- controls -------------------------------------------------------------------------

els.play.addEventListener("click", () => {
  state.playing = !state.playing;
  els.play.textContent = state.playing ? "\u23f8 pause" : "\u25b6 play";
});
els.restart.addEventListener("click", restart);
els.next.addEventListener("click", () => {
  if (state.scenario !== WAVES_SCENARIO && state.school !== null) {
    state.school.skipToNextManeuver();
  }
});
els.scenario.addEventListener("change", () => setScenario(els.scenario.value));
els.aircraft.addEventListener("change", () => {
  // The panel follows the chosen aircraft from here on (pin).
  const world = scene();
  if (world !== null) {
    world.selectLabel(els.aircraft.value);
    state.pinned = true;
  }
});

// --- the controls dropdown ----------------------------------------------------------

/** Whose hands fly the CURRENT aircraft — the sticky per-aircraft
 *  choice (AI, instructor, or a both-mode). Switching the aircraft
 *  dropdown leaves every other machine flying under whoever was
 *  chosen for it; the dropdown shows the current one's own mode. */
els.controls.addEventListener("change", () => {
  const mode = els.controls.value as ControlMode;
  if (state.scenario === WAVES_SCENARIO && state.waves !== null) {
    state.waves.setControlsMode(mode);
    return;
  }
  state.school?.setControlsMode(mode);
});

/** Keep the controls dropdown honest: it always shows the CURRENT
 *  aircraft's own sticky mode. */
function refreshControlsDropdown(): void {
  if (state.scenario === WAVES_SCENARIO && state.waves !== null) {
    const mode = state.waves.selectedControlsMode;
    if (mode !== null && els.controls.value !== mode) {
      els.controls.value = mode;
    }
    return;
  }
  const school = state.school;
  if (school !== null && els.controls.value !== school.controlsMode) {
    els.controls.value = school.controlsMode;
  }
}
els.sky.addEventListener("click", (event) => {
  // Clicking an aircraft on the sky selects it: it becomes the current
  // aircraft (the panel follows it) and the dropdown follows.
  const world = scene();
  if (world === null) {
    return;
  }
  const rect = els.sky.getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  const focus = followTarget() ?? world.current;
  if (focus === null) {
    return;
  }
  const W = els.sky.clientWidth;
  const H = els.sky.clientHeight;
  const scale = W / state.span;
  // hit-test against the drawn view (the camera, wherever it now looks)
  const cx = state.camera.x;
  const cy = state.camera.y;  let best: SceneAircraft | null = null;
  let bestDist = 16;                       // pick radius, px
  for (const entry of world.aircraft) {
    const p = entry.aircraft.body.position;
    const sxp = W / 2 + (p.y - cy) * scale;
    const syp = H / 2 - (p.x - cx) * scale;
    const d = Math.hypot(sxp - px, syp - py);
    if (d < bestDist) {
      bestDist = d;
      best = entry;
    }
  }
  if (best !== null) {
    world.select(best);
    state.pinned = true;
    els.aircraft.value = best.label;
  }
});
els.sky.addEventListener("wheel", (event) => {
  // Two-finger trackpad drag = 2D wheel: pan the view and TURN OFF the
  // automatic tracking right there. The view stays where you leave it
  // until "reset view". Pinch (ctrl+wheel) is not a pan; it is swallowed
  // so the page never zooms out from under the canvas.
  event.preventDefault();
  if (event.ctrlKey) {
    return;
  }
  const scale = els.sky.clientWidth / state.span;    // px per metre
  state.camera.tracking = false;
  state.camera.x -= event.deltaY / scale;
  state.camera.y += event.deltaX / scale;
}, { passive: false });
els.zoomIn.addEventListener("click", () => {
  state.span = Math.max(60.0, state.span / 1.3);
});els.zoomOut.addEventListener("click", () => {
  state.span = Math.min(4000.0, state.span * 1.3);
});

// --- the resize grips --------------------------------------------------------
// The bottom panel's height and the sidebar's width are the user's, by
// drag. The LAWS (fractions of the stage height / the window width):
//   BOTTOM: visible while >= 10%; drag it lower and it HIDES; from
//     hidden, a pull upwards of more than 10% opens it (it then tracks
//     the pointer, so it opens exactly as tall as you pulled).
//   SIDEBAR: visible while >= 10%; drag it narrower and it HIDES; from
//     hidden, a pull leftwards of more than 10% opens it back at its
//     LAST width (then it tracks the pointer from there).
//   MAP: the mirror law on the other side of the same grip — drag the
//     handle leftwards and the sidebar grows over the map; when the
//     map's width falls under 10% of the screen it HIDES (the sidebar
//     takes the whole width); from hidden, a pull rightwards of more
//     than 10% reopens it, tracking the pointer from its own floor.
const BOTTOM_HIDE = 0.10;
const BOTTOM_MAX = 0.85;
const SIDE_HIDE = 0.10;
const REOPEN = 0.10;
const GRIP_PX = 7.0;          // the grip's own track, px

let bottomFrac = 0.30;       // the bottom panel's height, of the stage
let bottomHidden = false;
let sideHidden = false;
let sideLastW = 0;           // the last width the sidebar showed
let mapHidden = false;

/** The stage's height: the app below the toolbar (the grips' tracks
 *  themselves are excluded — they sit between the regions). */
function stageHeight(): number {
  return els.app.clientHeight - els.bar.offsetHeight - 7.0;
}

/** Apply the bottom panel's height law to the DOM (guarded: writing
 *  the same height twice is free). */
let bottomPx = -1;
function layoutBottom(): void {
  const avail = stageHeight();
  const px = bottomHidden ? 0 : Math.round(bottomFrac * avail);
  if (px !== bottomPx) {
    bottomPx = px;
    els.results.style.height = `${px}px`;
  }
}

/** Apply the sidebar's width law to the DOM. */
function applySide(): void {
  els.app.classList.toggle("map-hidden", mapHidden);
  els.cockpit.classList.toggle("collapsed", sideHidden);
  if (sideWidth !== null) {
    const px = sideHidden ? 0 : Math.max(0, Math.round(sideWidth));
    els.cockpit.style.width = `${px}px`;
  }
  layoutSidebar();
}

/** Wire one grip: pointer-captured drags call back with (delta, start),
 *  where delta is the drag distance in the grip's GROW direction
 *  (up for the bottom panel, left for the sidebar). */
function gripDrag<T>(grip: HTMLElement, axis: "x" | "y",
                     startState: () => T,
                     onMove: (delta: number, start: T) => void): void {
  grip.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    grip.setPointerCapture(event.pointerId);
    grip.classList.add("dragging");
    const startPt = axis === "y" ? event.clientY : event.clientX;
    const start = startState();
    const move = (ev: PointerEvent): void => {
      const delta = axis === "y" ? startPt - ev.clientY : startPt - ev.clientX;
      onMove(delta, start);
    };
    const up = (): void => {
      grip.classList.remove("dragging");
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", up);
      grip.removeEventListener("pointercancel", up);
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
    grip.addEventListener("pointercancel", up);
  });
}

gripDrag(els.bottomGrip, "y",
  () => ({ px: bottomHidden ? 0.0 : bottomFrac * stageHeight() }),
  (dy, start): void => {
    const avail = stageHeight();
    const cand = Math.max(0.0, Math.min(avail * BOTTOM_MAX, start.px + dy));
    bottomFrac = cand / avail;
    bottomHidden = cand < avail * BOTTOM_HIDE;   // under 10%: it hides
    layoutBottom();
  });

gripDrag(els.sideGrip, "x",
  () => ({
    hidden: sideHidden,
    mapHidden,
    w: sideWidth ?? els.cockpit.offsetWidth,
    last: sideLastW > 0 ? sideLastW : els.cockpit.offsetWidth,
  }),
  (dx, start): void => {
    const sw = window.innerWidth;
    if (start.mapHidden) {
      // the map is hidden: a rightward pull of more than REOPEN
      // reopens it, tracking the pointer — the map opens exactly as
      // wide as the pull past the threshold, but never past the
      // sidebar's own 10% floor (the mirror of the sidebar's law)
      const pull = -dx;
      if (pull > sw * REOPEN) {
        const mapW = Math.min(
          Math.max(pull - sw * REOPEN, sw * SIDE_HIDE),
          (sw - GRIP_PX) - sw * SIDE_HIDE);
        mapHidden = false;
        sideHidden = false;
        sideWidth = (sw - GRIP_PX) - mapW;
        sideLastW = sideWidth;
      }
      applySide();
      return;
    }
    // from hidden the pull must cross REOPEN first; from visible the
    // width just follows the pointer
    const cand = start.hidden
      ? dx <= sw * REOPEN ? 0.0 : start.last + (dx - sw * REOPEN)
      : start.w + dx;
    if (cand >= sw * SIDE_HIDE) {
      sideHidden = false;
      // THE MAP'S LAW: the sidebar may grow until the map itself
      // would fall under 10% of the screen — there the map SNAPS
      // away (like the bottom panel at its own floor) and the
      // sidebar takes the whole width
      const mapFloor = (sw - GRIP_PX) - sw * SIDE_HIDE;
      if (cand > mapFloor) {
        mapHidden = true;
        sideWidth = sw - GRIP_PX;
      } else {
        mapHidden = false;
        sideWidth = Math.min(cand, mapFloor);
      }
      sideLastW = sideWidth;
    } else {
      sideHidden = true;                        // under 10%: it hides
      mapHidden = false;
    }
    applySide();
  });

layoutBottom();   // the initial 30%
els.zoomReset.addEventListener("click", () => {
  state.span = 260.0;
  state.camera.tracking = true;   // the view returns to the action
});
window.addEventListener("resize", () => {
  // the window changed: the panels re-fit, the grips' fractions re-apply
  layoutSidebar();
  layoutBottom();
  if (sideWidth !== null) {
    // the map law re-derives: the sidebar may own the whole width
    const sw = window.innerWidth;
    const mapFloor = (sw - GRIP_PX) - sw * SIDE_HIDE;
    sideWidth = Math.min(sideWidth, sw - GRIP_PX);
    mapHidden = sideWidth > mapFloor;
    applySide();
  }
});
window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    els.play.click();
  } else if (event.key === "r" || event.key === "R") {
    els.restart.click();
  } else if (event.key === "n" || event.key === "N") {
    els.next.click();
  }
});

// --- boot ------------------------------------------------------------------------------

async function fetchJson(url: string): Promise<any> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }
  return response.json();
}

/** The boot terminal's lesson table — one row per lesson, live:
 *  attempts, the EMA score, and the pass mark. Rendered after each chunk. */
function renderBootTable(rows: ReadonlyArray<{
  lesson: string; attempts: number; score: number | null;
  status: "training" | "done";
}>): void {
  const width = 26;
  const lines = rows.map((row) => {
    const name = row.lesson.padEnd(width).slice(0, width - 1);
    const score = row.score === null ? "     \u2013"
      : `${row.score.toFixed(2)}`.padStart(5) + "/6";
    const mark = row.status === "done" ? " done" : " \u00b7\u00b7\u00b7";
    return `${name}${String(row.attempts).padStart(2)}  ${score} ${mark}`;
  });
  els.bootLog.textContent =
    ["lesson".padEnd(width) + "demo   ema",
     "-".repeat(width + 17), ...lines].join("\n");
  // shrunk mode: keep the LAST line in view — the table scrolls to the
  // bottom (where the classroom currently is)
  if (!els.bootExpand.closest(".boot-dialog")?.classList.contains("expanded")) {
    els.bootLog.scrollTop = els.bootLog.scrollHeight;
  }
}

/** The boot dialog's expand button: the terminal grows to the full
 *  dialog, replacing the cubes; pressing it again brings them back. */
els.bootExpand.addEventListener("click", () => {
  const dialog = els.bootExpand.closest(".boot-dialog")!;
  const expanded = dialog.classList.toggle("expanded");
  els.bootExpand.textContent = expanded ? "\u29c9" : "\u29ca";
});

let frameRunning = false;

/**
 * THE OFFLINE PRE-LEARNING: the flight school trains the tensors before
 * the main scene ever appears — the boot dialog's terminal streams the
 * training and the bar fills left to right. When the classroom
 * completes, the dialog vanishes and the TRAINED BRAIN is transplanted
 * into the live world: the classroom's student AND every newborn drone.
 */
async function bootTraining(): Promise<void> {
  els.bootLog.textContent = "skynerd genesis \u2014 offline pre-learning\n";
  els.bootFill.style.width = "0%";
  const trainer = new BootTrainer();
  // THE DONE BUTTON: the classroom is ENDLESS — once every lesson has
  // had its repetitions the schedule deals another cycle and keeps
  // sharpening the tensors. Only the person's press stops it; the
  // graduate is whatever the brain has become by then.
  els.bootDone.disabled = false;
  els.bootDone.onclick = () => { trainer.finishNow(); };
  let done = false;
  const t0 = performance.now();
  while (!done) {
    // Chunked: the browser keeps painting the dialog between chunks —
    // the cubes rotate and the synapses pulse while the tensors fill.
    await new Promise((resolve) => setTimeout(resolve, 0));
    const progress = trainer.runChunk(14.0);
    renderBootTable(progress.rows);
    els.bootFill.style.width = `${(progress.fraction * 100.0).toFixed(1)}%`;
    drawBrainView(els.bootBrain, (performance.now() - t0) / 1000.0,
                  tensorCounts(trainer.policy));
    done = progress.done;
  }
  // THE BRAIN TRANSPLANT: separate copies of the trained tensors for the
  // classroom's student and for the swarm's newborns.
  state.brain = trainer.graduate;
  state.golden = trainer.golden();
  state.school = new FlightSchoolShow(trainer.school, trainer.graduate);
  populateScenarioDropdown([...trainer.school.lessons.keys()]);
  els.boot.classList.add("hidden");
  setScenario(els.scenario.value || "takeoff");
  if (!frameRunning) {
    frameRunning = true;
    requestAnimationFrame((ts) => {
      lastTs = ts;
      requestAnimationFrame(frame);
    });
  }
}

async function boot(): Promise<void> {
  const rngRef = await fetchJson("/reference/rng-ref.json");
  // The waves' master dice: the reference seed stream (seed 42).
  const seed42 = rngRef.seeds.find((s: { seed: number }) => s.seed === 42);
  state.seed = seed42 ?? rngRef.seeds[0];
  if (BOOT_SKIP) {
    startUntrainedWorld();
    return;
  }
  await bootTraining();
}

boot().catch((error) => {
  els.status.textContent = `boot failed: ${error}`;
});
