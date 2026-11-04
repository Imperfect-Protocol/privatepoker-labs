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
 *  - drone attack waves: the swarm as a conveyor — a wave is many
 *    drones in a line, the next wave lifts off right behind the last
 *    at a distance, no waiting; every drone connects to every nearby
 *    drone, and the dying scream their memory to whoever hears.
 *
 * The aircraft dropdown lists the scene's aircraft. Only a page reload
 * or the restart button ever starts anything over.
 *
 * The clock is a fixed-dt accumulator: wall time x speed becomes sim
 * ticks at exactly 1/50 s each — physics never sees a variable dt.
 */
import { mphOf, vsHundredsOf, altFtOf, rpmOf } from "./units.js";
import {
  drawCockpitView, drawControlsView, droneViewOf, type DroneView,
} from "./cockpit.js";
import { FlightSchoolShow } from "./school-show.js";
import { WavesShow } from "./waves-show.js";
import { BootTrainer } from "./boot-train.js";
import { FlightSchool } from "../learning/school.js";
import { FlightInstruments } from "../learning/instruments.js";
import { TRIM_SPEED } from "../learning/flight.js";
import { drawBrainView, tensorCounts } from "./boot-brain.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import type { Scene, SceneAircraft } from "./scene.js";

interface GoldenLesson {
  lesson: string;
  attempts: number;
  score: number;
  goals: { label: string; fraction: number }[];
}

const DT = 0.02;

const els = {
  sky: document.getElementById("sky") as HTMLCanvasElement,
  status: document.getElementById("status")!,
  narration: document.getElementById("narration")!,
  play: document.getElementById("play") as HTMLButtonElement,
  speed: document.getElementById("speed") as HTMLSelectElement,
  scenario: document.getElementById("scenario") as HTMLSelectElement,
  aircraft: document.getElementById("aircraft") as HTMLSelectElement,
  next: document.getElementById("next") as HTMLButtonElement,
  restart: document.getElementById("restart") as HTMLButtonElement,
  zoomIn: document.getElementById("zoomIn") as HTMLButtonElement,
  zoomOut: document.getElementById("zoomOut") as HTMLButtonElement,
  zoomReset: document.getElementById("zoomReset") as HTMLButtonElement,
  badge: document.getElementById("badge")!,
  schoolTbody: document.querySelector("#school tbody") as HTMLElement,
  results: document.getElementById("results") as HTMLElement,
  cockpit: document.getElementById("cockpit") as HTMLElement,
  cockpitTitle: document.getElementById("cockpit-title")!,
  cockpitFoot: document.getElementById("cockpit-foot")!,
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
  return school === null ? null : school.flyingEntry;
}

/** Keep the scene's selection honest: follow the action unless pinned. */
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
function layoutSidebar(): void {
  const aside = els.cockpit;
  const style = getComputedStyle(aside);
  const padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
  const gap = parseFloat(style.rowGap) || 8.0;
  // Everything in the column that is not a panel: its measured height
  // comes off the panels' budget, whatever the markup grows into.
  const panels = [els.cockpitCanvas, els.controlsCanvas];
  const others = [...aside.children].filter(
    (child): child is HTMLElement => !(child instanceof HTMLCanvasElement));
  const used = others.reduce((sum, child) => sum + child.offsetHeight, 0.0);
  const gaps = gap * (others.length + panels.length - 1);
  const avail = aside.clientHeight - padY;
  const panelH = Math.max(120.0, (avail - used - gaps) / 2.0);
  const panelW = Math.round(panelH * 1.5);
  for (const canvas of panels) {
    canvas.style.width = `${panelW}px`;
    canvas.style.height = `${Math.round(panelH)}px`;
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
  const r = Math.round(255 + (87 - 255) * t);
  const g = Math.round(122 + (212 - 122) * t);
  const b = Math.round(74 + (255 - 74) * t);
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
  ctx.strokeStyle = "#161c2a";
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

/** The classroom's sky: the instructor's demonstration ghosted behind
 *  the student's repeat, both machines drawn. */
function drawSchoolSky(ctx: CanvasRenderingContext2D,
                       sx: (y: number) => number,
                       sy: (x: number) => number): void {
  const school = state.school!;
  const instructor = school.instructorEntry;
  const student = school.studentEntry;
  // the demonstration path, ghosted behind the repeat
  const ghost = instructor.trail;
  if (school.moment.stage === "repeat" && ghost.length > 1) {
    ctx.strokeStyle = "rgba(139,150,171,0.30)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 6]);
    ctx.beginPath();
    ctx.moveTo(sx(ghost[0].y), sy(ghost[0].x));
    for (const p of ghost) {
      ctx.lineTo(sx(p.y), sy(p.x));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const entry of [instructor, student]) {
    drawTrail(ctx, entry, sx, sy);
    drawAircraftTop(ctx, entry, sx, sy, 10);
  }
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
  drawCockpitView(els.cockpitCanvas, title, view, [], foot,
                  els.cockpitTitle, els.cockpitFoot);
  drawControlsView(els.controlsCanvas, view);
}

// --- narration + status ------------------------------------------------------------

function renderNarration(): void {
  if (state.scenario === WAVES_SCENARIO && state.waves !== null) {
    const waves = state.waves;
    els.narration.textContent =
      `drone attack waves \u00b7 wave ${waves.waveCount} in the air \u00b7 ` +
      `${waves.aliveCount} flying \u00b7 the dying scream their memory to whoever hears`;
    els.narration.className = "narration stage-waves";
    els.status.textContent =
      `t ${waves.simSeconds.toFixed(1)}s \u00b7 wave ${waves.waveCount} \u00b7 ` +
      `${waves.aliveCount} flying \u00b7 ${waves.births} born \u00b7 ` +
      `${waves.obituaries} screams heard \u00b7 flock ${waves.flockSize}` +
      ` \u00b7 links ${waves.links().length}`;
    els.results.hidden = true;
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
  const intent = ` \u00b7 goal: ${moment.goalName} \u00b7 element: ${moment.elementName}`;
  const finishedText = moment.finished ? " \u00b7 flight finished" : "";
  const whoText = moment.bothFlying
    ? "both flying \u2014 student 10 s behind"
    : moment.stage === "demo"
      ? `student launches in ${moment.countdown.toFixed(0)} s`
      : "student at the controls";
  els.status.textContent =
    `maneuver ${moment.lessonIndex}/${moment.lessonCount} \u00b7 ${moment.lessonName}` +
    `${attempt}${scoreText}${intent}${finishedText} \u00b7 t ${moment.seconds.toFixed(1)}s \u00b7 ` +
    whoText;
  els.results.hidden = false;
  refreshSchoolTable();
}

function refreshSchoolTable(): void {
  const school = state.school;
  if (school === null) {
    return;
  }
  const moment = school.moment;
  const key = `${moment.lessonIndex}|${moment.lessonName}|` +
    `${moment.attempt}|${school.scores.size}|${school.goalReports.size}`;
  if (key === state.reportKey) {
    return;
  }
  state.reportKey = key;
  const rows = school.reportRows(state.golden ?? []);
  const body = rows.map((row) => {
    const score = row.score === null ? "\u2026"
      : `${row.score.toFixed(2)} (${row.grade ?? ""})`;
    const mark = row.exact === null
      ? `<span class="na">\u2026</span>`
      : row.exact ? `<span class="ok">\u2713 exact</span>`
        : `<span class="bad">\u2717 mismatch</span>`;
    return `<tr><td>${row.lesson.replace(/_/g, " ")}</td>` +
      `<td>${row.attempts}</td><td>${score}</td>` +
      `<td class="goals">${row.goals}</td><td>${mark}</td></tr>`;
  });
  els.schoolTbody.innerHTML = body.join("");
  updateBadge();
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
  drawSky();
  drawPanel();
  renderNarration();
  refreshAircraftDropdown();
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

/** The restart button: the one sanctioned fresh start of everything —
 *  the old student dies, the boot pre-learning runs again, and a new
 *  trained student is born (skip mode: just a fresh untrained world). */
function restart(): void {
  state.school = null;
  state.waves = null;
  state.brain = null;
  state.pinned = false;
  state.reportKey = "";
  if (BOOT_SKIP) {
    startUntrainedWorld();
    return;
  }
  els.boot.classList.remove("hidden");
  bootTraining().catch((error) => {
    els.status.textContent = `boot failed: ${error}`;
  });
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
els.zoomReset.addEventListener("click", () => {
  state.span = 260.0;
  state.camera.tracking = true;   // the view returns to the action
});
window.addEventListener("resize", () => {
  // the window changed: the sidebar's panels re-fit their 3:2 sizes
  layoutSidebar();
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
