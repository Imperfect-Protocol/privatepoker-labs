/**
 * Skynerd Genesis web app: the swarm's world, live.
 *
 * Two views of the same ported simulator core:
 * - FLIGHT SCHOOL: the classroom run live, one 50 Hz tick at a time — the
 *   instructor demonstrates each lesson, the student repeats it, the
 *   cockpit shows the flying drone's own instruments and controls, and the
 *   report card at the end is graded against the TS-frozen C172 reference
 *   (reference/c172-frozen.json).
 * - CARPET WAVES: the generations of the swarm (currently played back from
 *   a bit-exact recording; the live carpet rebuild is the next milestone),
 *   compared against the TS-frozen waves reference
 *   (reference/c172-waves-frozen.json).
 *
 * The clock is a fixed-dt accumulator: wall time x speed becomes sim ticks
 * at exactly 1/50 s each — physics never sees a variable dt.
 */
import { runTracedWaves, type WaveTrace, type SeedRef } from "./timeline.js";
import { mphOf, vsHundredsOf, altFtOf, rpmOf } from "./units.js";
import {
  drawCockpitView, drawControlsView, type DroneView,
} from "./cockpit.js";
import {
  FlightSchoolShow,
} from "./school-show.js";

interface SeedEntry extends SeedRef { seed: number }
interface GoldenWave { wave: number; survivals: number[]; deaths: number; flock_size: number }
interface GoldenLesson { lesson: string; score: number; stamps: number }

type Mode = "school" | "carpet";

const DT = 0.02;
const WORLD = { minX: -880, maxX: 880, minY: -270, maxY: 270 };

const els = {
  sky: document.getElementById("sky") as HTMLCanvasElement,
  chart: document.getElementById("chart") as HTMLCanvasElement,
  status: document.getElementById("status")!,
  narration: document.getElementById("narration")!,
  play: document.getElementById("play") as HTMLButtonElement,
  speed: document.getElementById("speed") as HTMLSelectElement,
  mode: document.getElementById("mode") as HTMLSelectElement,
  seed: document.getElementById("seed") as HTMLSelectElement,
  restart: document.getElementById("restart") as HTMLButtonElement,
  zoomIn: document.getElementById("zoomIn") as HTMLButtonElement,
  zoomOut: document.getElementById("zoomOut") as HTMLButtonElement,
  zoomReset: document.getElementById("zoomReset") as HTMLButtonElement,
  badge: document.getElementById("badge")!,
  tbody: document.querySelector("#waves tbody") as HTMLElement,
  schoolTbody: document.querySelector("#school tbody") as HTMLElement,
  cockpit: document.getElementById("cockpit") as HTMLElement,
  cockpitTitle: document.getElementById("cockpit-title")!,
  cockpitFoot: document.getElementById("cockpit-foot")!,
  cockpitClose: document.getElementById("cockpit-close") as HTMLButtonElement,
  cockpitCanvas: document.getElementById("cockpitCanvas") as HTMLCanvasElement,
  controlsCanvas: document.getElementById("controlsCanvas") as HTMLCanvasElement,
};

const state = {
  mode: "school" as Mode,
  // --- carpet (playback of the bit-exact recording, until the live rebuild)
  seeds: [] as SeedEntry[],
  golden: null as null | { perWave: GoldenWave[]; lessons: GoldenLesson[] },
  traces: [] as WaveTrace[],
  births: 0,
  obituaries: 0,
  waveIdx: 0,
  t: 0.0,
  done: false,
  selected: null as { slot: number } | null,
  // --- school (live)
  show: null as FlightSchoolShow | null,
  acc: 0.0,
  playing: true,
  reportKey: "",
  /** The school top-view zoom: metres spanned across the canvas. */
  span: 260.0,
};

function waveDuration(wave: WaveTrace): number {
  return wave.survivals.reduce((a, b) => Math.max(a, b), 0.0);
}

// --- carpet: recording + golden comparison -------------------------------------

function record(seed: SeedEntry): void {
  const run = runTracedWaves(seed, 5);
  state.traces = run.traces;
  state.births = run.flock.births;
  state.obituaries = run.flock.obituariesHeard;
  state.waveIdx = 0;
  state.t = 0.0;
  state.done = false;
  setSelected(null);
  els.tbody.innerHTML = "";
  updateBadge();
}

function goldenFor(waveIdx: number): GoldenWave | null {
  if (!state.golden || Number(els.seed.value) !== 42) {
    return null;
  }
  return state.golden.perWave[waveIdx] ?? null;
}

function waveMatchesGolden(waveIdx: number): boolean | null {
  const golden = goldenFor(waveIdx);
  if (!golden) {
    return null;
  }
  const trace = state.traces[waveIdx];
  return JSON.stringify(trace.survivals) === JSON.stringify(golden.survivals)
    && trace.deaths === golden.deaths
    && trace.flockSize === golden.flock_size;
}

function updateBadge(): void {
  if (state.mode === "school") {
    const show = state.show;
    if (show === null || !show.done) {
      els.badge.textContent = "";
      els.badge.className = "";
      return;
    }
    const rows = show.reportRows(state.golden?.lessons ?? []);
    const exact = rows.length > 0 && rows.every((r) => r.exact === true);
    els.badge.textContent = exact
      ? "golden: \u2713 bit-exact classroom"
      : "golden: \u2717 mismatch";
    els.badge.className = exact ? "ok" : "bad";
    return;
  }
  const results = state.traces.map((_, i) => waveMatchesGolden(i));
  if (Number(els.seed.value) !== 42) {
    els.badge.textContent = "golden: n/a (seed 42 only)";
    els.badge.className = "";
    return;
  }
  if (results.some((r) => r === false)) {
    els.badge.textContent = "golden: \u2717 MISMATCH";
    els.badge.className = "bad";
  } else if (results.every((r) => r === true)) {
    els.badge.textContent = "golden: \u2713 bit-exact (5/5 waves)";
    els.badge.className = "ok";
  } else {
    els.badge.textContent = "golden: \u2713 so far";
    els.badge.className = "ok";
  }
}

// --- shared canvas helpers ------------------------------------------------------

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

// --- school view (live) -----------------------------------------------------------

function drawSchoolSky(): void {
  const ctx = fitCanvas(els.sky);
  const W = els.sky.clientWidth;
  const H = els.sky.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const show = state.show;
  const view = show?.view ?? null;
  if (show === null || view === null) {
    return;
  }
  // follow camera: the classroom aircraft stays centred. FROM-SKY top view:
  // north (+X) points up the screen, east (+Y) points right.
  const span = state.span;                    // metres across the canvas (zoom)
  const scale = W / span;
  const halfY = H / 2 / scale;
  const sx = (y: number) => W / 2 + (y - view.y) * scale;
  const sy = (x: number) => H / 2 - (x - view.x) * scale;

  // world grid, fixed to the ground
  ctx.strokeStyle = "#161c2a";
  ctx.lineWidth = 1;
  for (let gy = Math.floor((view.y - span / 2) / 50) * 50;
       gy <= view.y + span / 2; gy += 50) {
    ctx.beginPath();
    ctx.moveTo(sx(gy), 0);
    ctx.lineTo(sx(gy), H);
    ctx.stroke();
  }
  for (let gx = Math.floor((view.x - halfY) / 50) * 50;
       gx <= view.x + halfY; gx += 50) {
    ctx.beginPath();
    ctx.moveTo(0, sy(gx));
    ctx.lineTo(W, sy(gx));
    ctx.stroke();
  }

  // the lesson's demonstration path, ghosted behind the repeat
  const ghost = show.ghost;
  if (ghost !== null && ghost.length > 1) {
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

  // this flight's trail, altitude-tinted
  ctx.lineWidth = 1.8;
  for (let i = 1; i < show.trail.length; i++) {
    const a = show.trail[i - 1];
    const b = show.trail[i];
    ctx.globalAlpha = 0.15 + 0.55 * (i / show.trail.length);
    ctx.strokeStyle = altitudeTint(b.z);
    ctx.beginPath();
    ctx.moveTo(sx(a.y), sy(a.x));
    ctx.lineTo(sx(b.y), sy(b.x));
    ctx.stroke();
  }
  ctx.globalAlpha = 1.0;

  // the aircraft
  const heading = Math.atan2(view.fwdY, view.fwdX);
  drawDroneTriangle(ctx, sx(view.y), sy(view.x), heading, 10,
                    view.alive ? altitudeTint(view.z) : "#ff4a3a");
  if (!view.alive) {
    ctx.strokeStyle = "#ff4a3a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx(view.y) - 6, sy(view.x) - 6);
    ctx.lineTo(sx(view.y) + 6, sy(view.x) + 6);
    ctx.stroke();
  }
}

function drawSchoolCockpit(): void {
  const show = state.show;
  const view = show?.view ?? null;
  if (show === null || view === null) {
    return;
  }
  const moment = show.moment;
  const phrase = moment.lessonName.replace(/_/g, " ");
  const title = `FLIGHT SCHOOL \u00b7 LESSON ${moment.lessonIndex}/${moment.lessonCount} \u00b7 ${phrase.toUpperCase()}`;
  const foot = view.alive
    ? `t ${moment.seconds.toFixed(1)}s \u00b7 ${altFtOf(view.z).toFixed(0)} ft \u00b7 ` +
      `${view.vs >= 0 ? "+" : ""}${vsHundredsOf(view.vs).toFixed(1)}\u00d7100fpm \u00b7 ` +
      `${mphOf(view.speed).toFixed(0)} mph \u00b7 ${rpmOf(view.throttle).toFixed(0)} rpm`
    : "crashed \u2014 the lesson ends on the ground";
  drawCockpitView(els.cockpitCanvas, title, view, [], foot,
                  els.cockpitTitle, els.cockpitFoot);
  drawControlsView(els.controlsCanvas, view);
}

function renderSchool(): void {
  const show = state.show;
  if (show === null) {
    return;
  }
  const moment = show.moment;
  drawSchoolSky();
  drawSchoolCockpit();

  els.narration.textContent = moment.narration;
  els.narration.className = `narration stage-${moment.stage}`
    + (moment.takeover ? " takeover" : "");

  const round = moment.round > 0 ? ` \u00b7 drill round ${moment.round}` : "";
  const score = show.session.scores.get(moment.lessonName);
  const scoreText = score === undefined ? "" : ` \u00b7 last score ${score.toFixed(2)}`;
  if (show.done) {
    const failed = show.session.failed;
    els.status.textContent = failed.length === 0
      ? `class graduates after ${show.session.rounds} drill round(s) \u2014 ` +
        `graduate memory ${show.graduateKeys ?? "?"} keys`
      : `${failed.length} lesson(s) below the pass mark after ` +
        `${show.session.rounds} rounds (${failed.join(", ")}) \u2014 ` +
        `graduate memory ${show.graduateKeys ?? "?"} keys`;
  } else {
    els.status.textContent =
      `lesson ${moment.lessonIndex}/${moment.lessonCount} \u00b7 ${moment.lessonName}` +
      `${round}${scoreText} \u00b7 t ${moment.seconds.toFixed(1)}s \u00b7 ` +
      `${moment.stage === "demo" ? "instructor at the controls"
        : moment.takeover ? "instructor has the controls"
        : "student at the controls"}`;
  }
  refreshSchoolTable();
}

function refreshSchoolTable(): void {
  const show = state.show;
  if (show === null) {
    return;
  }
  const moment = show.moment;
  const key = `${show.done}|${moment.lessonIndex}|${moment.lessonName}|` +
    `${moment.round}|${show.session.scores.size}|${show.session.counts.size}`;
  if (key === state.reportKey) {
    return;
  }
  state.reportKey = key;
  const rows = show.reportRows(state.golden?.lessons ?? []);
  const body = rows.map((row) => {
    const score = row.score === null ? "\u2026"
      : `${row.score.toFixed(2)} (${row.grade ?? ""})`;
    const mark = row.exact === null
      ? `<span class="na">\u2026</span>`
      : row.exact ? `<span class="ok">\u2713 exact</span>`
        : `<span class="bad">\u2717 mismatch</span>`;
    return `<tr><td>${row.lesson.replace(/_/g, " ")}</td>` +
      `<td>${row.stamps}</td><td>${score}</td>` +
      `<td>${row.goldenScore.toFixed(2)}</td><td>${mark}</td></tr>`;
  });
  els.schoolTbody.innerHTML = body.join("");
}

// --- carpet view (playback of the bit-exact recording) ----------------------------

function drawSky(): void {
  const ctx = fitCanvas(els.sky);
  const W = els.sky.clientWidth;
  const H = els.sky.clientHeight;
  const scale = Math.min(W / (WORLD.maxX - WORLD.minX), H / (WORLD.maxY - WORLD.minY));
  // FROM-SKY top view: north (+X) up, east (+Y) right.
  const sx = (y: number) => W / 2 + y * scale;
  const sy = (x: number) => H / 2 - x * scale;

  ctx.clearRect(0, 0, W, H);

  // grid
  ctx.strokeStyle = "#161c2a";
  ctx.lineWidth = 1;
  for (let gx = -800; gx <= 800; gx += 200) {
    ctx.beginPath();
    ctx.moveTo(sx(gx), 0);
    ctx.lineTo(sx(gx), H);
    ctx.stroke();
  }
  for (let gy = -200; gy <= 200; gy += 100) {
    ctx.beginPath();
    ctx.moveTo(0, sy(gy));
    ctx.lineTo(W, sy(gy));
    ctx.stroke();
  }
  // world boundary
  ctx.strokeStyle = "#5a2626";
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(sx(-800), 0);
  ctx.lineTo(sx(-800), H);
  ctx.moveTo(sx(800), 0);
  ctx.lineTo(sx(800), H);
  ctx.stroke();
  ctx.setLineDash([]);
  // runway / spawn ticks
  ctx.fillStyle = "#3a4258";
  for (let slot = 0; slot < 8; slot++) {
    const y = (slot - 8 / 2.0) * 40.0;
    ctx.fillRect(sx(y) - 8, sy(0) - 1.5, 16, 3);
  }

  if (state.traces.length === 0) {
    return;
  }
  const wave = state.traces[state.waveIdx];
  if (!wave) {
    return;
  }
  const t = state.t;

  for (let slot = 0; slot < wave.episodes.length; slot++) {
    const episode = wave.episodes[slot];
    const idx = Math.min(Math.floor(t / DT), episode.samples.length - 1);
    if (idx < 0) {
      continue;
    }
    // trail
    ctx.lineWidth = 1.6;
    const from = Math.max(0, idx - 45);
    for (let i = from + 1; i <= idx; i++) {
      const a = episode.samples[i - 1];
      const b = episode.samples[i];
      ctx.globalAlpha = 0.12 + 0.5 * ((i - from) / (idx - from + 1));
      ctx.strokeStyle = altitudeTint(b.z);
      ctx.beginPath();
      ctx.moveTo(sx(a.y), sy(a.x));
      ctx.lineTo(sx(b.y), sy(b.x));
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;

    const sample = episode.samples[idx];
    const px = sx(sample.y);
    const py = sy(sample.x);
    if (t < episode.survival) {
      const heading = Math.atan2(sample.fwdY, sample.fwdX);
      const size = 5 + sample.z * 0.045;
      drawDroneTriangle(ctx, px, py, heading, size, altitudeTint(sample.z));
    } else {
      // death marker
      const dx = sx(episode.deathY);
      const dy = sy(episode.deathX);
      ctx.strokeStyle = "#ff4a3a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(dx - 5, dy - 5);
      ctx.lineTo(dx + 5, dy + 5);
      ctx.moveTo(dx + 5, dy - 5);
      ctx.lineTo(dx - 5, dy + 5);
      ctx.stroke();
      const elapsed = t - episode.survival;
      if (elapsed < 0.7) {
        const p = elapsed / 0.7;
        ctx.globalAlpha = 1.0 - p;
        ctx.beginPath();
        ctx.arc(dx, dy, 6 + 34 * p, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    }
    // selection ring on the chosen drone (alive or crash site)
    if (state.selected !== null && state.selected.slot === slot) {
      ctx.strokeStyle = "#ffb84a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 12, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
}

function drawChart(): void {
  const ctx = fitCanvas(els.chart);
  const W = els.chart.clientWidth;
  const H = els.chart.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const maxBar = 20.0;
  const n = state.traces.length || 1;
  const slot = W / n;
  for (let i = 0; i < state.traces.length; i++) {
    const trace = state.traces[i];
    const barH = (trace.meanSurvival / maxBar) * (H - 16);
    const x = i * slot + slot * 0.22;
    const w = slot * 0.56;
    ctx.fillStyle = i <= state.waveIdx ? "#ff8c3a" : "#3a3327";
    ctx.fillRect(x, H - barH - 4, w, barH);
    const golden = goldenFor(i);
    if (golden) {
      const gMean = golden.survivals.reduce((a, b) => a + b, 0.0) / golden.survivals.length;
      const gy = H - (gMean / maxBar) * (H - 16) - 4;
      ctx.strokeStyle = "#58d47a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 2, gy);
      ctx.lineTo(x + w + 2, gy);
      ctx.stroke();
    }
    ctx.fillStyle = "#5b6878";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`w${i}`, x + w / 2 - 8, H - barH - 8 < 12 ? H - 2 : H - barH - 8);
  }
}

function updateHud(): void {
  if (state.traces.length === 0) {
    els.status.textContent = "no waves recorded";
    return;
  }
  const wave = state.traces[state.waveIdx];
  if (!wave) {
    els.status.textContent = "no waves recorded";
    return;
  }
  const totalDeaths = state.traces.slice(0, state.waveIdx + 1)
    .reduce((a, w) => a + w.deaths, 0);
  if (state.done) {
    els.status.textContent =
      `all ${state.traces.length} waves done — ${totalDeaths} deaths, ` +
      `flock ${wave.flockSize} survivor(s), ${state.births} births, ` +
      `${state.obituaries} obituaries heard — restart (R) to fly again`;
  } else {
    els.status.textContent =
      `wave ${state.waveIdx + 1}/${state.traces.length} · t ${state.t.toFixed(1)}s · ` +
      `deaths ${totalDeaths}/${state.births} born · ` +
      `flock ${wave.flockSize} · screams heard ${state.obituaries}`;
  }
  refreshTable();
}

function refreshTable(): void {
  const completed = state.done ? state.traces.length : state.waveIdx;
  const rows: string[] = [];
  for (let i = 0; i < completed; i++) {
    const trace = state.traces[i];
    const match = waveMatchesGolden(i);
    const mark = match === null
      ? `<span class="na">n/a</span>`
      : match ? `<span class="ok">✓ exact</span>` : `<span class="bad">✗ mismatch</span>`;
    rows.push(`<tr><td>wave ${i}</td><td>${trace.meanSurvival.toFixed(2)}</td>` +
      `<td>${trace.deaths}/8</td><td>${trace.flockSize}</td><td>${mark}</td></tr>`);
  }
  if (state.traces.length > 0 && !state.done && completed < state.traces.length) {
    rows.push(`<tr><td>wave ${completed} (flying)</td><td>…</td>` +
      `<td>…</td><td>…</td><td><span class="na">…</span></td></tr>`);
  }
  els.tbody.innerHTML = rows.join("");
}

// --- carpet cockpit (view adapter over the recording) -------------------------------

function carpetCockpit(): void {
  if (state.selected === null || els.cockpit.hidden) {
    return;
  }
  const wave = state.traces[state.waveIdx];
  if (!wave) {
    return;
  }
  const slot = state.selected.slot;
  const episode = wave.episodes[slot];
  if (!episode) {
    return;
  }
  const idx = Math.min(Math.floor(state.t / DT), episode.samples.length - 1);
  const s = episode.samples[idx];
  const alive = state.t < episode.survival;
  const view: DroneView = {
    x: s.x, y: s.y, z: s.z,
    fwdX: s.fwdX, fwdY: s.fwdY,
    pitch: s.pitch, bank: s.bank,
    speed: s.speed, vs: s.vs,
    velX: s.velX, velY: s.velY,
    throttle: s.throttle,
    ail: s.ail, elev: s.elev, rud: s.rud,
    alive,
  };
  const others: DroneView[] = [];
  for (let i = 0; i < wave.episodes.length; i++) {
    if (i === slot) {
      continue;
    }
    const other = wave.episodes[i];
    const os = other.samples[Math.min(idx, other.samples.length - 1)];
    others.push({
      x: os.x, y: os.y, z: os.z,
      fwdX: os.fwdX, fwdY: os.fwdY,
      pitch: os.pitch, bank: os.bank,
      speed: os.speed, vs: os.vs,
      velX: os.velX, velY: os.velY,
      throttle: os.throttle,
      ail: os.ail, elev: os.elev, rud: os.rud,
      alive: state.t < other.survival,
    });
  }
  const title = `DRONE ${slot + 1} \u00b7 WAVE ${state.waveIdx + 1}`;
  const foot = alive
    ? `t ${state.t.toFixed(1)}s \u00b7 ${altFtOf(s.z).toFixed(0)} ft \u00b7 ` +
      `${s.vs >= 0 ? "+" : ""}${vsHundredsOf(s.vs).toFixed(1)}\u00d7100fpm \u00b7 ` +
      `${mphOf(s.speed).toFixed(0)} mph \u00b7 ${rpmOf(s.throttle).toFixed(0)} rpm`
    : `crashed at ${episode.survival.toFixed(2)}s \u2014 the shard screamed to the flock`;
  drawCockpitView(els.cockpitCanvas, title, view, others, foot,
                  els.cockpitTitle, els.cockpitFoot);
  drawControlsView(els.controlsCanvas, view);
}

function setSelected(slot: number | null): void {
  state.selected = slot === null ? null : { slot };
  els.cockpit.hidden = state.selected === null;
  if (state.selected !== null) {
    carpetCockpit();
  }
}

/** Nearest drone (alive or crash site) to a canvas point, within pick radius. */
function hitTest(mx: number, my: number): number | null {
  const wave = state.traces[state.waveIdx];
  if (!wave) {
    return null;
  }
  const width = els.sky.clientWidth;
  const height = els.sky.clientHeight;
  const scale = Math.min(width / (WORLD.maxX - WORLD.minX),
                         height / (WORLD.maxY - WORLD.minY));
  const idx = Math.max(0, Math.floor(state.t / DT));
  let best: number | null = null;
  let bestDist = 16;                       // pick radius, px
  for (let i = 0; i < wave.episodes.length; i++) {
    const sample = wave.episodes[i].samples[
      Math.min(idx, wave.episodes[i].samples.length - 1)];
    const px = width / 2 + sample.y * scale;
    const py = height / 2 - sample.x * scale;
    const d = Math.hypot(px - mx, py - my);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

/**
 * The selected drone crashed: hand the cockpit to the next drone still
 * flying (next seat number, wrapping).
 */
function autoAdvanceSelection(): void {
  if (state.selected === null) {
    return;
  }
  const wave = state.traces[state.waveIdx];
  if (!wave) {
    return;
  }
  const current = wave.episodes[state.selected.slot];
  if (current === undefined || state.t < current.survival) {
    return;   // still flying
  }
  const n = wave.episodes.length;
  for (let step = 1; step <= n; step++) {
    const slot = (state.selected.slot + step) % n;
    if (state.t < wave.episodes[slot].survival) {
      setSelected(slot);
      return;
    }
  }
}

// --- mode plumbing -----------------------------------------------------------------

function setMode(mode: Mode): void {
  state.mode = mode;
  els.mode.value = mode;
  document.body.dataset.mode = mode;
  els.seed.disabled = mode === "school";
  els.narration.textContent = mode === "school" ? "" : "carpet waves — the swarm's generations";
  els.narration.className = "narration";
  state.playing = true;
  els.play.textContent = "\u23f8 pause";
  state.reportKey = "";
  els.cockpit.hidden = mode === "carpet";
  els.cockpitClose.hidden = mode === "school";
  if (mode === "school") {
    state.show = FlightSchoolShow.withFreshPolicy();
    state.acc = 0.0;
    updateBadge();
  } else {
    state.show = null;
    const seed = state.seeds.find((s) => s.seed === Number(els.seed.value))
      ?? state.seeds[0];
    record(seed);
  }
}

function restart(): void {
  setMode(state.mode);
}

// --- clock -------------------------------------------------------------------------

let lastTs = 0;
function frame(ts: number): void {
  const wall = Math.min(0.05, (ts - lastTs) / 1000);
  lastTs = ts;
  const speed = Number(els.speed.value);
  if (state.playing) {
    if (state.mode === "school" && state.show !== null) {
      state.acc += wall * speed;
      while (state.acc >= DT && !state.show.done) {
        state.show.step();
        state.acc -= DT;
      }
      if (state.show.done) {
        state.acc = 0.0;
        state.playing = false;
        els.play.textContent = "\u25b6 play";
        updateBadge();
      }
    } else if (state.mode === "carpet" && state.traces.length > 0) {
      state.t += wall * speed;
      const duration = waveDuration(state.traces[state.waveIdx]) + 1.5;
      if (state.t >= duration) {
        if (state.waveIdx + 1 < state.traces.length) {
          state.waveIdx += 1;
          state.t = 0.0;
          setSelected(null);
        } else {
          // Hold the final frame, fully drawn; never index past the last wave.
          state.done = true;
          state.playing = false;
          state.t = duration;
          els.play.textContent = "\u25b6 play";
          updateBadge();
        }
      }
    }
  }
  if (state.mode === "school") {
    renderSchool();
  } else {
    autoAdvanceSelection();
    drawSky();
    carpetCockpit();
    drawChart();
    updateHud();
  }
  requestAnimationFrame(frame);
}

// --- controls ------------------------------------------------------------------

els.play.addEventListener("click", () => {
  if (state.mode === "school" && state.show !== null && state.show.done) {
    return;
  }
  if (state.mode === "carpet" && state.done) {
    return;
  }
  state.playing = !state.playing;
  els.play.textContent = state.playing ? "\u23f8 pause" : "\u25b6 play";
});
els.restart.addEventListener("click", restart);
els.zoomIn.addEventListener("click", () => {
  state.span = Math.max(60.0, state.span / 1.3);
});
els.zoomOut.addEventListener("click", () => {
  state.span = Math.min(4000.0, state.span * 1.3);
});
els.zoomReset.addEventListener("click", () => {
  state.span = 260.0;
});
els.mode.addEventListener("change", () => setMode(els.mode.value as Mode));
els.seed.addEventListener("change", () => {
  if (state.mode === "carpet") {
    const seed = state.seeds.find((s) => s.seed === Number(els.seed.value))!;
    record(seed);
    els.play.textContent = "\u23f8 pause";
  }
});
els.sky.addEventListener("click", (event) => {
  if (state.mode !== "carpet") {
    return;
  }
  const rect = els.sky.getBoundingClientRect();
  setSelected(hitTest(event.clientX - rect.left, event.clientY - rect.top));
});
els.sky.addEventListener("mousemove", (event) => {
  if (state.mode !== "carpet") {
    els.sky.style.cursor = "default";
    return;
  }
  const rect = els.sky.getBoundingClientRect();
  els.sky.style.cursor =
    hitTest(event.clientX - rect.left, event.clientY - rect.top) !== null
      ? "pointer" : "default";
});
els.cockpitClose.addEventListener("click", () => setSelected(null));
window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    els.play.click();
  } else if (event.key === "r" || event.key === "R") {
    els.restart.click();
  }
});

// --- boot ----------------------------------------------------------------------

async function fetchJson(url: string): Promise<any> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }
  return response.json();
}

async function boot(): Promise<void> {
  const [rngRef, c172, wavesFrozen] = await Promise.all([
    fetchJson("/reference/rng-ref.json"),
    fetchJson("/reference/c172-frozen.json").catch(() => null),
    fetchJson("/reference/c172-waves-frozen.json").catch(() => null),
  ]);
  state.seeds = rngRef.seeds;
  els.seed.innerHTML = state.seeds
    .map((s: SeedEntry) => `<option value="${s.seed}"${s.seed === 42 ? " selected" : ""}>${s.seed}</option>`)
    .join("");
  // The TS-frozen C172 references supersede the Python goldens for BOTH
  // views: the flight school's report card (lessons) and the carpet's
  // traced waves (the Python golden-baseline.json records the old 1.5 kg
  // airframe, which the redesign invalidates).
  state.golden = {
    perWave: wavesFrozen?.traced?.per_wave ?? [],
    lessons: c172?.lessons ?? [],
  };
  setMode("school");
  requestAnimationFrame((ts) => {
    lastTs = ts;
    requestAnimationFrame(frame);
  });
}

boot().catch((error) => {
  els.status.textContent = `boot failed: ${error}`;
});
