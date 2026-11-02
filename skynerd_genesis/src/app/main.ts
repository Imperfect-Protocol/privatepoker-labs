/**
 * Skynerd Genesis web app: record carpet waves with the ported simulator and
 * play them back on a 2D canvas. Visualisation is not the world model — the
 * waves are recorded by src/app/timeline.ts (validated bit-exact against the
 * Python golden baseline) and this file only renders and drives the clock.
 */
import { runTracedWaves, type WaveTrace, type SeedRef } from "./timeline.js";
import { mphOf, vsHundredsOf, altFtOf, rpmOf } from "./units.js";

interface SeedEntry extends SeedRef { seed: number }
interface GoldenWave { wave: number; survivals: number[]; deaths: number; flock_size: number }

const DT = 0.02;
const WORLD = { minX: -880, maxX: 880, minY: -270, maxY: 270 };

const els = {
  sky: document.getElementById("sky") as HTMLCanvasElement,
  chart: document.getElementById("chart") as HTMLCanvasElement,
  status: document.getElementById("status")!,
  play: document.getElementById("play") as HTMLButtonElement,
  speed: document.getElementById("speed") as HTMLSelectElement,
  seed: document.getElementById("seed") as HTMLSelectElement,
  restart: document.getElementById("restart") as HTMLButtonElement,
  badge: document.getElementById("badge")!,
  tbody: document.querySelector("#waves tbody") as HTMLElement,
  cockpit: document.getElementById("cockpit") as HTMLElement,
  cockpitTitle: document.getElementById("cockpit-title")!,
  cockpitCanvas: document.getElementById("cockpitCanvas") as HTMLCanvasElement,
  controlsCanvas: document.getElementById("controlsCanvas") as HTMLCanvasElement,
  cockpitFoot: document.getElementById("cockpit-foot")!,
  cockpitClose: document.getElementById("cockpit-close") as HTMLButtonElement,
};

const state = {
  seeds: [] as SeedEntry[],
  golden: null as null | { perWave: GoldenWave[] },
  traces: [] as WaveTrace[],
  births: 0,
  obituaries: 0,
  waveIdx: 0,
  t: 0.0,
  playing: true,
  done: false,
  selected: null as { slot: number } | null,
};

function waveDuration(wave: WaveTrace): number {
  return wave.survivals.reduce((a, b) => Math.max(a, b), 0.0);
}

// --- recording ----------------------------------------------------------------

function record(seed: SeedEntry): void {
  const run = runTracedWaves(seed, 5);
  state.traces = run.traces;
  state.births = run.flock.births;
  state.obituaries = run.flock.obituariesHeard;
  state.waveIdx = 0;
  state.t = 0.0;
  state.playing = true;
  state.done = false;
  setSelected(null);
  els.tbody.innerHTML = "";
  updateBadge();
}

// --- golden comparison ---------------------------------------------------------

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

// --- canvas --------------------------------------------------------------------

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

function drawSky(): void {
  const ctx = fitCanvas(els.sky);
  const W = els.sky.clientWidth;
  const H = els.sky.clientHeight;
  const scale = Math.min(W / (WORLD.maxX - WORLD.minX), H / (WORLD.maxY - WORLD.minY));
  const sx = (x: number) => W / 2 + x * scale;
  const sy = (y: number) => H / 2 - y * scale;

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
    ctx.fillRect(sx(0) - 8, sy(y) - 1.5, 16, 3);
  }

  if (state.traces.length === 0) {
    return;
  }
  const wave = state.traces[state.waveIdx];
  if (!wave) {
    return;   // between waves or before boot: nothing to draw
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
      ctx.moveTo(sx(a.x), sy(a.y));
      ctx.lineTo(sx(b.x), sy(b.y));
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;

    const sample = episode.samples[idx];
    const px = sx(sample.x);
    const py = sy(sample.y);
    if (t < episode.survival) {
      // drone triangle, heading from the body frame forward
      const heading = Math.atan2(sample.fwdY, sample.fwdX);
      const size = 5 + sample.z * 0.045;
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(-heading);
      ctx.fillStyle = altitudeTint(sample.z);
      ctx.beginPath();
      ctx.moveTo(size * 1.6, 0);
      ctx.lineTo(-size, size * 0.8);
      ctx.lineTo(-size * 0.55, 0);
      ctx.lineTo(-size, -size * 0.8);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    } else {
      // death marker
      const dx = sx(episode.deathX);
      const dy = sy(episode.deathY);
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

// --- hud -----------------------------------------------------------------------

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
    const trace = state.traces[completed];
    rows.push(`<tr><td>wave ${completed} (flying)</td><td>…</td>` +
      `<td>…</td><td>…</td><td><span class="na">…</span></td></tr>`);
  }
  els.tbody.innerHTML = rows.join("");
}

// --- cockpit ---------------------------------------------------------------------

const FOV_H = (40 * Math.PI) / 180;
const FOV_V = (22 * Math.PI) / 180;
const wrapPi = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

function setSelected(slot: number | null): void {
  state.selected = slot === null ? null : { slot };
  els.cockpit.hidden = state.selected === null;
  if (state.selected !== null) {
    drawCockpit();
    drawControls();
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
    const px = width / 2 + sample.x * scale;
    const py = height / 2 - sample.y * scale;
    const d = Math.hypot(px - mx, py - my);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

function drawGauge(ctx: CanvasRenderingContext2D, x: number, y: number, r: number,
                   value: number, min: number, max: number, label: string): void {
  const a0 = Math.PI * 0.75;
  const sweep = Math.PI * 1.5;
  ctx.strokeStyle = "#3a4258";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(x, y, r, a0, a0 + sweep);
  ctx.stroke();
  for (let i = 0; i <= 4; i++) {
    const a = a0 + (i / 4) * sweep;
    ctx.beginPath();
    ctx.moveTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
    ctx.lineTo(x + Math.cos(a) * (r - 4), y + Math.sin(a) * (r - 4));
    ctx.stroke();
  }
  const t = Math.max(0.0, Math.min(1.0, (value - min) / (max - min)));
  const a = a0 + t * sweep;
  ctx.strokeStyle = "#ff8c3a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + Math.cos(a) * (r - 6), y + Math.sin(a) * (r - 6));
  ctx.stroke();
  ctx.fillStyle = "#5b6878";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText(label, x, y + r + 10);
  ctx.fillStyle = "#c9d4e3";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(value >= 100 ? value.toFixed(0) : value.toFixed(1), x, y + 3);
  ctx.textAlign = "left";
}

/** The selected drone's cockpit: ADI, traffic, gauges — its view of the world. */
function drawCockpit(): void {
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
  const ctx = fitCanvas(els.cockpitCanvas);
  const W = els.cockpitCanvas.clientWidth;
  const H = els.cockpitCanvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const DEG = 180 / Math.PI;
  const idx = Math.min(Math.floor(state.t / DT), episode.samples.length - 1);
  const s = episode.samples[idx];
  const alive = state.t < episode.survival;
  const heading = Math.atan2(s.fwdY, s.fwdX);

  const cx = W / 2;
  const cy = H * 0.5;
  const R = Math.min(W, H) * 0.33;
  const Ky = R / Math.tan(FOV_V);
  const Kx = R / Math.tan(FOV_H);
  const y0 = Math.tan(s.pitch) * Ky;         // horizon offset: nose up, horizon down

  // the round window: horizon card + traffic, clipped
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, 2 * Math.PI);
  ctx.clip();
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-s.bank);                       // the world tilts opposite the bank

  ctx.fillStyle = "#243356";                 // sky
  ctx.fillRect(-2 * R, y0 - 2 * R, 4 * R, 2 * R);
  ctx.fillStyle = "#4a3320";                 // ground
  ctx.fillRect(-2 * R, y0, 4 * R, 2 * R);

  // pitch ladder
  ctx.strokeStyle = "rgba(226,232,244,0.55)";
  ctx.fillStyle = "rgba(226,232,244,0.75)";
  ctx.font = "8px ui-monospace, monospace";
  ctx.lineWidth = 1;
  for (const d of [-20, -10, 10, 20]) {
    const ly = y0 - Math.tan((d * Math.PI) / 180) * Ky;
    const half = Math.abs(d) === 20 ? 26 : 15;
    ctx.beginPath();
    ctx.moveTo(-half, ly);
    ctx.lineTo(-6, ly);
    ctx.moveTo(6, ly);
    ctx.lineTo(half, ly);
    ctx.stroke();
    ctx.fillText(String(d), half + 4, ly + 3);
  }
  ctx.strokeStyle = "#f2f5fa";               // horizon line
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(-R, y0);
  ctx.lineTo(R, y0);
  ctx.stroke();

  // traffic: wave-mates in the forward view
  for (let i = 0; i < wave.episodes.length; i++) {
    if (i === slot || state.t >= wave.episodes[i].survival) {
      continue;
    }
    const other = wave.episodes[i];
    const os = other.samples[Math.min(idx, other.samples.length - 1)];
    const dx = os.x - s.x, dy = os.y - s.y, dz = os.z - s.z;
    const range = Math.hypot(dx, dy);
    if (range < 1.0) {
      continue;
    }
    const rel = wrapPi(Math.atan2(dy, dx) - heading);
    if (Math.abs(rel) >= FOV_H) {
      continue;
    }
    const elev = Math.atan2(dz, range) - s.pitch;
    if (Math.abs(elev) >= FOV_V) {
      continue;
    }
    const tx = Math.tan(rel) * Kx;
    const ty = y0 + Math.tan(elev) * Ky;
    const size = Math.max(3.0, Math.min(9.0, 260.0 / range));
    ctx.fillStyle = dz > 2 ? "#9fd8ff" : dz < -2 ? "#ffb066" : "#e8ecf4";
    ctx.beginPath();
    ctx.moveTo(tx, ty - size);
    ctx.lineTo(tx + size * 0.7, ty + size * 0.6);
    ctx.lineTo(tx - size * 0.7, ty + size * 0.6);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "rgba(200,210,228,0.7)";
    ctx.font = "8px ui-monospace, monospace";
    ctx.fillText(`${Math.round(range)}m`, tx + size + 2, ty + 3);
  }
  ctx.restore();                             // un-rotate

  // fixed centre symbol (the aircraft)
  ctx.strokeStyle = "#ffb84a";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx - R * 0.6, cy); ctx.lineTo(cx - R * 0.18, cy);
  ctx.moveTo(cx + R * 0.18, cy); ctx.lineTo(cx + R * 0.6, cy);
  ctx.moveTo(cx - R * 0.18, cy); ctx.lineTo(cx - R * 0.18, cy - 7);
  ctx.moveTo(cx + R * 0.18, cy); ctx.lineTo(cx + R * 0.18, cy - 7);
  ctx.stroke();
  ctx.fillStyle = "#ffb84a";
  ctx.beginPath();
  ctx.arc(cx, cy, 2.6, 0, 2 * Math.PI);
  ctx.fill();

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

  ctx.strokeStyle = "#3a4258";               // bezel
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, 2 * Math.PI);
  ctx.stroke();

  // top readouts
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "#8b96ab";
  ctx.fillText(`ALT ${altFtOf(s.z).toFixed(0)} ft`, 10, 14);
  ctx.textAlign = "center";
  ctx.fillText(`HDG ${String(Math.round(((heading * DEG) % 360 + 360) % 360)).padStart(3, "0")}\u00b0`, cx, 14);
  ctx.textAlign = "right";
  ctx.fillText(`SPD ${mphOf(s.speed).toFixed(0)} mph`, W - 10, 14);
  ctx.textAlign = "left";

  // gauges + throttle
  drawGauge(ctx, 52, H - 44, 26, mphOf(s.speed), 0, 170, "ASI mph");
  drawGauge(ctx, cx, H - 44, 26, vsHundredsOf(s.vs), -16, 16, "VSI \u00d7100 fpm");
  drawGauge(ctx, W - 52, H - 44, 26, rpmOf(s.throttle), 0, 2700, "RPM");

  // crashed overlay: the postmortem view
  if (!alive) {
    ctx.fillStyle = "rgba(255,58,48,0.18)";
    ctx.fillRect(0, cy - 15, W, 30);
    ctx.fillStyle = "#ff4a3a";
    ctx.font = "bold 13px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText("CRASHED", cx, cy + 5);
    ctx.textAlign = "left";
  }

  els.cockpitTitle.textContent = `DRONE ${slot + 1} \u00b7 WAVE ${state.waveIdx + 1}`;
  els.cockpitFoot.textContent = alive
    ? `t ${state.t.toFixed(1)}s \u00b7 ${altFtOf(s.z).toFixed(0)} ft \u00b7 ` +
      `${s.vs >= 0 ? "+" : ""}${vsHundredsOf(s.vs).toFixed(1)}\u00d7100fpm \u00b7 ` +
      `${mphOf(s.speed).toFixed(0)} mph \u00b7 ${rpmOf(s.throttle).toFixed(0)} rpm`
    : `crashed at ${episode.survival.toFixed(2)}s \u2014 the shard screamed to the flock`;
}

/** The selected drone's controls square: yoke, rudder pedals, throttle. */
function drawControls(): void {
  if (state.selected === null || els.cockpit.hidden) {
    return;
  }
  const wave = state.traces[state.waveIdx];
  if (!wave) {
    return;
  }
  const episode = wave.episodes[state.selected.slot];
  if (!episode) {
    return;
  }
  const ctx = fitCanvas(els.controlsCanvas);
  const W = els.controlsCanvas.clientWidth;
  const H = els.controlsCanvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  const idx = Math.min(Math.floor(state.t / DT), episode.samples.length - 1);
  const s = episode.samples[idx];
  const alive = state.t < episode.survival;
  const SURF_MAX = 0.35;    // mechanical limit, rad

  // --- yoke (aileron rotates the wheel, elevator pushes/pulls it) -----------
  const yx = W * 0.3;
  const yy = H * 0.4;
  const travel = 15;                    // px of push/pull travel
  const pull = (s.elev / SURF_MAX) * travel;    // nose up = pull = yoke toward pilot (down)
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
  ctx.strokeStyle = alive ? "#c9d4e3" : "#5b6878";
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
  ctx.fillStyle = alive ? "#8b96ab" : "#3d4658";
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

  // --- throttle lever ---------------------------------------------------------
  const tx = W * 0.78;
  const tyTop = H * 0.16;
  const tyBot = H * 0.62;
  const th = Math.max(0.0, Math.min(1.0, s.throttle));
  ctx.fillStyle = "#1a2030";
  ctx.fillRect(tx - 7, tyTop, 14, tyBot - tyTop);
  ctx.strokeStyle = "#3a4258";
  ctx.strokeRect(tx - 7, tyTop, 14, tyBot - tyTop);
  const knobY = tyBot - th * (tyBot - tyTop);
  ctx.fillStyle = alive ? "#ff8c3a" : "#5b4a30";
  ctx.beginPath();
  ctx.arc(tx, knobY, 9, 0, 2 * Math.PI);
  ctx.fill();
  ctx.strokeStyle = "#0b0e14";
  ctx.beginPath();
  ctx.arc(tx, knobY, 9, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.fillStyle = "#5b6878";
  ctx.fillText("THROTTLE", tx - 27, tyBot + 16);
  ctx.fillStyle = "#c9d4e3";
  ctx.fillText(`${Math.round(th * 100)}%`, tx - 10, tyBot + 28);

  // surface readouts, bottom right
  ctx.textAlign = "right";
  ctx.fillStyle = "#8b96ab";
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillText(`AIL ${s.ail >= 0 ? "+" : ""}${s.ail.toFixed(2)}`, W - 10, H - 34);
  ctx.fillText(`ELEV ${s.elev >= 0 ? "+" : ""}${s.elev.toFixed(2)}`, W - 10, H - 22);
  ctx.fillText(`RUD ${s.rud >= 0 ? "+" : ""}${s.rud.toFixed(2)}`, W - 10, H - 10);
  ctx.textAlign = "left";

  if (!alive) {
    ctx.fillStyle = "rgba(255,58,48,0.16)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#ff4a3a";
    ctx.font = "bold 11px ui-monospace, monospace";
    ctx.fillText("CONTROLS FROZEN", 12, H - 12);
  }
}

/**
 * The selected drone crashed: hand the cockpit to the next drone still
 * flying (next seat number, wrapping). When nobody in the wave is alive
 * the panel keeps the postmortem until the wave tail ends the show.
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

// --- clock ---------------------------------------------------------------------

let lastTs = 0;
function frame(ts: number): void {
  const dt = Math.min(0.05, (ts - lastTs) / 1000);
  lastTs = ts;
  if (state.playing && !state.done) {
    state.t += dt * Number(els.speed.value);
    const duration = waveDuration(state.traces[state.waveIdx]) + 1.5;
    if (state.t >= duration) {
      if (state.waveIdx + 1 < state.traces.length) {
        state.waveIdx += 1;
        state.t = 0.0;
        setSelected(null);       // a new wave is new drones: selection resets
      } else {
        // Hold the final frame, fully drawn; never index past the last wave.
        state.done = true;
        state.playing = false;
        state.t = duration;
        els.play.textContent = "▶ play";
        updateBadge();
      }
    }
  }
  autoAdvanceSelection();
  drawSky();
  drawCockpit();
  drawControls();
  drawChart();
  updateHud();
  requestAnimationFrame(frame);
}

// --- controls ------------------------------------------------------------------

els.play.addEventListener("click", () => {
  if (state.done) {
    return;
  }
  state.playing = !state.playing;
  els.play.textContent = state.playing ? "⏸ pause" : "▶ play";
});
els.restart.addEventListener("click", () => {
  const seed = state.seeds.find((s) => s.seed === Number(els.seed.value))!;
  record(seed);
  els.play.textContent = "⏸ pause";
});
els.seed.addEventListener("change", () => {
  const seed = state.seeds.find((s) => s.seed === Number(els.seed.value))!;
  record(seed);
  els.play.textContent = "⏸ pause";
});
els.sky.addEventListener("click", (event) => {
  const rect = els.sky.getBoundingClientRect();
  setSelected(hitTest(event.clientX - rect.left, event.clientY - rect.top));
});
els.sky.addEventListener("mousemove", (event) => {
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
  const [rngRef, goldenDoc] = await Promise.all([
    fetchJson("/reference/rng-ref.json"),
    fetchJson("/reference/golden-baseline.json").catch(() => null),
  ]);
  state.seeds = rngRef.seeds;
  els.seed.innerHTML = state.seeds
    .map((s: SeedEntry) => `<option value="${s.seed}"${s.seed === 42 ? " selected" : ""}>${s.seed}</option>`)
    .join("");
  if (goldenDoc) {
    state.golden = { perWave: goldenDoc.scenarios.waves.per_wave };
  }
  record(state.seeds[0]);
  requestAnimationFrame((ts) => {
    lastTs = ts;
    requestAnimationFrame(frame);
  });
}

boot().catch((error) => {
  els.status.textContent = `boot failed: ${error}`;
});
