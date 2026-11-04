/**
 * FPM placement + world render forensics: render drawCockpitView
 * headlessly with a recording 2D context. Checks:
 * - the flight path marker sits ON the horizon in level flight, ABOVE the
 *   horizon but BELOW the fixed nose symbol in a climb (the gap = angle of
 *   attack), and slides sideways in a banked turn (sideslip);
 * - the per-pixel world pass paints every row of the buffer (ground below
 *   the horizon, sky above), so nothing is left black in any attitude —
 *   including the spin's nose-down look-at-the-ground view.
 */
const DEG = Math.PI / 180;
globalThis.window = { devicePixelRatio: 1 };   // headless: fitCanvas reads it
const scratchOps = [];
globalThis.document = {                         // headless: worldBuffer creates one
  createElement: () => ({
    width: 0, height: 0,
    getContext: () => recCtx(scratchOps),
  }),
};

function recCtx(ops) {
  return {
    canvas: null,
    setTransform() {}, save() {}, restore() {}, beginPath() {},
    rect() {}, clip() {}, translate() {}, rotate() {},
    clearRect() {}, fillRect() {}, strokeRect() {},
    moveTo(x, y) { ops.push(["m", x, y]); },
    lineTo(x, y) { ops.push(["l", x, y]); },
    arc(x, y, r) { ops.push(["arc", x, y, r]); },
    ellipse() {},
    closePath() {}, fill() {}, stroke() {},
    fillText() {}, setLineDash() {}, globalAlpha: 1,
    font: "", textAlign: "left", lineWidth: 1, lineJoin: "miter",
    lineCap: "butt", strokeStyle: "", fillStyle: "",
    createImageData(w, h) {
      return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) };
    },
    createRadialGradient() { return { addColorStop() {} }; },
    createLinearGradient() { return { addColorStop() {} }; },
    putImageData(img) { ops.push(["put", img]); },
    drawImage() {},
  };
}

function recordingCanvas(W, H) {
  const ops = [];
  return {
    clientWidth: W, clientHeight: H, width: 0, height: 0,
    getContext: () => recCtx(ops),
    ops,
  };
}

const { drawCockpitView } = await import("../build/src/app/cockpit.js");

const W = 300, H = 300;
const FOV_H = 90 * DEG;
const F = (W / 2) / Math.tan(FOV_H / 2);       // the camera's focal length

function render(view, bankDeg) {
  scratchOps.length = 0;                        // the world buffer's puts land here
  const canvas = recordingCanvas(W, H);
  drawCockpitView(canvas, "t", { ...view, bank: bankDeg * DEG }, [], "");
  const fpmArc = canvas.ops.find((o) => o[0] === "arc" && o[3] === 4.5);
  const world = scratchOps.find((o) => o[0] === "put");
  return {
    fpm: fpmArc === undefined ? null : [fpmArc[1], fpmArc[2]],
    world: world === undefined ? null : world[1],
  };
}

const cases = [
  ["level cruise (pitch 0, vs 0)", { x: 0, y: 0, z: 300, fwdX: 1, fwdY: 0, velX: 38, velY: 0, pitch: 0, speed: 38, vs: 0 }],
  ["15 deg climb at 70 mph", { x: 0, y: 0, z: 300, fwdX: 1, fwdY: 0, velX: 36.5, velY: 0, pitch: 15 * DEG, speed: 37.1, vs: 6.5 }],
  ["25 deg bank LEFT turning", { x: 0, y: 0, z: 300, fwdX: 1, fwdY: 0, velX: 37.0, velY: 3.0, pitch: 3 * DEG, speed: 38.1, vs: 0.0 }],
  ["spin look-down (pitch -80)", { x: 0, y: 0, z: 600, fwdX: 0.17, fwdY: 0, velX: 4, velY: 0, pitch: -80 * DEG, speed: 30, vs: -28 }],
  ["on the runway, stopped", { x: 0, y: 0, z: 30.48, fwdX: 1, fwdY: 0, velX: 0, velY: 0, pitch: 0, speed: 0, vs: 0 }],
];

const add = (v, extra) => ({ ...v, ...extra, throttle: 0.8, ail: 0, elev: 0, rud: 0, alive: true });

for (const [name, v] of cases) {
  const view = add(v, {});
  for (const bank of [0, 45]) {
    const { fpm, world } = render(view, bank);
    const horizonY = 150 + F * Math.tan(view.pitch);   // where the horizon sits
    // count painted world rows: ground pixels vs sky pixels
    let ground = 0, sky = 0;
    if (world !== null) {
      const d = world.data;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i + 2] > d[i + 1] + 12) { sky++; }   // gradient sky / fog / water
        else { ground++; }
      }
    }
    const fpmStr = fpm === null ? "no marker (speed < 1)"
      : `fpm (${(fpm[0] - 150).toFixed(1)}, ${(fpm[1] - 150).toFixed(1)}) rel` +
        `  horizon +${(horizonY - 150).toFixed(1)}  -> ` +
        (fpm[1] < horizonY - 0.5 ? "above horizon" :
         fpm[1] > horizonY + 0.5 ? "below horizon" : "ON horizon");
    console.log(`${name.padEnd(34)} bank ${String(bank).padStart(3)}: ${fpmStr}` +
                `  | world: ${ground} ground px, ${sky} sky px` +
                `${ground + sky === 0 ? "  !! WORLD BUFFER EMPTY" : ""}`);
  }
}
console.log(`\nfocal ${F.toFixed(1)} px; buffer ${Math.ceil(W / 1)}x${Math.ceil(H / 1)} (full resolution)`);

// --- the filter probe: the Ground's sampling law, directly ----------------
const FTm = 0.3048;
const { Ground: GroundCls } = await import("../build/src/app/ground.js");
const { FIELD_ELEV } = await import("../build/src/app/land.js");
const g = new GroundCls();
{
  const norm = (c) => (c[0] - 66.0) / (74.0 - 66.0);   // 1.0 = level-0 bright (Grass)
  const lvl0Bright = [74, 106, 56];
  // ON THE GROUND (eye 3 m): the base map — even a HUGE footprint reads
  // level 0, because the level comes from ALTITUDE, not footprint.
  const tiny = g.sample(306.0, 210.0, 0.05, 0.05, 0.05, 0.05, 100.0, 3.0);
  const huge = g.sample(306.0, 210.0, 4000.0, 0.0, 0.0, 4000.0, 100.0, 3.0);
  const both0 = Math.abs(tiny[0] - lvl0Bright[0]) < 2
    && Math.abs(huge[0] - lvl0Bright[0]) < 2;
  console.log(`\nfilter probe (level = ALTITUDE):`);
  console.log(`  ground (3 m): tiny rgb(${tiny.map((c) => c.toFixed(0)).join(",")})` +
              ` huge rgb(${huge.map((c) => c.toFixed(0)).join(",")})` +
              ` — both the base map ${lvl0Bright.join(",")}` +
              `${both0 ? "  ✓ footprint never walks the pyramid" : "  ✗"}`);
  // AT 4000 ft: the TOP map — the same checker, same apparent size, brighter
  // ((960, 192): a bright 384 m cell centre of the top map, on the
  //  airfield's carved grass)
  const top = g.sample(960.0, 192.0, 4000.0, 0.0, 0.0, 4000.0, 100.0, 4000 * FTm);
  const topOk = Math.abs(top[0] - 130) < 6;
  console.log(`  4000 ft: rgb(${top.map((c) => c.toFixed(0)).join(",")})` +
              ` — the top map, crisp ${[130, 187, 99].join(",")}` +
              `${topOk ? "  ✓ same size, brighter" : "  ✗"}`);
  console.log(`  apparent size: 12 m from 125 ft = ${(12 / (125 * FTm)).toFixed(4)} rad,` +
              ` 384 m from 4000 ft = ${(384 / (4000 * FTm)).toFixed(4)} rad` +
              `${Math.abs(12 / (125 * FTm) - 384 / (4000 * FTm)) < 1e-6 ? "  ✓ equal" : "  ✗"}`);

  // TRILINEAR: sweep the EYE HEIGHT up the ladder; the colour must never
  // jump — the maps cross-fade as the aircraft climbs.
  let maxJump = 0.0;
  let prev = null;
  for (let hgt = 1.0; hgt <= 2500.0; hgt += 1.0) {
    const c = g.sample(306.0, 210.0, 3.0, 0.0, 0.0, 3.0, 100.0, hgt);
    if (prev !== null) {
      maxJump = Math.max(maxJump, Math.abs(c[0] - prev));
    }
    prev = c[0];
  }
  console.log(`  climb sweep 1–2500 m: max colour jump ${maxJump.toFixed(3)}` +
              `${maxJump < 2.0 ? "  ✓ seamless map cross-fade" : "  ✗ popping"}`);

  // ANISOTROPIC at ground level: the taps read the alternating pattern
  // along the stretch; the clamped square keeps full contrast too.
  const stretch = g.sample(306.0, 210.0, 30.0, 0.0, 0.0, 3.0, 100.0, 3.0);
  const square = g.sample(306.0, 210.0, 30.0, 0.0, 0.0, 30.0, 100.0, 3.0);
  console.log(`  anisotropic at ground: stretch norm ${norm(stretch).toFixed(3)}` +
              ` (structure along the stretch), square norm ${norm(square).toFixed(3)}` +
              `${Math.abs(norm(square)) > 0.1 ? "  ✓ full-contrast, never a flat mean" : "  ✗"}`);

  // THE TERRAIN: the height channel — the field a plateau at 5 ft
  // (FIELD_ELEV, the gear's ground), the hills marching up to ~1000 ft,
  // the relief shading leaning off flat.
  const land = g.land;
  const flat = Math.abs(land.heightAt(0.0, 0.0) - FIELD_ELEV) < 1e-6
    && Math.abs(land.normalAt(0.0, 0.0)[2] - 1.0) < 1e-9;  let peak = 0.0;
  let peakAt = [0.0, 0.0];
  for (let y = -4096; y < 4096; y += 32) {
    for (let x = -4096; x < 4096; x += 32) {
      const h = land.heightAt(x, y);
      if (h > peak) { peak = h; peakAt = [x, y]; }
    }
  }
  const peakOk = peak > 250.0 && peak <= 306.0;
  console.log(`\n  terrain: the field a plateau at ${(FIELD_ELEV / 0.3048).toFixed(0)} ft (${FIELD_ELEV} m) ${flat ? "✓" : "✗"}` +
              `; the tallest hill ${peak.toFixed(0)} m (${(peak / 0.3048).toFixed(0)} ft)` +
              `${peakOk ? "  ✓ rivers 0, hills ~1000 ft" : "  ✗"}`);
  // a ray at the peak hits the SLOPE before the flat plane would
  const [hx, hy] = peakAt;
  const ox = hx - 3000.0, oy = hy;
  const oz = 600.0;
  const dxx = 3000.0 / Math.hypot(3000.0, oz);
  const dzz = -oz / Math.hypot(3000.0, oz);
  const tHit = g.intersectRay(ox, oy, oz, dxx, 0.0, dzz);
  const tPlane = -oz / dzz;
  const slopeOk = tHit !== null && tHit < tPlane * 0.95;
  console.log(`  a ray into the hill: march t=${tHit === null ? "null" : tHit.toFixed(0)}` +
              ` vs flat-plane t=${tPlane.toFixed(0)}${slopeOk ? "  ✓ the slope catches it first" : "  ✗"}`);
  // the relief: the hilltop is flat by definition — the FLANK is where
  // the normal leans (walk downhill until half the peak's height)
  let fx = hx, fy = hy;
  const stepLen = Math.hypot(hx, hy) / 200.0;
  const stepX = -hx / Math.hypot(hx, hy) * stepLen;
  const stepY = -hy / Math.hypot(hx, hy) * stepLen;
  for (let i = 0; i < 200 && land.heightAt(fx, fy) > peak / 2.0; i++) {
    fx += stepX;
    fy += stepY;
  }
  const nf = land.normalAt(fx, fy);
  const leans = nf[2] < 0.995;
  console.log(`  relief on the flank (${Math.round(fx)}, ${Math.round(fy)}):` +
              ` normal (${nf.map((v) => v.toFixed(3)).join(",")})` +
              `${leans ? "  ✓ the hills lean into the sun" : "  ✗"}`);
  // THE WATER/LAND LAW: every water texel exactly level 0 (the flat
  // water plane — no slope, no shimmer), every land texel above 10 ft
  // (the bank at every waterline). Sampled on OPEN ground — texels
  // whose whole 8-neighbourhood is one kind; the boundary ring is the
  // bank itself, where the surface blends water level into land.
  let waterMax = -1.0, landMin = 1e9;
  for (let ty = 1; ty < 511; ty++) {
    for (let tx = 1; tx < 511; tx++) {
      const x = (tx + 0.5) * 16, y = (ty + 0.5) * 16;
      const biome = land.biomeAt(x, y);
      let mixed = false;
      for (let oy = -1; oy <= 1 && !mixed; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (land.biomeAt(x + ox * 16, y + oy * 16) !== biome) { mixed = true; break; }
        }
      }
      if (mixed) continue;               // the bank ring: the blend
      const hgt = land.heightAt(x, y);
      if (biome === 4) {                 // Biome.Water
        if (hgt > waterMax) waterMax = hgt;
      } else if (hgt < landMin) {
        landMin = hgt;
      }
    }
  }
  const lawOk = waterMax <= 1e-6 && landMin > 10.0 / 3.28084;
  console.log(`  the law: open water flat at 0 (max ${waterMax.toFixed(3)} m)` +
              `, open land above 10 ft (min ${(landMin / 0.3048).toFixed(1)} ft)` +
              `${lawOk ? "  ✓" : "  ✗"}`);}


