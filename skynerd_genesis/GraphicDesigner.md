# Graphic Designer — day log (2026-11-04)

Everything I did today, in order. My lane: all look and feel of the world
(the terrain material, the ground renderer, the cockpit's world view) plus
the rendering infrastructure underneath it. Where I crossed into the
Engineer's lane (physics datum, school wiring), I flagged it and wrote
`ENGINEER-NOTE.md`.

## 1. Terrain height mapping — the multidimensional ground material

The user's ask: the procedural 512×512 land-use texture should become a
PBR-style material — colour AND height. Rivers at 0, runway ground at
10–300 ft, hills to 1000 ft.

- `src/app/land.ts` — the generation loop now writes a `HEIGHT`
  Float32Array alongside the biome `MAP`: elev = TERRAIN_MAX·t^1.6,
  water at level 0, tallest hill 305 m (1001 ft). Exports: `SUN_DIR`
  (one sun for relief/shadows/glare), `TERRAIN_MAX`, `FIELD`, `FIELD_ELEV`.
- `LandUse.heightAt` — hermite (smoothstep) bilinear over the height
  channel, seamlessly tileable. `LandUse.normalAt` — central differences,
  used by the relief shading.
- The airfield carve: an ellipse at exactly level ground, rimmed by a
  smoothstep roll into natural terrain, so the flight school keeps its
  flat-ground physics parity while the world rolls up outside.
- `src/app/ground.ts` — `intersectRay` became a marching ray through the
  terrain band z ∈ [0, 305]: relative stride max(6, 0.09·t), 6 bisections
  to refine the crossing. `sample()` gained relief shading: surface normal
  against the sun, normalized so the flat plain keeps EXACTLY its approved
  palette (factor 1.0 on flat ground; only slopes brighten/darken).
- `src/app/cockpit.ts` — all world objects re-planted on the terrain via
  `heightAt` (tower base, tree trunks, houses per-corner foundations and
  ridge ends). Fixed a stale-`top` variable bug (resolved to `window.top`)
  by restoring the ridge height.
- `tools/fpm-debug.mjs` — terrain gates: the field flat, hills to 1001 ft,
  rays hit slopes before the flat plane would, flank normals lean into
  the sun.

## 2. Marching-ray performance (the renderer must hold 60 Hz)

First measurement: 255 ms per frame of pure marching at 1240×440 — about
4 fps. Ended at 101–165 ms worst case, 54–63 ms at cruise altitude, with
correctness unchanged against a fine reference march. The optimizations,
all exact (never skip a possible hit):

- **Coarse-max pyramid** (`MAXMIP_LEVELS` = 7) over the height channel:
  each block carries its tallest texel plus a one-texel ring (hermite's
  window). A ray provably above a block's max leaps to the block's far
  edge or where it would sink to that max.
- **Cascade**: from the coarsest block (2 km) down to the fine stride;
  the first level whose max the ray clears gives the longest provable leap.
- **Plateau fast paths**: `FIELD`/`FIELD_ELEV` constants +
  `LandUse.inField()`. A descending ray whose plane crossing lands inside
  the ellipse (with its in-band prefix inside too — the ellipse is convex)
  is answered analytically. Any probe inside the ellipse skips to the
  ray-ellipse exit in one closed-form step.
- **False-position refine** with adaptive bracket width, replacing the
  fixed 6-step bisection.
- **Fast wrapping**: 512 is a power of two — the double-modulo in
  `heightAt`/`biomeAt`/`maxHeightAt` became floor + `TEX_MASK`.

Ray-correctness sweep (2000 random rays vs a 2%-stride/14-bisection
reference): agreement 1897/2000 with the same 5 baseline misses at every
stage — the skips added zero new tunneling. Also fixed a real march bug
from the previous session's restructuring: the final probe at `end` was
never tested (`while (t < end)` exited early), collapsing ground-pixel
counts.

## 3. Water, rivers, and the altitude law set

The user then refined the laws:

- **Rivers were the water aliasing.** Lakes ramp smoothly from 0, but
  river texels were forced to 0 while the land beside sat at 60+ m — a
  16 m slit gorge whose cliff-wall normals glittered in the relief
  shading. Rivers now carve a wide smooth valley that meets its banks
  without a step and shallows to nothing before the height cutoff.
  Measured: shoreline normals mean z 0.979 (near-flat), later 0.892 with
  the wider valley — was cliff-steep.
- **The law set**: all water flat at 0 ft; all land above 10 ft
  (`LAND_MIN` = 3.2 m ≈ 10.5 ft — a bank at every waterline); runway a
  flat plateau at 95 ft; takeoff datum 100 ft = 95 ft terrain + 5 ft gear.
  The water plane is exactly 0 (open-water max measured 0.000 m).
- `src/core/world.ts` — new `AIRFIELD` export (elev 95 ft, ellipse
  cx 600 / rx 1050 / ry 420). `World.terrainHeight` returns the plateau
  inside the ellipse, the Gaussian hills outside. `src/app/land.ts`
  imports it — the physics ground and the visual ground are ONE shape.
- **Rim-clamp bug**: the plateau's rim blend was rolling toward water
  texels at 0, producing "land" below 10 ft where the field met a shore.
  The rim now blends toward max(natural terrain, LAND_MIN).
- `SYLLABUS.runwayM` → 100 ft (the datum), pin comment updated. The
  instruments read body z, so standing on the runway they read 100 ft —
  the 5 ft gear offset is the datum law itself.
- fpm-debug gained the water/land law gate: open-water and open-land
  sampling (skipping the boundary ring, which IS the bank blend).
  One float32 lesson: 1.524 stored as float32 is 1.5240000486 — the
  plateau check tolerance is 1e-6, not 1e-9.

## 4. The gear — physical wheels instead of the magic pin

The user saw the aircraft fall into the runway on the AI's own attempt
and asked for a physical gear:

> tangent deck (±15°), wheels within ±0.5 ft of rest: −1g upward.
> Nose-low −15..0°: lever torque to level, proportional to the lever
> force. Nose-up 0..15° on the ground: downforce, strong at the runway,
> relaxing completely by 50 ft.

New `src/physics/gear.ts` — one stateless `GearContact`, applied after
each physics tick in the school's shared `stepMachine` (every classroom
aircraft, the instructor twin included):

- **SUPPORT**: wheels at/below rest in the ±0.5 ft band, arrival ≤ 3 m/s:
  +1g cancels gravity, the oleo absorbs sink, the strut holds rest height.
  A 20 m/s dive collapses the gear and the crash proceeds to the school's
  grounded() line. Rising is free — the strut never pulls the aircraft
  down, so a held rotation lifts off unimpeded.
- **LEVELING**: lever torque toward level, ∝ nose-down depth, eased via
  a rate target so the accumulated rate never overshoots past level.
- **GLUE**: 0.25 g downforce at the runway, fading to zero at 50 ft,
  ∝ nose-up deck — the rotation cannot balloon.

Bugs found and fixed on the way: Vec3 is immutable (rebuild the vector,
the pin's precedent); gravity's per-tick displacement leaked the datum
down 3.4 mm/tick (cancel velocity AND displacement); the lever's
accumulated rate overshot nose-up into a stall-ish bounce (target-rate
easing); and — the subtle one — an early support draft clamped a RISING
aircraft back to the runway, an anti-lift-off law masquerading as a
gear (loaded = at/below rest AND sinking).

Verified: taxi holds rest with zero leak over 8 s; a 2 m/s arrival at
55 mph absorbs and settles in 1.1 s; the hard dive collapses and
crashes; a held full-throttle rotation is airborne in 0.7 s; the
instructor demo rotates clean (wheel rises 0.5 → 3.8 → 7.6 ft across
the rotation second, no balloon) and the AI's fumbled roll stays on the
wheels (min wheel −0.00 ft).

## 5. Test and harness upkeep (multi-agent reality)

- restart-check: added the `els.controls` stub (the Engineer's new
  controls dropdown crashed the harness at import), re-anchored to the
  one-aircraft classroom narration ("takeoff — element: roll · …",
  "AIRCRAFT · AIRBORNE"), then again to their instructor-first default.
  ALL PASS (29 checks).
- vec.test: the frozen Python terrain golden predates the field carve
  (all five points sit inside the ellipse) — the check now expects the
  plateau in-field and the Python hills outside, with a note that the
  Engineer can re-freeze the baseline with the field when convenient.
- physics.test builds no World (ground-effect-free goldens, unaffected);
  episode.test is replay-parity only (flight.ts untouched); school.test
  is the Engineer's placeholder.

## 6. Handoff

`ENGINEER-NOTE.md` (repo root) — the law set, every touch I made in
their files, the gear's constants for their review, the pin-vs-gear
overlap (they may want to retire the pin), the note that the waves
drones have no gear wired, and the open re-freezes (20-lesson
curriculum, vector baseline).

## Final state (all verified green today)

- `npm test` — 8 files, 1313 checks.
- `tools/restart-check.mjs` — ALL PASS.
- `tools/fpm-debug.mjs` — plateau 95 ft exact, open water exactly 0,
  open land min 10.5 ft, hills 1001 ft, relief leaning into the sun.
- Classroom takeoffs, both control modes: clean rotation, no fall.
