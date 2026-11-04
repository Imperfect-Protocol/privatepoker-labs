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

---

# Graphic Designer — addendum (the WebGPU pipeline, same day)

The user's afternoon asks, in order: use my LightAndGeometry as
structural inspiration (scene graph, clumps, geometries, lights —
explicitly NOT the raytracer); WebGPU, not WebGL; all math on GPU
(one-off host math allowed, massive only in kernels); put everything
in `src/pipeline` with a decent OOP shape; queue kernels and make the
conditionals happen IN the kernels on buffer data; render the terrain
as quads generated on the go (one buffer = the quad instances); use an
index map — project the render surface into tile→geometry indexes —
instead of visibility rays; biome instancing on GPU; variation via
noise textures.

## 7. src/pipeline — the GPU pipeline, built programmatically

```
src/pipeline/
  types.ts            the WebGPU surface as OUR structural interfaces
                      (zero npm deps; compiles under BOTH tsconfigs —
                      no ambient declares to collide with the DOM lib)
  core/
    gpu-device.ts     GpuDevice — the ONE device (plan §1), lost = CPU
    kernel-queue.ts   GpuBuffer, FrameRecord (the 176-B uniform),
                      KernelQueue — kernels queued, flushed as ONE
                      compute pass per frame
    wgsl.ts           the shared WGSL: Frame, the world's constants,
                      hash01v, heightAt (the CPU path's exact hermite),
                      normalAt, the checker law (ramp/squareMean/
                      checkerMip/levelPair/checkerColour), the biome
                      palettes, roads + runway paint, groundColour
  scene/
    camera.ts         Camera — owns the VP matrix (one-off CPU math
                      per frame); tests pin its half-edge projections
    light.ts          Light — the one sun (SUN_DIR)
    vertices.ts       Vertices — the programmatic vertex stream
    geometry.ts       Geometry — owns its GPU vertex/index buffers
    clump.ts          Clump — spatial group with bounds (L&G's TODO,
                      finished)
    scene.ts          Scene + RenderClump — the graph; generate()
                      then draw(); the clump order IS the draw order
    sky-clump.ts      the fullscreen sky triangle (gradient + sun)
  geometries/
    quad.ts           the unit quad — the terrain's cell
    cone.ts           the 8-segment cone — the tree
  kernels/
    terrain-quads.ts  TerrainClump — the on-the-go quad stream: near
                      160²@12 m + far 128²@160 m around the camera,
                      far interior SKIPPED by a kernel conditional;
                      the VS displaces corners off the height texture,
                      collapses index-flagged instances
    flora-instances.ts FloraClump — biome INSTANCING on GPU: walks the
                      20-m lattice, reads the biome texture, rolls the
                      same hash01 in WGSL, writes tree instances
    tile-index.ts     TileIndexMap — THE INDEX MAP: 32-px screen
                      tiles; each instance projects its bounds into
                      the grid (atomics), sets its visibility flag;
                      occlusion = the rasterizer's own depth buffer
  renderer/
    features.ts       RenderFeatures + defaults (haze on; relief,
                      glare, sunShadows, reflections off)
    world-renderer.ts the WorldRenderer seam (gpu if ready else cpu)
    gpu-world-renderer.ts  the conductor: textures uploaded once
                      (height r32float, biome r8uint, 256² variation
                      noise from hash01 value noise), group 0, the
                      scene, per frame: uniform write → kernel queue →
                      ONE render pass → submit
  util/
    intersect.ts      frustum planes, sphere culling, ray-AABB — the
                      CPU twins of the kernel tests (testable headless)
```

The WGSL mirrors the CPU laws function-for-function — hash01v is proven
bit-identical to land.ts's hash01 by test over ±600 lattice ranges;
heightAt is the same hermite bilinear; the checker law is the same
antiderivative box filter with the same 6-level ×1.12 pyramid; the
roads and runway paint are the same formulas. The parity probes
(plan §7) can diff exactly.

## 8. The app seam (my files, one touch in land.ts)

- `cockpit.ts` — `drawCockpitView` gained the optional `world`
  renderer param: GPU-ready → it renders the world layer and this
  canvas carries only the HUD ink (ladder, tapes, marker, overlays,
  glare per feature); otherwise today's CPU path, unchanged.
  `drawWorldObjects` takes `withTrees` — the GPU forest owns the trees.
- `main.ts` — creates `GpuWorldRenderer` (null-safe: the harness has
  no worldCanvas and no navigator.gpu → CPU path, zero new stubs),
  the render-feature checkboxes (also null-safe), sizes the new
  canvas stack.
- `land.ts` — two getters: `heightsOf()`, `biomesOf()` (the pipeline's
  texture uploads). No math touched.
- `web/index.html` — the `.cockpit-view` canvas stack (worldCanvas
  under cockpitCanvas) + the checkbox row next to RESET (relief, glare,
  haze, shadows, refl).
- `web/style.css` — the stack's positioning; the panel look moved to
  the wrapper.
- `test/pipeline.test.ts` — NEW suite: hash bit-parity, VP half-edge
  projections, frustum cull, ray-AABB, the geometries' shapes, the
  scene graph's frame order, the WGSL common's integrity (entry points
  present, no template holes, braces balance). 23 checks, all green.

## 9. Bugs the build caught (and the honest gaps)

- The VP matrix had −fwd in the view row while the projection expected
  +depth for w: every NDC came out sign-flipped — found by the test's
  half-edge probes, fixed (+fwd, z mapped to [0,1]).
- The near grid's half offset was (n−1)/2 — a 6-m slit at the field
  boundary where the far field's kept ring meets it; n/2 centres the
  grid exactly on its half-extent.
- The shared WGSL initially had no texture/sampler declarations —
  every module would have failed to compile; the review sweep caught
  it before any browser saw it.
- KNOWN NOT DONE (phases): the corner-ray shadow field (the sunShadows
  checkbox flows to the uniform; no kernel consumes it yet), the water
  reflections pass, GPU tensors (Engineer's §8). The GPU path itself
  has NOT been eyeballed in a browser yet — types, WGSL review and the
  headless battery are green; `npm run dev` is the next look.

## Final state (the pipeline build, all verified green)

- `npm test` — 9 files, 1359 checks (pipeline suite new, 23 checks).
- `tools/restart-check.mjs` — ALL PASS (the harness runs the compiled
  app with the pipeline present, picking the CPU path cleanly).
- `tools/fpm-debug.mjs` — all terrain gates unchanged.
- Both tsconfigs compile the whole tree with the pipeline in.

## 10. The browser eyeball — the class of bug no headless test can catch

Driven headless Chrome Canary (159) over the DevTools protocol with a
zero-dependency websocket driver (`/tmp/cdp-*.mjs` — session scratch,
not shipped): live console capture + DOM probe + screenshots + GPU
buffer readbacks. The plan §7 eyeball happened; every WGSL module now
compiles and the GPU path renders in a real browser (Metal).

Bugs the eyeball caught, all fixed and re-verified (npm test 236,
restart-check ALL PASS):

- **WGSL ternaries** — Canary's Tint rejects `?:` outright ("invalid
  character found"); even a trivial one. Rewrote all 7 sites to
  `select(false, true, cond)` (checkerRamp, foldDist, runwayShade ×2,
  road paint ×2, tree habits ×2).
- **`patch` is a reserved WGSL keyword** — the road patch roll renamed
  `paving`.
- **Vector comparisons** — `colour == base` on vec3f is a `vec3<bool>`;
  wrapped in `all(...)` (runway paint, 3 sites).
- **Quad record field** — the index entry read `q.z` on a
  `{x,y,size,pad}` record; `q.size` is the quad.
- **KernelQueue flush set only bind group 0** — every compute pipeline
  is [group0, own]; the queue now carries the full bind-group list.
- **Draw passes never set the index buffer** — `setIndexBuffer(raw,
  "uint32")` added in both clumps' drawPass (quad + cone).
- **THE BIG ONE — dispatch ordering**: all kernels flushed into ONE
  compute pass. WebGPU does NOT order dispatches within a pass — the
  tile-index kernel read the quad buffer while the gen kernel still
  wrote it, every frame. QueuedKernel now carries a PHASE; flush
  groups same-phase kernels and opens one compute pass per phase
  (pass boundaries are the barriers). Terrain gen + flora gen = phase
  0; both index kernels = phase 1 → 2 passes per frame.
- **THE consts bit-pattern bug** — `GenConsts.cols` is i32 in WGSL but
  was seeded through a `Float32Array([160, ...])`: the kernel read
  float-160's bits as int = 1,124,073,472, so every quad write landed
  out of bounds (robust buffers: silently dropped). The gen kernel
  had never written a quad — that was the entire "no terrain". The
  consts now seed through a mixed u32/f32 ArrayBuffer view.
- **The resize flicker + half-height view** — layoutSidebar measured
  `#cockpit-view` (the wrapper whose size it sets) as a fixed-height
  child: measure-my-own-output = two-frame oscillation between two
  sizes (the person's "buffer #1 vs buffer #2" diagnosis, exactly
  right). The wrapper is now excluded from the measured children; the
  view fills the column (381×215 → 574×323 on the same window).
- **The terrain waves ("ocean" artifact)** — the quad grid placed
  quads at camPos + slot offset, so every 12-m quad slid continuously
  across the heightfield as the camera moved: vertices sampled
  different heights each frame and the whole surface rippled. The grid
  origin now SNAPS to the world lattice (`floor(camPos/cell)*cell`,
  both fields; the far interior skip snaps to the near lattice the
  same way) — the geometry-clipmap law. Verified by readback: quad
  records bit-identical while the camera stays in one cell; shift by
  exactly 12 when it crosses. The surface is world-fixed.

GPU evidence trail (flags/instances readback via staging buffers,
MAP_READ probes): gen wrote nothing → after the consts fix, quads at
[-960,-960,12]... flags live, terrain green, trees visible, runway
painted, feature checkboxes shift pixels, zero console errors, both
tsconfigs clean, npm test + restart-check green throughout.

## 11. The sky's moods, the flares, the checker toggle, the crash floor

The person asked for four more toolbar things, all landed with the
parity law intact (one source of truth, both paths read it):

- **THE TIME OF DAY** — `src/pipeline/renderer/time-of-day.ts` is the
  single table both renderers read: 6 moods (dawn, sunrise, noon,
  sunset, dusk, night), each carrying the sun direction, zenith, fog,
  disc tint, disc size, brightness, ambient scale and cloud tint.
  `noon` IS the pre-mood constants exactly, so default parity holds.
  The Frame uniform grew 4 rows (176 → 240 B): zenith+discSize,
  fog+brightness, sunTint+ambient, extras (the checker toggle). The WGSL
  sky/terrain/tree read the mood from the uniform (÷255 — the table
  carries 0..255 like the CPU constants); the CPU cockpit and ground
  read `RENDER_STATE.mood` / `groundRender`. The relief's sun follows
  the mood on BOTH paths (the GPU always did; ground.ts now too).
- **THE LENS FLARES** — a `flare` dropdown (none/panavision/michael
  bay) drawing additive ink on the HUD canvas over whichever world
  layer rendered: panavision = the long thin blue anamorphic streak +
  warm halo + vertical whisper; michaelbay = white-hot core + cross
  streaks + ring + a chain of coloured ghosts marching through the
  frame centre. Both gate on sun-facing (align²) and track the mood's
  tint/size; both paths get them (the HUD canvas is shared).
- **THE CHECKER TOGGLE** — `checker` joins RenderFeatures (default
  on), flows to the uniform's extras.x; the WGSL checkerColour takes
  the flat per-level mean when off (no phase — the same pyramid, so
  the palette/brightness law is untouched); the CPU checkerTap's twin
  branch reads groundRender.checker.
- **THE CRASH FLOOR** — the background goes BLACK 15° below the
  horizon (SKY_FLOOR_SIN, both sky laws: the WGSL skyFs and the CPU
  per-pixel sky): a crashed camera looks into darkness, never into
  sky-through-the-ground. The terrain still covers the real ground
  plane; the floor only owns the background gaps under the world.

Verified live (Canary/Metal, the CDP feature probe): noon's luminance
signature UNCHANGED from the classic build (meanLum 101, max 247.4 —
exact palette parity); sunset warms (72,60,64 with a purple zenith
melting to the orange horizon — eyeballed), night is near-black
(18,22,28, dark moonlit ground — eyeballed), sunrise pink-tinged
(89,83,97); checker off shifts the ground (69→84 restored); the flare
families' streaks visibly bloom in the sunset shots (the Bay vertical
vs the Panavision horizontal — eyeballed both); zero console errors;
npm test (236 pipeline checks) + restart-check ALL PASS; both
tsconfigs clean.

## 12. The setting sun — blendy, chewed, glaring (both paths, parity)

The person asked for blendier sun edges and a more glaring, irregular
sunset. THE SETTING LAW, one formula on both renderers: the bigger the
disc, the lower the sun, and the more the atmosphere owns it —

- the SETTING FACTOR s = (discSize − 0.035)/0.055, 0 at noon's clean
  circle, 1 at the sunset swell (sunrise 0.727);
- THE WOBBLE: two harmonics chew the rim — 0.08·sin(4θ+1.7) +
  0.05·sin(9θ+0.6) — an irregular silhouette, not a circle;
- THE SQUASH: refraction flattens the disc vertically (÷(1+0.35s));
- THE BLENDY EDGE: solid at the wobbled core, fully melted just
  beyond the wobbled radius (never a hard rim) — the GPU mixes
  (radius−base)/(disc·0.32) clamped, the CPU twin draws the wobbled
  48-gon with a radial gradient that fades from 0.90 at 62% to 0 at
  the rim;
- THE GLARE: wider (×(1+0.8s)) and far stronger (0.28+1.1s — sunset
  glares ~5× noon) in the sky itself; the veil doubles too
  (×(1+1.2s)) and takes the mood's tint warmed toward white.

Verified with a camera pointed straight at the sun (noon/sunset/
sunrise PNGs off a scratch renderer): the sunset disc is a soft
glowing irregular blob — the close-up shows the dissolve rim and the
asymmetric chewed envelope, no hard edge anywhere; noon keeps its
clean bright circle (s = 0 — the classic look untouched). npm test
(236) + restart-check ALL PASS; both tsconfigs clean.

## 13. The sun matches its sky (the sticker fix, both paths)

The person's eye: the sunset disc read as a DARK ORANGE STICKER
against the bright glow around it — the mood's disc tint [255,140,64]
is darker in G/B than the sunset fog band it sits in. THE FIX, one law
on both renderers: the disc's FILL is the tint lifted 65% toward white
(GPU: `core = mix(tint, vec3f(1.0), 0.65)`; CPU twin in drawSun) — a
hot core that can never be darker than the sky around it, the blendy
rim then melting INTO the sky's own shade. noon's tint [255,253,242]
lifts to [255,255,248] — the classic disc, visually unchanged.

Pixel truth off a live sunset render (camera aimed at the sun, exact
canvas samples): disc centre [255,255,210], brighter than the sky 90px
away [255,255,196] (brighterThanSky: true) — the sun is now the
brightest thing in its sky, one blown-out warm field, no seam. npm
test (236) + restart-check ALL PASS; both tsconfigs clean.

## 14. Light is additive — the flare, the glare, the HUD glass

The person's screenshot caught the last flaw: over the sunset sky the
panavision streak read as a foreign washed-out line. TWO root causes,
both fixed:

- **THE COLOUR FAMILY** — the flare ink now derives ENTIRELY from the
  sun's tint (glare(lift) = tint lifted toward white — the same
  family skyFs's glow uses): warm orange streaks at sunset, near-
  white at noon. The Bay ghosts' cool/violet accents are MIXES OF THE
  TINT, never fixed RGB. One light source, one colour family.
- **THE COMPOSITE** — flare/glare can't match the sky while alpha-
  pasted over it. NEW #lightCanvas sits between the world canvas and
  the HUD, mix-blend-mode: plus-lighter (TRUE additive — screen
  family). The sun, the flare and the glare veil draw on it (GPU
  path; the CPU path keeps its in-canvas `lighter`, which is the
  same physics). Their ink now ADDS to the sky's own glow, exactly
  like the shader's col +=.
- **THE HUD GLASS** — #cockpitCanvas composites plus-lighter too:
  tapes, ladder, markers and objects glow additively over the world
  (the fighter-HUD look), ONLY on the GPU path — on the CPU path the
  HUD canvas holds the opaque world itself, so its blend flips back
  to normal per frame (adding it would double-brighten).
- the flare blooms with the hour: strength × (0.75 + 0.6·s) — the low
  sun's streaks bloom harder, matching its sky.

Live verdicts (Canary/Metal): sunset + bay hunt frames show the flare
additive over the world with its ghost chain visible and the HUD
"instrument overlay highly visible... blend smoothly with surrounding
sky colors"; the HUD glass reads as "glowing luminous overlays,
blending subtly with the world" (analyzer's own words). Zero console
errors; npm test (236) + restart-check ALL PASS; both tsconfigs clean.

## §15 the pilot's iris, the hour's grade, the map's law, and the curved windshield

**The iris (HDR adaptation)** — `IrisAdaptation` (cockpit.ts): the eye
tightens toward the sun on the glare's own align³ bloom (the veil and
the iris are one reflex), easing on the eye's time constants — a 0.2 s
flinch toward the light, a 1.0 s reopen away from it. One `iris` per
app; every pane renders through it. The exposure rides the Frame
uniform (extras.y, 240 → 256 B for the grade row) and multiplies
skyFs/terrainFs/treeFs's exits; the CPU compose and the sun/glare/
flare ink (drawSun's fill+glare alphas, drawSunGlare, drawLensFlare
strength) scale by the same number. Facing the sun dead-on the world
dims to ~45% and the glare recedes with it.

**The hour's grade (the user's colour list)** — the SkyMood table gained
`grade` (0..255, applied mean-1: tint without darkening): dawn dark
pinkish [214,186,202], sunrise bright pink [255,208,224], noon white
(the identity — parity exact), sunset orange [255,180,120], dusk
purple-orange [238,172,192], night purple-navy [148,152,224]. GPU: the
grade row (extras sibling) + `gradeFilter()` in WGSL_COMMON, applied in
terrainFs and treeFs after the ambient, before the haze. CPU: the same
mean-1 law in drawWorld's compose (gradeR/G/B × GROUND_GRADE=0.55) and
the haze blends to the fog from the GRADED colour. The CONTROLS strip
wears the hour too: `gradeControlsVeil` — a source-atop veil at 50%
(the drawn ink only, the frozen alarm stays red), under it the
instruments read like the hour's own light.

**The map's hide law (the user's 10%)** — SIDE_HIDE 0.30 → 0.10, and the
grip now owns BOTH sides of its track: dragging far left grows the
sidebar past the map's own 10% floor and the MAP column snaps away
(#app.map-hidden → the 0px track), the sidebar owning the whole width;
from hidden, a rightward pull past REOPEN reopens the map, tracking the
pointer, clamped to the sidebar's own floor (the poison-loop fix:
mapW can never exceed the track). The sidebar keeps its own mirror law
(under 10% it hides; a left pull reopens at its last width).

**THE CURVED WINDSHIELD (the user's own design)** — the view is one
area split into three seamless panes: worldCanvasL | world+light+HUD |
worldCanvasR, all inside #cockpit-view; the controls strip FLOATS on
the view's bottom-centre (opaque console slab, border removed after the
CDP seam probe caught its 1px border column crossing the left pane).
The continuity law: ONE shared focal = viewW/2/tan(FOV_HALF) (the whole
view spans the 90° law — every pane renders at the same px-per-degree),
and the side cameras yaw by atan(centreW/2f) + atan(sideW/2f) — exactly
half the centre pane's span plus half the side pane's span, so what the
centre pane's edge shows the side pane's edge continues. At equal
thirds the yaw is ±36.87° and the windshield sees 110° (the curved
projection's bonus). `yawedCameraOf` rotates fwd/right about the up
axis; drawCockpitView/drawSideViewport take the shared focal. The CPU
path (no GPU world) keeps one full-width pane — the reference view.

**THE CRASH FLOOR'S GATE (the takeoff black band)** — the user's
screenshot at 3–4 k ft showed the −15° band painting black across the
climb: the far terrain box ends (now 160×160 cells, ±12.8 km) and the
sky BELOW −15° went black although the camera was flying. The floor is
the CRASH view: `CRASH_FLOOR_EYE_M = 25` (world-renderer.ts) gates it —
extras.z on GPU, the eye-height twin on CPU; a flying camera keeps the
haze band there instead (the world dissolves into fog at its far edge,
never a black band), a crashed one looks into darkness. Pixel-probed:
row 0.75 pre-horizon fog rgb(184,200,214), band gone at 3 k and 4.9 k ft.

**The drag stall** — three costs found and killed: fitCanvas reassigned
canvas.width EVERY frame (a full clear+realloc of the ~10 M-pixel HUD
surface at Retina) — now guarded, assign only on change; the layout
writes are guarded the same way (unchanged placement writes nothing);
the panes are absolute inside the view, so a drag reflows nothing but
the two grid tracks. The hang probe (bottom hidden → map hidden → both
hidden, twelve-step drags) answers every CDP evaluate in 0–4 ms with
zero console errors — the page never stops answering.

Verification: CDP sides probe — panes 234×3 at natural width, 384×3
wide, L/R luminance 96/88 vs main 123 (the world renders in the sides);
map hides at the floor and reopens tracking; sidebar hides under 10%
and reopens at its last width; the seam scan shows the 1px border column
(fixed). Takeoff probe to 4.9 k ft: the band rows read fog, not black.
npm test: 9 suites green (pipeline 236). Both tsconfigs clean.
