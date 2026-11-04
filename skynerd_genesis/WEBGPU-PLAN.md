# WebGPU pipeline design — skynerd-genesis

For the user's asks: simplify the terrain texture (planar checker), move
3D rendering to WebGPU, feature checkboxes by RESET — and take
`src/LightAndGeometry` as the structural inspiration: a SCENE GRAPH of
clumps, geometries and lights, rendered with a standard raster pipeline
(NOT a raytracer — no per-pixel ray casting in the 3D path).

## 0. Grounded facts (what this builds on)

- The 3D world renders ONLY in the cockpit squares (today: per-pixel
  CPU march + ImageData, 100–165 ms/frame — the framerate blocker).
  The big stage view is a from-sky 2D map — it stays Canvas2D.
- The world model (`land.ts` heights/biomes, flora, `cameraOf` pinhole
  math) is pure CPU TS. The renderer's contract is
  `DroneView → pixels`.
- Node 24 (the whole test story) has no `navigator.gpu`.
- `package.json` already calls this a "WebGPU port". This is how.
- The servo tensor is tiny gather/scatter loops — GPU win there is
  batching (waves), not single decisions.

## 1. The one-device principle

`src/pipeline/core/gpu-device.ts` — **GpuDevice**: acquires adapter/device once,
owns the queue, buffer arena, and `device.lost` recovery (recreate, or
notify owners to fall back). Knows nothing about rendering or tensors.
Two clients register against it:

- `GpuWorldRenderer` (this design — Graphic Designer's lane)
- `GpuTensorService` (Engineer's lane, §8 — plumbing reserved now)

Per frame the app picks `gpu if runtime.ready else cpu` — never both.

## 2. Architecture: change the executor, not the model

```
interface WorldRenderer {
  renderFrame(canvas, cam: Camera, features: RenderFeatures): void;
}
class CpuWorldRenderer implements WorldRenderer   // today's path, kept
class GpuWorldRenderer  implements WorldRenderer   // same contract
```

The world model, camera math, flora queries, feature flags stay
CPU-owned TS. Tests, restart-check, fpm-debug and any machine without
WebGPU use the CPU path — determinism and the headless goldens are
untouched (§7 covers how the probes stay honest when pixels move to
the GPU).

**Canvas stack** (a canvas has one context type): each cockpit square
becomes a WebGPU `worldCanvas` with a 2D `hudCanvas` layered above —
ADI, HUD ink, titles stay today's crisp 2D code, zero GPU cost. The
controls strip stays 2D entirely.

## 3. THE SCENE — the LightAndGeometry shape, rasterized

L&G separates THE SCENE (what exists) from THE CAMERA (how it is seen)
from THE TRACER (how pixels are produced). We adopt the scene shape and
replace their tracer with a raster pipeline — a simulation frame is a
scene of draws with predictable cost, not a screen of rays.

```
WorldScene                      // the scene graph (L&G SceneGraph)
 ├─ Camera                      // cameraOf's pinhole, as a VP matrix
 ├─ Light: Sun (SUN_DIR)        // ONE directional light (L&G Light)
 │    └─ corner-ray shadow field // dropShadow rays, at square granularity
 ├─ Clump: TerrainField         // the heightfield — not a mesh list
 │    └─ near + far grids       //   (the ground IS the geometry)
 ├─ Clump: Flora                // L&G Clump: spatial group w/ bounds
 │    ├─ Geometry: Tree         // instanced family (L&G Geometry)
 │    ├─ Geometry: House
 │    └─ Clump: Airfield        // tower + runway furniture
 └─ Clump: Sky                  // gradient + sun disc + clouds
```

### The terrain: QUADS GENERATED ON THE GO

The terrain is ONE INFINITE GEOMETRY rendered as a STREAM OF QUADS —
not a static mesh, not rays. A compute kernel (`TerrainQuadField`)
writes the quad instances around the camera EVERY FRAME into a buffer
(one quad = one instance record: corner x, y + cell size; the vertex
shader displaces the corners off the height texture — the same hermite
`heightAt` as the CPU path, in WGSL):

- **Near field** — 160×160 quads at 12 m (one checker cell at level 0)
  centred on the camera, ~1.9 km of full-fidelity ground.
- **Far field** — 112×112 quads at 64 m to ~7.2 km, the kernel SKIPPING
  the interior covered by the near field (a conditional in the kernel,
  on the camera data in the uniform — the pipeline's conditional-ops
  law at work). The far field's fragment fades to full haze before its
  edge, so the world dissolves into sky with no visible rim.

The grid slots are FIXED (two buffers, sized once); the kernel re-fills
them as the camera moves, sampling WRAPPED (`mod 8192` — the field's own
period, seamless at any position). Fixed cost, no cracks (no LOD
transition seams — the two cell sizes are depth-tested, near wins),
no index buffer juggling. Terrain = the quad stream; the buffer IS the
terrain.

### The index map (screen tiles → geometry indexes — no rays)

The visibility idea, inverted: instead of casting camera→square rays,
each instance PROJECTS ITS OWN screen-space footprint into a TILE GRID
(32 px tiles) in a compute kernel (`TileIndexMap`): the quad/tree's
world centre and radius → clip-space bounds → the tiles it overlaps —
appended (packed tile+instance u32, `atomicAdd` on per-tile counters)
to per-tile LISTS. Every conditional is a kernel branch on buffer data:

- outside the frustum / behind the camera → not appended, visibility
  flag 0;
- occluded → NOT tested here at all: the rasterizer's own depth buffer
  IS the occlusion answer (the ray idea's third case, for free).

The flags buffer feeds the draw: the vertex shader collapses an
invisible instance to a degenerate triangle (conditional on the flag in
the buffer — no CPU compaction pass, nothing read back). The per-tile
lists are the FORWARD+ index: the hook for later per-pixel queries
(screen-space shadow lookup, tile-binned transparency), and they are
the "pixels of the render surface → indexes of specific geometries"
map the design asked for, built entirely on GPU.

### The flora: biome INSTANCING generated on GPU

Trees stop being a CPU-collected list. A compute kernel
(`FloraInstanceField`) walks the planting lattice around the camera
(20 m cells): each thread reads the BIOME from the biome texture,
rolls the deterministic hash (the same `hash01` as `land.ts`, in WGSL),
and writes tree instances (position on the ground — kernel samples
the height texture — height, radius, tint) to an instance buffer.
Dense in forest, lone on grass, none on rock/water — the CPU rules,
executed on GPU. The draw is one instanced cone per kind (conifer /
round), tinted and scaled per instance.

**Variation via noise textures**: a 256² tileable value-noise texture
(generated once from `hash01`, uploaded at startup) is sampled in the
terrain fragment (per-quad colour jitter — grass and forest mottle,
water shimmer) and in the tree shader (canopy tint variation) — the
"variation from noise textures" layer, no assets, all procedural.

### Pass structure (KernelQueue: kernels queued, then render passes)

One `KernelQueue` per frame records the compute kernels IN ORDER, then
flushes them as ONE compute pass; the render passes follow in one
command encoder. Every conditional lives in a kernel, branching on the
buffers/uniforms it is handed — the CPU never reads results back.

| step | what | how |
|---|---|---|
| 0a. terrain quads | the near + far quad instances around the camera | COMPUTE kernel `TerrainQuadField` → instance buffers (near, far), every frame |
| 0b. flora instances | the biome-planted trees around the camera | COMPUTE kernel `FloraInstanceField` → tree instance buffer, every frame |
| 0c. the index map | screen-tile lists + visibility flags for every instance (terrain quads + trees) | COMPUTE kernel `TileIndexMap` — project bounds, append to per-tile lists (atomics), set flags |
| 0d. shadow field (opt.) | the CHECKER-SQUARE CORNER RAYS: 4 rays per square at the CURRENT displayed level, toward the sun | COMPUTE kernel, on demand when the displayed level changes, cached per level — zero per-frame cost (§6) |
| B. sky | fullscreen quad | gradient + sun disc, analytic |
| C. terrain | the quad stream, depth-tested | fragment: planar checker §4, biome tint + noise variation, roads/runway analytic, optional relief, optional corner-shadow fetch, haze to full before the far edge |
| D. objects | instanced geometries (trees: cone per kind; tower/houses/clouds stay 2D overlays v1) | the visibility flags collapse hidden instances in the vertex shader |
| E. post | glare veil | the 2D HUD canvas's existing additive veil, over the world canvas |

### Resources (tiny, uploaded once at startup)

| resource | format | size |
|---|---|---|
| height field | r32float 512² | 1 MB |
| biome map | r8uint 512² | 256 KB |
| noise (variation) | rgba8unorm 256², tileable value noise from `hash01`, uploaded once | 256 KB |
| frame uniform | one buffer: viewProj, camera axes, sun, features, tile grid | ~200 B |
| terrain instances | storage, near 160² + far 112² quad records (16 B each), refilled per frame | ~1 MB |
| flora instances | storage, tree records (32 B), regenerated per frame | ~512 KB |
| tile index | per-tile u32 counters + packed item list + visibility flags | ~1 MB |
| corner-shadow fields | r8unorm per level, 683² down to 21² (the 8192 m period at each level's cell size), computed on demand, cached | ~0.5 MB total |

### WGSL module layout

`src/pipeline/**` — WGSL as exported template strings, co-located with the classes that own them: no file
loader, no fetch, tsc stays the only build step (works in the dev
server and any bundler alike).

## 4. The texture simplification (both paths)

Per the user: the checkerboard becomes plain **top-planar u,v** — the
fragment's world (x, y). The footprint/box-filter/aniso machinery is
retired.

- Cell 12 m × 2^level. **The standing altitude law survives**: the
  level still comes from EYE HEIGHT (one map per doubling above 125 ft,
  ×1.12 brighter per level) — as TWO explicit mip fetches with a manual
  bracket blend, so the law's "literal copies, never a flat mean"
  holds on GPU exactly as on CPU.
- Anti-aliasing: hardware bilinear within each level + the mesh's
  continuous vertices — the grazing-angle shimmer dies because
  there are no footprint approximations left to be wrong.
- The CPU fallback keeps the analytic box-filter antiderivative
  (already proven) — its `sample()` loses the aniso-tap loop, which is
  the expensive part; the simplification alone buys a large CPU
  framerate win before WebGPU even lands.

## 5. Feature checkboxes (toolbar, next to RESET)

`RenderFeatures` — one type, read by the toolbar, passed to whichever
renderer is active (uniform on GPU, branch on CPU):

- `relief` — the normal·sun hill shading (off by default)
- `glare` — sun disc + glare veil (off by default)
- `haze` — the ground→sky fog blend (ON: it's the horizon dissolve)
- `sunShadows` — hill shadows via the checker-square corner-ray field, bilinearly interpolated (off by default; §3 pass A)
- `waterReflections` — reserved, later phase (off by default)

Height, water, checker, roads, objects are always on — they're the
world, not a shading property. Defaults live in `RenderFeatures`.

## 6. From LightAndGeometry — scene structure, not the tracer

Read `src/LightAndGeometry/Raytracer` (C++ template raytracer). What we
took and where it went:

- **SceneGraph / Clump / Geometry / Light / Camera separation** — the
  whole §3 shape. The scene OWNS what exists; the camera only views it.
  In our case: `WorldScene` holds the TerrainField clump (a heightfield,
  not a mesh list — the one place our scene differs from L&G's list-of-
  geometries, because the ground is a field), the Flora clump with its
  instanced geometry families, and ONE directional Light.
- **Clump = a spatial group with bounds** (their AABBox/Mesh traits;
  their `Clump` TODO for bounds is something we finish). Each clump
  answers "collect instances near this camera" — the flora's lattice
  queries become the clump's collect method, unchanged CPU logic.
- **Camera as a first-class scene citizen** — `cameraOf`'s pinhole
  becomes the camera's view-projection; their frustum-delta ray
  generation survives only as the SKY quad's fullscreen-corner lerp
  (no rays anywhere else).
- **Their LIGHT model — raytraced right where it pays.** L&G's
  `dropShadow` casts a ray from each lit point toward the light; a
  shadow MAP was a raster distortion of that idea. The user's hybrid
  restores the ray, but at CHECKER-SQUARE granularity instead of
  per-pixel: 4 corner rays per square at the current displayed level
  (altitude already selects the level — on the ground 12 m squares, at
  4000 ft 384 m squares — the shadow detail always matches the visible
  texture detail), each ray a short march toward the sun through the
  coarse-max cascade (hills cap at 305 m, sun at ~20° → a ray needs
  ~850 m of reach), the 4 results stored per corner and BILINEARLY
  interpolated across the square — smooth, soft shadows for the price
  of a texture fetch. And the field is STATIC (terrain and sun never
  move), so each level's corner-shadow field is computed once on
  demand by a WebGPU compute pass and cached — the per-frame cost of
  `sunShadows` is literally zero. Softness beyond the interpolation
  can come from L&G's penumbra kernel: each corner tests a few jittered
  rays and stores the blocked FRACTION. Objects keep their existing
  painted ground-shadows (the corner rays test terrain only).
- **Everything built programmatically — procedural included.** L&G
  composes its ENTIRE renderer in code — camera, frustum, threads,
  FSAA, sampling are constructor calls and template parameters, no
  scene-file format, no markup, no runtime loader. WebGPU is the same
  philosophy in TS: every pipeline, bind group, buffer and pass is
  constructed in code (`createRenderPipeline`, `createBindGroup`,
  command encoders) — the whole renderer is reviewable, greppable
  source in `src/pipeline/**`, nothing between the repo and the screen.
  And the programmatic freedom runs all the way into the shaders:
  procedural textures and custom effects are plain WGSL (noise,
  palettes, the checker itself) — our world stays precomputed CPU-side
  (height, biome, palettes) for bit-parity with the CPU reference, but
  anything new can be generated in-shader without a single asset. The
  scene graph likewise is plain TS objects: the headless harness can
  build and inspect the entire scene without a GPU, so the probes
  gate plain data.
- **Explicitly NOT adopted:** the per-pixel tracer, the 16-ray FSAA,
  the scanline/thread work partition (the GPU's warps ARE the
  scanlines now), and mesh-based terrain (ours is a field).

## 7. Migration phases, each gated

- **Phase 0 — runtime + canvas stack + sky.** WebGpuRuntime, canvas
  split per cockpit square, sky quad. Gate: identical sky; CPU fallback
  green; `device.lost` handled.
- **Phase 1 — terrain grids.** The two displaced grids + planar
  checker + biome tint + haze (features: haze only). Gate: no seams at
  the 2 km ring or the 8192 wrap; hills silhouettes match the CPU
  march's horizon; frame < 8 ms for a 1240×440 square.
- **Phase 2 — objects + probes.** Instanced families from the flora
  clumps, depth compositing; glare post. The PROBES: fpm-debug's pixel
  classifiers stay on the CPU fallback (it never goes away); new
  renderer-independent checks gate the mesh (vertex heights ==
  `heightAt` sampled values; checker levels vs the CPU pyramid;
  instance sets == flora query results).
- **Phase 3 — checkboxes + the corner-ray shadow field.** All five
  features in the toolbar, plumbed both paths; the shadow-field compute
  pass (per-level, on demand, cached) + the bilinear fetch in the
  terrain shader. Gate: restart-check gains stub checkboxes; hill
  shadows read correctly against SUN_DIR's azimuth; the CPU fallback's
  own corner-ray sweep (same rays, same interpolation) matches the
  GPU field within tolerance.
- **Phase 4 (Engineer) — tensors.** §8, after the runtime is proven.

Rollback at any phase = the mode pick; the CPU path never goes away.

## 8. Tensor space (Engineer's lane — reserved plumbing)

`GpuTensorService` on the same WebGpuRuntime: storage-buffer arena,
WGSL compute modules per op family, one command encoder per decision
tick, one small readback per decision (`mapAsync` — decisions run at
10 Hz per aircraft; latency irrelevant; the AI stays authoritative on
CPU).

Honest sizing: today's servo work is microseconds on CPU — the win is
BATCHING the waves scenario (8+ drones × decide per tick) and later
patch growth. The service interface lands with the renderer phases
(the runtime must not assume render-only clients); implementations
after. **Determinism law:** frozen goldens stay on CPU forever; the GPU
tensor path is opt-in, tolerance-parity-tested (1e-6), same seeds; any
re-freeze with GPU math is a conscious Engineer decision.

## 9. Risks & answers

- **WebGPU unavailable** (Node harness, older browsers) → CPU path,
  features still toggle; tests stay on it by default.
- **LOD seam at the ring boundary** → far grid drawn first, near wins
  on depth; seam sits where haze already dominates; ring distance is
  a tunable if it ever shows.
- **8192 wrap seams** → wrapped sampling in the vertex shader (the
  field's own period), same math as CPU `heightAt`.
- **Vertex-shader textureLoad** — standard WebGPU (r32float,
  `textureSampleLevel`/`textureLoad` in VS); no exotic features
  required anywhere in this design.
- **Driver float differences vs CPU** → CPU path remains the
  pixel-exact reference; GPU adds tolerance-based parity probes only.
- **Two agents, one repo** → all pipeline code is new files (`src/pipeline/**`);
  `cockpit.ts` gains only the canvas-stack split and the renderer
  pick; the Engineer reviews the runtime before their service builds
  on it.
