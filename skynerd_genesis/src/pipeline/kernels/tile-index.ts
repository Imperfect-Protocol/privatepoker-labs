/**
 * TileIndexMap — THE INDEX MAP (plan §3): the render surface is cut
 * into 32-px tiles, and a compute kernel PROJECTS every instance's
 * screen-space bounds into the tile grid, appending its index to each
 * tile's list (packed u32, atomics) and setting its visibility flag.
 * No camera→square rays: the projection replaces them, and the
 * rasterizer's own depth buffer answers occlusion for free.
 *
 * Every conditional is a kernel branch on buffer data: outside the
 * frustum / behind the camera → no tiles, flag 0; the draw's vertex
 * shader collapses flagged-off instances to degenerate triangles.
 * The per-tile lists are the forward+ index — the hook for later
 * per-pixel queries (plan phase 3's screen-space work).
 */
import type { GpuDevice } from "../core/gpu-device.js";
import { GpuBuffer } from "../core/kernel-queue.js";
import { WGSL_COMMON } from "../core/wgsl.js";
import type {
  WgpuBindGroup, WgpuBindGroupLayout, WgpuComputePipeline,
} from "../types.js";
import { BUF, STAGE } from "../types.js";

/** The tile edge, framebuffer pixels. */
export const TILE_PX = 32;
/** Entries per tile before overflow (extras dropped — the flags, not
 *  the lists, drive the draw). */
export const TILE_CAP = 64;
/** The tile grid's fixed capacity: counters + items buffers are sized
 *  once for any sane canvas (128×64 tiles at 4096×2048 px). */
const TILES_MAX = 8192;

// One index-map module per instance-record KIND (the record struct and
// the entry point differ; the tile-append law — indexOne — is shared,
// and the quads carry their own corner law on top of it).
function indexWgsl(record: string, entry: "quads" | "trees"): string {
  const bindings = /* wgsl */ `
@group(1) @binding(0) var<storage, read> instances : Instances;
@group(1) @binding(1) var<storage, read_write> flags : array<atomic<u32>>;
@group(1) @binding(2) var<storage, read> countBuf : u32;
@group(1) @binding(3) var<storage, read_write> tileCounters : array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> tileItems : array<u32>;
const TILE_CAP : u32 = ${TILE_CAP};
`;
  const indexOne = /* wgsl */ `
// One SMALL instance's world centre + radius -> screen box -> tile
// range -> flags + list appends (the trees' law: their footprint is
// honestly inside the radius). Every branch reads the buffers/uniform:
// the conditional-ops law of this pipeline.
fn indexOne(gi : u32, centre : vec3f, radius : f32) {
  atomicStore(&flags[gi], 0u);
  let clip = frame.viewProj * vec4f(centre, 1.0);
  if (clip.w <= 0.25) {
    return;                               // behind the camera
  }
  let ndc = clip.xyz / clip.w;
  // small things far off-screen end here; everything else falls to
  // the tile clamp, which handles the frustum edges exactly
  if (radius < 1.0 && (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2)) {
    return;
  }
  let W = frame.up.w;
  let H = frame.fwd.w;
  let sx = (ndc.x * 0.5 + 0.5) * W;
  let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * H;
  let rpx = frame.right.w * radius / clip.w;   // projected radius, px
  let tilePx = frame.params.w;
  let cols = i32(frame.params.y);
  let rows = i32(frame.params.z);
  let tx0 = clamp(i32(floor((sx - rpx) / tilePx)), 0, cols - 1);
  let tx1 = clamp(i32(floor((sx + rpx) / tilePx)), 0, cols - 1);
  let ty0 = clamp(i32(floor((sy - rpx) / tilePx)), 0, rows - 1);
  let ty1 = clamp(i32(floor((sy + rpx) / tilePx)), 0, rows - 1);
  if (tx0 > tx1 || ty0 > ty1 || sx + rpx < 0.0 || sx - rpx > W
      || sy + rpx < 0.0 || sy - rpx > H) {
    return;                               // outside the frustum
  }
  atomicStore(&flags[gi], 1u);            // visible: render it
  for (var ty = ty0; ty <= ty1; ty++) {
    for (var tx = tx0; tx <= tx1; tx++) {
      let tile = u32(ty) * u32(cols) + u32(tx);
      let slot = atomicAdd(&tileCounters[tile], 1u);
      if (slot < TILE_CAP) {
        tileItems[tile * TILE_CAP + slot] = gi;
      }
    }
  }
}
`;

// THE QUAD'S OWN LAW: terrain is the camera's ground — the quad it
// stands on (or any quad crossing the near plane) is visible from a
// fragment of its footprint even when its CENTRE is behind the eye.
// A quad dies only when ALL FOUR corners (at their terrain heights)
// sit behind the near plane; the screen box comes from the corners
// in front, so a straddling quad still lands its tiles.
const indexQuad = /* wgsl */ `
fn indexQuad(gi : u32, q : Quad) {
  atomicStore(&flags[gi], 0u);
  var corners = array<vec3f, 4>(
    vec3f(q.x, q.y, heightAt(vec2f(q.x, q.y))),
    vec3f(q.x + q.size, q.y, heightAt(vec2f(q.x + q.size, q.y))),
    vec3f(q.x, q.y + q.size, heightAt(vec2f(q.x, q.y + q.size))),
    vec3f(q.x + q.size, q.y + q.size,
          heightAt(vec2f(q.x + q.size, q.y + q.size))));
  var anyFront = false;
  var minW = 1e9;
  var farX0 = 1e9; var farX1 = -1e9;
  var farY0 = 1e9; var farY1 = -1e9;
  for (var i = 0; i < 4; i = i + 1) {
    let clip = frame.viewProj * vec4f(corners[i], 1.0);
    if (clip.w <= 0.25) {
      continue;                        // this corner is behind the eye
    }
    anyFront = true;
    minW = min(minW, clip.w);
    if (clip.w > 1.0) {                // a stable screen position
      let ndc = clip.xyz / clip.w;
      let sx = (ndc.x * 0.5 + 0.5) * frame.up.w;
      let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * frame.fwd.w;
      farX0 = min(farX0, sx); farX1 = max(farX1, sx);
      farY0 = min(farY0, sy); farY1 = max(farY1, sy);
    }
  }
  if (!anyFront) {
    return;                            // the whole footprint is behind
  }
  let W = frame.up.w;
  let H = frame.fwd.w;
  // no corner far enough for a clean projection: the camera is IN the
  // quad's footprint at the near plane — it sweeps the whole view
  if (farX1 < farX0) {
    farX0 = 0.0; farX1 = W; farY0 = 0.0; farY1 = H;
  } else {
    // the clipped polygon also carries the near-plane crossings of
    // the quad's edges — pad the box by the footprint's projected
    // radius at the nearest front corner, the honest bound on that
    // sweep, so a straddling quad is never culled off-screen
    let pad = frame.right.w * q.size * 0.75 / max(minW, 0.25);
    farX0 -= pad; farX1 += pad; farY0 -= pad; farY1 += pad;
  }
  let tilePx = frame.params.w;
  let cols = i32(frame.params.y);
  let rows = i32(frame.params.z);
  let tx0 = clamp(i32(floor(farX0 / tilePx)), 0, cols - 1);
  let tx1 = clamp(i32(floor(farX1 / tilePx)), 0, cols - 1);
  let ty0 = clamp(i32(floor(farY0 / tilePx)), 0, rows - 1);
  let ty1 = clamp(i32(floor(farY1 / tilePx)), 0, rows - 1);
  if (tx0 > tx1 || ty0 > ty1 || farX1 < 0.0 || farX0 > W
      || farY1 < 0.0 || farY0 > H) {
    return;                            // the whole box is off-screen
  }
  atomicStore(&flags[gi], 1u);
  for (var ty = ty0; ty <= ty1; ty++) {
    for (var tx = tx0; tx <= tx1; tx++) {
      let tile = u32(ty) * u32(cols) + u32(tx);
      let slot = atomicAdd(&tileCounters[tile], 1u);
      if (slot < TILE_CAP) {
        tileItems[tile * TILE_CAP + slot] = gi;
      }
    }
  }
}
`;
  const quadEntry = /* wgsl */ `
@compute @workgroup_size(64)
fn indexQuads(@builtin(global_invocation_id) gid : vec3u) {
  let gi = gid.x;
  if (gi >= countBuf) {
    return;
  }
  indexQuad(gi, instances.quads[gi]);
}
`;
  const treeEntry = /* wgsl */ `
@compute @workgroup_size(64)
fn indexTrees(@builtin(global_invocation_id) gid : vec3u) {
  let gi = gid.x;
  let n = min(countBuf, TREE_MAX);
  if (gi >= n) {
    if (gi < TREE_MAX) {
      atomicStore(&flags[gi], 0u);        // stale slot: never renders
    }
    return;
  }
  let tr = instances.trees[gi];
  let centre = vec3f(tr.x, tr.y, tr.z + tr.h * 0.5);
  indexOne(gi, centre, max(tr.r * 1.5, tr.h) * 0.6);
}
`;
  const entries = entry === "quads" ? indexQuad + quadEntry : treeEntry;
  return WGSL_COMMON + record + bindings + indexOne + entries;
}

const QUAD_RECORD = /* wgsl */ `
struct Quad { x: f32, y: f32, size: f32, pad: f32 };
struct Instances { quads: array<Quad> };
`;
const TREE_RECORD = /* wgsl */ `
struct Tree { x: f32, y: f32, z: f32, h: f32,
              r: f32, shade: f32, kind: f32, pad: f32 };
struct Instances { trees: array<Tree> };
const TREE_MAX : u32 = 8192u;
`;

/** The index map: one tile grid, shared by every instance set in the
 *  scene (terrain quads, trees) — each set registers through a bind
 *  group and the clumps queue the dispatches. */
export class TileIndexMap {
  readonly quadIndexPipeline: WgpuComputePipeline;
  readonly treeIndexPipeline: WgpuComputePipeline;
  private readonly layout: WgpuBindGroupLayout;
  private readonly counters: GpuBuffer;
  private readonly items: GpuBuffer;
  private readonly zeros: Uint32Array;

  constructor(private readonly gpu: GpuDevice,
              group0Layout: WgpuBindGroupLayout) {
    const dev = gpu.device;
    this.layout = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: STAGE.COMPUTE,
          buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: STAGE.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 2, visibility: STAGE.COMPUTE,
          buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: STAGE.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 4, visibility: STAGE.COMPUTE,
          buffer: { type: "storage" } },
      ],
    });
    const layout = dev.createPipelineLayout({
      bindGroupLayouts: [group0Layout, this.layout],
    });
    this.quadIndexPipeline = dev.createComputePipeline({
      layout,
      compute: {
        module: dev.createShaderModule({ code: indexWgsl(QUAD_RECORD, "quads") }),
        entryPoint: "indexQuads",
      },
    });
    this.treeIndexPipeline = dev.createComputePipeline({
      layout,
      compute: {
        module: dev.createShaderModule({ code: indexWgsl(TREE_RECORD, "trees") }),
        entryPoint: "indexTrees",
      },
    });
    // the tile grid's own storage: counters (cleared every frame from
    // CPU — a zero-fill, never a read-back) + the packed item lists
    this.counters = GpuBuffer.create(gpu, TILES_MAX * 4,
                                     BUF.STORAGE | BUF.COPY_DST);
    this.items = GpuBuffer.create(gpu, TILES_MAX * TILE_CAP * 4,
                                  BUF.STORAGE);
    this.zeros = new Uint32Array(TILES_MAX);
  }

  /** The bind group for one quad instance set (terrain near/far). */
  quadGroup(instances: GpuBuffer, flags: GpuBuffer,
            count: number): WgpuBindGroup {
    const countBuf = GpuBuffer.seeded(this.gpu, new Uint32Array([count]),
                                      BUF.STORAGE);
    return this.group(instances, flags, countBuf);
  }

  /** The bind group for the tree set (count comes from the flora
   *  kernel's own buffer — read live in the kernel). */
  treeGroup(instances: GpuBuffer, flags: GpuBuffer,
            countBuf: GpuBuffer): WgpuBindGroup {
    return this.group(instances, flags, countBuf);
  }

  /** Clear the per-tile counters for a fresh frame (a plain zero
   *  write; the lists are overwritten by slot order). */
  clear(): void {
    this.gpu.queue.writeBuffer(this.counters.raw, 0, this.zeros);
  }

  private group(instances: GpuBuffer, flags: GpuBuffer,
                countBuf: GpuBuffer): WgpuBindGroup {
    return this.gpu.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: instances.raw } },
        { binding: 1, resource: { buffer: flags.raw } },
        { binding: 2, resource: { buffer: countBuf.raw } },
        { binding: 3, resource: { buffer: this.counters.raw } },
        { binding: 4, resource: { buffer: this.items.raw } },
      ],
    });
  }
}
