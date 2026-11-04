/**
 * TerrainQuadField — the terrain clump: ONE INFINITE GEOMETRY rendered
 * as a STREAM OF QUADS GENERATED ON THE GO (plan §3). A compute kernel
 * rewrites the quad instance buffers around the camera EVERY FRAME —
 * a near field at checker resolution, a far field at 160 m cells whose
 * interior (covered by the near field) the kernel SKIPS as a
 * conditional on the camera data. The buffer IS the terrain.
 *
 * The draw is the unit quad, instanced: the vertex shader displaces
 * each corner off the height texture (the shared WGSL heightAt — the
 * CPU path's exact hermite), collapses instances the index map flagged
 * invisible, and the fragment shader paints the ground's full law
 * (checker, biome, noise variation, roads, runway, relief, haze).
 */
import { WGSL_CHECKER, WGSL_COMMON } from "../core/wgsl.js";
import type { GpuDevice } from "../core/gpu-device.js";
import { GpuBuffer } from "../core/kernel-queue.js";
import type { KernelQueue } from "../core/kernel-queue.js";
import { QuadGeometry } from "../geometries/quad.js";
import { RenderClump } from "../scene/scene.js";
import type {
  WgpuBindGroup, WgpuBindGroupLayout, WgpuComputePipeline,
  WgpuRenderPassEncoder, WgpuRenderPipeline,
} from "../types.js";
import { BUF, STAGE } from "../types.js";
import type { TileIndexMap } from "./tile-index.js";

// --- the field's shape (plan §3: near + far, fixed cost) -------------

export const NEAR_N = 160;        // near grid slots per side
export const NEAR_CELL = 12.0;    // one checker cell at level 0, m
export const FAR_N = 160;         // far grid slots per side
export const FAR_CELL = 160.0;   // far cells, m
/** The near field's half-extent, m — the far field's interior skip. */
export const NEAR_HALF = (NEAR_N * NEAR_CELL) / 2.0;   // 960 m
/** The far field's half-extent — where the ground fully melts to fog. */
export const FAR_HALF = (FAR_N * FAR_CELL) / 2.0;      // 10240 m

/** One field's buffers + the three groups that view them. */
interface QuadSlot {
  instances: GpuBuffer;
  flags: GpuBuffer;
  count: number;
  genGroup: WgpuBindGroup;
  indexGroup: WgpuBindGroup;
  drawGroup: WgpuBindGroup;
}

const GEN_WGSL = /* wgsl */ `
struct Quad { x: f32, y: f32, size: f32, pad: f32 };
struct GenConsts { cols: i32, cell: f32, half: f32, nearHalf: f32,
                   nearCell: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(1) @binding(0) var<storage, read_write> quads : array<Quad>;
@group(1) @binding(1) var<storage, read> consts : GenConsts;

// One thread per grid slot: place the quad on the WORLD-SNAPPED grid
// around the camera. The origin snaps to the cell lattice — the grid
// re-anchors only when the camera crosses a cell boundary, so every
// quad keeps its world rectangle frame to frame and the tessellation
// never swims over the height field (the ocean-wave artifact). The
// far field's interior skip is a CONDITIONAL on the consts — the
// slots the near field covers are left unwritten.
@compute @workgroup_size(8, 8)
fn terrainQuadGen(@builtin(global_invocation_id) gid : vec3u) {
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  if (ix >= consts.cols || iy >= consts.cols) {
    return;
  }
  let ox = floor(frame.camPos.x / consts.cell) * consts.cell;
  let oy = floor(frame.camPos.y / consts.cell) * consts.cell;
  let wx = ox + (f32(ix) - consts.half) * consts.cell;
  let wy = oy + (f32(iy) - consts.half) * consts.cell;
  // skip only quads FULLY inside the near field's world coverage:
  // straddling quads stay (no holes); the near field is the finer
  // surface and wins the depth test where they overlap
  if (consts.nearHalf > 0.0) {
    let nx = floor(frame.camPos.x / consts.nearCell) * consts.nearCell;
    let ny = floor(frame.camPos.y / consts.nearCell) * consts.nearCell;
    if (wx >= nx - consts.nearHalf && wx + consts.cell <= nx + consts.nearHalf
        && wy >= ny - consts.nearHalf && wy + consts.cell <= ny + consts.nearHalf) {
      return;
    }
  }
  let index = u32(iy) * u32(consts.cols) + u32(ix);
  quads[index] = Quad(wx, wy, consts.cell, 0.0);
}
`;

const DRAW_WGSL = /* wgsl */ `
struct TerrainVOut {
  @builtin(position) pos : vec4f,
  @location(0) world : vec3f,
};
@group(1) @binding(0) var<storage, read> flags : array<u32>;

// The quad: corners from the unit geometry, placement + size from the
// instance record, height from the shared heightAt — and the index
// map's flag: an invisible instance collapses to a degenerate triangle
// (a conditional on buffer data, in the shader, no CPU round trip).
@vertex
fn terrainVs(@location(0) corner : vec3f,
             @location(1) inst : vec3f,
             @builtin(instance_index) ii : u32) -> TerrainVOut {
  var out : TerrainVOut;
  if (flags[ii] == 0u) {
    out.pos = vec4f(0.0, 0.0, 0.5, 1.0);      // zero-area: clipped
    out.world = vec3f(0.0);
    return out;
  }
  let wx = inst.x + corner.x * inst.z;
  let wy = inst.y + corner.y * inst.z;
  let h = heightAt(vec2f(wx, wy));
  out.world = vec3f(wx, wy, h);
  out.pos = frame.viewProj * vec4f(out.world, 1.0);
  return out;
}

@fragment
fn terrainFs(v : TerrainVOut) -> @location(0) vec4f {
  let p = v.world;
  let t = distance(p, frame.camPos.xyz);
  // the biome that owns this ground point (the map's own texel)
  let bx = (i32(floor(p.x / TEXEL)) & (TEX - 1));
  let by = (i32(floor(p.y / TEXEL)) & (TEX - 1));
  let biome = i32(textureLoad(biomeTex, vec2<i32>(bx, by), 0).r);
  // the pixel's ground footprint (the CPU path's per-pixel steps, by
  // derivative) drives every box filter below — the exact CPU filter
  let wU = max(fwidth(p.x), 1e-4);
  let wV = max(fwidth(p.y), 1e-4);
  var col = groundColour(p.xy, wU, wV, t, biome, frame.camPos.w);
  // VARIATION from the noise texture: the land mottles, the water
  // shimmers — per-biome amounts, all procedural
  let n = textureSample(noiseTex, noiseSamp, p.xy / 512.0);
  if (biome == 4) {
    col *= 0.94 + 0.12 * n.y;
  } else {
    col *= 0.90 + 0.20 * n.x;
  }
  // relief (optional): the surface normal against the sun, normalised
  // so the flat plain keeps its palette exactly
  if (frame.features.y > 0.5) {
    let nrm = normalAt(p.xy);
    let ndl = max(0.0, dot(nrm, frame.sunDir.xyz));
    let flat = SHADE_AMBIENT + SHADE_SUN * frame.sunDir.z;
    col *= (SHADE_AMBIENT + SHADE_SUN * ndl) / flat;
  }
  // the mood's ambient: the whole composed ground (checker, roads,
  // runway, relief) dims with the hour — before the haze, which
  // always melts toward the mood's own fog
  col *= frame.sunTint.w;
  // THE HOUR'S GRADE: the ground wears the mood's colour (mean-1
  // filter — tint, not darken), before the haze melts it to the fog
  col *= gradeFilter(frame.grade.w);
  // haze: the ground melts to full fog BEFORE the far field's edge,
  // so the world dissolves into the sky with no visible rim
  if (frame.features.x > 0.5) {
    let mixF = clamp((t - ${FAR_HALF.toFixed(1)} * 0.55)
                     / (${FAR_HALF.toFixed(1)} * 0.40), 0.0, 1.0);
    col = mix(col, frame.fog.xyz / 255.0, mixF);
  }
  // THE PILOT'S IRIS: the tightened eye dims the ground with it
  col *= frame.extras.y;
  return vec4f(col, 1.0);
}
`;

export class TerrainClump extends RenderClump {
  readonly nearCount = NEAR_N * NEAR_N;
  readonly farCount = FAR_N * FAR_N;
  private readonly quad = new QuadGeometry();
  private readonly genPipeline: WgpuComputePipeline;
  private readonly renderPipeline: WgpuRenderPipeline;
  private readonly genLayout: WgpuBindGroupLayout;
  private readonly drawLayout: WgpuBindGroupLayout;
  private readonly near: QuadSlot;
  private readonly far: QuadSlot;

  constructor(private readonly gpu: GpuDevice,
              canvasFormat: string,
              private readonly group0: WgpuBindGroup,
              group0Layout: WgpuBindGroupLayout,
              private readonly tiles: TileIndexMap) {
    super("terrain");
    const dev = gpu.device;

    // --- the generation pipeline (compute) --------------------------
    this.genLayout = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: STAGE.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 1, visibility: STAGE.COMPUTE,
          buffer: { type: "read-only-storage" } },
      ],
    });
    this.genPipeline = dev.createComputePipeline({
      layout: dev.createPipelineLayout({
        bindGroupLayouts: [group0Layout, this.genLayout],
      }),
      compute: {
        module: dev.createShaderModule({
          code: WGSL_COMMON + GEN_WGSL,
        }),
        entryPoint: "terrainQuadGen",
      },
    });

    // --- the draw pipeline (raster) ----------------------------------
    this.drawLayout = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: STAGE.VERTEX,
          buffer: { type: "read-only-storage" } },
      ],
    });
    this.renderPipeline = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({
        bindGroupLayouts: [group0Layout, this.drawLayout],
      }),
      vertex: {
        module: dev.createShaderModule({
          code: WGSL_COMMON + WGSL_CHECKER + DRAW_WGSL,
        }),
        entryPoint: "terrainVs",
        buffers: [
          { arrayStride: this.quad.stride, attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
          ] },
          { arrayStride: 16, stepMode: "instance", attributes: [
            { shaderLocation: 1, offset: 0, format: "float32x3" },
          ] },
        ],
      },
      fragment: {
        module: dev.createShaderModule({
          code: WGSL_COMMON + WGSL_CHECKER + DRAW_WGSL,
        }),
        entryPoint: "terrainFs",
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    // --- the two fields' buffers + groups ----------------------------
    this.near = this.fieldSlot(NEAR_N, NEAR_CELL, 0.0);
    this.far = this.fieldSlot(FAR_N, FAR_CELL, NEAR_HALF);
  }

  /** One field's instances, flags, and the three groups that view
   *  them: generation (write), the tile index (read + flag write),
   *  the draw (flag read). One buffer set, three views, one owner. */
  private fieldSlot(n: number, cell: number,
                    nearHalf: number): QuadSlot {
    const count = n * n;
    const instances = GpuBuffer.create(this.gpu, count * 16,
                                       BUF.STORAGE | BUF.VERTEX
                                       | BUF.COPY_SRC);
    const flags = GpuBuffer.create(this.gpu, count * 4,
                                    BUF.STORAGE | BUF.COPY_SRC);
    // half = n/2: the grid spans exactly -half*cell..+half*cell,
    // so the near field's edges meet the far field's kept ring
    // with no slit (the far skip test uses the same half-extent).
    // cols is an i32 in the shader: seeded through a u32 view so
    // the kernel reads the integer bits, not a float pattern
    const consts = new ArrayBuffer(32);
    const words = new Uint32Array(consts);
    const floats = new Float32Array(consts);
    words[0] = n;                    // cols (i32 in WGSL)
    floats[1] = cell;
    floats[2] = n / 2.0;
    floats[3] = nearHalf;            // 0.0 for near: skip disabled
    floats[4] = NEAR_CELL;           // the near lattice (far skip)
    const constsBuf = GpuBuffer.seeded(this.gpu, new Uint8Array(consts),
                                       BUF.STORAGE);
    const genGroup = this.gpu.device.createBindGroup({
      layout: this.genLayout,
      entries: [
        { binding: 0, resource: { buffer: instances.raw } },
        { binding: 1, resource: { buffer: constsBuf.raw } },
      ],
    });
    return {
      instances, flags, count,
      genGroup,
      indexGroup: this.tiles.quadGroup(instances, flags, count),
      drawGroup: this.gpu.device.createBindGroup({
        layout: this.drawLayout,
        entries: [
          { binding: 0, resource: { buffer: flags.raw } },
        ],
      }),
    };
  }

  override enqueueKernels(queue: KernelQueue): void {
    const genWg = Math.ceil(NEAR_N / 8);
    const genWgF = Math.ceil(FAR_N / 8);
    // generation, then each field's index-map pass — all queued in
    // order, flushed as one compute pass
    queue.enqueue(this.genPipeline, [this.group0, this.near.genGroup],
                  [genWg, genWg, 1]);
    queue.enqueue(this.genPipeline, [this.group0, this.far.genGroup],
                  [genWgF, genWgF, 1]);
    const inN = Math.ceil(this.nearCount / 64);
    const inF = Math.ceil(this.farCount / 64);
    // phase 1: the index pass reads what the gen pass wrote — the
    // pass boundary between phases is the ordering barrier
    queue.enqueue(this.tiles.quadIndexPipeline,
                  [this.group0, this.near.indexGroup],
                  [inN, 1, 1], 1);
    queue.enqueue(this.tiles.quadIndexPipeline,
                  [this.group0, this.far.indexGroup],
                  [inF, 1, 1], 1);
  }

  override drawPass(pass: WgpuRenderPassEncoder): void {
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.group0);
    const vertexBuf = this.quad.vertexBufferOf(this.gpu);
    pass.setVertexBuffer(0, vertexBuf.raw);
    // the far field first — the near field wins the depth test where
    // the boundary quads overlap, and it is the finer surface
    const indices = this.quad.indexBufferOf(this.gpu);
    if (indices !== null) {
      pass.setIndexBuffer(indices.raw, "uint32");
    }
    pass.setBindGroup(1, this.far.drawGroup);
    pass.setVertexBuffer(1, this.far.instances.raw);
    pass.drawIndexed(this.quad.indexCount, this.far.count);
    pass.setBindGroup(1, this.near.drawGroup);
    pass.setVertexBuffer(1, this.near.instances.raw);
    pass.drawIndexed(this.quad.indexCount, this.near.count);
  }
}
