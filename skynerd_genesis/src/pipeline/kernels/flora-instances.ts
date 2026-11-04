/**
 * FloraClump — the biome's INSTANCING, generated on GPU (plan §3):
 * a compute kernel walks the planting lattice around the camera (20 m
 * cells), reads the BIOME from the biome texture, rolls the same
 * deterministic hash01 as land.ts in WGSL, and writes tree instances
 * straight to a buffer — dense in forest, lone on grass, none on
 * rock/water/roads/runway. The CPU rules, executed on GPU.
 *
 * The draw is one instanced cone: conifers narrow and tall, round
 * canopies wide and squashed (kind from the instance), tinted by the
 * instance's hash and the noise texture's variation, index-map flags
 * collapsing the hidden ones.
 */
import type { GpuDevice } from "../core/gpu-device.js";
import { GpuBuffer } from "../core/kernel-queue.js";
import type { KernelQueue } from "../core/kernel-queue.js";
import { WGSL_CHECKER, WGSL_COMMON } from "../core/wgsl.js";
import { ConeGeometry } from "../geometries/cone.js";
import { RenderClump } from "../scene/scene.js";
import type {
  WgpuBindGroup, WgpuBindGroupLayout, WgpuComputePipeline,
  WgpuRenderPassEncoder, WgpuRenderPipeline,
} from "../types.js";
import { BUF, STAGE } from "../types.js";
import type { TileIndexMap } from "./tile-index.js";
import { FAR_HALF } from "./terrain-quads.js";   // the shared haze law's extent

/** Tree instance slots (the dense-forest worst case in the region). */
export const TREE_MAX = 8192;
/** The planting region's half-extent around the camera, m. */
const REGION = 768.0;
/** The planting lattice, m (land.ts SP). */
const LATTICE = 20.0;
const GEN_WGSL = /* wgsl */ `
struct Tree { x: f32, y: f32, z: f32, h: f32,
              r: f32, shade: f32, kind: f32, pad: f32 };
struct GenConsts { region : f32, lattice : f32, pad0 : u32, pad1 : u32 };
@group(1) @binding(0) var<storage, read_write> trees : array<Tree>;
@group(1) @binding(1) var<storage, read_write> treeCount : atomic<u32>;
@group(1) @binding(2) var<storage, read> consts : GenConsts;
const TREE_MAX : u32 = ${TREE_MAX};

// the tree density per biome (land.ts TREE_DENSITY):
// forest .85, grass .02, dry .008, rocky and water bare
const TREE_DENSITY = array<f32, 5>(0.02, 0.008, 0.0, 0.85, 0.0);

fn biomeAt(p : vec2f) -> u32 {
  let bx = i32(floor(p.x / TEXEL)) & (TEX - 1);
  let by = i32(floor(p.y / TEXEL)) & (TEX - 1);
  return textureLoad(biomeTex, vec2<i32>(bx, by), 0).r;
}

// nothing plants on the airfield's clear zone or the roads (land.ts
// Flora.excluded — the same numbers)
fn excluded(x : f32, y : f32) -> bool {
  if (x > -360.0 && x < 1560.0 && abs(y) < 125.0) {
    return true;
  }
  let d = roadDistances(vec2f(x, y));
  return abs(d.x) < 13.0 || abs(d.y) < 13.0;
}

@compute @workgroup_size(8, 8)
fn floraGen(@builtin(global_invocation_id) gid : vec3u) {
  let lx = i32(floor((frame.camPos.x - consts.region) / consts.lattice))
           + i32(gid.x);
  let ly = i32(floor((frame.camPos.y - consts.region) / consts.lattice))
           + i32(gid.y);
  // the cell's centre picks the biome, exactly as the CPU lattice does
  let centre = vec2f((f32(lx) + 0.5) * consts.lattice,
                     (f32(ly) + 0.5) * consts.lattice);
  let biome = biomeAt(centre);
  if (hash01v(lx, ly, 1) >= TREE_DENSITY[biome]) {
    return;
  }
  let x = (f32(lx) + 0.15 + 0.7 * hash01v(lx, ly, 2)) * consts.lattice;
  let y = (f32(ly) + 0.15 + 0.7 * hash01v(lx, ly, 3)) * consts.lattice;
  let x0 = frame.camPos.x - consts.region;
  let x1 = frame.camPos.x + consts.region;
  let y0 = frame.camPos.y - consts.region;
  let y1 = frame.camPos.y + consts.region;
  if (x < x0 || x > x1 || y < y0 || y > y1 || excluded(x, y)) {
    return;
  }
  if (biomeAt(vec2f(x, y)) == 4u) {
    return;                                // never a tree in the water
  }
  let slot = atomicAdd(&treeCount, 1u);
  if (slot >= TREE_MAX) {
    return;                                // the region overflowed
  }
  let conifer = hash01v(lx, ly, 6) < 0.45;
  trees[slot] = Tree(
    x, y,
    heightAt(vec2f(x, y)),
    7.0 + 4.0 * hash01v(lx, ly, 4),
    3.0 + 1.5 * hash01v(lx, ly, 5),
    hash01v(lx, ly, 7),
    select(0.0, 1.0, conifer),
    0.0);
}
`;

const DRAW_WGSL = /* wgsl */ `
struct TreeVOut {
  @builtin(position) pos : vec4f,
  @location(0) world : vec3f,
  @location(1) normal : vec3f,
  @location(2) shade : f32,
};
@group(1) @binding(0) var<storage, read> flags : array<u32>;

@vertex
fn treeVs(@location(0) p : vec3f,
          @location(1) n : vec3f,
          @location(2) a : vec4f,   // x, y, z, h
          @location(3) b : vec4f,   // r, shade, kind, pad
          @builtin(instance_index) ii : u32) -> TreeVOut {
  var out : TreeVOut;
  if (flags[ii] == 0u) {
    out.pos = vec4f(0.0, 0.0, 0.5, 1.0);
    out.world = vec3f(0.0);
    out.normal = vec3f(0.0, 0.0, 1.0);
    out.shade = 0.0;
    return out;
  }
  // the two habits: conifers narrow and tall, round canopies wide
  // and squashed — the instance's own dice (land.ts's kinds)
  let conifer = b.z >= 0.5;
  let sr = select(b.x * 1.35, b.x * 0.85, conifer);
  let sh = select(a.w * 0.7, a.w, conifer);
  out.world = a.xyz + vec3f(p.x * sr, p.y * sr, p.z * sh);
  out.normal = n;
  out.shade = b.y;
  out.pos = frame.viewProj * vec4f(out.world, 1.0);
  return out;
}

@fragment
fn treeFs(v : TreeVOut) -> @location(0) vec4f {
  let t = distance(v.world, frame.camPos.xyz);
  // the canopy: two greens mixed by the instance's hash
  var col = mix(vec3f(0.10, 0.17, 0.07), vec3f(0.15, 0.26, 0.10),
                v.shade);
  // VARIATION from the noise texture — every canopy its own mottle
  let n = textureSample(noiseTex, noiseSamp, v.world.xy / 512.0);
  col *= 0.80 + 0.40 * n.x;
  // the one sun's lambert, flat-shaded per cone facet
  let ndl = max(0.0, dot(normalize(v.normal), frame.sunDir.xyz));
  col *= 0.55 + 0.45 * ndl;
  // the mood's ambient: the forest dims with the hour, like its ground
  col *= frame.sunTint.w;
  // THE HOUR'S GRADE: the forest wears its ground's colour filter
  col *= gradeFilter(frame.grade.w);
  // the same haze law as the terrain: the forest melts with its ground
  if (frame.features.x > 0.5) {
    let mixF = clamp((t - ${FAR_HALF.toFixed(1)} * 0.55)
                     / (${FAR_HALF.toFixed(1)} * 0.40), 0.0, 1.0);
    col = mix(col, frame.fog.xyz / 255.0, mixF);
  }
  // THE PILOT'S IRIS: the forest dims with its ground
  col *= frame.extras.y;
  return vec4f(col, 1.0);
}
`;

export class FloraClump extends RenderClump {
  private readonly cone = new ConeGeometry(8);
  private readonly genPipeline: WgpuComputePipeline;
  private readonly renderPipeline: WgpuRenderPipeline;
  private readonly genLayout: WgpuBindGroupLayout;
  private readonly drawLayout: WgpuBindGroupLayout;
  readonly treeCountBuf: GpuBuffer;
  private readonly genGroup: WgpuBindGroup;
  private readonly indexGroup: WgpuBindGroup;
  private readonly drawGroup: WgpuBindGroup;
  private readonly instances: GpuBuffer;
  private readonly flags: GpuBuffer;
  private readonly zeroWord = new Uint32Array(1);

  constructor(private readonly gpu: GpuDevice,
              canvasFormat: string,
              private readonly group0: WgpuBindGroup,
              group0Layout: WgpuBindGroupLayout,
              private readonly tiles: TileIndexMap) {
    super("flora");
    const dev = gpu.device;

    // --- generation (compute) ----------------------------------------
    this.genLayout = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: STAGE.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 1, visibility: STAGE.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 2, visibility: STAGE.COMPUTE,
          buffer: { type: "read-only-storage" } },
      ],
    });
    this.genPipeline = dev.createComputePipeline({
      layout: dev.createPipelineLayout({
        bindGroupLayouts: [group0Layout, this.genLayout],
      }),
      compute: {
        module: dev.createShaderModule({
          code: WGSL_COMMON + WGSL_CHECKER + GEN_WGSL,
        }),
        entryPoint: "floraGen",
      },
    });

    // --- draw (raster) ------------------------------------------------
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
          code: WGSL_COMMON + DRAW_WGSL,
        }),
        entryPoint: "treeVs",
        buffers: [
          { arrayStride: this.cone.stride, attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
          ] },
          { arrayStride: 32, stepMode: "instance", attributes: [
            { shaderLocation: 2, offset: 0, format: "float32x4" },
            { shaderLocation: 3, offset: 16, format: "float32x4" },
          ] },
        ],
      },
      fragment: {
        module: dev.createShaderModule({
          code: WGSL_COMMON + DRAW_WGSL,
        }),
        entryPoint: "treeFs",
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    // --- the instance set ---------------------------------------------
    this.instances = GpuBuffer.create(gpu, TREE_MAX * 32,
                                      BUF.STORAGE | BUF.VERTEX);
    this.flags = GpuBuffer.create(gpu, TREE_MAX * 4, BUF.STORAGE);
    this.treeCountBuf = GpuBuffer.create(gpu, 4, BUF.STORAGE
                                            | BUF.COPY_DST);
    const consts = new Float32Array([REGION, LATTICE, 0, 0]);
    const constsBuf = GpuBuffer.seeded(gpu, consts, BUF.STORAGE);
    this.genGroup = dev.createBindGroup({
      layout: this.genLayout,
      entries: [
        { binding: 0, resource: { buffer: this.instances.raw } },
        { binding: 1, resource: { buffer: this.treeCountBuf.raw } },
        { binding: 2, resource: { buffer: constsBuf.raw } },
      ],
    });
    this.indexGroup = tiles.treeGroup(this.instances, this.flags,
                                      this.treeCountBuf);
    this.drawGroup = dev.createBindGroup({
      layout: this.drawLayout,
      entries: [
        { binding: 0, resource: { buffer: this.flags.raw } },
      ],
    });
  }

  override enqueueKernels(queue: KernelQueue): void {
    // reset the count (a zero write; the kernel counts live), then
    // plant, then index — queued in order, one compute pass
    this.gpu.queue.writeBuffer(this.treeCountBuf.raw, 0, this.zeroWord);
    const cells = Math.ceil((2.0 * REGION) / LATTICE) + 2;
    const wg = Math.ceil(cells / 8);
    queue.enqueue(this.genPipeline, [this.group0, this.genGroup], [wg, wg, 1]);
    const idxWg = Math.ceil(TREE_MAX / 64);
    // phase 1: the tree index reads the trees the gen pass planted
    queue.enqueue(this.tiles.treeIndexPipeline,
                  [this.group0, this.indexGroup],
                  [idxWg, 1, 1], 1);
  }

  override drawPass(pass: WgpuRenderPassEncoder): void {
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.group0);
    const indices = this.cone.indexBufferOf(this.gpu);
    if (indices !== null) {
      pass.setIndexBuffer(indices.raw, "uint32");
    }
    pass.setVertexBuffer(0, this.cone.vertexBufferOf(this.gpu).raw);
    pass.setVertexBuffer(1, this.instances.raw);
    pass.setBindGroup(1, this.drawGroup);
    // every slot draws; the flags collapse the empty/stale/hidden ones
    pass.drawIndexed(this.cone.indexCount, TREE_MAX);
  }
}
