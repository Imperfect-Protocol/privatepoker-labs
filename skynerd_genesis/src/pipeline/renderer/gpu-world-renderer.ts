/**
 * GpuWorldRenderer — the pipeline's conductor (plan §1–3). One frame:
 *   1. the camera adopts the cockpit's pinhole (one-off CPU math) and
 *      the Frame uniform is written — everything else is GPU;
 *   2. the scene's clumps queue their kernels (terrain quad
 *      generation, flora planting, the tile index map) and the kernel
 *      queue flushes them as ONE compute pass;
 *   3. one render pass: sky, then the terrain's quad stream, then the
 *      flora's instanced cones — every conditional decided in kernels.
 *
 * Built entirely programmatically — every pipeline, bind group,
 * buffer, texture and pass is constructed here in code (plan §6, the
 * LightAndGeometry philosophy). Node's harness has no navigator.gpu:
 * `create` returns null and the cockpit keeps its CPU path.
 */
import { GpuDevice } from "../core/gpu-device.js";
import { FrameRecord, GpuBuffer, KernelQueue } from "../core/kernel-queue.js";
import { hash01v } from "../core/wgsl.js";
import { FloraClump } from "../kernels/flora-instances.js";
import { TerrainClump } from "../kernels/terrain-quads.js";
import { TILE_PX, TileIndexMap } from "../kernels/tile-index.js";
import type { ViewCamera } from "../scene/camera.js";
import { Light } from "../scene/light.js";
import { Scene } from "../scene/scene.js";
import { SkyClump } from "../scene/sky-clump.js";
import type {
  WgpuCanvasContext, WgpuTexture, WgpuTextureView,
} from "../types.js";
import { BUF, STAGE, TEXU } from "../types.js";
import type { RenderFeatures } from "./features.js";
import { DEFAULT_FEATURES } from "./features.js";
import {
  DEFAULT_TIME_OF_DAY, GROUND_GRADE, SKY_MOODS,
  type SkyMood, type TimeOfDay,
} from "./time-of-day.js";
import { CRASH_FLOOR_EYE_M, type WorldRenderer } from "./world-renderer.js";

/** The checker law's altitude rung (ground.ts's private twin). */
const ALT_PER_LEVEL = 125.0 * 0.3048;
const MIP_LEVELS = 6;
/** The CPU path's draw distance — kept as the uniform's K. */
const K = 24000.0;

/** The world data the renderer uploads: the land-use plan's height
 *  and biome channels (land.ts owns their generation — the pipeline
 *  stays app-agnostic). */
export interface WorldTextures {
  heights: Float32Array;   // 512², metres above level 0
  biomes: Uint8Array;      // 512², Biome values
}

/** Periodic value noise for the variation texture — land.ts's vnoise
 *  scheme (integer frequency wraps, so the texture tiles with its
 *  repeat address mode). */
function vnoise(u: number, v: number, freq: number,
                salt: number): number {
  const gx = u * freq;
  const gy = v * freq;
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const smooth = (t: number): number =>
    t * t * t * (t * (t * 6 - 15) + 10);
  const fx = smooth(gx - x0);
  const fy = smooth(gy - y0);
  const wrap = (i: number): number => i - Math.floor(i / freq) * freq;
  const n00 = hash01v(wrap(x0), wrap(y0), salt);
  const n10 = hash01v(wrap(x0 + 1), wrap(y0), salt);
  const n01 = hash01v(wrap(x0), wrap(y0 + 1), salt);
  const n11 = hash01v(wrap(x0 + 1), wrap(y0 + 1), salt);
  const a = n00 + (n10 - n00) * fx;
  const b = n01 + (n11 - n01) * fx;
  return a + (b - a) * fy;
}

export class GpuWorldRenderer implements WorldRenderer {
  private readonly queue = new KernelQueue();
  private readonly record = new FrameRecord();
  private readonly scene = new Scene();
  private readonly tiles: TileIndexMap;
  private readonly frameUniform: GpuBuffer;
  private readonly sun = Light.sun();
  private depthTexture: WgpuTexture | null = null;
  private depthView: WgpuTextureView | null = null;
  private fbW = 0;
  private fbH = 0;
  private lost = false;
  private features: RenderFeatures = { ...DEFAULT_FEATURES };
  /** The sky's mood — noon by default (the classic look, exactly). */
  private mood: SkyMood = SKY_MOODS[DEFAULT_TIME_OF_DAY];
  /** The pilot's iris this frame — 1.0 wide open (the uniform's
   *  extras.y; the whole world scales by it). */
  private exposure = 1.0;

  private constructor(private readonly gpu: GpuDevice,
                       private readonly canvas: HTMLCanvasElement,
                       private readonly context: WgpuCanvasContext,
                       private readonly format: string,
                       world: WorldTextures) {
    const dev = gpu.device;
    this.frameUniform = GpuBuffer.create(gpu, FrameRecord.BYTES,
                                         BUF.UNIFORM | BUF.COPY_DST);

    // --- the world's three textures, uploaded once -------------------
    const allStages = STAGE.VERTEX | STAGE.FRAGMENT | STAGE.COMPUTE;
    const heightTexture = dev.createTexture({
      size: { width: 512, height: 512 },
      format: "r32float",
      usage: TEXU.TEXTURE_BINDING | TEXU.COPY_DST,
    });
    this.upload(heightTexture, world.heights, 512 * 4, 512, 512);
    const biomeTexture = dev.createTexture({
      size: { width: 512, height: 512 },
      format: "r8uint",
      usage: TEXU.TEXTURE_BINDING | TEXU.COPY_DST,
    });
    this.upload(biomeTexture, world.biomes, 512, 512, 512);
    const noiseTexture = dev.createTexture({
      size: { width: 256, height: 256 },
      format: "rgba8unorm",
      usage: TEXU.TEXTURE_BINDING | TEXU.COPY_DST,
    });
    this.upload(noiseTexture, this.noiseData(), 256 * 4, 256, 256);

    // --- group 0: the frame uniform + the world's textures ------------
    const group0Layout = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: allStages,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: allStages,
          texture: { sampleType: "unfilterable-float" } },
        { binding: 2, visibility: allStages,
          texture: { sampleType: "uint" } },
        { binding: 3, visibility: STAGE.FRAGMENT,
          texture: { sampleType: "float" } },
        { binding: 4, visibility: STAGE.FRAGMENT,
          sampler: { type: "filtering" } },
      ],
    });
    const group0 = dev.createBindGroup({
      layout: group0Layout,
      entries: [
        { binding: 0, resource: { buffer: this.frameUniform.raw } },
        { binding: 1, resource: heightTexture.createView() },
        { binding: 2, resource: biomeTexture.createView() },
        { binding: 3, resource: noiseTexture.createView() },
        { binding: 4,
          resource: dev.createSampler({
            addressModeU: "repeat", addressModeV: "repeat",
            magFilter: "linear", minFilter: "linear",
          }) },
      ],
    });

    // --- the scene: sky, terrain, flora — the clump order IS the
    //     draw order -----------------------------------------------------
    this.tiles = new TileIndexMap(gpu, group0Layout);
    this.scene.setLights([this.sun]);
    this.scene.addClump(
      new SkyClump(gpu, this.format, group0, group0Layout));
    this.scene.addClump(
      new TerrainClump(gpu, this.format, group0, group0Layout, this.tiles));
    this.scene.addClump(
      new FloraClump(gpu, this.format, group0, group0Layout, this.tiles));
  }

  /** Acquire the device and build the pipeline, or null when WebGPU
   *  is unavailable (headless Node, older browsers) — the caller then
   *  keeps the CPU path. Never throws. A `shared` device (another
   *  renderer's, e.g. the side viewports') is reused instead of
   *  acquiring a new one: one device, many canvases. */
  static async create(canvas: HTMLCanvasElement | null,
                      world: WorldTextures,
                      shared?: GpuDevice | null):
    Promise<GpuWorldRenderer | null> {
    if (canvas === null) {
      return null;
    }
    const gpu = shared !== null && shared !== undefined
      ? shared
      : await GpuDevice.create();
    if (gpu === null) {
      return null;
    }
    const context = (canvas as unknown as
      { getContext(kind: "webgpu"): WgpuCanvasContext | null })
      .getContext("webgpu");
    if (context === null) {
      return null;
    }
    const format = gpu.canvasFormat;
    context.configure({ device: gpu.device, format,
                        alphaMode: "opaque" });
    const renderer = new GpuWorldRenderer(gpu, canvas, context, format,
                                          world);
    gpu.onLost(() => {
      renderer.lost = true;               // ready flips: CPU fallback
    });
    return renderer;
  }

  get ready(): boolean {
    return !this.lost && this.gpu.ready;
  }

  /** The device this renderer runs on — the side viewports' own
   *  renderers are created on it (one device, many canvases). */
  get gpuDevice(): GpuDevice {
    return this.gpu;
  }

  /** The toolbar's toggles (plan §5) — read into the frame uniform. */
  setFeatures(features: RenderFeatures): void {
    this.features = { ...features };
  }

  /** The time-of-day dropdown: the mood the uniform carries. */
  setTimeOfDay(tod: TimeOfDay): void {
    this.mood = SKY_MOODS[tod];
  }

  /** The pilot's iris: the exposure every clump's shader scales by. */
  setExposure(level: number): void {
    this.exposure = level;
  }

  get featuresOf(): RenderFeatures {
    return { ...this.features };
  }

  renderFrame(cam: ViewCamera, eyeHeight: number): void {
    if (!this.ready) {
      return;
    }
    this.resize();
    // the camera adopts this frame's pinhole in FRAMEBUFFER units: the
    // focal scales from CSS px, and the record's W/H follow it — the VP
    // and every screen-space law share one consistent space
    const focalFb = cam.f * (this.fbW / Math.max(1, cam.W));
    this.scene.camera.adopt(
      { ...cam, f: focalFb, W: this.fbW, H: this.fbH }, focalFb);
    // the altitude law: the checker's level from the eye height
    const eyeLevel = Math.max(0.0, Math.min(MIP_LEVELS - 1.0,
      Math.log2(Math.max(eyeHeight, 1e-6) / ALT_PER_LEVEL)));
    const pos = this.scene.camera.position;
    this.record.viewProj = this.scene.camera.viewProj;
    this.record.setCamPos([pos[0], pos[1], pos[2]], eyeLevel);
    this.record.setRight([cam.rx, cam.ry, cam.rz], focalFb);
    this.record.setUp([cam.ux, cam.uy, cam.uz], this.fbW);
    this.record.setFwd([cam.fx, cam.fy, cam.fz], this.fbH);
    this.record.setParams(K, Math.ceil(this.fbW / TILE_PX),
                          Math.ceil(this.fbH / TILE_PX), TILE_PX);
    this.record.setSunDir(this.mood.sunDir);
    this.record.setFeatures([
      this.features.haze ? 1 : 0, this.features.relief ? 1 : 0,
      this.features.sunShadows ? 1 : 0,
      this.features.waterReflections ? 1 : 0,
    ]);
    this.record.setZenith(this.mood.zenith, this.mood.discSize);
    this.record.setFog(this.mood.fog, this.mood.brightness);
    this.record.setSunTint(this.mood.disc, this.mood.ambient);
    this.record.setExtras(this.features.checker ? 1 : 0,
                          this.exposure,
                          eyeHeight < CRASH_FLOOR_EYE_M ? 1 : 0);
    this.record.setGrade(this.mood.grade, GROUND_GRADE);
    this.gpu.queue.writeBuffer(this.frameUniform.raw, 0,
                               this.record.data);
    this.tiles.clear();

    // phase one: the kernel queue — generation + the index map, one
    // compute pass; phase two: the one render pass, clumps in order
    const encoder = this.gpu.device.createCommandEncoder();
    this.scene.generate(this.queue, encoder);
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.72, g: 0.78, b: 0.84, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depthView as WgpuTextureView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    this.scene.draw(pass);
    pass.end();
    this.gpu.queue.submit([encoder.finish()]);
  }

  /** Match the canvas's framebuffer to its client size (dpr-aware),
   *  recreating the depth attachment when it changes. */
  private resize(): void {
    const dpr = (globalThis as { devicePixelRatio?: number })
      .devicePixelRatio ?? 1;
    const w = Math.max(2, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(2, Math.round(this.canvas.clientHeight * dpr));
    if (w === this.fbW && h === this.fbH) {
      return;
    }
    this.fbW = w;
    this.fbH = h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.depthTexture?.destroy();
    this.depthTexture = this.gpu.device.createTexture({
      size: { width: w, height: h },
      format: "depth24plus",
      usage: TEXU.RENDER_ATTACHMENT,
    });
    this.depthView = this.depthTexture.createView();
  }

  /** One texture's one-off upload: staging buffer → copy → submit. */
  private upload(texture: WgpuTexture, data: ArrayBufferView,
                  bytesPerRow: number, width: number, height: number): void {
    const staging = GpuBuffer.seeded(this.gpu, data, BUF.COPY_SRC);
    const encoder = this.gpu.device.createCommandEncoder();
    encoder.copyBufferToTexture(
      { buffer: staging.raw, bytesPerRow },
      { texture },
      { width, height });
    this.gpu.queue.submit([encoder.finish()]);
    staging.destroy();
  }

  /** The variation texture (256², tileable): broad mottle in r, fine
   *  grain in g, per-texel dither in b — the noise layer the terrain
   *  and canopy shaders sample (plan §3's GPU biome variation). */
  private noiseData(): Uint8Array {
    const n = 256;
    const data = new Uint8Array(n * n * 4);
    for (let y = 0; y < n; y++) {
      const v = (y + 0.5) / n;
      for (let x = 0; x < n; x++) {
        const u = (x + 0.5) / n;
        const i = (y * n + x) * 4;
        data[i] = Math.round(255 * vnoise(u, v, 8, 101));
        data[i + 1] = Math.round(255 * vnoise(u, v, 16, 102));
        data[i + 2] = Math.round(255 * hash01v(x, y, 103));
        data[i + 3] = 255;
      }
    }
    return data;
  }
}
