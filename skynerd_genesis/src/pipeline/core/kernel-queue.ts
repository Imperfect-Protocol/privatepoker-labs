/**
 * The kernel queue — the pipeline's compute side (plan §3 pass table,
 * step 0). Kernels are QUEUED in order and flushed as ONE compute
 * pass; every conditional happens inside a kernel, branching on the
 * data in the buffers it is handed. The CPU never reads results back.
 *
 * Also here: GpuBuffer (a typed, lifetime-owned wrapper) and
 * FrameRecord (the one uniform every kernel and shader reads — the
 * per-frame camera, sun and feature data).
 */
import { BUF, type WgpuBuffer } from "../types.js";
import type { GpuDevice } from "./gpu-device.js";

/** A GPU buffer with an owner: created once, written many times. */
export class GpuBuffer {
  private constructor(readonly raw: WgpuBuffer) {}

  /** Create an empty buffer with the given usage flags. */
  static create(device: GpuDevice, size: number, usage: number): GpuBuffer {
    return new GpuBuffer(device.device.createBuffer({
      size: Math.max(16, Math.ceil(size / 16) * 16),  // WebGPU aligns
      usage,
    }));
  }

  /** Create and immediately fill from CPU data (one-off uploads:
   *  the height/biome/noise textures' staging, uniform init). */
  static seeded(device: GpuDevice, data: ArrayBufferView,
                usage: number): GpuBuffer {
    const raw = device.device.createBuffer({
      size: Math.max(16, Math.ceil(data.byteLength / 16) * 16),
      usage: usage | BUF.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(raw.getMappedRange()).set(
      data instanceof Uint8Array ? data
        : new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    raw.unmap();
    return new GpuBuffer(raw);
  }

  /** Queue a CPU-side write (per-frame uniforms, instance data). */
  write(device: GpuDevice, offset: number, data: ArrayBufferView): void {
    device.queue.writeBuffer(this.raw, offset, data);
  }

  destroy(): void {
    this.raw.destroy();
  }
}

/** The per-frame uniform record — one 240-byte buffer, every kernel
 *  and shader reads it as `Frame`. Field layout is byte-identical to
 *  the WGSL `struct Frame` in core/wgsl.ts (std140-ish, 16-byte rows):
 *  the camera and its VP, the tile grid, the sun, the features, and
 *  the four time-of-day rows (zenith+disc size, fog+brightness, sun
 *  tint+ambient, extras — the checker toggle and the iris exposure). */
export class FrameRecord {
  static readonly BYTES = 256;
  private readonly words = new Float32Array(FrameRecord.BYTES / 4);

  /** The VP matrix (column-major, WGSL mat4x4f). */
  set viewProj(m: ArrayLike<number>) {
    for (let i = 0; i < 16; i++) {
      this.words[i] = m[i];
    }
  }

  /** Camera position + the checker's eye LEVEL (the altitude law). */
  setCamPos(p: readonly [number, number, number], eyeLevel: number): void {
    const o = 16;
    this.words[o] = p[0];
    this.words[o + 1] = p[1];
    this.words[o + 2] = p[2];
    this.words[o + 3] = eyeLevel;
  }

  setRight(v: readonly [number, number, number], focal: number): void {
    const o = 20;
    this.words[o] = v[0];
    this.words[o + 1] = v[1];
    this.words[o + 2] = v[2];
    this.words[o + 3] = focal;
  }

  setUp(v: readonly [number, number, number], width: number): void {
    const o = 24;
    this.words[o] = v[0];
    this.words[o + 1] = v[1];
    this.words[o + 2] = v[2];
    this.words[o + 3] = width;
  }

  setFwd(v: readonly [number, number, number], height: number): void {
    const o = 28;
    this.words[o] = v[0];
    this.words[o + 1] = v[1];
    this.words[o + 2] = v[2];
    this.words[o + 3] = height;
  }

  /** K draw distance, then the tile grid (x count, y count, px size). */
  setParams(k: number, tilesX: number, tilesY: number,
            tilePx: number): void {
    const o = 32;
    this.words[o] = k;
    this.words[o + 1] = tilesX;
    this.words[o + 2] = tilesY;
    this.words[o + 3] = tilePx;
  }

  setSunDir(v: readonly [number, number, number]): void {
    const o = 36;
    this.words[o] = v[0];
    this.words[o + 1] = v[1];
    this.words[o + 2] = v[2];
  }

  /** The feature toggles (haze, relief, sunShadows, waterRefl). */
  setFeatures(f: readonly [number, number, number, number]): void {
    const o = 40;
    this.words[o] = f[0];
    this.words[o + 1] = f[1];
    this.words[o + 2] = f[2];
    this.words[o + 3] = f[3];
  }

  /** The sky mood: zenith colour + the sun disc's angular size. */
  setZenith(c: readonly [number, number, number],
            discSize: number): void {
    const o = 44;
    this.words[o] = c[0];
    this.words[o + 1] = c[1];
    this.words[o + 2] = c[2];
    this.words[o + 3] = discSize;
  }

  /** The sky mood: fog colour + the sun's brightness. */
  setFog(c: readonly [number, number, number],
          brightness: number): void {
    const o = 48;
    this.words[o] = c[0];
    this.words[o + 1] = c[1];
    this.words[o + 2] = c[2];
    this.words[o + 3] = brightness;
  }

  /** The sky mood: the sun's tint + the ambient light scale. */
  setSunTint(c: readonly [number, number, number],
              ambient: number): void {
    const o = 52;
    this.words[o] = c[0];
    this.words[o + 1] = c[1];
    this.words[o + 2] = c[2];
    this.words[o + 3] = ambient;
  }

  /** The extras row: x = the checker toggle, y = the iris exposure
   *  (the HDR adaptation — the whole world scales by it), z = the
   *  crash-floor gate (1: the camera is low — black below −15°). */
  setExtras(checker: number, exposure: number,
            crashFloor: number): void {
    const o = 56;
    this.words[o] = checker;
    this.words[o + 1] = exposure;
    this.words[o + 2] = crashFloor;
    this.words[o + 3] = 0;
  }

  /** The hour's grade row: rgb = the ground's colour filter (0..255,
   *  normalised to mean 1 at its use sites), w = its strength. */
  setGrade(c: readonly [number, number, number], strength: number): void {
    const o = 60;
    this.words[o] = c[0];
    this.words[o + 1] = c[1];
    this.words[o + 2] = c[2];
    this.words[o + 3] = strength;
  }

  get data(): Float32Array {
    return this.words;
  }
}

/** One queued kernel: a pipeline + its bind groups + the workgroup
 *  grid + its PHASE. WebGPU orders dispatches BETWEEN compute passes
 *  (each pass boundary is a barrier), but NOT within one pass — so
 *  kernels that read what an earlier kernel writes must sit in a later
 *  phase. flush() opens one compute pass per phase. */
interface QueuedKernel {
  pipeline: import("../types.js").WgpuComputePipeline;
  bindGroups: readonly import("../types.js").WgpuBindGroup[];
  workgroups: readonly [number, number, number];
  phase: number;
}

/** The compute pipeline's frame queue. */
export class KernelQueue {
  private readonly queue: QueuedKernel[] = [];

  /** Queue one kernel dispatch for this frame's flush. Kernels in the
   *  same phase share one compute pass; a higher phase waits for the
   *  whole previous pass (its writes) to complete. */
  enqueue(pipeline: import("../types.js").WgpuComputePipeline,
          bindGroups: readonly import("../types.js").WgpuBindGroup[],
          workgroups: readonly [number, number, number],
          phase = 0): void {
    this.queue.push({ pipeline, bindGroups, workgroups, phase });
  }

  /** The queued kernels grouped into one compute pass per PHASE —
   *  pass boundaries are the write→read barriers. Kernels sharing a
   *  phase never read each other's writes, so the group order inside
   *  a pass does not matter; a later phase always waits for the
   *  whole earlier pass. */
  flush(encoder: import("../types.js").WgpuCommandEncoder): void {
    if (this.queue.length === 0) {
      return;
    }
    // stable partition by phase: all phase-0 kernels, then phase-1, ...
    const phases = new Map<number, QueuedKernel[]>();
    for (const k of this.queue) {
      const bucket = phases.get(k.phase);
      if (bucket === undefined) {
        phases.set(k.phase, [k]);
      } else {
        bucket.push(k);
      }
    }
    this.queue.length = 0;
    const order = [...phases.keys()].sort((a, b) => a - b);
    for (const phase of order) {
      const pass = encoder.beginComputePass();
      for (const k of phases.get(phase)!) {
        pass.setPipeline(k.pipeline);
        for (let i = 0; i < k.bindGroups.length; i++) {
          pass.setBindGroup(i, k.bindGroups[i]);
        }
        pass.dispatchWorkgroups(k.workgroups[0], k.workgroups[1],
                                 k.workgroups[2]);
      }
      pass.end();
    }
  }
}
