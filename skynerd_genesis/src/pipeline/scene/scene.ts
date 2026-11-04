/**
 * Scene — the scene graph's root (LightAndGeometry's SceneGraph, plan
 * §3): it OWNS what exists — the camera, the lights, the top-level
 * clumps in DRAW ORDER — and nothing about how pixels are produced.
 * The renderer walks it; the headless harness can build and inspect
 * it without a GPU (plan §6: plain TS objects all the way down).
 */
import type { GpuDevice } from "../core/gpu-device.js";
import type { KernelQueue } from "../core/kernel-queue.js";
import type {
  WgpuCommandEncoder, WgpuRenderPassEncoder,
} from "../types.js";
import { Camera } from "./camera.js";
import { Clump } from "./clump.js";
import type { Light } from "./light.js";

/** A clump that knows how to render: it queues its compute kernels
 *  (the on-the-go generation) and records its draw into the frame's
 *  one render pass. The renderer calls these in the scene's clump
 *  order — the order IS the draw order. */
export abstract class RenderClump extends Clump {
  /** Queue this clump's compute kernels for the frame (step 0 of the
   *  pass table): generate instances, build index-map entries. Each
   *  clump owns its device — the queue is the only handoff. */
  abstract enqueueKernels(queue: KernelQueue): void;

  /** Record this clump's draw commands into the render pass. */
  abstract drawPass(pass: WgpuRenderPassEncoder): void;
}

export class Scene {
  readonly camera: Camera = new Camera();
  readonly clumps: RenderClump[] = [];
  private lights: Light[] = [];

  /** The clump order is the draw order (sky first, terrain, flora). */
  addClump(clump: RenderClump): void {
    this.clumps.push(clump);
  }

  /** The scene's lights (today: one sun). */
  get lightsOf(): readonly Light[] {
    return this.lights;
  }

  setLights(lights: readonly Light[]): void {
    this.lights = [...lights];
  }

  /** One frame, phase one: every clump's kernels, in clump order,
   *  flushed as ONE compute pass — encoded BEFORE the render pass
   *  opens (passes record sequentially on one encoder). */
  generate(queue: KernelQueue, encoder: WgpuCommandEncoder): void {
    for (const clump of this.clumps) {
      clump.enqueueKernels(queue);
    }
    queue.flush(encoder);
  }

  /** One frame, phase two: every clump's draw, in clump order — the
   *  clump order IS the draw order. */
  draw(pass: WgpuRenderPassEncoder): void {
    for (const clump of this.clumps) {
      clump.drawPass(pass);
    }
  }
}
