/**
 * GpuDevice — the ONE device principle (plan §1).
 *
 * Acquires adapter and device ONCE, owns the queue, reports `ready`,
 * and surfaces `device.lost` as a fallback event. Knows nothing about
 * rendering or tensors — the scene passes and the Engineer's tensor
 * service both register against this one object.
 */
import {
  webGpuOf, type Wgpu, type WgpuAdapter, type WgpuDevice,
} from "../types.js";

export class GpuDevice {
  private lostFlag = false;
  private readonly lostHooks: (() => void)[] = [];

  private constructor(private readonly gpu: Wgpu,
                      private readonly adapter: WgpuAdapter,
                      readonly device: WgpuDevice) {
    // the loss event is the pipeline's fallback trigger: mark, notify,
    // and let every client pick its CPU path on the next frame
    void device.lost.then(() => {
      this.lostFlag = true;
      for (const hook of this.lostHooks) {
        hook();
      }
    });
  }

  /** Acquire the device, or null when WebGPU is not available (the
   *  headless Node harness, older browsers). Never throws. */
  static async create(): Promise<GpuDevice | null> {
    const gpu = webGpuOf();
    if (gpu === null) {
      return null;
    }
    try {
      const adapter = await gpu.requestAdapter();
      if (adapter === null) {
        return null;
      }
      const device = await adapter.requestDevice();
      if (device === null) {
        return null;
      }
      return new GpuDevice(gpu, adapter, device);
    } catch {
      return null;                    // any acquisition failure: CPU path
    }
  }

  get ready(): boolean {
    return !this.lostFlag;
  }

  /** The canvas format the platform prefers ("bgra8unorm" typically). */
  get canvasFormat(): string {
    return this.gpu.getPreferredCanvasFormat();
  }

  /** Register a callback for device loss (the renderer falls back). */
  onLost(hook: () => void): void {
    this.lostHooks.push(hook);
  }

  get queue(): WgpuDevice["queue"] {
    return this.device.queue;
  }
}
