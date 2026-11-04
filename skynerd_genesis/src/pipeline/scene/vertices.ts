/**
 * Vertices — one geometry's vertex stream, built programmatically
 * (LightAndGeometry composes every mesh in code; so do we — plan §6).
 * A builder owns named channels of packed float records; `geometry`
 * freezes it into an interleaved buffer + optional index list.
 */

/** The vertex channel formats this pipeline uses. */
export type VertexChannel = "position" | "normal";

const CHANNEL_SLOTS: Readonly<Record<VertexChannel, number>> = {
  position: 3,
  normal: 3,
};

/** An append-only vertex stream: per-channel arrays, frozen on
 *  `interleaved()` into one GPU-ready Float32Array. */
export class Vertices {
  private readonly channels = new Map<VertexChannel, number[]>();
  private count = 0;

  constructor(private readonly layout: readonly VertexChannel[]) {}

  /** Append one vertex's channel values (in layout order). */
  push(...values: number[]): void {
    let slot = 0;
    for (const channel of this.layout) {
      const width = CHANNEL_SLOTS[channel];
      const store = this.storeOf(channel);
      for (let w = 0; w < width; w++) {
        store.push(values[slot + w] ?? 0.0);
      }
      slot += width;
    }
    this.count += 1;
  }

  get length(): number {
    return this.count;
  }

  /** The interleaved buffer: one record per vertex, channels in layout
   *  order — the exact bytes the vertex shader's layout describes. */
  interleaved(): Float32Array {
    const width = this.stride / 4;
    const out = new Float32Array(this.count * width);
    let offset = 0;
    for (const channel of this.layout) {
      const store = this.channels.get(channel)!;
      const slots = CHANNEL_SLOTS[channel];
      for (let v = 0; v < this.count; v++) {
        for (let s = 0; s < slots; s++) {
          out[v * width + offset + s] = store[v * slots + s];
        }
      }
      offset += slots;
    }
    return out;
  }

  /** The stride of one vertex record, BYTES. */
  get stride(): number {
    let slots = 0;
    for (const channel of this.layout) {
      slots += CHANNEL_SLOTS[channel];
    }
    return slots * 4;
  }

  /** The attribute descriptors in vertex-buffer-layout order. */
  attributes(): { shaderLocation: number; offset: number; format: string }[] {
    const out: { shaderLocation: number; offset: number; format: string }[] = [];
    let offset = 0;
    for (const channel of this.layout) {
      out.push({
        shaderLocation: out.length,
        offset,
        format: "float32x3",
      });
      offset += CHANNEL_SLOTS[channel] * 4;
    }
    return out;
  }

  private storeOf(channel: VertexChannel): number[] {
    let store = this.channels.get(channel);
    if (store === undefined) {
      store = [];
      this.channels.set(channel, store);
    }
    return store;
  }
}
