/**
 * Geometry — one drawable mesh (LightAndGeometry's Geometry/Mesh):
 * a vertex stream plus an optional index list, owning its GPU vertex
 * and index buffers (uploaded once, on first use — the buffers are
 * lifetime-owned here, the renderer never manages them).
 */
import type { GpuDevice } from "../core/gpu-device.js";
import { GpuBuffer } from "../core/kernel-queue.js";
import { BUF } from "../types.js";
import { Vertices } from "./vertices.js";

export class Geometry {
  private vertexBuf: GpuBuffer | null = null;
  private indexBuf: GpuBuffer | null = null;

  constructor(readonly name: string,
              private readonly vertices: Vertices,
              private readonly indices?: readonly number[]) {}

  /** The vertex buffer, uploaded on first use (usage: VERTEX |
   *  COPY_DST — written once, read every draw). */
  vertexBufferOf(device: GpuDevice): GpuBuffer {
    if (this.vertexBuf === null) {
      this.vertexBuf = GpuBuffer.seeded(device, this.vertices.interleaved(),
                                        BUF.VERTEX);
    }
    return this.vertexBuf;
  }

  /** The index buffer, or null when the mesh draws its vertices in
   *  order. Indices widen to u32 (u16 caps at 65k verts). */
  indexBufferOf(device: GpuDevice): GpuBuffer | null {
    if (this.indices === undefined) {
      return null;
    }
    if (this.indexBuf === null) {
      this.indexBuf = GpuBuffer.seeded(device,
        new Uint32Array(this.indices), BUF.INDEX);
    }
    return this.indexBuf;
  }

  get vertexCount(): number {
    return this.vertices.length;
  }

  get indexCount(): number {
    return this.indices?.length ?? 0;
  }

  get stride(): number {
    return this.vertices.stride;
  }

  attributes(): { shaderLocation: number; offset: number; format: string }[] {
    return this.vertices.attributes();
  }
}
