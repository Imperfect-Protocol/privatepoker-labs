/**
 * The WebGPU API surface this pipeline uses — declared as OUR OWN
 * structural interfaces, not ambient declarations.
 *
 * Why: the repo has zero npm dependencies and compiles under BOTH
 * tsconfigs — the Node build (no DOM lib, no navigator) and the web
 * build (DOM lib, which may itself ship WebGPU types). Ambient
 * `declare interface GPUDevice` would collide with the DOM lib in
 * the web build and dangle in Node; our own names collide with
 * nothing. The runtime objects satisfy these shapes structurally —
 * the one cast happens at device acquisition (gpu-device.ts) and
 * everything downstream is strictly typed against this file.
 *
 * The usage constants are the WebGPU spec's flag values (they are
 * stable, spec-fixed numbers — the browser globals GPUBufferUsage
 * and GPUTextureUsage hold exactly these).
 */

// --- the flag constants (spec values, hex as in the WebGPU spec) ------

/** Buffer usage flags. */
export const BUF = {
  COPY_SRC: 0x04,
  COPY_DST: 0x08,
  INDEX: 0x10,
  VERTEX: 0x20,
  UNIFORM: 0x40,
  STORAGE: 0x80,
  INDIRECT: 0x100,
} as const;

/** Texture usage flags. */
export const TEXU = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  RENDER_ATTACHMENT: 0x10,
} as const;

/** Shader stage visibility flags. */
export const STAGE = {
  VERTEX: 0x1,
  FRAGMENT: 0x2,
  COMPUTE: 0x4,
} as const;

/** Buffer map modes (for mapAsync). */
export const MAP = { READ: 0x1, WRITE: 0x2 } as const;

// --- the device family ------------------------------------------------

export interface WgpuAdapter {
  requestDevice(): Promise<WgpuDevice | null>;
}

export interface WgpuDevice {
  readonly queue: WgpuQueue;
  /** Resolves when the device is lost — the pipeline's fallback hook. */
  readonly lost: Promise<unknown>;
  createShaderModule(descriptor: { code: string }): WgpuShaderModule;
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): WgpuBuffer;
  createTexture(descriptor: {
    size: { width: number; height: number };
    format: string;
    usage: number;
  }): WgpuTexture;
  createSampler(descriptor: {
    addressModeU?: string;
    addressModeV?: string;
    magFilter?: string;
    minFilter?: string;
  }): WgpuSampler;
  createBindGroupLayout(descriptor: {
    entries: WgpuBindGroupLayoutEntry[];
  }): WgpuBindGroupLayout;
  createPipelineLayout(descriptor: {
    bindGroupLayouts: WgpuBindGroupLayout[];
  }): WgpuPipelineLayout;
  createComputePipeline(descriptor: {
    layout: WgpuPipelineLayout;
    compute: { module: WgpuShaderModule; entryPoint: string };
  }): WgpuComputePipeline;
  createRenderPipeline(descriptor: WgpuRenderPipelineDescriptor): WgpuRenderPipeline;
  createBindGroup(descriptor: {
    layout: WgpuBindGroupLayout;
    entries: WgpuBindGroupEntry[];
  }): WgpuBindGroup;
  createCommandEncoder(): WgpuCommandEncoder;
  destroy(): void;
}

export interface WgpuQueue {
  writeBuffer(buffer: WgpuBuffer, bufferOffset: number,
              data: ArrayBufferView): void;
  submit(commandBuffers: WgpuCommandBuffer[]): void;
}

// --- buffers ----------------------------------------------------------

export interface WgpuBuffer {
  readonly size: number;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  mapAsync(mode: number): Promise<void>;
  unmap(): void;
  destroy(): void;
}

// --- textures ---------------------------------------------------------

export interface WgpuTexture {
  createView(): WgpuTextureView;
  destroy(): void;
}

export type WgpuTextureView = object;
export type WgpuSampler = object;

// --- shaders, layouts, pipelines --------------------------------------

export type WgpuShaderModule = object;
export type WgpuBindGroupLayout = object;
export type WgpuPipelineLayout = object;

export interface WgpuBindGroupLayoutEntry {
  binding: number;
  visibility: number;
  buffer?: { type: string };
  texture?: { sampleType: string };
  sampler?: { type: string };
  /** Non-zero only for storage-texture bindings (unused today). */
  storageTexture?: { access: string; format: string };
}

export interface WgpuBindGroupEntry {
  binding: number;
  resource: { buffer: WgpuBuffer; offset?: number; size?: number }
           | WgpuTextureView | WgpuSampler;
}

export interface WgpuBindGroup {
  setPipeline(pipeline: WgpuComputePipeline
              | WgpuRenderPipeline): void;
  // (placeholder — never called; bind groups are opaque handles)
}

export interface WgpuComputePipeline {
  /* opaque */
}

export interface WgpuVertexAttribute {
  shaderLocation: number;
  offset: number;
  format: string;
}

export interface WgpuVertexBufferLayout {
  arrayStride: number;
  stepMode?: string;
  attributes: WgpuVertexAttribute[];
}

export interface WgpuRenderPipelineDescriptor {
  layout: WgpuPipelineLayout;
  vertex: {
    module: WgpuShaderModule;
    entryPoint: string;
    buffers?: WgpuVertexBufferLayout[];
  };
  fragment: {
    module: WgpuShaderModule;
    entryPoint: string;
    targets: { format: string }[];
  };
  primitive?: { topology: string; cullMode?: string };
  depthStencil?: {
    format: string;
    depthWriteEnabled: boolean;
    depthCompare: string;
  };
}

export interface WgpuRenderPipeline {
  /* opaque */
}

// --- command encoding ---------------------------------------------------

export interface WgpuComputePassEncoder {
  setPipeline(pipeline: WgpuComputePipeline): void;
  setBindGroup(index: number, group: WgpuBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface WgpuRenderPassEncoder {
  setPipeline(pipeline: WgpuRenderPipeline): void;
  setBindGroup(index: number, group: WgpuBindGroup): void;
  setVertexBuffer(slot: number, buffer: WgpuBuffer, offset?: number): void;
  setIndexBuffer(buffer: WgpuBuffer, format: string): void;
  draw(vertexCount: number, instanceCount?: number): void;
  drawIndexed(indexCount: number, instanceCount?: number): void;
  end(): void;
}

export interface WgpuCommandEncoder {
  beginComputePass(): WgpuComputePassEncoder;
  beginRenderPass(descriptor: {
    colorAttachments: {
      view: WgpuTextureView;
      clearValue: { r: number; g: number; b: number; a: number };
      loadOp: string;
      storeOp: string;
    }[];
    depthStencilAttachment?: {
      view: WgpuTextureView;
      depthClearValue: number;
      depthLoadOp: string;
      depthStoreOp: string;
    };
  }): WgpuRenderPassEncoder;
  copyBufferToTexture(
    source: { buffer: WgpuBuffer; bytesPerRow: number },
    destination: { texture: WgpuTexture },
    copySize: { width: number; height: number },
  ): void;
  finish(): WgpuCommandBuffer;
}

export interface WgpuCommandBuffer {
  /* opaque */
}

// --- the canvas context -------------------------------------------------

export interface WgpuCanvasContext {
  configure(descriptor: { device: WgpuDevice; format: string;
                          alphaMode?: string }): void;
  unconfigure(): void;
  getCurrentTexture(): WgpuTexture;
}

export interface WgpuCanvas {
  getContext(kind: "webgpu"): WgpuCanvasContext | null;
}

/** The `navigator.gpu` entry point, as a structural type. */
export interface Wgpu {
  requestAdapter(): Promise<WgpuAdapter | null>;
  getPreferredCanvasFormat(): string;
}

/** The single probe: find navigator.gpu without naming `navigator`
 *  (the identifier does not exist in the Node build's type space). */
export function webGpuOf(): Wgpu | null {
  const nav = (globalThis as unknown as
    { navigator?: { gpu?: Wgpu } }).navigator;
  return nav?.gpu ?? null;
}
