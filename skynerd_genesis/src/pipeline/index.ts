/**
 * The pipeline's public surface — what the app imports. Everything
 * else (WGSL, buffer plumbing, kernel internals) stays behind these
 * doors: the app sees the renderer, the features and the scene types.
 */
export { GpuDevice } from "./core/gpu-device.js";
export type { RenderFeatures } from "./renderer/features.js";
export { DEFAULT_FEATURES } from "./renderer/features.js";
export type { WorldRenderer } from "./renderer/world-renderer.js";
export { GpuWorldRenderer } from "./renderer/gpu-world-renderer.js";
export type { WorldTextures } from "./renderer/gpu-world-renderer.js";
export { CRASH_FLOOR_EYE_M } from "./renderer/world-renderer.js";
export type { ViewCamera } from "./scene/camera.js";
export {
  DEFAULT_TIME_OF_DAY, GROUND_GRADE, SKY_MOODS,
} from "./renderer/time-of-day.js";
export type { SkyMood, TimeOfDay } from "./renderer/time-of-day.js";
