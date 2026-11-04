/**
 * WorldRenderer — the executor seam (plan §2): the cockpit hands one
 * frame's camera + eye height to whichever renderer is active. The GPU
 * implementation lives in gpu-world-renderer.ts; the CPU implementation
 * is the cockpit's own proven per-pixel path (the headless harness and
 * every WebGPU-less machine keep it forever — the pixel-exact
 * reference). The pick happens per frame: `gpu if ready else cpu`.
 */
import type { ViewCamera } from "../scene/camera.js";
import type { RenderFeatures } from "./features.js";
import type { TimeOfDay } from "./time-of-day.js";

/** THE CRASH FLOOR'S GATE (metres above the terrain): the sky goes
 *  black below −15° only when the camera is this low — the crashed
 *  view. A flying camera keeps the horizon's haze there: the world
 *  dissolves into fog at its far edge, never into a black band (the
 *  far field's box ends long before the eye's reach at altitude). */
export const CRASH_FLOOR_EYE_M = 25.0;

export interface WorldRenderer {
  /** False after device loss — the caller falls back to the CPU path. */
  readonly ready: boolean;

  /** The toolbar's current toggles (the glare gate lives here). */
  readonly featuresOf: RenderFeatures;

  /** Render one frame of the world through the given camera. The eye
   *  height (the altitude law's input) picks the checker's level. */
  renderFrame(cam: ViewCamera, eyeHeight: number): void;

  /** The time-of-day dropdown: which sky mood the world wears. */
  setTimeOfDay(tod: TimeOfDay): void;

  /** THE PILOT'S IRIS (HDR adaptation): the exposure the whole world
   *  scales by this frame — 1.0 wide open, tightened facing the sun.
   *  The renderer carries it in the frame uniform (extras.y). */
  setExposure(level: number): void;
}
