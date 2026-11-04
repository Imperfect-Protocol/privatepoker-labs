/**
 * The ground's render state — the mood-dependent rows of the CPU
 * ground law (the GPU carries them in the Frame uniform; this is the
 * reference path's twin). One small record, set from the toolbar,
 * read by every sample: the relief's sun and the ambient scale.
 */
import type { SkyMood } from "../pipeline/index.js";

export const groundRender = {
  /** The biome checkerboard (the GPU uniform's extras.x). */
  checker: true,
  /** The mood the relief's sun and the ambient follow. */
  mood: null as SkyMood | null,

  /** The toolbar's mood pick: the relief sun + ambient follow it. */
  setMood(mood: SkyMood): void {
    this.mood = mood;
  },
};
