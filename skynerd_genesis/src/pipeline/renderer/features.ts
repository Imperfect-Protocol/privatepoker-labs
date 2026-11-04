/**
 * RenderFeatures — the toolbar's toggles (plan §5), one type read by
 * the checkboxes and passed to whichever renderer is active (a uniform
 * on GPU, branch points on CPU). Height, water, roads and objects are
 * always on — they are the world, not a shading property.
 */
export interface RenderFeatures {
  /** The normal·sun hill shading (off by default). */
  relief: boolean;
  /** The sun disc + the glare veil (off by default). */
  glare: boolean;
  /** The ground→sky fog blend — ON: it is the horizon dissolve. */
  haze: boolean;
  /** Hill shadows via the corner-ray field (off; plan phase 3). */
  sunShadows: boolean;
  /** Reserved for the water mirror pass (off). */
  waterReflections: boolean;
  /** The biome checkerboard pattern — ON: it is the ground's law. */
  checker: boolean;
}

export const DEFAULT_FEATURES: RenderFeatures = {
  relief: false,
  glare: false,
  haze: true,
  sunShadows: false,
  waterReflections: false,
  checker: true,
};
