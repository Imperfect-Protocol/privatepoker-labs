/**
 * Light — the scene's lights (LightAndGeometry's Light, plan §6).
 * The world has ONE light today: the sun, a fixed directional light
 * shared with the CPU path's SUN_DIR (land.ts) — the terrain relief,
 * the tree shading and the corner-ray shadow field all read it.
 */

/** A light source in the scene graph. Directional today; the class
 *  carries what a light IS (kind, direction, colour, intensity), not
 *  what any one pass does with it. */
export class Light {
  constructor(
      readonly kind: "directional",
      readonly direction: readonly [number, number, number],
      readonly colour: readonly [number, number, number] = [1.0, 1.0, 1.0],
      readonly intensity: number = 1.0) {}

  /** The one sun the whole landscape obeys (land.ts SUN_DIR — the
   *  import stays on the app side; the value is identical). */
  static sun(): Light {
    const v = [-0.88, -0.32, 0.35];
    const l = Math.hypot(v[0], v[1], v[2]);
    return new Light("directional",
                     [v[0] / l, v[1] / l, v[2] / l]);
  }
}
