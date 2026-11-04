/**
 * Clump — a spatial group in the scene graph (LightAndGeometry's
 * Clump, plan §6 — their TODO for bounds is something we finish).
 * A clump owns children (nested clumps) and the geometries planted in
 * its space, and answers for its bounds. The pipeline's renderable
 * clumps (terrain, flora, sky) extend this with their kernel/draw
 * hooks — the graph stays backend-agnostic below that line.
 */
import type { Geometry } from "./geometry.js";

/** An axis-aligned box in world metres. */
export interface Aabb {
  readonly lo: readonly [number, number, number];
  readonly hi: readonly [number, number, number];
}

export class Clump {
  readonly children: Clump[] = [];
  protected readonly geometries: Geometry[] = [];

  constructor(readonly name: string,
              protected bounds: Aabb = {
                lo: [-Infinity, -Infinity, -Infinity],
                hi: [Infinity, Infinity, Infinity],
              }) {}

  /** Add a nested clump. */
  add(clump: Clump): void {
    this.children.push(clump);
  }

  /** Plant a geometry in this clump's space. */
  plant(geometry: Geometry): void {
    this.geometries.push(geometry);
  }

  /** The geometries planted here (not recursing — each clump draws
   *  what it owns; the renderer walks the tree). */
  geometriesOf(): readonly Geometry[] {
    return this.geometries;
  }

  /** The clump's bounds (subclasses recompute as their contents move —
   *  the terrain clump's bounds follow the camera). */
  boundsOf(): Aabb {
    return this.bounds;
  }

  /** Union of this box and a child's, for recomputation. */
  protected static union(a: Aabb, b: Aabb): Aabb {
    const lo: [number, number, number] = [
      Math.min(a.lo[0], b.lo[0]),
      Math.min(a.lo[1], b.lo[1]),
      Math.min(a.lo[2], b.lo[2]),
    ];
    const hi: [number, number, number] = [
      Math.max(a.hi[0], b.hi[0]),
      Math.max(a.hi[1], b.hi[1]),
      Math.max(a.hi[2], b.hi[2]),
    ];
    return { lo, hi };
  }
}
