/**
 * The tree geometry — one instanced cone (the canopy; the trunk hides
 * under it), unit-sized: apex at (0, 0, 1), base circle of radius 1
 * at z = 0. The flora kernel scales it per instance (height h up,
 * radius r out — world Z is up, as everywhere in this codebase).
 * Built programmatically, like every mesh here (plan §6).
 */
import { Geometry } from "../scene/geometry.js";
import { Vertices } from "../scene/vertices.js";

export class ConeGeometry extends Geometry {
  constructor(segments = 8) {
    const verts = new Vertices(["position", "normal"]);
    const push = (x: number, y: number, z: number,
                   nx: number, ny: number, nz: number): void =>
      verts.push(x, y, z, nx, ny, nz);
    // side ring + apex: the base verts first (angle = i/segments)
    const idx: number[] = [];
    const baseFirst = 0;
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * 2.0 * Math.PI;
      const c = Math.cos(a);
      const s = Math.sin(a);
      // the cone's side normal (unit): slope 1:1 cone of height = radius
      const nl = 1.0 / Math.SQRT2;
      push(c, s, 0.0, c * nl, s * nl, nl);
    }
    const apex = verts.length;
    push(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    for (let i = 0; i < segments; i++) {
      idx.push(baseFirst + i, apex, baseFirst + i + 1);
    }
    // the base cap (drawn from below — rare, but the spin can look up)
    const capCentre = verts.length;
    push(0.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    for (let i = 0; i < segments; i++) {
      idx.push(baseFirst + i, baseFirst + i + 1, capCentre);
    }
    super("cone", verts, idx);
  }
}
