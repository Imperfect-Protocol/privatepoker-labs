/**
 * The unit quad — the pipeline's workhorse geometry: the terrain's
 * cell (one quad = one checker cell region, instanced by the million
 * by the terrain kernel) and, untextured, the sky's fullscreen
 * triangle companion. Corners span (0,0)..(1,1); the terrain vertex
 * shader places and displaces them per instance.
 */
import { Geometry } from "../scene/geometry.js";
import { Vertices } from "../scene/vertices.js";

export class QuadGeometry extends Geometry {
  constructor() {
    const verts = new Vertices(["position"]);
    verts.push(0.0, 0.0, 0.0);
    verts.push(1.0, 0.0, 0.0);
    verts.push(1.0, 1.0, 0.0);
    verts.push(0.0, 1.0, 0.0);
    // two triangles, both faces (terrain seen from above; the sky quad
    // draws its own fullscreen triangle instead)
    super("quad", verts, [0, 1, 2, 0, 2, 3]);
  }
}
