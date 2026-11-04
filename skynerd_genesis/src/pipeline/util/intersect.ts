/**
 * Intersection utils — the CPU twins of the kernel-side tests
 * (LightAndGeometry's Intersect.h, plan §6). The GPU kernels do their
 * own tests inline in WGSL (projection into screen tiles); these TS
 * forms serve the headless harness, the probes and any future CPU
 * culling — same math, one place, testable without a GPU.
 */
import type { Aabb } from "../scene/clump.js";

/** One view-projection matrix (column-major, as the Frame uniform
 *  carries it) unrolled into six frustum planes, each [x, y, z, w]
 *  with the plane's normal pointing INWARD (inside: dot ≥ 0). */
export function frustumPlanes(vp: readonly number[]):
    [number, number, number, number][] {
  // row r of M from column-major storage: m[r], m[4+r], m[8+r], m[12+r]
  const row = (r: number): [number, number, number, number] =>
    [vp[r], vp[4 + r], vp[8 + r], vp[12 + r]];
  const [r0, r1, r2, r3] = [row(0), row(1), row(2), row(3)];
  const sub = (a: number[], b: number[]): [number, number, number, number] =>
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
  const add = (a: number[], b: number[]): [number, number, number, number] =>
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
  return [add(r3, r0), sub(r3, r0), add(r3, r1), sub(r3, r1),
          add(r3, r2), sub(r3, r2)];
}

/** Whether a sphere (centre + radius) is wholly outside the frustum —
 *  the tile-index kernel's per-instance test, in CPU form for tests. */
export function sphereCulled(planes: readonly (readonly number[])[],
                              cx: number, cy: number, cz: number,
                              radius: number): boolean {
  for (const p of planes) {
    if (p[0] * cx + p[1] * cy + p[2] * cz + p[3] < -radius) {
      return true;                       // behind one plane entirely
    }
  }
  return false;
}

/** Ray–AABB slab intersection, or null when the ray misses. The
 *  shadow-field kernel's march uses the WGSL twin of this test. */
export function rayAabb(ox: number, oy: number, oz: number,
                        dx: number, dy: number, dz: number,
                        box: Aabb): number | null {
  let tNear = 0.0;
  let tFar = Infinity;
  const o = [ox, oy, oz];
  const d = [dx, dy, dz];
  const lo = box.lo;
  const hi = box.hi;
  for (let a = 0; a < 3; a++) {
    if (Math.abs(d[a]) < 1e-12) {
      if (o[a] < lo[a] || o[a] > hi[a]) {
        return null;
      }
      continue;
    }
    const inv = 1.0 / d[a];
    let t0 = (lo[a] - o[a]) * inv;
    let t1 = (hi[a] - o[a]) * inv;
    if (t0 > t1) {
      const swap = t0;
      t0 = t1;
      t1 = swap;
    }
    tNear = Math.max(tNear, t0);
    tFar = Math.min(tFar, t1);
    if (tNear > tFar) {
      return null;
    }
  }
  return tNear;
}

/** Point–AABB containment. */
export function pointInBox(x: number, y: number, z: number,
                           box: Aabb): boolean {
  return x >= box.lo[0] && x <= box.hi[0]
      && y >= box.lo[1] && y <= box.hi[1]
      && z >= box.lo[2] && z <= box.hi[2];
}
