/**
 * World coordinates: gravity, air, terrain, bounds. Ported from the
 * coordinate/model half of `skynerd/physics/world.py` (Genesis Milestone 1).
 *
 * Terrain is flat ground plus optional smooth hills (sum of Gaussians).
 * The raycast half of the Python module belongs to the physics milestone
 * (Genesis Milestone 3) and is intentionally not here yet.
 */
import { GRAVITY, RHO_SEA_LEVEL } from "../physics/aero.js";
import { Vec3 } from "./vec.js";

export { RHO_SEA_LEVEL, GRAVITY } from "../physics/aero.js";

/** The airfield's plateau: the runway's flat ground at 95 ft — the
 *  datum the aircraft's 100 ft takeoff altitude stands on (95 ft of
 *  terrain + 5 ft of landing gear). One source of truth: the visual
 *  ground (app/land.ts) carves the very same ellipse. */
export const AIRFIELD = {
  elev: 95.0 / 3.28084,   // 95 ft in metres
  cx: 600.0,              // the ellipse's centre, world x
  rx: 1050.0,             // its semi-axis along x, m
  ry: 420.0,              // its semi-axis along y, m
};

/** A smooth hill: height * exp(-d^2 / (2 sigma^2)) around (centerX, centerY). */
export class Hill {
  readonly centerX: number;
  readonly centerY: number;
  readonly height: number;
  readonly radius: number; // Gaussian sigma, metres

  constructor(centerX: number, centerY: number, height: number, radius: number) {
    this.centerX = centerX;
    this.centerY = centerY;
    this.height = height;
    this.radius = radius;
  }
}

export interface WorldOptions {
  readonly hills?: readonly Hill[];
  readonly gravity?: Vec3;
  readonly rho?: number;
  readonly halfExtent?: number;
}

export class World {
  readonly hills: readonly Hill[];
  readonly gravity: Vec3;
  readonly rho: number;
  readonly halfExtent: number;

  constructor(options: WorldOptions = {}) {
    this.hills = options.hills ?? [];
    this.gravity = options.gravity ?? GRAVITY;
    this.rho = options.rho ?? RHO_SEA_LEVEL;
    this.halfExtent = options.halfExtent ?? 1000.0;
  }
  /** The terrain's height at (x, y): the airfield's flat plateau at
   *  95 ft inside its ellipse (the runway and its clear zone — exact,
   *  no rim: nothing the aircraft flies reaches the rim's blend), the
   *  smooth hills outside it. */
  terrainHeight(x: number, y: number): number {
    const fx = (x - AIRFIELD.cx) / AIRFIELD.rx;
    const fy = y / AIRFIELD.ry;
    if (fx * fx + fy * fy < 1.0) {
      return AIRFIELD.elev;
    }
    let height = 0.0;
    for (const hill of this.hills) {
      const d2 = (x - hill.centerX) ** 2 + (y - hill.centerY) ** 2;
      height += hill.height * Math.exp(-d2 / (2.0 * hill.radius * hill.radius));
    }
    return height;
  }
  isInsideBounds(position: Vec3): boolean {
    return (
      Math.abs(position.x) <= this.halfExtent &&
      Math.abs(position.y) <= this.halfExtent &&
      position.z >= -10.0
    );
  }
}
