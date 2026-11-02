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
  terrainHeight(x: number, y: number): number {
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
