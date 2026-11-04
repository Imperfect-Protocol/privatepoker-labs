/**
 * 3D vector math, heading frames, and binning for Skynerd.
 *
 * Pure math, no simulator dependencies. Ported 1:1 from `skynerd/vec.py`
 * (Genesis Milestone 1); the reference values in `reference/vector-ref.json`
 * were produced by the Python implementation and this module must agree
 * with them within tolerance.
 *
 * - `Vec3`: small immutable 3D vector.
 * - `Heading`: a body frame built from a forward vector (right/up derived,
 *   roll-free v1 — good enough for bank-to-turn fixed-wing).
 * - Angle binning: azimuth bins around the horizontal plane (bin 0 = +X
 *   forward, positive clockwise seen from above) and elevation bins from
 *   -90..+90 deg. Range bins are log-spaced so nearby objects get fine
 *   resolution.
 * - Fuzzy bin weights: a direction never belongs to exactly one bin; it is
 *   a normalized distribution over the two nearest bins.
 *
 * Angles are radians everywhere in code; bins are the only integer view.
 */

export class Vec3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;

  constructor(x: number, y: number, z: number) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  static of(a: readonly number[]): Vec3 {
    return new Vec3(a[0], a[1], a[2]);
  }

  static zero(): Vec3 {
    return new Vec3(0.0, 0.0, 0.0);
  }

  asArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  add(other: Vec3): Vec3 {
    return new Vec3(this.x + other.x, this.y + other.y, this.z + other.z);
  }

  sub(other: Vec3): Vec3 {
    return new Vec3(this.x - other.x, this.y - other.y, this.z - other.z);
  }

  scale(factor: number): Vec3 {
    return new Vec3(this.x * factor, this.y * factor, this.z * factor);
  }

  dot(other: Vec3): number {
    return this.x * other.x + this.y * other.y + this.z * other.z;
  }

  cross(other: Vec3): Vec3 {
    return new Vec3(
      this.y * other.z - this.z * other.y,
      this.z * other.x - this.x * other.z,
      this.x * other.y - this.y * other.x,
    );
  }

  length(): number {
    return Math.sqrt(this.dot(this));
  }

  distanceTo(other: Vec3): number {
    return this.sub(other).length();
  }

  normalized(): Vec3 {
    const n = this.length();
    if (n === 0.0) {
      return Vec3.zero();
    }
    return this.scale(1.0 / n);
  }

  rotatedAboutZ(radians: number): Vec3 {
    const c = Math.cos(radians);
    const s = Math.sin(radians);
    return new Vec3(this.x * c - this.y * s, this.x * s + this.y * c, this.z);
  }

  toString(): string {
    return `Vec3(${this.x.toFixed(3)}, ${this.y.toFixed(3)}, ${this.z.toFixed(3)})`;
  }
}

/** A roll-free body frame: forward, right, up derived from a forward vector.
 *
 * Up is world +Z as long as forward is not vertical; a fixed-wing swarm in
 * bank-to-turn flight never needs roll in its perception frame for v1.
 */
export class Heading {
  readonly forward: Vec3;
  readonly right: Vec3;
  readonly up: Vec3;

  constructor(forward: Vec3) {
    let f = forward.normalized();
    if (f.length() === 0.0 || Math.abs(f.z) > 0.999) {
      f = new Vec3(1.0, 0.0, 0.0); // degenerate: face +X
    }
    const worldUp = new Vec3(0.0, 0.0, 1.0);
    let right = f.cross(worldUp).normalized();
    if (right.length() === 0.0) {
      right = new Vec3(0.0, 1.0, 0.0);
    }
    const up = right.cross(f).normalized();
    this.forward = f;
    this.right = right;
    this.up = up;
  }

  static fromYawPitch(yaw: number, pitch: number): Heading {
    // yaw: rotation about +Z from +X; pitch: positive nose-up.
    const cp = Math.cos(pitch);
    const forward = new Vec3(cp * Math.cos(yaw), cp * Math.sin(yaw), Math.sin(pitch));
    return new Heading(forward);
  }
}

/** Signed azimuth of offset in [-pi, pi); 0 = +X, positive clockwise from above. */
export function azimuthRadians(offset: Vec3): number {
  return Math.atan2(offset.y, offset.x);
}

/** Signed elevation of offset in [-pi/2, pi/2]; 0 = horizontal, positive up. */
export function elevationRadians(offset: Vec3): number {
  const horizontal = Math.hypot(offset.x, offset.y);
  return Math.atan2(offset.z, horizontal);
}

/** Bin 0 centred on +X, positive clockwise from above. */
export function azimuthBin(offset: Vec3, bins: number): number {
  const angle = azimuthRadians(offset);
  const shifted = ((angle + Math.PI / bins) % (2.0 * Math.PI) + 2.0 * Math.PI) % (2.0 * Math.PI);
  return Math.floor(shifted / ((2.0 * Math.PI) / bins)) % bins;
}

/** Bins span -pi/2..+pi/2; bin 0 is the lowest (nadir), bins-1 zenith. */
export function elevationBin(offset: Vec3, bins: number): number {
  const angle = elevationRadians(offset);
  const shifted = (angle + Math.PI / 2.0) / Math.PI; // 0..1
  return Math.min(bins - 1, Math.max(0, Math.floor(shifted * bins)));
}

/** Normalized weights over the two bins nearest to `angle`.
 *
 * `span` is the total angular width the bins divide (2*pi for azimuth,
 * pi for elevation). Bin centres sit at multiples of the bin width, so bin 0
 * is centred on angle 0 — the same convention as `azimuthBin`. Returns
 * [bin, weight] pairs with weights summing to 1. A direction is a fuzzy
 * hint, never a hard assignment.
 */
export function fuzzyBinWeights(angle: number, span: number, bins: number): [number, number][] {
  const width = span / bins;
  const x = (((angle % span) + span) % span) / width; // bin centres at integers
  const nearest = Math.floor(x + 0.5); // matches azimuthBin's rounding
  const t = x - nearest; // in [-0.5, 0.5)
  const wNear = 1.0 - Math.abs(t);
  const other = t < 0.0 ? ((nearest - 1) % bins + bins) % bins : (nearest + 1) % bins;
  return [[((nearest % bins) + bins) % bins, wNear], [other, Math.abs(t)]];
}

/** Log-spaced range bins over (min_range, max_range]. Bin 0 is the closest. */
export class RangeBins {
  readonly minRange: number;
  readonly maxRange: number;
  readonly bins: number;
  private readonly logMin: number;
  private readonly logMax: number;

  constructor(maxRange: number, bins: number, minRange = 1.0) {
    if (minRange <= 0.0 || maxRange <= minRange) {
      throw new Error("range bins need 0 < min_range < max_range");
    }
    this.minRange = minRange;
    this.maxRange = maxRange;
    this.bins = bins;
    this.logMin = Math.log(minRange);
    this.logMax = Math.log(maxRange);
  }

  binOf(distance: number): number {
    const d = Math.min(Math.max(distance, this.minRange), this.maxRange);
    const t = (Math.log(d) - this.logMin) / (this.logMax - this.logMin);
    return Math.min(this.bins - 1, Math.floor(t * this.bins));
  }

  /** Geometric centre distance of a bin (the representative range). */
  binCenter(index: number): number {
    const t = (index + 0.5) / this.bins;
    return Math.exp(this.logMin + t * (this.logMax - this.logMin));
  }
}
