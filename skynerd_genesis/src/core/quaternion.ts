/**
 * Unit quaternion for attitude. Ported 1:1 from `skynerd/vec.py`.
 *
 * Body convention: X forward, Y right, Z up. Positive rotation about +Y
 * pitches the nose DOWN (right-handed axes with Z up), positive about +Z
 * yaws the nose right (+X toward +Y), positive about +X rolls left. Control
 * surfaces are defined in the learner-friendly convention (positive elevator
 * = nose up) and the physics layer translates.
 */
import { Vec3 } from "./vec.js";

export class Quaternion {
  readonly w: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;

  constructor(w = 1.0, x = 0.0, y = 0.0, z = 0.0) {
    this.w = w;
    this.x = x;
    this.y = y;
    this.z = z;
  }

  normalized(): Quaternion {
    const n = Math.sqrt(this.w * this.w + this.x * this.x + this.y * this.y + this.z * this.z);
    if (n === 0.0) {
      return new Quaternion();
    }
    return new Quaternion(this.w / n, this.x / n, this.y / n, this.z / n);
  }

  multiply(other: Quaternion): Quaternion {
    return new Quaternion(
      this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z,
      this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
      this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
      this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w,
    );
  }

  /** Rotate a world vector by this attitude (body-to-world for attitudes). */
  rotate(vector: Vec3): Vec3 {
    const u = new Vec3(this.x, this.y, this.z);
    const t = u.cross(vector).scale(2.0);
    return vector.add(t.scale(this.w)).add(u.cross(t));
  }

  static fromAxisAngle(axis: Vec3, radians: number): Quaternion {
    const a = axis.normalized();
    const half = radians / 2.0;
    const s = Math.sin(half);
    return new Quaternion(Math.cos(half), a.x * s, a.y * s, a.z * s);
  }

  /** ZYX Euler composition: yaw about +Z, then pitch about +Y, then roll about +X. */
  static fromEuler(yaw: number, pitch: number, roll: number): Quaternion {
    return Quaternion.fromAxisAngle(new Vec3(0.0, 0.0, 1.0), yaw)
      .multiply(Quaternion.fromAxisAngle(new Vec3(0.0, 1.0, 0.0), pitch))
      .multiply(Quaternion.fromAxisAngle(new Vec3(1.0, 0.0, 0.0), roll));
  }

  /** (forward, right, up): this attitude applied to +X, +Y, +Z. */
  bodyAxes(): [Vec3, Vec3, Vec3] {
    return [
      this.rotate(new Vec3(1.0, 0.0, 0.0)),
      this.rotate(new Vec3(0.0, 1.0, 0.0)),
      this.rotate(new Vec3(0.0, 0.0, 1.0)),
    ];
  }

  /** Advance attitude by body angular rates (rad/s) over dt; renormalizes. */
  integrateBodyRates(p: number, q: number, r: number, dt: number): Quaternion {
    // q_dot = 0.5 * q ⊗ (0, omega_body); explicit Euler is fine at small dt.
    const half = 0.5 * dt;
    const dq = new Quaternion(
      -half * (p * this.x + q * this.y + r * this.z),
      half * (p * this.w + q * this.z - r * this.y),
      half * (q * this.w + r * this.x - p * this.z),
      half * (r * this.w + p * this.y - q * this.x),
    );
    return new Quaternion(this.w + dq.w, this.x + dq.x, this.y + dq.y, this.z + dq.z)
      .normalized();
  }
}
