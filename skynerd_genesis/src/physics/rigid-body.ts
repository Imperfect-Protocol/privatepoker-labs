/**
 * 6-DOF rigid body flight at a fixed tick.
 * Ported 1:1 from `skynerd/physics/rigid_body.py` (Genesis Milestone 3).
 *
 * State: position, velocity (world frame, +Z up), attitude (quaternion,
 * body X forward / Y right / Z up), and body angular rates (p, q, r).
 *
 * Integration at each fixed dt:
 * 1. Aero + propulsion force from velocity, attitude, surfaces, tables.
 * 2. Velocity += (force / mass + gravity) * dt; position += velocity * dt.
 * 3. Control/stability moment; angular rates get damping integrated
 *    IMPLICITLY (unconditionally stable at any dt), control explicitly.
 * 4. Attitude advances by body rates and renormalizes.
 *
 * Determinism: same state + same inputs + same dt => identical trajectory.
 * No randomness lives here; chaos enters the architecture at the tensor
 * pipeline, not in the physics.
 */
import { Vec3 } from "../core/vec.js";
import { Quaternion } from "../core/quaternion.js";
import {
  AeroDiag, AeroTables, GRAVITY, RHO_SEA_LEVEL, aeroForce,
  controlStabilityMoment, dampingSlopePerAxis,
} from "./aero.js";
import { SurfacesState } from "./surfaces.js";

export class RigidBody {
  position: Vec3;
  velocity: Vec3;
  attitude: Quaternion;
  rateP = 0.0;   // roll rate, rad/s, about body +X
  rateQ = 0.0;   // pitch rate, rad/s, about body +Y
  rateR = 0.0;   // yaw rate, rad/s, about body +Z
  /** Seconds spent below stall speed (episode bookkeeping, like Python's dynamic attribute). */
  slowSeconds = 0.0;
  readonly tables: AeroTables;

  constructor(position: Vec3, velocity: Vec3, yaw = 0.0, pitch = 0.0, roll = 0.0,
              tables: AeroTables | null = null) {
    this.position = position;
    this.velocity = velocity;
    this.attitude = Quaternion.fromEuler(yaw, pitch, roll);
    this.tables = tables ?? new AeroTables();
  }

  get bodyAxes(): [Vec3, Vec3, Vec3] {
    return this.attitude.bodyAxes();
  }

  /** Advance one fixed physics tick. Returns aero diagnostics. */
  step(servos: SurfacesState, dt: number,
       gravity: Vec3 = GRAVITY, rho: number = RHO_SEA_LEVEL): AeroDiag {
    if (dt <= 0.0) {
      throw new Error("dt must be positive");
    }
    const tables = this.tables;
    const [forward, right, up] = this.bodyAxes;

    const { force, diag } = aeroForce(this.velocity, forward, right, up,
                                      servos, tables, rho);
    this.velocity = this.velocity.add(
      force.scale(1.0 / tables.mass).add(gravity).scale(dt));
    this.position = this.position.add(this.velocity.scale(dt));

    const [alpha, beta, qDyn] = [diag.alpha, diag.beta, diag.qDyn];
    const moment = controlStabilityMoment(alpha, beta, servos, tables, qDyn);
    const [kx, ky, kz] = dampingSlopePerAxis(qDyn, diag.airspeed, tables, rho);
    // Implicit damping: rate_new = (rate + explicit_rate_delta) / (1 + k*dt).
    const [ix, iy, iz] = [tables.inertiaX, tables.inertiaY, tables.inertiaZ];
    this.rateP = (this.rateP + (moment.x / ix) * dt) / (1.0 + kx * dt);
    this.rateQ = (this.rateQ + (moment.y / iy) * dt) / (1.0 + ky * dt);
    this.rateR = (this.rateR + (moment.z / iz) * dt) / (1.0 + kz * dt);

    this.attitude = this.attitude.integrateBodyRates(
      this.rateP, this.rateQ, this.rateR, dt);
    return diag;
  }
}
