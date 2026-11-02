/**
 * Airframe aerodynamic model: coefficient tables, forces, moments.
 * Ported 1:1 from `skynerd/physics/aero.py` (Genesis Milestone 3).
 *
 * A deliberately simple fixed-wing UAV model — qualitatively right, not
 * CFD-exact. The learned layers never see these coefficients; they see
 * positions, bearings, and flight-instrument errors. What matters is that
 * the envelope is honest: lift needs airspeed, drag grows with induced
 * lift, the airframe is weathervane-stable, controls have authority, and
 * stall exists.
 *
 * Sign conventions (body frame X forward, Y right, Z up):
 * - alpha (angle of attack) > 0: relative wind from below the nose.
 * - beta (sideslip) > 0: relative wind from the right.
 * - Positive rotation about +Y is nose DOWN; about +Z is nose RIGHT; about +X
 *   is roll LEFT.
 * - Surface channels: [aileron, elevator, rudder, throttle], with
 *   learner-friendly signs: positive aileron = roll right, positive elevator
 *   = nose up, positive rudder = nose right, throttle in [0, 1].
 */
import { Vec3 } from "../core/vec.js";

export const RHO_SEA_LEVEL = 1.225;
export const GRAVITY = new Vec3(0.0, 0.0, -9.81);

/** Airframe constants. One instance per airframe design. */
export class AeroTables {
  readonly mass: number = 1.5;              // kg
  readonly wingArea: number = 0.5;          // m^2
  readonly wingspan: number = 1.5;          // m
  readonly inertiaX: number = 0.3;          // kg m^2 (roll)
  readonly inertiaY: number = 1.0;          // kg m^2 (pitch)
  readonly inertiaZ: number = 1.2;          // kg m^2 (yaw)

  // Lift. CL is clipped to [clMin, clMax]; beyond alphaStall the wing is
  // separated and lift stops growing with alpha (hard clip in v1).
  readonly cl0: number = 0.30;
  readonly clAlpha: number = 5.0;           // per rad
  readonly alphaStall: number = 0.35;      // rad (~20 deg)
  readonly clMin: number = -0.6;
  readonly clMax: number = 1.2;

  // Drag: parasite + induced + alpha-dependent separation drag.
  readonly cd0: number = 0.035;
  readonly kInduced: number = 0.06;
  readonly cdAlpha: number = 0.10;

  // Side force.
  readonly cyBeta: number = 0.15;

  // Pitch: positive Cm is nose DOWN (body +Y moment), which inverts the
  // textbook (z-down) sign. cmAlpha > 0 is pitch-stable here. cmDe maps the
  // friendly elevator sign (positive = nose up). cm0 is a slight nose-UP
  // bias so neutral trim is a gentle glide, not a dive.
  readonly cm0: number = -0.01;
  readonly cmAlpha: number = 0.35;
  readonly cmDe: number = 0.9;              // per rad, applied as -cmDe * elevator
  readonly cmQ: number = 12.0;              // damping magnitude, applied as decay

  // Roll: positive aileron = roll right = negative moment about +X.
  readonly clDa: number = 0.15;
  readonly clP: number = 0.45;              // damping magnitude

  // Yaw: positive rudder = nose right = positive moment about +Z.
  readonly cnDr: number = 0.08;
  readonly cnBeta: number = 0.12;
  readonly cnR: number = 0.25;              // damping magnitude

  // Propulsion: thrust along body forward, throttle in [0, 1].
  readonly thrustMax: number = 9.0;         // N

  get chord(): number {
    return this.wingArea / this.wingspan;
  }
}

/** Diagnostics for one integration step (tests, instruments, metrics). */
export interface AeroDiag {
  airspeed: number;
  alpha: number;
  beta: number;
  qDyn: number;
  liftCoefficient: number;
  lift: number;
  drag: number;
  thrust: number;
}

/** Anything with surface positions ([aileron, elevator, rudder, throttle]). */
export interface SurfacePositions {
  positions: ArrayLike<number>;
}

function bodyAirspeed(velocity: Vec3, forward: Vec3, right: Vec3, up: Vec3):
    [number, number, number] {
  return [velocity.dot(forward), velocity.dot(right), velocity.dot(up)];
}

function liftCoefficient(alpha: number, tables: AeroTables): number {
  const cl = tables.cl0 + tables.clAlpha * alpha;
  return Math.max(tables.clMin, Math.min(tables.clMax, cl));
}

/** World-frame aerodynamic + propulsion force. Returns force and diagnostics. */
export function aeroForce(velocity: Vec3, forward: Vec3, right: Vec3, up: Vec3,
                          servos: SurfacePositions, tables: AeroTables,
                          rho: number = RHO_SEA_LEVEL):
    { force: Vec3; diag: AeroDiag } {
  const speed = velocity.length();
  const throttle = servos.positions[3];
  const thrustVec = forward.scale(throttle * tables.thrustMax);
  if (speed < 0.5) {
    // Too slow for meaningful aerodynamics; propulsion and gravity only.
    return {
      force: thrustVec,
      diag: { airspeed: speed, alpha: 0.0, beta: 0.0, qDyn: 0.0,
              liftCoefficient: 0.0, lift: 0.0, drag: 0.0,
              thrust: throttle * tables.thrustMax },
    };
  }

  const [vx, vy, vz] = bodyAirspeed(velocity, forward, right, up);
  const alpha = Math.atan2(-vz, Math.max(vx, 1e-6));
  const beta = Math.atan2(vy, Math.hypot(vx, vz));
  const qDyn = 0.5 * rho * speed * speed;

  const cl = liftCoefficient(alpha, tables);
  const cd = tables.cd0 + tables.kInduced * cl * cl + tables.cdAlpha * Math.abs(alpha);

  const lift = qDyn * tables.wingArea * cl;
  const drag = qDyn * tables.wingArea * cd;
  const side = qDyn * tables.wingArea * tables.cyBeta * beta;

  const velUnit = velocity.scale(1.0 / speed);
  // Lift direction: body-up projected perpendicular to the velocity,
  // i.e. (vel x up) x vel = up_perp. (Order matters: the reversed cross
  // product yields -up_perp, an anti-lift airframe.)
  let liftDir = velUnit.cross(up).cross(velUnit);
  liftDir = liftDir.length() > 1e-9 ? liftDir.normalized() : up;
  const dragDir = velUnit.scale(-1.0);
  const sideDir = right.scale(-1.0); // beta > 0 (wind from right) pushes left

  const force = liftDir.scale(lift)
    .add(dragDir.scale(drag))
    .add(sideDir.scale(side))
    .add(thrustVec);
  const diag: AeroDiag = {
    airspeed: speed, alpha, beta, qDyn,
    liftCoefficient: cl, lift, drag,
    thrust: throttle * tables.thrustMax,
  };
  return { force, diag };
}

/** Body-frame aerodynamic moment WITHOUT damping (damping is implicit). */
export function controlStabilityMoment(alpha: number, beta: number,
                                       servos: SurfacePositions,
                                       tables: AeroTables, qDyn: number): Vec3 {
  const aileron = servos.positions[0];
  const elevator = servos.positions[1];
  const rudder = servos.positions[2];
  // Beyond stall the moment curve flattens: clip alpha's influence so the
  // restoring moment cannot pump energy during separated flow.
  const momentAlpha = Math.max(-2.0 * tables.alphaStall,
                               Math.min(2.0 * tables.alphaStall, alpha));
  const momentBeta = Math.max(-tables.alphaStall,
                              Math.min(tables.alphaStall, beta));
  const refMoment = qDyn * tables.wingArea * tables.chord;
  const refRoll = qDyn * tables.wingArea * tables.wingspan;
  const cx = refRoll * (-tables.clDa * aileron);
  const cy = refMoment * (tables.cm0 + tables.cmAlpha * momentAlpha
                          - tables.cmDe * elevator);
  const cz = refMoment * (tables.cnDr * rudder + tables.cnBeta * momentBeta);
  return new Vec3(cx, cy, cz);
}

/** Per-axis linear damping slopes (1/s) for implicit integration. */
export function dampingSlopePerAxis(qDyn: number, airspeed: number,
                                    tables: AeroTables,
                                    rho: number = RHO_SEA_LEVEL):
    [number, number, number] {
  const refMoment = tables.chord;
  const refRoll = tables.wingspan;
  if (airspeed < 0.5) {
    return [0.05, 0.05, 0.05];
  }
  const denom = 2.0 * airspeed;
  const kx = qDyn * tables.wingArea * refRoll * refRoll * tables.clP
    / denom / tables.inertiaX;
  const ky = qDyn * tables.wingArea * refMoment * refMoment * tables.cmQ
    / denom / tables.inertiaY;
  const kz = qDyn * tables.wingArea * refMoment * refMoment * tables.cnR
    / denom / tables.inertiaZ;
  return [kx + 0.05, ky + 0.05, kz + 0.05];
}
