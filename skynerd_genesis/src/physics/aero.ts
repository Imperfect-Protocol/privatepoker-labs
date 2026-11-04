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
  readonly mass: number = 1111.0;             // kg — the battle-Cessna (MTOW weight class)
  readonly wingArea: number = 16.17;          // m² — C172 (174 sq ft)
  readonly wingspan: number = 11.0;           // m — C172 (36 ft 1 in)
  readonly inertiaX: number = 1285.0;         // kg m² (roll)
  readonly inertiaY: number = 1825.0;         // kg m² (pitch)
  readonly inertiaZ: number = 2667.0;         // kg m² (yaw)

  // Lift. CL is clipped to [clMin, clMax]; beyond alphaStall the wing is
  // separated and lift stops growing with alpha (hard clip in v1).
  // clMax 2.2 puts the 1 g stall at ~50 mph at this wing loading — the
  // POH-style slow-speed end of the C172 envelope.
  readonly cl0: number = 0.30;
  readonly clAlpha: number = 5.0;             // per rad
  readonly alphaStall: number = 0.35;       // rad (~20 deg)
  readonly clMin: number = -1.0;
  readonly clMax: number = 2.2;

  // Drag: parasite + induced + alpha-dependent separation drag.
  // Balanced so cruise settles at ~85 mph at 2300 RPM (throttle 0.8).
  readonly cd0: number = 0.047;
  readonly kInduced: number = 0.053;
  readonly cdAlpha: number = 0.10;

  // Side force.
  readonly cyBeta: number = 0.15;

  // Pitch: positive Cm is nose DOWN (body +Y moment), which inverts the
  // textbook (z-down) sign. cmAlpha > 0 is pitch-stable here. cmDe maps the
  // friendly elevator sign (positive = nose up).
  readonly cm0: number = -0.01;
  readonly cmAlpha: number = 0.35;
  readonly cmDe: number = 0.9;                // per rad, applied as -cmDe * elevator
  readonly cmQ: number = 12.0;               // damping magnitude, applied as decay

  // Roll: positive aileron = roll right = negative moment about +X.
  readonly clDa: number = 0.15;
  readonly clP: number = 0.45;               // damping magnitude

  // Yaw: positive rudder = nose right = positive moment about +Z.
  readonly cnDr: number = 0.20;
  readonly cnBeta: number = 0.12;
  readonly cnR: number = 2.0;                // damping magnitude

  // Propulsion: a fixed-pitch prop behind a Lycoming-class engine.
  // Shaft power grows with the CUBE of the throttle (real props: power
  // tracks RPM^3), and thrust falls with airspeed as the prop unloads —
  // so full throttle is strong on the runway and modest at cruise, and
  // cruise at 2300 RPM (throttle 0.8) settles at 80-90 mph.
  // 3000 N static: the battle-C172 tune — enough to sustain the 2 g
  // (60 deg bank) level steep turn the syllabus demands.
  readonly thrustMax: number = 3000.0;        // N static, full throttle
  readonly propFalloffSpeed: number = 110.0; // m/s — thrust fades above this
  readonly propFloor: number = 0.30;         // never below 30% thrust

  // Ground effect: close to the ground the aircraft is a bit GLUED — a
  // suction pulling it down plus a nose-down pitching moment — easing off
  // with height until it is gone by 100 ft AGL. (The pilot holds the yoke
  // aft to keep the nose up until through it.)
  readonly groundEffectHeight: number = 30.48;   // m — 100 ft AGL, zero above
  readonly groundEffectPull: number = 0.12;      // max downforce, fraction of weight
  readonly groundEffectPitch: number = 0.08;     // max nose-down moment coefficient

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
  groundEffect: number;    // 0 far from the ground, 1 glued to it
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
                          rho: number = RHO_SEA_LEVEL,
                          agl: number = Number.POSITIVE_INFINITY):
    { force: Vec3; diag: AeroDiag } {
  const speed = velocity.length();
  const throttle = servos.positions[3];
  // Ground effect: the glue. Strongest at the surface, easing off with
  // height, gone by groundEffectHeight (100 ft). Infinity (no ground
  // data) means no effect at all.
  const geFactor = agl < tables.groundEffectHeight
    ? ((tables.groundEffectHeight - agl) / tables.groundEffectHeight) ** 2
    : 0.0;
  // A fixed-pitch prop: shaft power tracks the throttle cubed (power ∝
  // RPM³), and thrust unloads with airspeed. Static = thrustMax; cruise
  // gets a fraction of it.
  const power = throttle * throttle * throttle;
  const falloff = Math.max(tables.propFloor,
                           1.0 - (speed / tables.propFalloffSpeed) ** 2);
  const thrust = power * tables.thrustMax * falloff;
  const thrustVec = forward.scale(thrust);
  if (speed < 0.5) {
    // Too slow for meaningful aerodynamics; propulsion and gravity only.
    return {
      force: thrustVec,
      diag: { airspeed: speed, alpha: 0.0, beta: 0.0, qDyn: 0.0,
              liftCoefficient: 0.0, lift: 0.0, drag: 0.0,
              thrust: thrust, groundEffect: geFactor },
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
  // The glue: ground-effect suction pressing the aircraft toward the
  // ground, at most groundEffectPull of its weight, easing off aloft.
  const glue = geFactor * tables.groundEffectPull * tables.mass * 9.81;
  const glued = force.add(new Vec3(0.0, 0.0, -glue));
  const diag: AeroDiag = {
    airspeed: speed, alpha, beta, qDyn,
    liftCoefficient: cl, lift, drag,
    thrust,
    groundEffect: geFactor,
  };
  return { force: glued, diag };
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
