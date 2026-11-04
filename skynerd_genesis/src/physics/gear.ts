/**
 * THE GEAR: the wheels' contact laws, one focused type owning the
 * three forces a tricycle gear exerts on the airframe near the ground.
 *
 * The DATUM: the body's origin rides 5 ft above the terrain when the
 * wheels rest on it (the gear's own height — the school's runway
 * law: standing on the 95 ft plateau the instruments read 100 ft).
 * `agl` passed in is the datum's height above the terrain, measured
 * BEFORE the physics tick; the wheels' height after the tick is
 * derived exactly (the integrator advances position by the NEW
 * velocity, so the two agree to the tick).
 *
 * The three laws (all pitch-gated on the deck pitch, nose-up positive,
 * active only while the deck sits within ±15° of the ground plane):
 * 1. SUPPORT — wheels at or below resting height, within ±0.5 ft of
 *    it, arriving no faster than the oleo's limit: the gear carries
 *    its load — +1g upward cancelling gravity — the strut holds rest
 *    height and absorbs the sink rate. A harder arrival collapses the
 *    gear and the crash proceeds. Rising is free: the strut never
 *    pulls the aircraft down, so the wing lifts it off unimpeded.
 * 2. LEVELING — deck nose-low at contact: the ground acts as a lever,
 *    torqueing the nose toward level. Proportional to the lever
 *    force (zero at level, full at -15°), easing into level so the
 *    accumulated rate never carries the deck past it.
 * 3. GLUE — deck nose-up within the first 50 ft of the ground: a
 *    downforce toward the ground, relatively strong at the runway
 *    and relaxing completely by 50 ft — the rotation cannot balloon
 *    off the deck.
 */
import { Vec3 } from "../core/vec.js";
import type { RigidBody } from "./rigid-body.js";

const FT = 0.3048;
const DEG = Math.PI / 180.0;
const GRAVITY = 9.81;

/** The gear's height: the datum rides this far above the terrain when
 *  the wheels rest on it (5 ft — the runway law's instrument offset). */
export const GEAR_M = 5.0 * FT;
/** The wheels' contact band, around resting height. */
export const GEAR_BAND_M = 0.5 * FT;
/** The deck's tangent band: within ±15° the deck sits on its wheels. */
export const GEAR_PITCH = 15.0 * DEG;
/** The oleo's sink limit: a firmer arrival collapses the gear. */
export const GEAR_SINK_MAX_MPS = 3.0;
/** The glue's reach: gone by 50 ft above the resting height. */
export const GEAR_GLUE_AGL_M = 50.0 * FT;
/** The glue's strength at the runway, in g. */
export const GEAR_GLUE_G = 0.25;
/** The leveling's ease into level: the rate target's time constant. */
export const GEAR_LEVEL_EASE_S = 0.9;
/** How fast the deck rate chases the level-off target. */
export const GEAR_LEVEL_RESPONSE = 6.0;

/** The wheels: applied after each physics tick, wherever an aircraft
 *  flies near the terrain. Stateless — every law is a pure function
 *  of the body's state and its height above the ground. */
export class GearContact {
  /** One tick of the wheels' forces on `body`, whose datum sits `agl`
   *  metres above the terrain (measured before the tick). */
  apply(body: RigidBody, agl: number, dt: number): void {
    const deck = body.deckPitch;
    if (deck < -GEAR_PITCH || deck > GEAR_PITCH) {
      return;                       // outside the tangent band: no gear law
    }
    // THE WHEELS' HEIGHT above the ground: 0 at rest. `agl` predates
    // the tick; the tick moved the datum by velocity·dt, so the
    // after-tick height is exact — and restZ (the datum's resting
    // height) is invariant under the same shift.
    const wheel = agl - GEAR_M;
    const wheelNow = wheel + body.velocity.z * dt;
    const restZ = body.position.z - wheelNow;
    const inBand = Math.abs(wheelNow) <= GEAR_BAND_M;
    // 1. SUPPORT: wheels at or below rest, within the band, inside
    //    the oleo's limit — the strut carries the load at rest height
    if (inBand && wheelNow <= 0.0 && body.velocity.z >= -GEAR_SINK_MAX_MPS) {
      let vz = body.velocity.z + GRAVITY * dt;   // the +1g, cancelling gravity
      if (vz < 0.0) {
        vz = 0.0;                  // the oleo absorbs the sink rate
      }
      body.velocity = new Vec3(body.velocity.x, body.velocity.y, vz);
      body.position = new Vec3(body.position.x, body.position.y, restZ);
    }
    // 2. LEVELING: nose-low deck at contact — the lever rights the
    //    nose, proportional to the nose-down depth (zero at level,
    //    full at -15°), easing into level with no overshoot
    if (inBand && wheelNow <= 0.0 && deck < 0.0) {
      const targetRate = deck / GEAR_LEVEL_EASE_S;   // nose-up, ∝ depth
      const blend = Math.min(1.0, GEAR_LEVEL_RESPONSE * dt);
      body.rateQ += (targetRate - body.rateQ) * blend;
    }
    // 3. GLUE: nose-up deck within the first 50 ft — a downforce
    //    toward the ground, strong at the runway, gone by 50 ft
    if (deck > 0.0) {
      const fade = Math.max(0.0, 1.0 - Math.max(0.0, wheelNow) / GEAR_GLUE_AGL_M);
      if (fade > 0.0) {
        const vz = body.velocity.z - GRAVITY * GEAR_GLUE_G
          * (deck / GEAR_PITCH) * fade * dt;
        body.velocity = new Vec3(body.velocity.x, body.velocity.y, vz);
      }
    }
  }
}
