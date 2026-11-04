/**
 * THE GUARDIAN — the instructor's takeover judgment, shared by the
 * classroom and the swarm. Who holds the levers is a property of the
 * AIRCRAFT (sticky per machine, switchable any tick, mid-air or on
 * the ground):
 *
 *   ai            — the AI's own hands fly the machine;
 *   instructor    — the instructor flies the AI's current intent;
 *   both-suggest  — the AI suggests, the instructor flies, and the
 *                    error wave teaches the AI continuously;
 *   both-guarded  — the AI flies; the instructor watches; a dangerous
 *                    state hands the levers over (suggest mode while
 *                    recovering), a stable stretch hands them back.
 *
 * The danger predicate is the instructor's own judgment: the stall
 * family, an overbanked attitude, a nose dive, a hard sink, or
 * overspeed. The recovery is the twin condition: not dangerous for a
 * sustained window.
 */
import { RigidBody } from "../physics/rigid-body.js";

export type ControlMode =
  | "ai"
  | "instructor"
  | "both-suggest"
  | "both-guarded";

export const CONTROL_MODES: readonly ControlMode[] = [
  "ai", "instructor", "both-suggest", "both-guarded",
];

/** The stick-and-rudder label of a mode (the UI's status line and the
 *  controls dropdown). The values stay the machine's own; only what the
 *  person reads changes: SOLO = one pilot aboard, PAIR = both pilots
 *  aboard (the AI suggesting or flying under the instructor's
 *  guardian), named by WHOSE HANDS FLY. */
export function controlModeLabel(mode: ControlMode): string {
  switch (mode) {
    case "ai": return "Solo: AI";
    case "instructor": return "Solo: Instructor";
    case "both-suggest": return "Pair: Instructor";
    case "both-guarded": return "Pair: AI";
  }
}

const MPH_PER_MPS = 2.23694;
const DEG = Math.PI / 180.0;

/** The stall's edge: below this the wing is about to give up. */
const DANGER_SPEED_MPH = 55.0;
/** Overspeed: past Vne the airframe is failing. */
const DANGER_OVERSPEED_MPH = 145.0;
/** An overbanked spiral's edge. */
const DANGER_BANK_RAD = 50.0 * DEG;
/** A nose dive steeper than a recovery descent. */
const DANGER_PITCH_RAD = -15.0 * DEG;
/** A sink that is a spiral dive's signature, m/s (about −2400 fpm). */
const DANGER_SINK_MPS = -12.0;

/** The instructor's danger judgment of one machine's state. */
export function isDangerous(body: RigidBody): boolean {
  const speed = body.velocity.length() * MPH_PER_MPS;
  const bank = Math.abs(body.bank);
  const pitch = body.deckPitch;
  const sink = body.velocity.z;
  return speed < DANGER_SPEED_MPH
      || speed > DANGER_OVERSPEED_MPH
      || bank > DANGER_BANK_RAD
      || pitch < DANGER_PITCH_RAD
      || sink < DANGER_SINK_MPS;
}

/** How long the machine must stay out of danger before the guarded
 *  mode hands the levers back to the AI, in seconds. */
export const RECOVERY_HOLD_S = 2.0;

/** The stabilizer's altitude law: a cascade, altitude -> target pitch
 *  -> elevator. The pitch target carries the turn's back-pressure
 *  (a banked wing needs the nose up to hold height), and the
 *  elevator chases the pitch error at the hand's authority. */
const ALT_TO_PITCH = 0.004;         // rad of target pitch per metre
const VS_TO_PITCH = 0.06;            // rate damping: sink raises the nose
const PITCH_CLAMP = 0.26;            // ±15°
const TURN_BACK_PRESSURE = 0.35;     // pitch per radian of bank
const ELEV_GAIN = 0.4;              // elevator per radian of pitch error
const ELEV_CLAMP = 0.35;
const TRIM_ELEV = 0.02485;
const TRIM_THROTTLE = 0.8;
/** Beyond this radius from the world's origin the stabilizer banks
 *  toward home — a wings-level drone in a bounded world would fly
 *  straight out of it. Early and steep enough to turn back: the
 *  boundary is 800 m, the turn radius at this bank about 220 m. */
const HOME_RADIUS_M = 400.0;
/** The homing bank, radians (about 34° — a decisive turn home). */
const HOME_BANK = 0.60;
/** The bank law's aileron gain (deflection per radian of bank error). */
const BANK_GAIN = 0.8;
const BANK_CLAMP = 0.175;

/**
 * THE STABILIZER — the instructor's free-flight law (the swarm has no
 *  lesson chain): hold the captured altitude, wings level, ball
 *  centered, trim speed. Continuous targets the easing hand chases.
 * Far from the world's origin, a decisive bank steers it home.
 */
export function stabilizeTargets(capturedAltM: number,
                                 body: RigidBody): Float64Array {
  const altError = capturedAltM - body.position.z;
  const distance = Math.hypot(body.position.x, body.position.y);
  let targetBank = 0.0;
  if (distance > HOME_RADIUS_M) {
    // Steer toward the origin: bank into whichever side the nose must
    // turn (positive bank turns right).
    const [forward] = body.bodyAxes;
    const heading = Math.atan2(forward.y, forward.x);
    const bearingHome = Math.atan2(-body.position.y, -body.position.x);
    let relative = bearingHome - heading;
    while (relative > Math.PI) {
      relative -= 2.0 * Math.PI;
    }
    while (relative < -Math.PI) {
      relative += 2.0 * Math.PI;
    }
    targetBank = Math.sign(relative) * HOME_BANK;
  }
  const bankError = targetBank - body.bank;
  const aileron = Math.max(-BANK_CLAMP,
    Math.min(BANK_CLAMP, BANK_GAIN * bankError));
  // The pitch target: the altitude error and the sink rate (a steady
  // sink must raise the nose before it becomes a descent), plus the
  // turn's back-pressure — a banked wing holds height nose-high.
  const targetPitch = Math.max(-PITCH_CLAMP, Math.min(PITCH_CLAMP,
    ALT_TO_PITCH * altError - VS_TO_PITCH * body.velocity.z))
    + TURN_BACK_PRESSURE * Math.abs(body.bank);
  const pitchError = targetPitch - body.deckPitch;
  const elevator = TRIM_ELEV + Math.max(-ELEV_CLAMP,
    Math.min(ELEV_CLAMP, ELEV_GAIN * pitchError));
  return new Float64Array([aileron, elevator, 0.0, TRIM_THROTTLE]);
}
