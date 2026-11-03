/**
 * The flight school: an instructor demonstrates, the drone only remembers.
 * Redesigned for the C172 airframe (1111 kg, stall ~50 mph, cruise 85 mph
 * at 2300 RPM): each lesson is now a PHASE-BASED PROCEDURE — a real
 * takeoff (yoke neutral on the roll, pitch 0 on the gear, rotate at
 * 60 mph into a 15° climb to 1000 ft), full 180° turns held
 * level with the elevator, steep turns with top rudder, and the complete
 * stall / spin recovery sequences — flown by feedback laws, smoothly.
 *
 * The instructor is a controller with a HAND MODEL: raw phase targets are
 * eased toward at bounded rates (rad/s), so every input on the flight
 * controls is smooth — no steps, no jerks, even at phase transitions. The
 * student never sees the controller; it sees instrument state -> servo
 * nudges (push / ease / hold), stamped into the servo tensor at the demo's
 * established phases only (the stall/spin ENTRY is never stamped — the
 * student must never learn to enter a departure; the recovery is the
 * lesson).
 *
 * The graduation exam is flown by the student alone (memory-only replay),
 * graded 0-6 on mean panel drift from the demo. Failing lessons are
 * drilled: the student repeats, the instructor takes the controls at every
 * drift out of the demo band and stamps the correction at the student's
 * actual state (double strength), until the class graduates or the round
 * cap hits. Every flight is step-wise (one tick per step()) so the browser
 * runs the classroom live; teach() drives the same flights headlessly.
 */
import { Vec3 } from "../core/vec.js";
import { Quaternion } from "../core/quaternion.js";
import { World } from "../core/world.js";
import { NumpyPcg64 } from "../core/rng.js";
import { RigidBody } from "../physics/rigid-body.js";
import { SurfacesState, SURFACE_LIMITS } from "../physics/surfaces.js";
import { AeroTables, GRAVITY, RHO_SEA_LEVEL } from "../physics/aero.js";
import { FlightInstruments } from "./instruments.js";
import { ServoPolicy, HOLD, PUSH, EASE_OFF, HOLD_ALL_SURFACE } from "./servo-tensor.js";
import { TRIM_SPEED, TRIM_COMMAND, springTowardNeutral } from "./flight.js";

// Desire slots the school names in the will. After a lesson, waking that
// desire means wanting that maneuver.
export const DESIRE_LEFT = 0;
export const DESIRE_RIGHT = 1;
export const DESIRE_CLIMB = 2;
export const DESIRE_DESCEND = 3;
export const DESIRE_LEVEL = 4;
export const DESIRE_RECOVER = 5;

export const DECISION_EVERY = 10;        // same 5 Hz decision clock as flight
const AILERON_LIMIT = SURFACE_LIMITS[0][1];
const ELEVATOR_LIMIT = SURFACE_LIMITS[1][1];
const RUDDER_LIMIT = SURFACE_LIMITS[2][1];

// Level-off capture: the instructor's pitch-down to break a climb. The
// push is FIRMLY proportional to the remaining nose-up excess and EASES
// OFF continuously as the nose settles — the yoke reads as one decisive
// forward input, gradually released, and the nose arrives at the level
// attitude with the pitch rate dying to zero: no overshoot, no bobbing.
const CAPTURE_GAIN = 0.8;     // commanded nose-down rate per excess (1/s)
const CAPTURE_RATE_MAX = 0.12; // rad/s — the firm nose-down, ~7 deg/s
const CAPTURE_KD = 1.5;       // rate tracking: how hard the push chases it
const CAPTURE_DONE = 0.03;    // rad (~1.7 deg) excess: capture -> normal hold
const CAPTURE_RATE_DEAD = 0.02; // rad/s: the arrival is gentle — rate dead too
// The lead before the level-off altitude where the capture starts — the
// climb bleeds off over the capture, so the arrival lands ON the altitude.
export const LEVEL_OFF_LEAD_S = 2.7;   // seconds of remaining climb to lead by

// The graduation exam is graded 0-6 on mean panel drift.
export const QUALITY_CAP = 6.0;
export const QUALITY_SLOPE = 2.0;
export const DEFAULT_PASS_SCORE = 4.0;
export const GRADUATION_ROUNDS = 100;

export const GRADE_NAMES: readonly (readonly [number, string])[] = [
  [5.5, "exceptional"],
  [4.5, "very good"],
  [3.5, "good"],
  [2.5, "pass"],
  [1.5, "below expectations"],
  [0.5, "must repeat"],
];

/** The report-card word for a 0-6 score. */
export function gradeName(score: number): string {
  for (const [threshold, name] of GRADE_NAMES) {
    if (score >= threshold) {
      return name;
    }
  }
  return "failed badly";
}

export const DEMO_BAND = 0.15;   // panel drift that hands the controls to the instructor
export const STRETCH_DECISIONS = 15;   // one takeover lasts at most 3 s

// The syllabus numbers, in SI (the user's C172 procedure spec).
export const SYLLABUS = {
  rotateMps: 60.0 / 2.23694,        // 60 mph
  stallMps: 50.0 / 2.23694,         // 50 mph
  stallEntryMps: 51.0 / 2.23694,    // decelerate through 51 mph
  diveOutMps: 65.0 / 2.23694,       // 65 mph
  spinPullUpMps: 60.0 / 2.23694,    // 60 mph
  runwayM: 10.0 / 3.28084,          // 10 ft
  liftOffToM: 100.0 / 3.28084,      // 100 ft AGL — hold the pull through ground effect
  climbToM: 1000.0 / 3.28084,       // 1000 ft
  stallAltM: 3000.0 / 3.28084,      // 3000 ft
  bankTurn: 25.0 * Math.PI / 180.0,   // 20-30 deg band, aim 25
  bankSteep: 60.0 * Math.PI / 180.0,  pitchClimb: 15.0 * Math.PI / 180.0,
  pitchDive: -30.0 * Math.PI / 180.0,
  pitchRecovery: 15.0 * Math.PI / 180.0,    // both recoveries climb back at 15 deg
  turnAmount: Math.PI,              // 180 horizontal degrees
  turnLeadOut: 0.18,               // rad: start the roll-out early so the
                                   // heading-hold lands exactly on 180
  steepLeadOut: 0.52,             // rad: same, for the 60 deg steep turn
                                   // (bigger bank turns longer to level)
  cruiseSpeedMps: 38.0,            // 85 mph — the speed held in turns
  steepSpeedMps: 48.0,             // 107 mph — the 2 g steep-turn equilibrium
  cruiseThrottle: 0.8,             // 2300 RPM
  idleThrottle: 0.1,
} as const;

// numpy.random.default_rng(0) PCG64 state, frozen: the classroom's dice.
const RNG0_STATE = 35399562948360463058890781895381311971n;
const RNG0_INC = 87136372517582989555478159403783844777n;

function clamp(value: number, limit: number): number {
  return Math.max(-limit, Math.min(limit, value));
}

function wrapPi(angle: number): number {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

/** Current bank angle in radians (positive = right). */
export function bankRadTargetOf(body: RigidBody): number {
  return body.bank;
}

/** Deck pitch, positive = nose up. */
export function pitchOf(body: RigidBody): number {
  return body.deckPitch;
}

/**
 * The trimmed cruise deck angle: lift = weight at TRIM_SPEED, sea level.
 * (The constructor's pitch parameter is nose-DOWN-positive; spawns that
 * need a nose-up attitude pass the negative of this.)
 */
export function cruiseTrimAlpha(): number {
  const t = new AeroTables();
  const cl = (2.0 * t.mass * 9.81) / (1.225 * TRIM_SPEED * TRIM_SPEED * t.wingArea);
  return (cl - t.cl0) / t.clAlpha;
}

/** The airframe every school helper computes against. */
const SCHOOL_TABLES = new AeroTables();

/**
 * The LEVEL-ATTITUDE trim: the deck pitch at which lift's vertical
 * component carries the weight at this speed and bank. This is the
 * "nose slightly above the horizon" the instructor holds to keep VSI at
 * 0 — the attitude rises as the turn loads the wing and as speed bleeds.
 */
export function levelAttitudeRad(airspeed: number, bank: number): number {
  const t = SCHOOL_TABLES;
  const cosBank = Math.max(0.2, Math.cos(bank));
  const speed = Math.max(airspeed, 1.0);
  const cl = (2.0 * t.mass * 9.81)
    / (RHO_SEA_LEVEL * speed * speed * t.wingArea * cosBank);
  const alpha = (cl - t.cl0) / t.clAlpha;
  return Math.max(0.0, Math.min(t.alphaStall, alpha));
}

/**
 * Where the controls rest when a flight begins. On the runway the aircraft
 * starts parked: yoke neutral, throttle idle. Anywhere else it is already
 * flying at the trimmed cruise, so the controls start at the trim.
 */
export function startCommandsFor(
    startKind: "runway" | "cruise" | "cruise1000" | "cruise3000"): Float64Array {
  return startKind === "runway"
    ? Float64Array.of(0.0, 0.0, 0.0, 0.0)
    : Float64Array.from(TRIM_COMMAND);
}

/**
 * The landing gear while the wheels are on the runway: wings level, and
 * the deck no lower than the ground attitude (pitch 0). While no phase
 * commands rotation the deck is HELD level — that is the takeoff roll,
 * yoke neutral, the pitch gauge reading 0. Once a phase pitches up (the
 * rotation) only the floor applies: the nose comes up off the gear freely.
 */
function holdGroundDeck(body: RigidBody, deckLocked: boolean): void {
  const pitch = pitchOf(body);
  const bank = bankRadTargetOf(body);
  const heldPitch = deckLocked ? 0.0 : Math.max(0.0, pitch);
  body.rateP = 0.0;                                  // the wheels hold wings level
  body.rateQ = deckLocked || pitch < 0.0 ? 0.0 : Math.min(0.0, body.rateQ);
  if (Math.abs(bank) < 1e-9 && Math.abs(pitch - heldPitch) < 1e-9) {
    return;
  }
  const [forward] = body.bodyAxes;
  const yaw = Math.atan2(forward.y, forward.x);
  body.attitude = Quaternion.fromEuler(yaw, -heldPitch, 0.0);
}

/** What a phase law sees of the aircraft. */
export interface PhaseReadings {
  airspeed: number;      // m/s
  altitudeM: number;
  bank: number;          // rad, positive right
  pitch: number;         // rad, positive nose up
  rateP: number;
  rateQ: number;
  rateR: number;
  vsMps: number;
  /** Heading change since this phase started, rad (positive = right turn). */
  turned: number;
  /** Heading rate, rad/s (positive = turning right). */
  headingRate: number;
  /** Ticks (1/50 s) spent in this phase. */
  ticks: number;
  /** Angle of attack, rad (for stall detection). */
  alpha: number;
}

/** Rudder jobs: a fixed deflection, top rudder in a bank, or anti-spin. */
export type RudderLaw = number | "top" | "antiSpin" | "proSpin";

/** One stage of a maneuver: a control law plus an exit condition. */
export interface ManeuverPhase {
  readonly name: string;
  /** Stamp this phase's readings into the student's memory. */
  readonly stamp: boolean;
  /** Hold the aircraft on the runway while in this phase (takeoff roll). */
  readonly pin?: boolean;
  /** Hold this bank (rad) with the ailerons, if defined. */
  readonly bankTarget?: number;
  /** Carry the heading accumulator across this phase (turn roll-in/out). */
  readonly carryTurn?: boolean;
  /** Roll out onto exactly entry-heading ± 180°: the bank targets the
   * remaining heading, so the turn finishes on the nose. */
  readonly headingHold?: boolean;
  /** Hold vertical speed (m/s) with the elevator, if defined. */
  readonly vsTarget?: number;
  /** LEVEL-OFF CAPTURE: break the climb with a firm nose-down that eases
   * off with the remaining pitch excess — gentle arrival, no oscillation. */
  readonly pitchCapture?: boolean;
  /** Hold this airspeed (m/s) with the throttle, if defined. */
  readonly speedTarget?: number;
  /** Hold deck pitch (rad, nose-up positive) with the elevator, if defined. */
  readonly pitchTarget?: number;
  /** Fixed deflection override: aileron (used by the spin entry). */
  readonly aileronOverride?: number;
  /** Elevator override (used by stall-break neutrals and spin back pressure). */
  readonly elevatorOverride?: number;
  readonly rudder: RudderLaw;
  readonly throttle: number;
  /** Phase exit condition. */
  readonly done: (r: PhaseReadings) => boolean;
}

/**
 * The instructor's hand: eases commanded deflections toward the raw phase
 * targets at bounded rates. All inputs on the flight controls are smooth —
 * the yoke never jumps, even when a phase changes the law underneath.
 */
export class InstructorHand {
  readonly commands: Float64Array;
  /** Per-channel easing rates: aileron, elevator, rudder (rad/s), throttle (/s). */
  private static readonly RATES = [0.6, 0.4, 0.6, 0.4];

  constructor(commands: ArrayLike<number> = TRIM_COMMAND) {
    this.commands = Float64Array.from(commands);
  }

  /** One tick of easing toward the raw targets. */
  easeToward(targets: ArrayLike<number>, dt: number): Float64Array {
    for (let channel = 0; channel < 4; channel++) {
      const rate = InstructorHand.RATES[channel] * dt;
      const delta = clamp(targets[channel] - this.commands[channel], rate);
      this.commands[channel] += delta;
    }
    return this.commands;
  }
}

/** Tracks which phase a flight is in and what it has accumulated there. */
export class PhaseCursor {
  index = 0;
  ticks = 0;
  turned = 0.0;
  pitchIntegral = 0.0;   // residual only — the stick feedforward carries the trim
  /** Turn direction while a heading-hold roll-out is flying (+1 right, −1 left). */
  turnDirection = 1;
  private lastHeading = 0.0;

  constructor(readonly phases: readonly ManeuverPhase[], body: RigidBody) {
    const [forward] = body.bodyAxes;
    this.lastHeading = Math.atan2(forward.y, forward.x);
  }

  get current(): ManeuverPhase {
    return this.phases[Math.min(this.index, this.phases.length - 1)];
  }

  get finished(): boolean {
    return this.index >= this.phases.length;
  }

  /** Advance per-tick bookkeeping; returns true when the phase completed. */
  advance(body: RigidBody): boolean {
    if (this.finished) {
      return false;
    }
    const [forward] = body.bodyAxes;
    const heading = Math.atan2(forward.y, forward.x);
    this.turned += wrapPi(heading - this.lastHeading);
    this.lastHeading = heading;
    this.ticks += 1;
    const phase = this.current;
    if (phase.done(this.readings(body))) {
      const next = this.phases[this.index + 1];
      // A turn's 180 degrees span the roll-in, the turn, AND the roll-out
      // (which lands exactly on the target heading): phases flagged
      // carryTurn keep the heading accumulator.
      const carry = next !== undefined && next.carryTurn === true;
      if (next !== undefined && next.headingHold === true) {
        this.turnDirection = this.turned >= 0.0 ? 1 : -1;
      }
      this.index += 1;
      this.ticks = 0;
      this.pitchIntegral = 0.0;   // the stick feedforward retrims instantly
      if (!carry) {
        this.turned = 0.0;
      }
      return true;
    }
    return false;
  }

  /** The readings the current phase law and exit conditions see. */
  readings(body: RigidBody): PhaseReadings {
    const [forward, , up] = body.bodyAxes;
    const speed = body.velocity.length();
    // alpha: velocity against the deck (positive = wind from below)
    const vFwd = body.velocity.dot(forward);
    const vUp = body.velocity.dot(up);
    const alpha = speed > 0.5 ? Math.atan2(-vUp, Math.max(vFwd, 1e-6)) : 0.0;
    const heading = Math.atan2(forward.y, forward.x);
    // The heading rate, from the physics of the turn itself (stateless, so
    // no call-order traps): a banked aircraft turns at g·tan(bank)/V_h.
    const bank = bankRadTargetOf(body);
    const vH = Math.hypot(body.velocity.x, body.velocity.y);
    const turnRate = vH > 5.0
      ? 9.81 * Math.tan(Math.max(-1.4, Math.min(1.4, bank))) / vH
      : 0.0;
    return {
      airspeed: speed,
      altitudeM: body.position.z,
      bank: bankRadTargetOf(body),
      pitch: pitchOf(body),
      rateP: body.rateP,
      rateQ: body.rateQ,
      rateR: body.rateR,
      vsMps: body.velocity.z,
      turned: this.turned,
      headingRate: turnRate,
      ticks: this.ticks,
      alpha,
    };
  }

  /** The instructor's raw deflection targets for the current phase. */
  phaseTargets(body: RigidBody): Float64Array {
    const phase = this.current;
    const r = this.readings(body);
    // The bank being held: the phase's, or — in a heading-hold roll-out —
    // the bank that closes the remaining turn onto exactly ±180°.
    const bankTarget = phase.headingHold === true
      ? clamp(1.4 * wrapPi(this.turnDirection * Math.PI - r.turned), AILERON_LIMIT)
      : phase.bankTarget;
    // aileron: bank-hold (or a fixed deflection in the spin entry) with
    // TURN-KINEMATICS compensation: in a banked, nose-up turn the heading
    // rotation bleeds into the Euler bank (p = φ̇ − ψ̇ sinθ), so holding
    // bank needs continuous into-turn roll. The feedforward supplies that
    // roll and the damping acts on the true bank rate — the P loop then
    // holds the bank with no steady error.
    let aileron: number;
    if (phase.aileronOverride !== undefined) {
      aileron = phase.aileronOverride;
    } else if (bankTarget === undefined) {
      aileron = 0.0;
    } else {
      // The roll feedforward needs flying airspeed; parked on the runway
      // (qDyn ~ 0) it is undefined and the plain bank-hold flies the roll.
      if (r.airspeed < 5.0) {
        aileron = clamp(0.8 * (bankTarget - r.bank) + 0.9 * r.rateP,
                        AILERON_LIMIT);
      } else {
      const qDyn = 0.5 * RHO_SEA_LEVEL * r.airspeed * r.airspeed;
      const kx = qDyn * SCHOOL_TABLES.wingArea * SCHOOL_TABLES.wingspan
        * SCHOOL_TABLES.wingspan * SCHOOL_TABLES.clP
        / (2.0 * r.airspeed * SCHOOL_TABLES.inertiaX) + 0.05;
      const requiredP = r.headingRate * Math.sin(r.pitch);  // ψ̇·sin(pitch)
      const rollFF = -requiredP * SCHOOL_TABLES.inertiaX * kx
        / (qDyn * SCHOOL_TABLES.wingArea * SCHOOL_TABLES.wingspan
           * SCHOOL_TABLES.clDa);
      aileron = clamp(0.8 * (bankTarget - r.bank)
                      + 0.9 * (r.rateP - requiredP) + rollFF,
                      AILERON_LIMIT);
      }
    }
    // elevator: pitch-hold, vs-hold (with a slow integral), or an override
    let elevator: number;
    if (phase.elevatorOverride !== undefined) {
      elevator = phase.elevatorOverride;
    } else {
      // The LEVEL-OFF CAPTURE: coming down from a climb, the instructor
      // applies a FIRM nose-down and then REDUCES it gradually — the
      // commanded pitch rate is proportional to the remaining nose-up
      // excess (positive q is nose-down in this body convention), so the
      // input eases off as the nose settles and the arrival is gentle:
      // pitch rate -> 0 exactly as pitch -> level.
      const levelPitch = levelAttitudeRad(r.airspeed, r.bank);
      const excess = r.pitch - levelPitch;
      if (phase.pitchCapture === true
          && (excess > CAPTURE_DONE || Math.abs(r.rateQ) > CAPTURE_RATE_DEAD)) {
        // The FIRM INPUT, REDUCED GRADUALLY: the instructor pushes the
        // nose down at a commanded rate (positive q is nose-down) — a real,
        // visible push while the rotation builds — and the commanded rate
        // DECAYS with the remaining excess, so the input eases off
        // continuously and the nose arrives at level with the rate dying
        // to zero: no overshoot, no bobbing, no phugoid. No vs feedback:
        // the flight path bleeds off smoothly with the attitude.
        const commandedRate = Math.min(CAPTURE_GAIN * excess, CAPTURE_RATE_MAX);
        const stickFF = (SCHOOL_TABLES.cm0 + SCHOOL_TABLES.cmAlpha * levelPitch)
          / SCHOOL_TABLES.cmDe;
        // The trim integral stays frozen through the capture (the stick
        // feedforward holds the trim); the normal hold resumes it, from 0.
        elevator = clamp(stickFF + CAPTURE_KD * (r.rateQ - commandedRate),
                         ELEVATOR_LIMIT);
      } else {
      // The pitch target: the phase's commanded attitude, or — in the
      // "VSI 0" phases — the level trim at the CURRENT speed and bank: the
      // nose slightly above the horizon, lift's vertical component
      // carrying the weight. Holding that attitude IS holding VSI at 0.
      // A sinking flight path raises the nose (back pressure against the
      // sink) — the outer VS loop closing what the feedforward leaves.
      const feedforward = phase.pitchTarget !== undefined
        ? phase.pitchTarget
        : levelAttitudeRad(r.airspeed, r.bank)
          + clamp(0.06 * ((phase.vsTarget ?? 0.0) - r.vsMps), 0.25);
      const targetPitch = feedforward;
      // Attitude hold with a STICK-POSITION FEEDFORWARD: the static
      // elevator that trims this alpha is (cm0 + cmAlpha·alpha)/cmDe —
      // exactly the cruise trim 0.0249 at the cruise alpha. With the trim
      // placed instantly, the slow integral only mops up the residual
      // (dynamic-pressure shifts, the ground-effect glue), so the nose
      // does not sag through roll-ins and phase changes.
      const stickFF = (SCHOOL_TABLES.cm0 + SCHOOL_TABLES.cmAlpha * targetPitch)
        / SCHOOL_TABLES.cmDe;
      this.pitchIntegral = Math.max(-0.05, Math.min(0.15,
        this.pitchIntegral + 0.30 * (targetPitch - r.pitch) * (1.0 / 50.0)));
      elevator = clamp(0.9 * (targetPitch - r.pitch) + this.pitchIntegral
                       + 0.25 * r.rateQ + stickFF,
                       ELEVATOR_LIMIT);
      }
    }
    // rudder: fixed, top rudder (steep turns), anti-spin, or pro-spin entry
    let rudder: number;
    if (phase.rudder === "top") {
      // A GENTLE top rudder: enough to see on the pedals (right rudder in
      // a left steep turn), not enough to crab the nose off the path.
      rudder = phase.bankTarget === undefined ? 0.0
        : -Math.sign(phase.bankTarget) * 0.05;
    } else if (phase.rudder === "antiSpin") {
      rudder = clamp(-0.5 * r.rateR, RUDDER_LIMIT);
    } else if (phase.rudder === "proSpin") {
      rudder = -RUDDER_LIMIT;
    } else {
      rudder = phase.rudder;
    }
    // throttle: the phase's setting, nudged by a speed hold when the phase
    // asks for one — the instructor flies attitude on the yoke and SPEED
    // ON THE POWER, the way a real pilot does.
    const throttle = phase.speedTarget === undefined
      ? phase.throttle
      : Math.max(0.0, Math.min(1.0,
          phase.throttle + 0.05 * (phase.speedTarget - r.airspeed)));
    return Float64Array.of(aileron, elevator, rudder, throttle);
  }
}

/**
 * One standard maneuver: a spawn state plus the phase procedure that flies
 * it. `desire`/`targets` carry the lesson's intent (the will handshake is
 * a later milestone); `durationCap` bounds the flight even if a phase
 * never completes.
 */
export class Lesson {
  readonly phases: readonly ManeuverPhase[];
  readonly durationCap: number;

  constructor(readonly name: string, readonly desire: number,
              readonly targets: readonly number[],
              phases: readonly ManeuverPhase[], options: {
                durationCap?: number,
                startKind?: "runway" | "cruise" | "cruise1000" | "cruise3000",
              } = {}) {
    this.phases = phases;
    this.durationCap = options.durationCap ?? 60.0;
    this.startKind = options.startKind ?? "cruise";
  }
  readonly startKind: "runway" | "cruise" | "cruise1000" | "cruise3000";

  /** The classroom start state for this maneuver. */
  spawn(): RigidBody {
    const trim = cruiseTrimAlpha();
    switch (this.startKind) {
      case "runway":
        return new RigidBody(new Vec3(0.0, 0.0, SYLLABUS.runwayM),
                              new Vec3(0.0, 0.0, 0.0));
      case "cruise1000":
        return new RigidBody(new Vec3(0.0, 0.0, SYLLABUS.climbToM),
                              new Vec3(TRIM_SPEED, 0.0, 0.0),
                              0.0, -trim, 0.0);
      case "cruise3000":
        return new RigidBody(new Vec3(0.0, 0.0, SYLLABUS.stallAltM),
                              new Vec3(TRIM_SPEED, 0.0, 0.0),
                              0.0, -trim, 0.0);
      default:
        return new RigidBody(new Vec3(0.0, 0.0, SYLLABUS.climbToM),
                              new Vec3(TRIM_SPEED, 0.0, 0.0),
                              0.0, -trim, 0.0);
    }
  }
}

export interface DemoTickOptions {
  takeover?: boolean;
  studentThrottle?: number | null;
  timeEpoch?: number;
  amount?: number | null;
  demoNearTrim?: boolean;
}

export interface DemoTickResult {
  stamped: boolean;
}

/** Advance one fixed tick; returns false when the flight has ended. */
export interface Stepper {
  step(): boolean;
  readonly body: RigidBody;
  readonly servos: SurfacesState;
  readonly commands: Float64Array;
  readonly lesson: Lesson;
  readonly cursor: PhaseCursor;
  readonly seconds: number;
  readonly finished: boolean;
}

/** Shared tick machinery: phase advance, the pin, ground contact. */
  abstract class ClassroomFlight implements Stepper {
  readonly body: RigidBody;
  readonly servos = new SurfacesState();
  readonly cursor: PhaseCursor;
  tick = 0;
  protected ended = false;
  protected readonly steps: number;

  constructor(readonly lesson: Lesson, protected readonly school: FlightSchool,
              protected readonly policy: ServoPolicy) {
    this.body = lesson.spawn();
    this.cursor = new PhaseCursor(lesson.phases, this.body);
    this.steps = Math.floor(lesson.durationCap / school.dt);
  }

  get seconds(): number {
    return this.tick * this.school.dt;
  }

  get finished(): boolean {
    return this.ended;
  }

  get commands(): Float64Array {
    return this.handCommands();
  }

  protected abstract handCommands(): Float64Array;

  abstract step(): boolean;

  /** One physics tick with the phase bookkeeping and the runway pin. */
  protected tickPhysics(commands: Float64Array): boolean {
    const dt = this.school.dt;
    this.servos.command(commands, dt);
    // Height above the terrain: ground effect lives in the physics.
    const agl = this.body.position.z
      - this.school.world.terrainHeight(this.body.position.x, this.body.position.y);
    this.body.step(this.servos, dt, GRAVITY, RHO_SEA_LEVEL, agl);
    const phase = this.cursor.current;
    if (phase.pin === true) {
      // The runway: hold the aircraft at 10 ft while it cannot fly (vz
      // would go negative) — the takeoff roll, and the sag after rotation
      // until the wing carries it. It unpins itself the moment it flies.
      if (this.body.position.z < SYLLABUS.runwayM + 0.001) {
        this.body.position = new Vec3(this.body.position.x, this.body.position.y,
                                      SYLLABUS.runwayM);
        this.body.velocity = new Vec3(this.body.velocity.x, this.body.velocity.y,
                                      Math.max(0.0, this.body.velocity.z));
        // The gear: wings level, deck at the ground attitude until a phase
        // commands rotation — a pitch target or a held elevator pull. The
        // roll shows pitch 0, yoke neutral.
        holdGroundDeck(this.body, phase.pitchTarget === undefined
                       && (phase.elevatorOverride ?? 0.0) <= 0.0);
      }
    }
    this.cursor.advance(this.body);
    this.tick += 1;
    if (this.school.grounded(this.body) && phase.pin !== true) {
      this.ended = true;
      return false;
    }
    if (this.cursor.finished || this.tick >= this.steps) {
      this.ended = true;
      return false;
    }
    return true;
  }
}

/** The instructor demonstrates the lesson once; stamps land per phase. */
export class DemoFlight extends ClassroomFlight {
  readonly hand: InstructorHand;
  stamped = 0;

  constructor(school: FlightSchool, lesson: Lesson, policy: ServoPolicy) {
    super(lesson, school, policy);
    // Parked on the runway, the hand starts neutral-idle; already flying,
    // it starts at the trim.
    this.hand = new InstructorHand(startCommandsFor(lesson.startKind));
  }

  protected handCommands(): Float64Array {
    return this.hand.commands;
  }

  step(): boolean {
    if (this.ended) {
      return false;
    }
    const dt = this.school.dt;
    // The phase law runs every tick (the hand chases live targets); the
    // stamp decision happens at decision ticks, reading the same raw.
    const raw = this.cursor.phaseTargets(this.body);
    if (this.tick % DECISION_EVERY === 0) {
      const demo = this.school.demoTick(this.policy, this.cursor, this.body,
                                       this.servos, this.seconds, raw, {});
      this.stamped += demo.stamped ? 1 : 0;
    }
    const smooth = this.hand.easeToward(raw, dt);
    return this.tickPhysics(Float64Array.from(smooth));
  }
}

/**
 * The student flies the exam: memory-only replay, no exploration, the
 * stick spring active. Measures mean panel drift from the demo — the 0-6
 * grade. Its own phase cursor drives the runway pin.
 */
export class ExamFlight extends ClassroomFlight {
  private commandsNow: Float64Array;
  private readonly rng = new NumpyPcg64(RNG0_STATE, RNG0_INC);
  private readonly demo: readonly (readonly number[])[];
  private total = 0.0;
  private ticks = 0;
  private readonly savedEpsilon: number;

  constructor(school: FlightSchool, lesson: Lesson, policy: ServoPolicy) {
    super(lesson, school, policy);
    this.demo = school.demoPanel(lesson);
    this.savedEpsilon = policy.epsilon;
    policy.epsilon = 0.0;              // replay the exam exactly
    this.commandsNow = Float64Array.from(startCommandsFor(lesson.startKind));
  }

  protected handCommands(): Float64Array {
    return this.commandsNow;
  }

  get meanError(): number {
    return this.total / Math.max(this.ticks, 1);
  }

  release(): void {
    if (!this.ended) {
      this.ended = true;
      this.policy.epsilon = this.savedEpsilon;
    }
  }

  step(): boolean {
    if (this.ended) {
      return false;
    }
    const dt = this.school.dt;
    if (this.tick % DECISION_EVERY === 0) {
      const stateKey = this.school.instruments.stateKey(this.body, this.commandsNow[3]);
      const actions = this.policy.choose(stateKey, this.rng);
      this.commandsNow = this.policy.commandFrom(this.commandsNow, actions);
    }
    const goal = this.demo[Math.min(Math.floor(this.tick / DECISION_EVERY),
                                    this.demo.length - 1)];
    const errors = this.school.instruments.errors(this.body);
    this.total += Math.abs(errors.x - goal[0])
      + Math.abs(errors.y - goal[1])
      + Math.abs(errors.z - goal[2]);
    this.ticks++;
    const more = this.tickPhysics(this.commandsNow);
    this.commandsNow = springTowardNeutral(this.commandsNow, dt);
    if (!more) {
      this.release();
    }
    return more;
  }
}

/**
 * Show-and-repeat with corrective takeover: the student repeats the exam;
 * whenever its panel drifts out of the demo's band the instructor takes the
 * controls AT THAT STATE (the hand easing from the student's own commands,
 * so even the takeover is smooth), stamps the firm correction there, flies
 * the recovery until the panel is back on the demo's line, then hands back.
 */
export class RepeatFlight extends ClassroomFlight {
  private commandsNow: Float64Array;
  private readonly rng = new NumpyPcg64(RNG0_STATE, RNG0_INC);
  private readonly demo: readonly (readonly number[])[];
  private readonly hand: InstructorHand;
  stamps = 0;
  takeovers = 0;
  /** True while the instructor has the controls (for narration). */
  takeoverActive = false;
  /** This decision tick's panel drift from the demo (for narration). */
  drift = 0.0;
  private stretch = 0;
  private studentThrottle: number | null = null;
  private readonly savedEpsilon: number;

  constructor(school: FlightSchool, lesson: Lesson, policy: ServoPolicy,
              private readonly drillEpoch: number) {
    super(lesson, school, policy);
    this.demo = school.demoPanel(lesson);
    this.savedEpsilon = policy.epsilon;
    policy.epsilon = 0.0;              // the repeat is a memory replay
    const start = startCommandsFor(lesson.startKind);
    this.commandsNow = Float64Array.from(start);
    this.hand = new InstructorHand(start);
  }

  protected handCommands(): Float64Array {
    return this.takeoverActive ? this.hand.commands : this.commandsNow;
  }

  get seconds(): number {
    return this.tick * this.school.dt;
  }

  release(): void {
    if (!this.ended) {
      this.ended = true;
      this.policy.epsilon = this.savedEpsilon;
    }
  }

  step(): boolean {
    if (this.ended) {
      return false;
    }
    const dt = this.school.dt;
    const seconds = this.tick * dt;
    if (this.tick % DECISION_EVERY === 0) {
      const errors = this.school.instruments.errors(this.body);
      const goal = this.demo[Math.min(Math.floor(this.tick / DECISION_EVERY),
                                      this.demo.length - 1)];
      this.drift = Math.max(Math.abs(errors.x - goal[0]),
                            Math.abs(errors.y - goal[1]),
                            Math.abs(errors.z - goal[2]));
      const demoNearTrim = Math.abs(goal[0]) < 0.15
        && Math.abs(goal[1]) < 0.15 && Math.abs(goal[2]) < 0.2;
      if (this.stretch === 0 && this.drift > DEMO_BAND) {
        this.takeovers += 1;           // the student has it wrong
        this.stretch = STRETCH_DECISIONS;
        // Address the coming corrections to the student's own lever
        // position, so its exam-time lookups find them.
        this.studentThrottle = this.commandsNow[3];
        // Un-teach the hands-off reflex at this state: the drift proves
        // doing nothing here was wrong. Only hold-all is weakened; the
        // anti-stamp stops at a reserve.
        const takeoverKey = this.school.instruments.stateKey(
          this.body, this.studentThrottle);
        const holdallKey = [...takeoverKey, HOLD_ALL_SURFACE, HOLD];
        if (this.policy.tensor.score(holdallKey, 0, 1.0) > 2.0) {
          this.policy.tensor.reward(holdallKey, 0, -1.5);
        }
      }
      if (this.stretch > 0) {
        if (this.drift <= DEMO_BAND) {
          this.stretch = 0;            // back on the demo line: hands off
          this.takeoverActive = false;
        } else {
          const demo = this.school.demoTick(
            this.policy, this.cursor, this.body, this.servos, seconds,
            this.cursor.phaseTargets(this.body), {
              takeover: true,
              studentThrottle: this.studentThrottle,
              timeEpoch: this.drillEpoch,
              amount: 2.0 * this.school.imprintAmount,
              demoNearTrim,
            });
          this.stamps += demo.stamped ? 1 : 0;
          this.takeoverActive = true;
          this.stretch -= 1;
        }
      }
      if (this.stretch === 0) {
        this.takeoverActive = false;
        const stateKey = this.school.instruments.stateKey(this.body,
                                                          this.commandsNow[3]);
        const actions = this.policy.choose(stateKey, this.rng);
        this.commandsNow = this.policy.commandFrom(this.commandsNow, actions);
      }
    }
    // The instructor's hand eases from the student's commands at takeover;
    // at handback the student resumes from wherever the hand left them.
    const raw = this.takeoverActive
      ? this.cursor.phaseTargets(this.body)
      : this.commandsNow;
    const smooth = this.takeoverActive
      ? this.hand.easeToward(raw, dt)
      : this.commandsNow;
    const more = this.tickPhysics(Float64Array.from(smooth));
    this.commandsNow = springTowardNeutral(this.commandsNow, dt);
    if (!more) {
      this.release();
    }
    return more;
  }
}

export interface TeachReport {
  counts: Map<string, number>;
  rounds: number;
  failed: string[];
  graduate: ServoPolicy;
}

/**
 * Runs the curriculum against a student's servo tensor: demonstrations,
 * first exams, then corrective drill rounds until graduation (or the cap).
 * All flights are handed out one at a time so the browser can run the
 * classroom live (SchoolSession); teach() is the headless fast path.
 */
export class FlightSchool {
  readonly instruments: FlightInstruments;
  readonly world: World;
  readonly imprintAmount = 1.0;
  readonly dt = 1.0 / 50.0;
  graduate: ServoPolicy | null = null;
  readonly lessons = new Map<string, Lesson>();
  private readonly demoPanels = new Map<string, (readonly number[])[]>();

  constructor(instruments: FlightInstruments, world?: World) {
    this.instruments = instruments;
    this.world = world ?? new World({ halfExtent: 800.0 });
    this.defineCurriculum();
  }

  private defineCurriculum(): void {
    const S = SYLLABUS;
    const lessons: [string, number, readonly number[], ManeuverPhase[],
                   { durationCap: number, startKind: Lesson["startKind"] }][] = [];

    // --- takeoff: parked at 10 ft, yoke neutral to 60 mph, rotate and HOLD
    // THE PULL through ground effect to 100 ft, then a 15° climb to 1000 ft -
    lessons.push(["takeoff", DESIRE_CLIMB, [0.1, 0.0, 1.0], [
      { name: "roll", stamp: false, pin: true, bankTarget: 0.0,
        elevatorOverride: 0.0, rudder: 0.0, throttle: 1.0,
        done: (r) => r.airspeed >= S.rotateMps },
      // Holding 15° while slow in ground effect takes a HELD PULL — the
      // PI's integral parks the yoke well aft (the elevator trim this
      // slow speed needs); it eases forward as the speed builds.
      { name: "lift off", stamp: true, pin: true, pitchTarget: S.pitchClimb,
        bankTarget: 0.0, rudder: 0.0, throttle: 1.0,
        done: (r) => r.altitudeM >= S.runwayM + S.liftOffToM },
      { name: "climb", stamp: true, pitchTarget: S.pitchClimb, bankTarget: 0.0,
        rudder: 0.0, throttle: 1.0,
        done: (r) => r.altitudeM >= S.climbToM - LEVEL_OFF_LEAD_S * r.vsMps },
      { name: "level off", stamp: true, pitchCapture: true, vsTarget: 0.0,
        bankTarget: 0.0, rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => r.ticks >= 150 && Math.abs(r.vsMps) < 0.3 },
    ], { durationCap: 150.0, startKind: "runway" }]);

    // --- level cruise: the 80-90 mph / 2300 RPM hands-off lesson -----------
    lessons.push(["level_cruise", DESIRE_LEVEL, [0.0, 0.0, 0.0], [
      { name: "cruise", stamp: true, vsTarget: 0.0, bankTarget: 0.0,
        speedTarget: S.cruiseSpeedMps,
        rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => r.ticks >= 1000 },
    ], { durationCap: 25.0, startKind: "cruise1000" }]);

    // --- turns: 180 deg, 25 deg bank, nose held on the horizon. The roll-out
    // is a HEADING HOLD: it lands the aircraft on entry heading ± 180.0°. --
    const turnPhases = (bank: number): ManeuverPhase[] => [
      { name: "roll in", stamp: false, carryTurn: true, bankTarget: bank,
        vsTarget: 0.0, speedTarget: S.cruiseSpeedMps,
        rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => Math.abs(r.bank) >= 0.95 * Math.abs(bank) },
      { name: "turn", stamp: true, carryTurn: true, bankTarget: bank,
        vsTarget: 0.0, speedTarget: S.cruiseSpeedMps,
        rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => Math.abs(r.turned) >= S.turnAmount - S.turnLeadOut },
      { name: "roll out", stamp: false, carryTurn: true, headingHold: true,
        vsTarget: 0.0, speedTarget: S.cruiseSpeedMps,
        rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => Math.abs(wrapPi(Math.PI - Math.abs(r.turned))) <= 0.014
                       && Math.abs(r.bank) <= 0.022 },
      { name: "settle", stamp: true, vsTarget: 0.0, bankTarget: 0.0,
        speedTarget: S.cruiseSpeedMps,
        rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => r.ticks >= 150 && Math.abs(r.vsMps) < 0.5 },
    ];
    lessons.push(["turn_left_180", DESIRE_LEFT, [0.0, -S.bankTurn, 0.0],
                  turnPhases(-S.bankTurn),
                  { durationCap: 60.0, startKind: "cruise1000" }]);
    lessons.push(["turn_right_180", DESIRE_RIGHT, [0.0, S.bankTurn, 0.0],
                  turnPhases(S.bankTurn),
                  { durationCap: 60.0, startKind: "cruise1000" }]);

    // --- steep turns: 180 deg, 60 deg bank, top rudder --------------------
    // A 2 g level turn needs ENERGY: enter at cruise, full throttle, and
    // let the speed ride up to the steep-turn band (~100 mph).
    const steepPhases = (bank: number): ManeuverPhase[] => [
      { name: "roll in", stamp: false, carryTurn: true, bankTarget: bank,
        vsTarget: 0.0, speedTarget: S.steepSpeedMps,
        rudder: "top", throttle: 1.0,
        done: (r) => Math.abs(r.bank) >= 0.95 * Math.abs(bank) },
      { name: "turn", stamp: true, carryTurn: true, bankTarget: bank,
        vsTarget: 0.0, speedTarget: S.steepSpeedMps,
        rudder: "top", throttle: 1.0,
        done: (r) => Math.abs(r.turned) >= S.turnAmount - S.steepLeadOut },
      { name: "roll out", stamp: false, carryTurn: true, headingHold: true,
        vsTarget: 0.0, speedTarget: S.steepSpeedMps,
        rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => Math.abs(wrapPi(Math.PI - Math.abs(r.turned))) <= 0.014
                       && Math.abs(r.bank) <= 0.022 },
      { name: "settle", stamp: true, vsTarget: 0.0, bankTarget: 0.0,
        speedTarget: S.cruiseSpeedMps,
        rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => r.ticks >= 150 && Math.abs(r.vsMps) < 0.5 },
    ];
    lessons.push(["steep_turn_left_180", DESIRE_LEFT, [0.0, -S.bankSteep, 0.0],
                  steepPhases(-S.bankSteep),
                  { durationCap: 45.0, startKind: "cruise1000" }]);
    lessons.push(["steep_turn_right_180", DESIRE_RIGHT, [0.0, S.bankSteep, 0.0],
                  steepPhases(S.bankSteep),
                  { durationCap: 45.0, startKind: "cruise1000" }]);

    // --- stall recovery: idle, hold altitude to the stall, dive, recover --
    lessons.push(["stall_recovery", DESIRE_RECOVER, [0.0, 0.0, 1.0], [
      { name: "entry", stamp: false, vsTarget: 0.0, bankTarget: 0.0,
        rudder: 0.0, throttle: S.idleThrottle,
        done: (r) => r.airspeed <= S.stallEntryMps },
      { name: "wait", stamp: false, vsTarget: 0.0, bankTarget: 0.0,
        rudder: 0.0, throttle: S.idleThrottle,
        done: (r) => r.alpha >= 0.35 || r.vsMps < -3.0 },
      { name: "break", stamp: false, elevatorOverride: 0.0, bankTarget: 0.0,
        rudder: 0.0, throttle: S.idleThrottle,
        done: (r) => r.pitch < 0.0 },
      { name: "push", stamp: true, pitchTarget: S.pitchDive, bankTarget: 0.0,
        rudder: 0.0, throttle: S.idleThrottle,
        done: (r) => r.airspeed >= S.diveOutMps },
      { name: "pull up", stamp: true, pitchTarget: S.pitchRecovery, bankTarget: 0.0,
        rudder: 0.0, throttle: 1.0,
        done: (r) => r.altitudeM >= S.stallAltM - LEVEL_OFF_LEAD_S * r.vsMps
                       && r.vsMps >= -0.2 },
      { name: "level off", stamp: true, pitchCapture: true, vsTarget: 0.0,
        bankTarget: 0.0, rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => r.ticks >= 150 && Math.abs(r.vsMps) < 0.3 },
    ], { durationCap: 120.0, startKind: "cruise3000" }]);

    // --- spin recovery: stall with a dropped wing, opposite rudder --------
    lessons.push(["spin_recovery", DESIRE_RECOVER, [0.0, 0.0, 1.0], [
      { name: "entry", stamp: false, vsTarget: 0.0, bankTarget: 0.0,
        rudder: 0.0, throttle: S.idleThrottle,
        done: (r) => r.airspeed <= S.stallEntryMps },
      { name: "wait", stamp: false, vsTarget: 0.0, bankTarget: 0.0,
        rudder: 0.0, throttle: S.idleThrottle,
        done: (r) => r.alpha >= 0.35 || r.vsMps < -3.0 },
      { name: "drop wing", stamp: false, aileronOverride: -AILERON_LIMIT,
        elevatorOverride: 0.1, rudder: "proSpin", throttle: S.idleThrottle,
        done: (r) => r.rateR <= -0.6 || r.ticks >= 200 },
      { name: "spin", stamp: false, aileronOverride: -0.2,
        elevatorOverride: 0.12, rudder: "proSpin", throttle: S.idleThrottle,
        done: (r) => r.ticks >= 100 },
      { name: "opposite rudder", stamp: true, elevatorOverride: 0.0,
        rudder: "antiSpin", throttle: S.idleThrottle,
        done: (r) => Math.abs(r.rateR) <= 0.1 },
      { name: "dive", stamp: true, elevatorOverride: 0.0, bankTarget: 0.0,
        rudder: 0.0, throttle: S.idleThrottle,
        done: (r) => r.airspeed >= S.spinPullUpMps },
      { name: "pull up", stamp: true, pitchTarget: S.pitchRecovery, bankTarget: 0.0,
        rudder: 0.0, throttle: 1.0,
        done: (r) => r.altitudeM >= S.stallAltM - LEVEL_OFF_LEAD_S * r.vsMps
                       && r.vsMps >= -0.2 },
      { name: "level off", stamp: true, pitchCapture: true, vsTarget: 0.0,
        bankTarget: 0.0, rudder: 0.0, throttle: S.cruiseThrottle,
        done: (r) => r.ticks >= 150 && Math.abs(r.vsMps) < 0.3 },
    ], { durationCap: 140.0, startKind: "cruise3000" }]);

    for (const [name, desire, targets, phases, options] of lessons) {
      this.lessons.set(name, new Lesson(name, desire, targets, phases, options));
    }
  }

  /** Ground contact ends every classroom flight. */
  grounded(body: RigidBody): boolean {
    return body.position.z
      <= this.world.terrainHeight(body.position.x, body.position.y) + 1.0;
  }

  /**
   * One instructor demonstration tick: the phase law produces the raw
   * deflection targets (always returned to fly); the anti-poisoning gates
   * decide whether the reading is stamped. Phase-aware: entry phases never
   * stamp (the stall/spin entry must never be taught), and a takeover
   * correction in a bank-holding phase may only roll the bank TOWARD the
   * phase's band — never deepen it — and never while in a dive signature
   * (a spiral dive reads like a shallow turn in these bins).
   */
  demoTick(policy: ServoPolicy, cursor: PhaseCursor, body: RigidBody,
           servos: SurfacesState, seconds: number,
           rawTargets: Float64Array,
           options: DemoTickOptions): DemoTickResult {
    const takeover = options.takeover ?? false;
    const phase = cursor.current;
    const raw = rawTargets;
    if (!phase.stamp && !takeover) {
      return { stamped: false };
    }
    if (takeover && phase.bankTarget !== undefined) {
      const band = 0.15;
      const lo = phase.bankTarget - band;
      const hi = phase.bankTarget + band;
      const bank = bankRadTargetOf(body);
      const errors = this.instruments.errors(body);
      if (!(lo <= bank && bank <= hi)) {
        const inDive = errors.x < -0.3 || errors.z > 0.3;
        if (inDive) {
          return { stamped: false };
        }
        if (bank > hi && raw[0] >= 0) {
          return { stamped: false };
        }
        if (bank < lo && raw[0] <= 0) {
          return { stamped: false };
        }
      }
    }
    const throttle = options.studentThrottle == null
      ? raw[3] : options.studentThrottle;
    const stateKey = this.instruments.stateKey(body, throttle);
    const panel = this.instruments.errors(body);
    // Feel the plane: near trim the lesson is hands off. At a takeover the
    // rule reads the DEMO instead.
    const nearTrim = Math.abs(panel.x) < 0.15 && Math.abs(panel.y) < 0.15
      && Math.abs(panel.z) < 0.2;
    const demoNearTrim = options.demoNearTrim ?? true;
    let actions: Int32Array;
    if (nearTrim && (demoNearTrim || !takeover)) {
      actions = new Int32Array(4).fill(HOLD);
    } else {
      actions = FlightSchool.translateTargets(raw, servos);
    }
    const amount = options.amount == null
      ? this.imprintAmount : options.amount;
    policy.imprint(stateKey, actions, amount, options.timeEpoch ?? 0);
    return { stamped: true };
  }

  /** Instructor deflections -> the student's push/ease/hold decisions. */
  private static translateTargets(targets: Float64Array,
                                  servos: SurfacesState): Int32Array {
    const actions = new Int32Array(4).fill(HOLD);
    for (let surface = 0; surface < 4; surface++) {
      const delta = targets[surface] - servos.positions[surface];
      if (delta > 0.04) {
        actions[surface] = PUSH;
      } else if (delta < -0.04) {
        actions[surface] = EASE_OFF;
      }
    }
    return actions;
  }

  /** The instructor's panel readings at each decision tick of the demo. */
  demoPanel(lesson: Lesson): (readonly number[])[] {
    const cached = this.demoPanels.get(lesson.name);
    if (cached !== undefined) {
      return cached;
    }
    const flight = new DemoFlight(this, lesson, new ServoPolicy());
    const panel: (readonly number[])[] = [];
    while (flight.step()) {
      if (flight.tick % DECISION_EVERY === 0) {
        const errors = this.instruments.errors(flight.body);
        panel.push([errors.x, errors.y, errors.z]);
      }
    }
    if (panel.length === 0) {
      panel.push([0.0, 0.0, 0.0]);
    }
    this.demoPanels.set(lesson.name, panel);
    return panel;
  }

  /** How well the student repeats the maneuver, graded 0-6 (the exam). */
  studentQuality(policy: ServoPolicy, lesson: Lesson): number {
    const exam = new ExamFlight(this, lesson, policy);
    while (exam.step()) { /* run to completion */ }
    return Math.max(0.0, Math.min(QUALITY_CAP,
      QUALITY_CAP - QUALITY_SLOPE * exam.meanError));
  }

  /** Imprint the curriculum, then drill until the student graduates. */
  teach(policy: ServoPolicy, options: {
    lessons?: readonly string[];
    bar?: number;
    maxRounds?: number;
  } = {}): TeachReport {
    const session = new SchoolSession(this, policy, options);
    session.runToCompletion();
    this.graduate = session.graduate;
    return {
      counts: session.counts,
      rounds: session.rounds,
      failed: session.failed,
      graduate: session.graduate!,
    };
  }
}

/** Quality points from a mean panel drift: 6 - 2 * error, clamped 0-6. */
export function qualityOf(meanError: number): number {
  return Math.max(0.0, Math.min(QUALITY_CAP, QUALITY_CAP - QUALITY_SLOPE * meanError));
}

/** Where a session flight sits in the curriculum. */
export type SessionStage = "demo" | "exam" | "repeat" | "done";

/** One flight handed out by a SchoolSession. */
export interface SchoolFlight {
  readonly flight: DemoFlight | ExamFlight | RepeatFlight;
  readonly lesson: Lesson;
  readonly stage: SessionStage;
  /** 0 for the demonstration pass and the first exams; 1.. for drill rounds. */
  readonly round: number;
}

/**
 * The teach() loop, resumable one flight at a time so the browser can run
 * the classroom live: demonstrations, the first exam of every lesson, then
 * drill rounds (corrective show-and-repeat + re-grade) on the failing
 * lessons until the class graduates or the round cap hits.
 */
export class SchoolSession {
  readonly school: FlightSchool;
  readonly policy: ServoPolicy;
  readonly bar: number;
  readonly maxRounds: number;
  private readonly names: readonly string[];
  readonly counts = new Map<string, number>();
  readonly scores = new Map<string, number>();
  readonly takeovers = new Map<string, number>();
  graduate: ServoPolicy | null = null;
  rounds = 0;
  failed: string[] = [];
  private stage: "demo" | "firstExam" | "drill" | "done" = "demo";
  private index = 0;
  private currentFailing: string[] = [];
  private awaitingExam: string | null = null;
  private pending: SchoolFlight | null = null;

  constructor(school: FlightSchool, policy: ServoPolicy, options: {
    lessons?: readonly string[];
    bar?: number;
    maxRounds?: number;
  } = {}) {
    this.school = school;
    this.policy = policy;
    this.bar = options.bar ?? DEFAULT_PASS_SCORE;
    this.maxRounds = options.maxRounds ?? GRADUATION_ROUNDS;
    this.names = options.lessons ?? [...school.lessons.keys()];
    for (const name of this.names) {
      if (!school.lessons.has(name)) {
        throw new Error(`unknown lesson: ${name}`);
      }
      this.counts.set(name, 0);
      this.takeovers.set(name, 0);
    }
  }

  get done(): boolean {
    return this.stage === "done";
  }

  /** Absorb the previous flight and hand out the next one (null = done). */
  nextFlight(): SchoolFlight | null {
    this.absorbPending();
    return this.advance();
  }

  /** Run every remaining flight headlessly (the teach() fast path). */
  runToCompletion(): void {
    let next = this.nextFlight();
    while (next !== null) {
      while (next.flight.step()) { /* fly */ }
      next = this.nextFlight();
    }
  }

  private absorbPending(): void {
    const pending = this.pending;
    if (pending === null) {
      return;
    }
    this.pending = null;
    const name = pending.lesson.name;
    if (pending.stage === "demo") {
      this.counts.set(name, (pending.flight as DemoFlight).stamped);
      this.index += 1;
      if (this.index === this.names.length) {
        // The graduate the flock copies is the base curriculum: the clean
        // demonstration reflexes. Drilled corrections are classroom polish.
        const graduate = new ServoPolicy();
        graduate.tensor.restore(this.policy.tensor.snapshot());
        this.graduate = graduate;
        this.school.graduate = graduate;
        this.stage = "firstExam";
        this.index = 0;
      }
      return;
    }
    if (pending.stage === "exam") {
      const score = qualityOf((pending.flight as ExamFlight).meanError);
      this.scores.set(name, score);
      if (this.stage === "firstExam") {
        this.index += 1;
        if (this.index === this.names.length) {
          this.currentFailing = this.names.filter(
            (lessonName) => this.scores.get(lessonName)! < this.bar);
          if (this.currentFailing.length === 0 || this.maxRounds === 0) {
            this.finish(this.currentFailing);
          } else {
            this.stage = "drill";
            this.index = 0;
          }
        }
      } else {
        this.awaitingExam = null;
        this.index += 1;
        if (this.index === this.currentFailing.length) {
          this.rounds += 1;
          const stillFailing = this.currentFailing.filter(
            (lessonName) => this.scores.get(lessonName)! < this.bar);
          if (stillFailing.length === 0 || this.rounds >= this.maxRounds) {
            this.finish(stillFailing);
          } else {
            this.currentFailing = stillFailing;
            this.index = 0;
          }
        }
      }
      return;
    }
    const repeat = pending.flight as RepeatFlight;
    this.counts.set(name, this.counts.get(name)! + repeat.stamps);
    this.takeovers.set(name, this.takeovers.get(name)! + repeat.takeovers);
    this.awaitingExam = name;
  }

  private advance(): SchoolFlight | null {
    if (this.stage === "done") {
      return null;
    }
    if (this.stage === "demo") {
      const name = this.names[this.index];
      const lesson = this.school.lessons.get(name)!;
      return this.hand(new DemoFlight(this.school, lesson, this.policy),
                       lesson, "demo", 0);
    }
    if (this.stage === "firstExam") {
      const name = this.names[this.index];
      const lesson = this.school.lessons.get(name)!;
      return this.hand(new ExamFlight(this.school, lesson, this.policy),
                       lesson, "exam", 0);
    }
    if (this.awaitingExam !== null) {
      const lesson = this.school.lessons.get(this.awaitingExam)!;
      return this.hand(new ExamFlight(this.school, lesson, this.policy),
                       lesson, "exam", this.rounds + 1);
    }
    const name = this.currentFailing[this.index];
    const lesson = this.school.lessons.get(name)!;
    const drillEpoch = 1 + this.rounds % 7;
    return this.hand(new RepeatFlight(this.school, lesson, this.policy, drillEpoch),
                     lesson, "repeat", this.rounds + 1);
  }

  private hand(flight: DemoFlight | ExamFlight | RepeatFlight, lesson: Lesson,
               stage: SessionStage, round: number): SchoolFlight {
    this.pending = { flight, lesson, stage, round };
    return this.pending;
  }

  private finish(failed: string[]): void {
    this.failed = failed;
    this.stage = "done";
  }
}
