/**
 * The flight school: the instructor demonstrates, the student repeats.
 * Redesigned for the C172 airframe (1111 kg, stall ~50 mph, cruise 85 mph
 * at 2300 RPM): each lesson is a PHASE-BASED PROCEDURE — a real
 * takeoff (yoke neutral on the roll, pitch 0 on the gear, rotate at
 * 60 mph into a 15° climb to 1000 ft), full 180° turns held
 * level with the elevator, steep turns with top rudder, and the complete
 * stall / spin recovery sequences — flown by feedback laws, smoothly.
 *
 * The classroom runs in the flight-school loop, one maneuver at a time:
 *   1. the instructor demonstrates the maneuver (stamping what its hands
 *      do into the student's servo tensor),
 *   2. the student repeats it alone,
 *   3. the repeat is GRADED AGAINST THE MANEUVER'S KNOWN GOALS —
 *      "rotate at 60 mph", "climb at 15° pitch", "bank 20-30°",
 *      "recover to 3000 ft"… — and the loop moves to the next maneuver on
 *      a pass, or the instructor demonstrates the same one again on a
 *      fail, until the lesson passes or its attempt cap is reached.
 *
 * The instructor is a controller with a HAND MODEL: raw phase targets are
 * eased toward at bounded rates (rad/s), so every input on the flight
 * controls is smooth — no steps, no jerks, even at phase transitions. The
 * student never sees the controller; it sees instrument state -> servo
 * nudges (push / ease / hold), stamped into the servo tensor at the
 * demo's established phases only (the stall/spin ENTRY is never stamped —
 * the student must never learn to enter a departure; the recovery is the
 * lesson). Where an entry cannot be replayed from memory at all, the
 * instructor sets the aircraft up into the maneuver and hands it over at
 * that phase boundary (Lesson.repeatFrom) — the student repeats the part
 * it was taught.
 *
 * The grading is goal-based: every taught phase carries named goals, each
 * weighted by the demonstration's own decision-tick count in that phase,
 * and the student's repeat earns the fraction its flying satisfies — so
 * a skipped, blown, or never-reached phase costs exactly its share of the
 * 0-6 grade.
 */
import { Vec3 } from "../core/vec.js";
import { Quaternion } from "../core/quaternion.js";
import { World } from "../core/world.js";
import { NumpyPcg64 } from "../core/rng.js";
import { RigidBody } from "../physics/rigid-body.js";
import { SurfacesState, SURFACE_LIMITS } from "../physics/surfaces.js";
import { AeroTables, RHO_SEA_LEVEL } from "../physics/aero.js";
import { Aircraft } from "../physics/aircraft.js";import { FlightInstruments, type SenseReading } from "./instruments.js";
import { ServoPolicy, HOLD, PUSH, EASE_OFF, HOLD_ALL_SURFACE,
         ELEMENT_FIRE_BAR, ACTION_DELTA,
       } from "./servo-tensor.js";
import { Refractory } from "./neurons.js";
import { TRIM_SPEED, TRIM_COMMAND, springStickTowardNeutral } from "./flight.js";

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
export const GRADUATION_ATTEMPTS = 12;

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

// GOAL-BASED GRADING: every maneuver is scored against its named goals —
// one per taught phase ("roll on the wheels to 60 mph", "hold 15° pitch",
// "hold 25° bank", "level off smoothly at 1000 ft"…). Each goal phase of
// the demonstration contributes its decision-tick count as weight; the
// student's repeat earns ok-ticks in that phase, capped at the nominal
// weight, so skipping or stalling a phase scores against it. The score is
// 0-6 = 6 * earned / nominal.
export const GOAL_TOLERANCES = {
  pitchHold: 0.10,      // ±5.7° on a held pitch target
  pitchDive: -0.20,     // nose at least this far down in the stall push
  bankHold: 0.10,       // ±5.7° on a held bank target (inside 20-30 / 60 band)
  bankRoll: 0.12,      // overshoot allowance while rolling in/out
  wingsLevel: 0.09,    // ±5° wings level
  vsLevel: 0.8,        // ±160 fpm on "VSI ~0"
  vsTurn: 1.2,        // ±240 fpm through roll-in/roll-out
  speedHold: 4.0,      // ±9 mph on a held airspeed target
  spinStopped: 0.5,    // rad/s yaw rate: the spin has stopped
} as const;

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

/**
 * THE ELEMENT VOCABULARY — the intention tensor's element axis. Every
 * atomic chunk a maneuver decomposes into, one word each. The same word
 * in two maneuvers (the takeoff's climb and the stall recovery's climb)
 * is the SAME element: the goal axis contextualizes, the element word
 * carries the skill.
 */
export const ELEMENT_VOCABULARY = [
  "free flight",                       // reserved: the swarm's element-less flight
  "roll",                              // takeoff roll on the gear
  "rotate",                            // rotation and lift-off through ground effect
  "climb",                             // the held 15 deg climb
  "level off",                         // the smooth capture onto an altitude
  "cruise",                            // the hands-off level cruise
  "roll in",                           // rolling into the bank
  "turn",                              // the held-bank turn
  "roll out",                          // the heading-hold roll-out
  "settle",                            // settling level after the roll-out
  "stall entry",                       // shown, never taught, never graded
  "wait",                              // holding altitude into the stall break
  "break",                             // letting the nose drop through the stall
  "push",                              // the stall-recovery dive
  "drop wing",                         // the spin entry's dropped wing
  "spin",                              // the developed spin
  "opposite rudder",                   // anti-spin rudder to the stop
  "dive",                              // the spin-recovery dive
] as const;
export type ElementName = typeof ELEMENT_VOCABULARY[number];

/** The element axis bin a vocabulary word sits at. */
export function elementIdOf(name: ElementName): number {
  return ELEMENT_VOCABULARY.indexOf(name);
}

/** The control law an element flies: which channels it commands, and how. */
interface ElementLaw {
  /** Hold this bank (rad) with the ailerons, if defined. */
  readonly bankTarget?: number;
  /** Carry the heading accumulator across this element (turn roll-in/out). */
  readonly carryTurn?: boolean;
  /** Roll out onto exactly entry-heading ± 180° (a heading hold). */
  readonly headingHold?: boolean;
  /** Hold vertical speed (m/s) with the elevator, if defined. */
  readonly vsTarget?: number;
  /** LEVEL-OFF CAPTURE: a firm nose-down that eases off with the pitch excess. */
  readonly pitchCapture?: boolean;
  /** Hold this airspeed (m/s) with the throttle, if defined. */
  readonly speedTarget?: number;
  /** Hold deck pitch (rad) with the elevator, if defined. */
  readonly pitchTarget?: number;
  /** Fixed aileron override (the spin entry). */
  readonly aileronOverride?: number;
  /** Fixed elevator override (stall-break neutrals, spin back pressure). */
  readonly elevatorOverride?: number;
  readonly rudder: RudderLaw;
  readonly throttle: number;
}

/**
 * ONE ATOMIC ELEMENT of a maneuver — the smallest chunk a student is
 * intentionally asked to execute ("roll", "rotate", "climb", "level
 * off"). The INSTRUCTOR flies it with a control law and a decision-tree
 * exit; the STUDENT chooses it from memory (the intention tensor's
 * element axis) and is graded on its named goals.
 */
export abstract class FlightElement {
  /** The vocabulary word — the intention tensor's element axis bin. */
  abstract readonly name: ElementName;
  /** Stamp this element's readings into the student's memory. */
  abstract readonly stamp: boolean;
  /** The known goals this element is graded on (the report card's law). */
  goals: readonly ManeuverGoal[] = [];
  /** Hold the aircraft on the runway while in this element (the roll). */
  readonly pin: boolean = false;

  constructor(protected readonly law: ElementLaw) {}

  get elementId(): number {
    return elementIdOf(this.name);
  }

  get carryTurn(): boolean {
    return this.law.carryTurn === true;
  }

  get headingHold(): boolean {
    return this.law.headingHold === true;
  }

  /** Does this element command rotation off the gear? (the deck law) */
  get rotates(): boolean {
    return this.law.pitchTarget !== undefined
      || (this.law.elevatorOverride ?? 0.0) > 0.0;
  }

  /** Element exit condition — the instructor's decision tree. */
  abstract done(r: PhaseReadings): boolean;

  /**
   * The instructor's raw deflection targets for this element: the shared
   * feedback law over the element's commanded channels.
   */
  targets(r: PhaseReadings, cursor: PhaseCursor): Float64Array {
    // The bank being held: the element's, or — in a heading-hold
    // roll-out — the bank that closes the remaining turn onto ±180°.
    const bankTarget = this.law.headingHold === true
      ? clamp(1.4 * wrapPi(cursor.turnDirection * Math.PI - r.turned), AILERON_LIMIT)
      : this.law.bankTarget;
    // aileron: bank-hold (or a fixed deflection in the spin entry) with
    // TURN-KINEMATICS compensation: in a banked, nose-up turn the heading
    // rotation bleeds into the Euler bank (p = φ̇ − ψ̇ sinθ), so holding
    // bank needs continuous into-turn roll. The feedforward supplies that
    // roll and the damping acts on the true bank rate — the P loop then
    // holds the bank with no steady error.
    let aileron: number;
    if (this.law.aileronOverride !== undefined) {
      aileron = this.law.aileronOverride;
    } else if (bankTarget === undefined) {
      aileron = 0.0;
    } else {
      // The roll feedforward needs flying airspeed; parked on the runway
      // (qDyn ~ 0) it is undefined and the plain bank-hold flies the roll.
      if (r.airspeed < 5.0) {
        aileron = clamp(0.8 * (bankTarget - r.bank) + 0.9 * r.rateP, AILERON_LIMIT);
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
    if (this.law.elevatorOverride !== undefined) {
      elevator = this.law.elevatorOverride;
    } else {
      // The LEVEL-OFF CAPTURE: coming down from a climb, the instructor
      // applies a FIRM nose-down and then REDUCES it gradually — the
      // commanded pitch rate is proportional to the remaining nose-up
      // excess (positive q is nose-down in this body convention), so the
      // input eases off as the nose settles and the arrival is gentle:
      // pitch rate -> 0 exactly as pitch -> level.
      const levelPitch = levelAttitudeRad(r.airspeed, r.bank);
      const excess = r.pitch - levelPitch;
      if (this.law.pitchCapture === true
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
      // The pitch target: the element's commanded attitude, or — in the
      // "VSI 0" elements — the level trim at the CURRENT speed and bank:
      // the nose slightly above the horizon, lift's vertical component
      // carrying the weight. Holding that attitude IS holding VSI at 0.
      // A sinking flight path raises the nose (back pressure against the
      // sink) — the outer VS loop closing what the feedforward leaves.
      const feedforward = this.law.pitchTarget !== undefined
        ? this.law.pitchTarget
        : levelAttitudeRad(r.airspeed, r.bank)
          + clamp(0.06 * ((this.law.vsTarget ?? 0.0) - r.vsMps), 0.25);
      const targetPitch = feedforward;
      // Attitude hold with a STICK-POSITION FEEDFORWARD: the static
      // elevator that trims this alpha is (cm0 + cmAlpha·alpha)/cmDe —
      // exactly the cruise trim 0.0249 at the cruise alpha. With the trim
      // placed instantly, the slow integral only mops up the residual
      // (dynamic-pressure shifts, the ground-effect glue), so the nose
      // does not sag through roll-ins and element changes.
      const stickFF = (SCHOOL_TABLES.cm0 + SCHOOL_TABLES.cmAlpha * targetPitch)
        / SCHOOL_TABLES.cmDe;
      cursor.pitchIntegral = Math.max(-0.05, Math.min(0.15,
        cursor.pitchIntegral + 0.30 * (targetPitch - r.pitch) * (1.0 / 50.0)));
      elevator = clamp(0.9 * (targetPitch - r.pitch) + cursor.pitchIntegral
                       + 0.25 * r.rateQ + stickFF,
                       ELEVATOR_LIMIT);
      }
    }
    // rudder: fixed, top rudder (steep turns), anti-spin, or pro-spin entry
    let rudder: number;
    if (this.law.rudder === "top") {
      // A GENTLE top rudder: enough to see on the pedals (right rudder in
      // a left steep turn), not enough to crab the nose off the path.
      rudder = this.law.bankTarget === undefined ? 0.0
        : -Math.sign(this.law.bankTarget) * 0.05;
    } else if (this.law.rudder === "antiSpin") {
      rudder = clamp(-0.5 * r.rateR, RUDDER_LIMIT);
    } else if (this.law.rudder === "proSpin") {
      rudder = -RUDDER_LIMIT;
    } else {
      rudder = this.law.rudder;
    }
    // throttle: the element's setting, nudged by a speed hold when the
    // element asks for one — the instructor flies attitude on the yoke
    // and SPEED ON THE POWER, the way a real pilot does.
    const throttle = this.law.speedTarget === undefined
      ? this.law.throttle
      : Math.max(0.0, Math.min(1.0,
          this.law.throttle + 0.05 * (this.law.speedTarget - r.airspeed)));
    return Float64Array.of(aileron, elevator, rudder, throttle);
  }
}

/**
 * The instructor's hand: eases commanded deflections toward the raw phase
 * targets at bounded rates. All inputs on the flight controls are smooth —
 * the yoke never jumps, even when a phase changes the law underneath.
 */
export class InstructorHand {
  readonly commands: Float64Array;
  /**
   * Per-channel easing rates: the instructor demonstrates at the
   * STUDENT'S hand speed — one ACTION_DELTA per decision tick window
   * (0.05 rad per 0.2 s = 0.25/s). A faster hand carves a pull the
   * student's ±0.05/decision vocabulary cannot track: the demo's
   * stamps land on a path through (reading, yoke) space the student
   * physically cannot follow, and every replay diverges from the
   * lesson it was shown. The instructor shows the maneuver the way a
   * student's hands can fly it.
   */
  private static readonly RATES = [
    ACTION_DELTA * 5.0, ACTION_DELTA * 5.0, ACTION_DELTA * 5.0,
    ACTION_DELTA * 5.0,
  ];

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

  constructor(readonly phases: readonly FlightElement[], body: RigidBody,
              startIndex = 0) {
    const [forward] = body.bodyAxes;
    this.lastHeading = Math.atan2(forward.y, forward.x);
    this.index = Math.min(startIndex, phases.length - 1);
  }

  get current(): FlightElement {
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
    const element = this.current;
    if (element.done(this.readings(body))) {
      const next = this.phases[this.index + 1];
      // A turn's 180 degrees span the roll-in, the turn, AND the roll-out
      // (which lands exactly on the target heading): elements flagged
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

  /** The instructor's raw deflection targets for the current element. */
  phaseTargets(body: RigidBody): Float64Array {
    return this.current.targets(this.readings(body), this);
  }
}

/**
 * One standard maneuver: a spawn state plus the phase procedure that flies
 * it. `desire`/`targets` carry the lesson's intent (the will handshake is
 * a later milestone); `durationCap` bounds the flight even if a phase
 * never completes.
 */
export class Lesson {
  readonly phases: readonly FlightElement[];
  readonly durationCap: number;
  /** The intention tensor's goal axis bin: which higher-level goal this
   * maneuver realizes ("take off from the runway"). Follow-instructor (0)
   * and free flight (1) are reserved; the curriculum starts at 2. */
  readonly goalId: number;
  /** The HSI bearing the maneuver turns onto (the turn lessons' ±180°);
   * null = no bearing target, the needle reads centered. */
  readonly targetBearing: number | null;

  constructor(readonly name: string, readonly desire: number,
              readonly targets: readonly number[],
              phases: readonly FlightElement[], options: {
                durationCap?: number,
                startKind?: "runway" | "cruise" | "cruise1000" | "cruise3000",
                repeatFrom?: number,
                goalId?: number,
                targetBearing?: number | null,
              } = {}) {
    this.phases = phases;
    this.durationCap = options.durationCap ?? 60.0;
    this.startKind = options.startKind ?? "cruise";
    this.repeatFrom = options.repeatFrom ?? 0;
    this.goalId = options.goalId ?? 1;
    this.targetBearing = options.targetBearing ?? null;
  }
  readonly startKind: "runway" | "cruise" | "cruise1000" | "cruise3000";
  /**
   * The phase the STUDENT's repeat starts from. 0 = from the maneuver's
   * start. Above 0 the instructor sets the aircraft up into the maneuver
   * and hands it over at that phase's boundary — used where the entry
   * cannot be replayed from memory: the stall/spin entry is never taught
   * (the student must never learn to enter a departure), and a turn's
   * roll-in from level cruise collides with the hold-all reflex the
   * level-cruise lesson stamps at the same instrument bins.
   */
  readonly repeatFrom: number;

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
  timeEpoch?: number;
}

export interface DemoTickResult {
  stamped: boolean;
}

/** Advance one fixed tick; returns false when the flight has ended.
 *  The machine is sealed (physics/aircraft.ts): `commands` is a COPY of
 *  the hand's own input state, the actuators are not exposed at all. */
export interface Stepper {
  step(): boolean;
  readonly body: RigidBody;
  readonly commands: Float64Array;
  readonly lesson: Lesson;
  readonly cursor: PhaseCursor;
  readonly seconds: number;
  readonly finished: boolean;
}
/**
 * The state the instructor hands the student when the repeat starts inside
 * a maneuver (repeatFrom > 0): the aircraft's body, its rates, and the
 * controls as the instructor left them ("your controls").
 */
export interface HandoffState {
  readonly position: Vec3;
  readonly velocity: Vec3;
  readonly attitude: Quaternion;
  readonly rateP: number;
  readonly rateQ: number;
  readonly rateR: number;
  readonly commands: Float64Array;
  readonly phaseIndex: number;
}

/** A flight start either from the lesson's spawn or an instructor handoff. */
export interface FlightStart {
  readonly body: RigidBody;
  readonly phaseIndex: number;
  readonly commands: Float64Array;
}

/** The hand's decisions as the controls panel shows them
 *  (see ClassroomFlight.handInputs): the commands are the aircraft's own
 *  (Aircraft.commands); these are the choices on top of them. */
export interface HandInputs {
  /** The student's last per-channel decision (EASE_OFF / HOLD / PUSH),
   *  or null while the instructor eases (no decisions happen). */
  readonly choices: Int32Array | null;
  /** Whether a decision window just opened — the 5 Hz brain beat. */
  readonly fresh: boolean;
}

/** Shared tick machinery: phase advance, the pin, ground contact.
 *  The machine is ONE Aircraft (physics/aircraft.ts) — the instructor
 *  and the student both fly this same class; the flight class is only
 *  the hand at its controls. */
abstract class ClassroomFlight implements Stepper {
  readonly aircraft: Aircraft;
  readonly cursor: PhaseCursor;
  tick = 0;
  protected ended = false;
  protected readonly steps: number;

  constructor(readonly lesson: Lesson, protected readonly school: FlightSchool,
              protected readonly policy: ServoPolicy,
              start: FlightStart | null = null) {
    const body = start === null ? lesson.spawn() : start.body;
    this.aircraft = new Aircraft(body);
    this.cursor = new PhaseCursor(lesson.phases, this.aircraft.body,
                                  start === null ? 0 : start.phaseIndex);
    this.steps = Math.floor(lesson.durationCap / school.dt);
  }

  /** The physical body of the flight's one aircraft. */
  get body(): RigidBody {
    return this.aircraft.body;
  }

  get seconds(): number {
    return this.tick * this.school.dt;
  }

  get finished(): boolean {
    return this.ended;
  }

  /** The hand's own input state — a COPY (the machine is sealed; the
   *  handoff capture and nothing else reads this). */
  get commands(): Float64Array {
    return Float64Array.from(this.handCommands());
  }

  protected abstract handCommands(): Float64Array;

  abstract step(): boolean;

  /**
   * What the hand at the controls is INPUTTING right now — the controls
   * panel's truth, so the viewer can SEE the student's inputs as they
   * happen:
   *  - the commanded targets live on the aircraft itself
   *    (Aircraft.commands — the hand's inputs);
   *  - choices: the student's LAST per-channel decision, one of
   *    EASE_OFF / HOLD / PUSH per surface — null while the instructor's
   *    hand eases its smooth demonstration path (no decisions happen);
   *  - fresh: whether a decision window just opened — the 5 Hz brain
   *    beat made visible.
   */
  abstract get handInputs(): HandInputs;

  /** One physics tick with the phase bookkeeping and the runway pin. */
  protected tickPhysics(commands: Float64Array): boolean {
    const dt = this.school.dt;
    this.aircraft.command(commands, dt);
    // Height above the terrain: ground effect lives in the physics.
    const agl = this.aircraft.body.position.z
      - this.school.world.terrainHeight(this.aircraft.body.position.x,
                                        this.aircraft.body.position.y);
    this.aircraft.step(dt, agl);
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
        // The gear: wings level, deck at the ground attitude until an
        // element commands rotation — a pitch target or a held elevator
        // pull. The roll shows pitch 0, yoke neutral.
        holdGroundDeck(this.body, !phase.rotates);
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

  constructor(school: FlightSchool, lesson: Lesson, policy: ServoPolicy,
              private readonly timeEpoch = 0) {
    super(lesson, school, policy);
    // Parked on the runway, the hand starts neutral-idle; already flying,
    // it starts at the trim.
    this.hand = new InstructorHand(startCommandsFor(lesson.startKind));
  }

  protected handCommands(): Float64Array {
    return this.hand.commands;
  }

  /** The instructor's hand: smooth eased inputs, no decisions to show. */
  get handInputs(): HandInputs {
    return { choices: null, fresh: false };
  }
  step(): boolean {
    if (this.ended) {
      return false;
    }
    const dt = this.school.dt;
    // The element's control law runs every tick (the hand chases live
    // targets); the stamp decision happens at decision ticks, reading
    // the same raw.
    const raw = this.cursor.phaseTargets(this.body);
    if (this.tick % DECISION_EVERY === 0) {
      const demo = this.school.demoTick(this.policy, this.lesson, this.cursor,
                                       this.body, this.aircraft.surfacePositions,
                                       raw, { timeEpoch: this.timeEpoch });
      this.stamped += demo.stamped ? 1 : 0;
    }
    const smooth = this.hand.easeToward(raw, dt);
    return this.tickPhysics(Float64Array.from(smooth));
  }
}
/**
 * One recorded tick of a student's repeat, for goal scoring. The same
 * readings the phase laws see, snapshotted at each decision tick with the
 * phase the student's own cursor was in.
 */
export interface GoalSample {
  readonly phaseIndex: number;
  readonly airspeed: number;
  readonly altitudeM: number;
  readonly bank: number;
  readonly pitch: number;
  readonly vsMps: number;
  readonly turned: number;
  readonly alpha: number;
  readonly rateR: number;
}

/** The student's repeat, recorded at every decision tick. */
export class GoalTrace {
  readonly samples: GoalSample[] = [];

  record(phaseIndex: number, r: PhaseReadings): void {
    this.samples.push({
      phaseIndex, airspeed: r.airspeed, altitudeM: r.altitudeM,
      bank: r.bank, pitch: r.pitch, vsMps: r.vsMps, turned: r.turned,
      alpha: r.alpha, rateR: r.rateR,
    });
  }

  ofPhase(phaseIndex: number): readonly GoalSample[] {
    return this.samples.filter((s) => s.phaseIndex === phaseIndex);
  }

  lastOfPhase(phaseIndex: number): GoalSample | null {
    const ofPhase = this.ofPhase(phaseIndex);
    return ofPhase.length === 0 ? null : ofPhase[ofPhase.length - 1];
  }

  /** The flight's final sample (null before the first decision tick). */
  last(): GoalSample | null {
    return this.samples.length === 0 ? null : this.samples[this.samples.length - 1];
  }
}

/** A goal judge: what fraction (0..1) of the goal the repeat earned. */
export type GoalJudge = (trace: GoalTrace, phaseIndex: number) => number;

/** One known, named goal of a maneuver. */
export interface ManeuverGoal {
  readonly label: string;
  readonly judge: GoalJudge;
}

/**
 * The repeat flies pure memory: the student flies exactly what it
 * remembers from the demonstration. The exploration lottery (5% random
 * commands) turned out to cost more than it gave — one random jab
 * mid-rotation or mid-turn is enough to fail a hold goal, and a retry
 * after a re-demonstration differs anyway when the student's own flight
 * diverges from the demo it watched.
 */
export const REPEAT_EPSILON = 0.0;

/** Goal helpers: the band-hold and the phase-event shapes. */
export const GoalChecks = {
  /** The fraction of a phase's samples that satisfy the band check. */
  hold: (label: string, ok: (s: GoalSample) => boolean): ManeuverGoal => ({
    label,
    judge: (trace, phaseIndex) => {
      const ofPhase = trace.ofPhase(phaseIndex);
      if (ofPhase.length === 0) {
        return 0.0;
      }
      let good = 0;
      for (const s of ofPhase) {
        if (ok(s)) {
          good += 1;
        }
      }
      return good / ofPhase.length;
    },
  }),
  /** All-or-nothing on the phase's last recorded sample. */
  atPhaseEnd: (label: string, ok: (s: GoalSample) => boolean): ManeuverGoal => ({
    label,
    judge: (trace, phaseIndex) => {
      const s = trace.lastOfPhase(phaseIndex);
      return s !== null && ok(s) ? 1.0 : 0.0;
    },
  }),
  /** All-or-nothing on the flight's last recorded sample. */
  atFlightEnd: (label: string, ok: (s: GoalSample) => boolean): ManeuverGoal => ({
    label,
    judge: (trace) => {
      const s = trace.last();
      return s !== null && ok(s) ? 1.0 : 0.0;
    },
  }),
  /**
   * All-or-nothing on the first sample anywhere in the repeat that meets
   * the trigger — used for "rotate at 60 mph": the first nose-up is the
   * rotation, and its airspeed is the grade.
   */
  firstWhere: (label: string, trigger: (s: GoalSample) => boolean,
               ok: (s: GoalSample) => boolean): ManeuverGoal => ({
    label,
    judge: (trace) => {
      for (const s of trace.samples) {
        if (trigger(s)) {
          return ok(s) ? 1.0 : 0.0;
        }
      }
      return 0.0;
    },
  }),
};

// ============================================================================
// THE ATOMIC ELEMENTS — one small class per chunk of a maneuver, each
// owning its vocabulary word (the tensor's element axis), its control
// law parameters, its exit condition, and its named goals.
// ============================================================================

/**
 * TAKEOFF ROLL: full throttle, yoke neutral, wings level on the gear —
 * roll on the ground until rotation speed.
 */
class RunwayRollElement extends FlightElement {
  readonly name: ElementName = "roll";
  readonly stamp = true;
  readonly pin = true;

  constructor() {
    super({ bankTarget: 0.0, elevatorOverride: 0.0, rudder: 0.0, throttle: 1.0 });
  }

  readonly goals = [GoalChecks.hold(
    "roll on the ground, wings level, nose on the wheels, until 60 mph",
    (s) => Math.abs(s.pitch) <= 0.06
           && Math.abs(s.bank) <= GOAL_TOLERANCES.wingsLevel)];

  done(r: PhaseReadings): boolean {
    return r.airspeed >= SYLLABUS.rotateMps;
  }
}

/**
 * ROTATION: at 60 mph pull the yoke and HOLD THE PULL through ground
 * effect — the deck comes up to the 15° climb attitude while the wing
 * starts to carry.
 */
class RotateElement extends FlightElement {
  readonly name: ElementName = "rotate";
  readonly stamp = true;
  readonly pin = true;

  constructor() {
    super({ pitchTarget: SYLLABUS.pitchClimb, bankTarget: 0.0,
            rudder: 0.0, throttle: 1.0 });
  }

  readonly goals = [
    GoalChecks.firstWhere(
      "rotate at 60 mph (not before, not much after)",
      (s) => s.pitch > 0.07,
      (s) => s.airspeed >= 0.9 * SYLLABUS.rotateMps
             && s.airspeed <= 1.2 * SYLLABUS.rotateMps),
    GoalChecks.hold("hold 15° pitch through ground effect to 100 ft",
      (s) => Math.abs(s.pitch - SYLLABUS.pitchClimb) <= GOAL_TOLERANCES.pitchHold),
  ];

  done(r: PhaseReadings): boolean {
    return r.altitudeM >= SYLLABUS.runwayM + SYLLABUS.liftOffToM;
  }
}

/**
 * CLIMB: hold the 15° climb attitude, wings level, full throttle — the
 * takeoff's climb to 1000 ft and the stall/spin recovery's pull-up back
 * to 3000 ft are the SAME skill at different target altitudes.
 */
class ClimbElement extends FlightElement {
  readonly name: ElementName = "climb";
  readonly stamp = true;

  constructor(private readonly toAltitudeM: number) {
    super({ pitchTarget: SYLLABUS.pitchClimb, bankTarget: 0.0,
            rudder: 0.0, throttle: 1.0 });
  }

  readonly goals = [GoalChecks.hold("climb at 15° pitch, wings level",
    (s) => Math.abs(s.pitch - SYLLABUS.pitchClimb) <= GOAL_TOLERANCES.pitchHold
           && Math.abs(s.bank) <= GOAL_TOLERANCES.wingsLevel)];

  done(r: PhaseReadings): boolean {
    return r.altitudeM >= this.toAltitudeM - LEVEL_OFF_LEAD_S * r.vsMps
           && r.vsMps >= -0.2;
  }
}

/**
 * LEVEL OFF: break the climb with the firm, gradually-released nose-down
 * and settle onto the target altitude, smooth — 1000 ft after takeoff,
 * 3000 ft after the stall/spin recovery.
 */
class LevelOffElement extends FlightElement {
  readonly name: ElementName = "level off";
  readonly stamp = true;
  private readonly atAltitudeM: number;

  constructor(atAltitudeM: number, altitudeName: string) {
    super({ pitchCapture: true, vsTarget: 0.0, bankTarget: 0.0,
            rudder: 0.0, throttle: SYLLABUS.cruiseThrottle });
    this.atAltitudeM = atAltitudeM;
    this.goals = [GoalChecks.atFlightEnd(
      `level off smoothly at ${altitudeName}`,
      (s) => Math.abs(s.altitudeM - this.atAltitudeM) <= 25.0
             && Math.abs(s.vsMps) <= GOAL_TOLERANCES.vsLevel)];
  }

  done(r: PhaseReadings): boolean {
    return r.ticks >= 150 && Math.abs(r.vsMps) < 0.3;
  }
}

/** LEVEL CRUISE: hands-off, 85 mph, 2300 RPM, wings level, VSI 0. */
class CruiseElement extends FlightElement {
  readonly name: ElementName = "cruise";
  readonly stamp = true;

  constructor() {
    super({ vsTarget: 0.0, bankTarget: 0.0,
            speedTarget: SYLLABUS.cruiseSpeedMps,
            rudder: 0.0, throttle: SYLLABUS.cruiseThrottle });
  }

  readonly goals = [
    GoalChecks.hold("hold altitude (VSI ~0)",
      (s) => Math.abs(s.vsMps) <= GOAL_TOLERANCES.vsLevel),
    GoalChecks.hold("wings level",
      (s) => Math.abs(s.bank) <= GOAL_TOLERANCES.wingsLevel),
    GoalChecks.hold("hold 85 mph",
      (s) => Math.abs(s.airspeed - SYLLABUS.cruiseSpeedMps)
             <= GOAL_TOLERANCES.speedHold),
  ];

  done(r: PhaseReadings): boolean {
    return r.ticks >= 1000;
  }
}

/** ROLL IN: roll to the target bank and start the turn. */
class RollInElement extends FlightElement {
  readonly name: ElementName = "roll in";
  readonly stamp = true;

  constructor(bank: number, speedTarget: number, throttle: number,
              rudder: RudderLaw) {
    super({ bankTarget: bank, carryTurn: true, vsTarget: 0.0,
            speedTarget, rudder, throttle });
  }

  done(r: PhaseReadings): boolean {
    return Math.abs(r.bank) >= 0.95 * Math.abs(this.law.bankTarget!);
  }
}

/** TURN: hold the bank, nose on the horizon, through the full 180°. */
class TurnElement extends FlightElement {
  readonly name: ElementName = "turn";
  readonly stamp = true;
  private readonly leadOut: number;
  private readonly speedTarget: number;
  private readonly steep: boolean;

  constructor(bank: number, speedTarget: number, throttle: number,
              rudder: RudderLaw, leadOut: number, steep: boolean) {
    super({ bankTarget: bank, carryTurn: true, vsTarget: 0.0,
            speedTarget, rudder, throttle });
    this.leadOut = leadOut;
    this.speedTarget = speedTarget;
    this.steep = steep;
    this.goals = [
      steep
        ? GoalChecks.hold("hold the bank at 60° (steep turn)",
            (s) => Math.abs(s.bank) >= 0.85 * SYLLABUS.bankSteep
                   && Math.abs(s.bank) <= 1.15 * SYLLABUS.bankSteep)
        : GoalChecks.hold("hold the bank in the 20-30° band",
            (s) => Math.abs(s.bank) >= 0.9 * SYLLABUS.bankTurn
                   && Math.abs(s.bank) <= 1.3 * SYLLABUS.bankTurn),
      GoalChecks.hold(
        steep
          ? "hold altitude through the steep turn (elevator + top rudder)"
          : "hold altitude through the turn (elevator)",
        (s) => Math.abs(s.vsMps) <= (steep ? 1.5 : 1.0) * GOAL_TOLERANCES.vsTurn),
      GoalChecks.hold(
        steep ? "hold ~107 mph through the steep turn"
              : "hold 85 mph through the turn",
        (s) => Math.abs(s.airspeed - speedTarget) <= (steep ? 5.0 : 4.0)),
    ];
  }

  done(r: PhaseReadings): boolean {
    return Math.abs(r.turned) >= SYLLABUS.turnAmount - this.leadOut;
  }
}

/** ROLL OUT: the heading-hold roll-out onto exactly ±180°, wings level.
 *  The roll-out COMPLETES the turn — it carries the heading accumulator
 *  and lands on the target — so the 180° goal is graded here, at the
 *  end of the element that finishes the maneuver. */
class RollOutElement extends FlightElement {
  readonly name: ElementName = "roll out";
  readonly stamp = true;

  constructor(speedTarget: number, throttle: number) {
    super({ headingHold: true, carryTurn: true, vsTarget: 0.0,
            speedTarget, rudder: 0.0, throttle });
    this.goals = [
      GoalChecks.atPhaseEnd("turn a full 180°",
        (s) => Math.abs(s.turned) >= 0.87 * SYLLABUS.turnAmount
               && Math.abs(s.turned) <= 1.06 * SYLLABUS.turnAmount),
    ];
  }

  done(r: PhaseReadings): boolean {
    return Math.abs(wrapPi(Math.PI - Math.abs(r.turned))) <= 0.014
           && Math.abs(r.bank) <= 0.022;
  }
}

/** SETTLE: hold level and let the aircraft settle after the roll-out. */
class SettleElement extends FlightElement {
  readonly name: ElementName = "settle";
  readonly stamp = true;

  constructor() {
    super({ vsTarget: 0.0, bankTarget: 0.0,
            speedTarget: SYLLABUS.cruiseSpeedMps,
            rudder: 0.0, throttle: SYLLABUS.cruiseThrottle });
  }

  done(r: PhaseReadings): boolean {
    return r.ticks >= 150 && Math.abs(r.vsMps) < 0.5;
  }
}

/** STALL ENTRY: idle, hold altitude, bleed off to the stall. Shown only. */
class StallEntryElement extends FlightElement {
  readonly name: ElementName = "stall entry";
  readonly stamp = false;

  constructor() {
    super({ vsTarget: 0.0, bankTarget: 0.0, rudder: 0.0,
            throttle: SYLLABUS.idleThrottle });
  }

  done(r: PhaseReadings): boolean {
    return r.airspeed <= SYLLABUS.stallEntryMps;
  }
}

/** WAIT: hold it into the break. Shown only. */
class StallWaitElement extends FlightElement {
  readonly name: ElementName = "wait";
  readonly stamp = false;

  constructor() {
    super({ vsTarget: 0.0, bankTarget: 0.0, rudder: 0.0,
            throttle: SYLLABUS.idleThrottle });
  }

  done(r: PhaseReadings): boolean {
    return r.alpha >= 0.35 || r.vsMps < -3.0;
  }
}

/** BREAK: neutral yoke, let the nose drop through the stall. Shown only. */
class StallBreakElement extends FlightElement {
  readonly name: ElementName = "break";
  readonly stamp = false;

  constructor() {
    super({ elevatorOverride: 0.0, bankTarget: 0.0, rudder: 0.0,
            throttle: SYLLABUS.idleThrottle });
  }

  done(r: PhaseReadings): boolean {
    return r.pitch < 0.0;
  }
}

/** PUSH: the stall-recovery dive — nose down, break the stall, dive out. */
class PushElement extends FlightElement {
  readonly name: ElementName = "push";
  readonly stamp = true;

  constructor() {
    super({ pitchTarget: SYLLABUS.pitchDive, bankTarget: 0.0,
            rudder: 0.0, throttle: SYLLABUS.idleThrottle });
  }

  readonly goals = [
    GoalChecks.hold("push the nose down (break the stall)",
      (s) => s.pitch <= GOAL_TOLERANCES.pitchDive && s.vsMps < -2.0),
    GoalChecks.atPhaseEnd("dive out to 65 mph",
      (s) => s.airspeed >= 0.95 * SYLLABUS.diveOutMps),
  ];

  done(r: PhaseReadings): boolean {
    return r.airspeed >= SYLLABUS.diveOutMps;
  }
}

/** DROP WING: the spin entry's stalled, dropping wing. Shown only. */
class DropWingElement extends FlightElement {
  readonly name: ElementName = "drop wing";
  readonly stamp = false;

  constructor() {
    super({ aileronOverride: -AILERON_LIMIT, elevatorOverride: 0.1,
            rudder: "proSpin", throttle: SYLLABUS.idleThrottle });
  }

  done(r: PhaseReadings): boolean {
    return r.rateR <= -0.6 || r.ticks >= 200;
  }
}

/** SPIN: the developed, self-sustaining rotation. Shown only. */
class SpinElement extends FlightElement {
  readonly name: ElementName = "spin";
  readonly stamp = false;

  constructor() {
    super({ aileronOverride: -0.2, elevatorOverride: 0.12,
            rudder: "proSpin", throttle: SYLLABUS.idleThrottle });
  }

  done(r: PhaseReadings): boolean {
    return r.ticks >= 100;
  }
}

/**
 * OPPOSITE RUDDER: full anti-spin rudder and HOLD it until the rotation
 * stops.
 */
class OppositeRudderElement extends FlightElement {
  readonly name: ElementName = "opposite rudder";
  readonly stamp = true;

  constructor() {
    super({ elevatorOverride: 0.0, rudder: "antiSpin",
            throttle: SYLLABUS.idleThrottle });
  }

  readonly goals = [GoalChecks.atPhaseEnd(
    "opposite rudder until the rotation stops",
    (s) => Math.abs(s.rateR) <= GOAL_TOLERANCES.spinStopped)];

  done(r: PhaseReadings): boolean {
    return Math.abs(r.rateR) <= 0.1;
  }
}

/** DIVE: neutral yoke, wings level, let it build 60 mph. */
class DiveElement extends FlightElement {
  readonly name: ElementName = "dive";
  readonly stamp = true;

  constructor() {
    super({ elevatorOverride: 0.0, bankTarget: 0.0, rudder: 0.0,
            throttle: SYLLABUS.idleThrottle });
  }

  readonly goals = [
    GoalChecks.hold("let it dive: nose low, wings level",
      (s) => s.pitch <= 0.05 && Math.abs(s.bank) <= 0.2),
    GoalChecks.atPhaseEnd("build 60 mph in the dive",
      (s) => s.airspeed >= 0.95 * SYLLABUS.spinPullUpMps),
  ];

  done(r: PhaseReadings): boolean {
    return r.airspeed >= SYLLABUS.spinPullUpMps;
  }
}

/**
 * The student repeats the maneuver alone: pure memory replay, the stick
 * spring active, scored 0-6 against the lesson's NAMED GOALS. Every
 * taught phase contributes its decision-tick weight (the demonstration's
 * own tick count in that phase), and each goal's judge earns a fraction
 * of it — so a skipped, blown, or never-reached phase costs exactly its
 * share of the grade.
 */
export class ExamFlight extends ClassroomFlight {
  private commandsNow: Float64Array;
  readonly trace = new GoalTrace();
  private readonly spec: LessonGoalSpec;
  private readonly savedEpsilon: number;
  /**
   * The element the student has CHOSEN to execute right now — the
   * intentional "currently climb" of the two-level goal/element plan.
   * Starts at the handoff element (or the maneuver's first).
   */
  chosenIndex: number;
  /**
   * The element selector's refractory period: it fires its choice, then
   * waits one decision stunned while the servo neuron acts on it — the
   * pulsating dynamic (one neuron fires, the other waits; the choice
   * travels in time, not in an instantaneous loop).
   */
  private readonly elementRefractory = new Refractory(1);
  /** The servo neuron's stun: 1 decision after each element fire. */
  private servoStunned = 0;
  /** The last decision the servo neuron chose, per channel — the panel's
   *  view of the student's inputs. */
  private lastChoices: Int32Array | null = null;

  constructor(school: FlightSchool, lesson: Lesson, policy: ServoPolicy,
              private readonly rng: NumpyPcg64) {
    super(lesson, school, policy, school.preparedStart(lesson));
    this.savedEpsilon = policy.epsilon;
    policy.epsilon = REPEAT_EPSILON;   // fly what it remembers
    const start = school.preparedStart(lesson);
    this.commandsNow = start === null
      ? Float64Array.from(startCommandsFor(lesson.startKind))
      : Float64Array.from(start.commands);
    this.spec = school.goalSpec(lesson);
    this.chosenIndex = this.cursor.index;
  }

  protected handCommands(): Float64Array {
    return this.commandsNow;
  }

  /** The student's inputs: the LAST decision per channel (push / ease /
   *  hold) with the decision window's beat — the commanded targets are
   *  the aircraft's own (Aircraft.commands). What the panel shows. */
  get handInputs(): HandInputs {
    return {
      choices: this.lastChoices,
      fresh: this.tick % DECISION_EVERY < DECISION_EVERY / 2,
    };
  }
  /** The element the student is executing now (for narration + grading). */
  get chosenElement(): FlightElement {
    return this.lesson.phases[Math.min(this.chosenIndex,
                                       this.lesson.phases.length - 1)];
  }

  /** The goal score this repeat earned, 0-6. */
  get goalScore(): number {
    let earned = 0.0;
    let nominal = 0.0;
    for (let i = 0; i < this.spec.goals.length; i++) {
      earned += this.spec.goals[i].judge(this.trace, this.spec.phaseIndices[i])
        * this.spec.nominal[i];
      nominal += this.spec.nominal[i];
    }
    return nominal === 0.0 ? 0.0 : QUALITY_CAP * earned / nominal;
  }

  /** Per-goal fractions for the report card, in curriculum order. */
  goalVerdicts(): Array<{ label: string; fraction: number }> {
    return this.spec.goals.map((goal, i) => ({
      label: goal.label,
      fraction: goal.judge(this.trace, this.spec.phaseIndices[i]),
    }));
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
      // THE PULSATING INTENTION. The panel pulses in every decision; the
      // neurons take turns firing, so the system never runs a feedback
      // loop — it oscillates: choose, act, choose, act.
      const sense = this.school.instruments.senseReading(this.body,
                                                        this.aircraft.surfacePositions,
                                                        this.lesson.targetBearing);
      // 1. The element neuron fires (when its refractory period ends):
      //    WHICH element of the maneuver to execute — the higher-level
      //    goal is the assigned lesson ("take off from the runway").
      //    The fire startles the servo neuron for one decision — but
      //    only when the intention actually CHANGES: a re-confirmation
      //    of the same element startles nothing, so the student keeps
      //    acting at full cadence while it holds an element, and the
      //    pulse travels the synapse only on a transition.
      if (this.elementRefractory.tick()) {
        const next = this.intend(sense);
        if (next !== this.chosenIndex) {
          this.servoStunned = 1;   // the new intention startles the servo neuron
        }
        this.chosenIndex = next;
      }
      // 2. The servo neuron fires (when its stun relaxes): WHAT TO DO
      //    inside the chosen element — the remembered push/ease/hold.
      if (this.servoStunned > 0) {
        this.servoStunned -= 1;
      } else {
        const actions = this.policy.choose(this.chosenElement.elementId,
                                           sense, this.rng);
        this.commandsNow = this.policy.commandFrom(this.commandsNow, actions);
        this.lastChoices = actions;
      }
      // Grade the chosen element's flying against its named goals.
      this.trace.record(this.chosenIndex,
                        this.cursor.readings(this.body));
    }
    const more = this.tickPhysics(this.commandsNow);
    this.commandsNow = springStickTowardNeutral(this.commandsNow, dt);
    if (!more) {
      this.release();
    }
    return more;
  }

  /**
   * The element choice — intention, not obedience. Among the maneuver's
   * TAUGHT elements (the stamped ones; entries are shown but never
   * taught), the student executes the FURTHEST one its OWN memory
   * supports at this reading — an element fires only on its own cells
   * (raw evidence above the bar), never on a neighbor's blur leak, and
   * the fired (denoised) projection must confirm it. The furthest rule
   * is the progression: at the rotation boundary roll and rotate both
   * fire, and the student moves on to rotate. The choice is MONOTONE:
   * a fired element never regresses the plan below what the student
   * already chose — a maneuver's elements are an ordered chain (roll
   * until the speed, climb until the altitude, level off), and the
   * reading noise at a boundary must not bounce the student back down
   * the chain it has already climbed. Where nothing fires the student
   * KEEPS the current element — a student does not un-choose
   * mid-flight because a reading drifted off the demonstration's exact
   * profile.
   */
  private intend(sense: SenseReading): number {
    const evidence = this.policy.elementEvidence(this.lesson.goalId, sense);
    const projection = this.policy.elementProjection(this.lesson.goalId,
                                                     sense, this.rng);
    let chosen = -1;
    for (let i = 0; i < this.lesson.phases.length; i++) {
      const element = this.lesson.phases[i];
      if (!element.stamp) {
        continue;   // shown but never taught: never a choice
      }
      const own = evidence[element.elementId];
      const fired = projection[element.elementId];
      if (own > ELEMENT_FIRE_BAR && fired > ELEMENT_FIRE_BAR) {
        chosen = i;
      }
    }
    if (chosen < 0) {
      return Math.min(this.chosenIndex, this.lesson.phases.length - 1);
    }
    return Math.max(chosen, this.chosenIndex);   // monotone: no regression
  }
}

/** The lesson's known goals: judges in curriculum order, with weights. */
export interface LessonGoalSpec {
  readonly goals: readonly ManeuverGoal[];
  /** The phase each goal grades (the judge receives it). */
  readonly phaseIndices: readonly number[];
  /** The demonstration's decision-tick count in that phase — the weight. */
  readonly nominal: readonly number[];
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
  private readonly prepared = new Map<string, {
    phaseTicks: number[];
    handoff: HandoffState | null;
  }>();
  private readonly goalSpecs = new Map<string, LessonGoalSpec>();

  constructor(instruments: FlightInstruments, world?: World) {
    this.instruments = instruments;
    this.world = world ?? new World({ halfExtent: 800.0 });
    this.defineCurriculum();
  }

  private defineCurriculum(): void {
    const S = SYLLABUS;
    const lessons: [string, number, readonly number[], FlightElement[],
                   { durationCap: number, startKind: Lesson["startKind"],
                     repeatFrom?: number, goalId?: number,
                     targetBearing?: number | null }][] = [];

    // Goal ids on the intention tensor's goal axis: follow-instructor (0)
    // and free flight (1) are reserved; the curriculum takes 2 upward.
    let goalId = 2;
    const nextGoalId = (): number => goalId++;

    // --- TAKEOFF: roll on the ground to 60 mph, rotate and hold the pull
    // through ground effect to 100 ft, climb at 15° to 1000 ft, level off
    // smoothly there. The student is taught the elements roll -> rotate
    // -> climb -> level off, and chooses each from memory. ---------------
    lessons.push(["takeoff", DESIRE_CLIMB, [0.1, 0.0, 1.0], [
      new RunwayRollElement(),
      new RotateElement(),
      new ClimbElement(S.climbToM),
      new LevelOffElement(S.climbToM, "1000 ft"),
    ], { durationCap: 150.0, startKind: "runway", goalId: nextGoalId() }]);

    // --- LEVEL CRUISE: the 80-90 mph / 2300 RPM hands-off lesson. -------
    lessons.push(["level_cruise", DESIRE_LEVEL, [0.0, 0.0, 0.0], [
      new CruiseElement(),
    ], { durationCap: 25.0, startKind: "cruise1000", goalId: nextGoalId() }]);

    // --- TURNS: 180° at 25° bank, nose held on the horizon. The roll-out
    // is a HEADING HOLD that lands on entry heading ± 180°. The student
    // repeats from the ESTABLISHED BANK (repeatFrom): rolling in from
    // level cruise cannot be replayed from memory — the readings there
    // belong to the level-cruise hold-all reflex. The roll-in is still
    // STAMPED: the student re-banks whenever the bank decays below the
    // 20° band floor. The HSI target is the 180°-off bearing. -----------
    const turnElements = (bank: number): FlightElement[] => [
      new RollInElement(bank, S.cruiseSpeedMps, S.cruiseThrottle, 0.0),
      new TurnElement(bank, S.cruiseSpeedMps, S.cruiseThrottle, 0.0,
                      S.turnLeadOut, false),
      new RollOutElement(S.cruiseSpeedMps, S.cruiseThrottle),
      new SettleElement(),
    ];
    lessons.push(["turn_left_180", DESIRE_LEFT, [0.0, -S.bankTurn, 0.0],
                  turnElements(-S.bankTurn),
                  { durationCap: 60.0, startKind: "cruise1000", repeatFrom: 1,
                    goalId: nextGoalId(), targetBearing: Math.PI }]);
    lessons.push(["turn_right_180", DESIRE_RIGHT, [0.0, S.bankTurn, 0.0],
                  turnElements(S.bankTurn),
                  { durationCap: 60.0, startKind: "cruise1000", repeatFrom: 1,
                    goalId: nextGoalId(), targetBearing: Math.PI }]);

    // --- STEEP TURNS: 180° at 60° bank with top rudder. A 2 g level turn
    // needs ENERGY: enter at cruise, full throttle, speed rides up to the
    // steep-turn band (~107 mph). Like the normal turns, the student
    // repeats from the established 60° bank. ----------------------------
    const steepElements = (bank: number): FlightElement[] => [
      new RollInElement(bank, S.steepSpeedMps, 1.0, "top"),
      new TurnElement(bank, S.steepSpeedMps, 1.0, "top",
                      S.steepLeadOut, true),
      new RollOutElement(S.steepSpeedMps, S.cruiseThrottle),
      new SettleElement(),
    ];
    lessons.push(["steep_turn_left_180", DESIRE_LEFT, [0.0, -S.bankSteep, 0.0],
                  steepElements(-S.bankSteep),
                  { durationCap: 45.0, startKind: "cruise1000", repeatFrom: 1,
                    goalId: nextGoalId(), targetBearing: Math.PI }]);
    lessons.push(["steep_turn_right_180", DESIRE_RIGHT, [0.0, S.bankSteep, 0.0],
                  steepElements(S.bankSteep),
                  { durationCap: 45.0, startKind: "cruise1000", repeatFrom: 1,
                    goalId: nextGoalId(), targetBearing: Math.PI }]);

    // --- STALL RECOVERY: the instructor stalls it and hands over at the
    // break; the student flies the RECOVERY: push to the dive, climb back
    // at 15°, level off at 3000 ft. The ENTRY is never taught. -----------
    lessons.push(["stall_recovery", DESIRE_RECOVER, [0.0, 0.0, 1.0], [
      new StallEntryElement(),
      new StallWaitElement(),
      new StallBreakElement(),
      new PushElement(),
      new ClimbElement(S.stallAltM),
      new LevelOffElement(S.stallAltM, "3000 ft"),
    ], { durationCap: 120.0, startKind: "cruise3000", repeatFrom: 3,
         goalId: nextGoalId() }]);

    // --- SPIN RECOVERY: the instructor spins it and hands over inside
    // the developed rotation; the student recovers: opposite rudder to the
    // stop, dive to 60 mph, climb at 15°, back to 3000 ft level. --------
    lessons.push(["spin_recovery", DESIRE_RECOVER, [0.0, 0.0, 1.0], [
      new StallEntryElement(),
      new StallWaitElement(),
      new DropWingElement(),
      new SpinElement(),
      new OppositeRudderElement(),
      new DiveElement(),
      new ClimbElement(S.stallAltM),
      new LevelOffElement(S.stallAltM, "3000 ft"),
    ], { durationCap: 140.0, startKind: "cruise3000", repeatFrom: 4,
         goalId: nextGoalId() }]);

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
   * One instructor demonstration tick: the element's control law produces
   * the raw deflection targets (always returned to fly); the
   * anti-poisoning gates decide what is stamped. Element-aware: entry
   * elements never stamp (the stall/spin entry must never be taught).
   *
   * This is the INSTRUCTION FLIGHT, governed by the higher-level goal
   * "listen to the instructor and do what they ask": the instructor chose
   * the element and the actions; the student records what it watches —
   * filed under the maneuver's goal and the element being flown, at the
   * sense reading the panel shows, so the solo repeat finds it.
   *
   * The stamp law mirrors how a real student learns from the instructor's
   * hands:
   * - A control the instructor is MOVING stamps the movement (push/ease),
   *   keyed at the actual control positions — so the student, whose own
   *   surfaces pass the same bins, can chase it.
   * - A control the instructor is HOLDING AWAY from the cruise trim
   *   stamps the held pressure: the student's stick springs toward trim,
   *   so "hold" only means hold when the hand rests at the trim. A parked
   *   climb pull or a held spin correction is taught as the continuous
   *   push that keeps it there.
   * - Only a genuinely hands-off reading (everything at trim) stamps the
   *   hold-all reflex.
   */
  demoTick(policy: ServoPolicy, lesson: Lesson, cursor: PhaseCursor,
           body: RigidBody, positions: ArrayLike<number>,
           rawTargets: Float64Array,
           options: DemoTickOptions): DemoTickResult {
    const element = cursor.current;
    const raw = rawTargets;
    if (!element.stamp) {
      return { stamped: false };
    }
    const sense = this.instruments.senseReading(body, positions,
                                               lesson.targetBearing);
    let actions = FlightSchool.translateTargets(raw, positions);
    if (!actions.some((a) => a !== HOLD)) {
      actions = FlightSchool.heldPressures(positions);
    }
    policy.imprint(lesson.goalId, element.elementId, sense, actions,
                   this.imprintAmount, options.timeEpoch ?? 0);
    return { stamped: true };
  }

  /** What the instructor's resting hands teach: held pressures, or
   *  hold-all only when every control rests at the cruise trim. */
  private static heldPressures(positions: ArrayLike<number>): Int32Array {
    const actions = new Int32Array(4).fill(HOLD);
    let holding = false;
    for (let surface = 0; surface < 4; surface++) {
      const parked = positions[surface];
      const trim = TRIM_COMMAND[surface];
      if (Math.abs(parked - trim) > 0.05) {
        actions[surface] = parked > trim ? PUSH : EASE_OFF;
        holding = true;
      }
    }
    if (!holding) {
      return new Int32Array(4).fill(HOLD);   // hold-all
    }
    return actions;
  }

  /** Instructor deflections -> the student's push/ease/hold decisions. */
  private static translateTargets(targets: Float64Array,
                                  positions: ArrayLike<number>): Int32Array {
    const actions = new Int32Array(4).fill(HOLD);
    for (let surface = 0; surface < 4; surface++) {
      const delta = targets[surface] - positions[surface];
      if (delta > 0.04) {
        actions[surface] = PUSH;
      } else if (delta < -0.04) {
        actions[surface] = EASE_OFF;
      }
    }
    return actions;
  }

  /**
   * One demonstration pass per lesson, cached: the per-phase decision-tick
   * counts (the goal weights) and, when the student's repeat starts inside
   * the maneuver, the handoff state at that phase boundary.
   */
  private prepareLesson(lesson: Lesson): {
    phaseTicks: number[];
    handoff: HandoffState | null;
  } {
    const cached = this.prepared.get(lesson.name);
    if (cached !== undefined) {
      return cached;
    }
    const flight = new DemoFlight(this, lesson, new ServoPolicy());
    const phaseTicks = lesson.phases.map(() => 0);
    let handoff: HandoffState | null = null;
    if (lesson.repeatFrom === 0) {
      handoff = null;
    }
    while (flight.step()) {
      if (flight.tick % DECISION_EVERY === 0) {
        const index = Math.min(flight.cursor.index, lesson.phases.length - 1);
        phaseTicks[index] += 1;
        if (handoff === null && lesson.repeatFrom > 0
            && index === lesson.repeatFrom) {
          handoff = {
            position: flight.body.position,
            velocity: flight.body.velocity,
            attitude: flight.body.attitude,
            rateP: flight.body.rateP,
            rateQ: flight.body.rateQ,
            rateR: flight.body.rateR,
            commands: Float64Array.from(flight.commands),
            phaseIndex: lesson.repeatFrom,
          };
        }
      }
    }
    const result = { phaseTicks, handoff };
    this.prepared.set(lesson.name, result);
    return result;
  }

  /**
   * The start of a student's repeat: null when it flies the maneuver from
   * its spawn, or a fresh aircraft built from the instructor's handoff at
   * repeatFrom ("your controls").
   */
  preparedStart(lesson: Lesson): FlightStart | null {
    if (lesson.repeatFrom === 0) {
      return null;
    }
    const { handoff } = this.prepareLesson(lesson);
    if (handoff === null) {
      // The demonstration never reached the handoff phase (a broken
      // phase procedure): fall back to the lesson's spawn.
      return null;
    }
    const body = new RigidBody(handoff.position, handoff.velocity);
    body.attitude = handoff.attitude;
    body.rateP = handoff.rateP;
    body.rateQ = handoff.rateQ;
    body.rateR = handoff.rateR;
    return {
      body,
      phaseIndex: handoff.phaseIndex,
      commands: Float64Array.from(handoff.commands),
    };
  }

  /** The lesson's known goals with their phase weights, in curriculum order. */
  goalSpec(lesson: Lesson): LessonGoalSpec {
    const cached = this.goalSpecs.get(lesson.name);
    if (cached !== undefined) {
      return cached;
    }
    const { phaseTicks } = this.prepareLesson(lesson);
    const goals: ManeuverGoal[] = [];
    const phaseIndices: number[] = [];
    const nominal: number[] = [];
    for (let i = lesson.repeatFrom; i < lesson.phases.length; i++) {
      const phase = lesson.phases[i];
      if (phase.goals === undefined) {
        continue;
      }
      const weight = Math.max(phaseTicks[i], 1);
      for (const goal of phase.goals) {
        goals.push(goal);
        phaseIndices.push(i);
        nominal.push(weight);
      }
    }
    const spec: LessonGoalSpec = { goals, phaseIndices, nominal };
    this.goalSpecs.set(lesson.name, spec);
    return spec;
  }

  /** How well the student repeats the maneuver, graded 0-6 on the goals. */
  studentQuality(policy: ServoPolicy, lesson: Lesson): number {
    const exam = new ExamFlight(this, lesson, policy,
                                new NumpyPcg64(RNG0_STATE, RNG0_INC));
    while (exam.step()) { /* run to completion */ }
    return exam.goalScore;
  }

  /** Imprint the curriculum, then run the show-and-repeat loop to the end. */
  teach(policy: ServoPolicy, options: {
    lessons?: readonly string[];
    bar?: number;
    maxAttempts?: number;
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

/** Where a session flight sits in the classroom loop. */
export type SessionStage = "demo" | "repeat" | "done";

/** One flight handed out by a SchoolSession. */
export interface SchoolFlight {
  readonly flight: DemoFlight | ExamFlight;
  readonly lesson: Lesson;
  readonly stage: SessionStage;
  /** How many times the instructor has demonstrated THIS lesson so far. */
  readonly attempt: number;
}

/**
 * The element being flown right now — the instructor's current element on
 * a demonstration, the student's CHOSEN element on a repeat: the
 * "currently climb" of the lesson's two-level plan.
 */
export function currentElementName(flight: SchoolFlight): string {
  return flight.stage === "demo"
    ? flight.flight.cursor.current.name
    : (flight.flight as ExamFlight).chosenElement.name;
}

/** One goal's verdict on a repeat, for the report card. */
export interface GoalReportRow {
  readonly label: string;
  readonly fraction: number;
}

/**
 * The classroom loop, resumable one flight at a time so the browser can
 * run it live. Per lesson, exactly the format a flight school runs:
 *
 *   1. the instructor demonstrates the maneuver,
 *   2. the student repeats it alone,
 *   3. the repeat is graded against the maneuver's known goals —
 *      at or above the bar: on to the next maneuver,
 *      below the bar: back to step 1 (the instructor shows it again),
 *      until the lesson passes or its attempt cap is reached.
 *
 * The loop interleaves: instructor, student, instructor, student — never
 * a block of demonstrations followed by a block of exams.
 *
 * Two endings exist. Bounded (loop: false, the headless default) the
 * course graduates and the session is done — teach() and the freeze tools
 * rely on this. Endless (loop: true, the browser classroom) the school
 * NEVER closes: when the curriculum runs out it wraps and keeps teaching,
 * the student's memory carrying over lap to lap, and skipLesson() moves
 * to the next maneuver on demand.
 */
export class SchoolSession {
  readonly school: FlightSchool;
  readonly policy: ServoPolicy;
  readonly bar: number;
  readonly maxAttempts: number;
  /** The endless classroom flag (see the constructor options). */
  readonly loop: boolean;
  private readonly names: readonly string[];
  /** Demonstration stamps absorbed per lesson. */
  readonly counts = new Map<string, number>();
  /** The student's latest goal score per lesson, 0-6. */
  readonly scores = new Map<string, number>();
  /** How many demonstrations each lesson has had (1 = passed first try). */
  readonly attempts = new Map<string, number>();
  /** The latest repeat's per-goal fractions per lesson (the report card). */
  readonly goalReports = new Map<string, GoalReportRow[]>();
  /** Lessons the student passed, in passing order. */
  readonly passed: string[] = [];
  graduate: ServoPolicy | null = null;
  /** How many times the class went back to step 1. */
  rounds = 0;
  /** How many times the endless classroom wrapped the curriculum
   *  (loop mode: 0 during the first lap, 1 once it begins again…). */
  laps = 0;
  failed: string[] = [];
  private stage: SessionStage = "demo";
  private index = 0;
  private pending: SchoolFlight | null = null;
  /** The classroom's TIME: one fresh epoch per demonstration, handed out
   *  monotonically across the whole school career. The neurons' EMA
   *  stacks decay in this time — each new demonstration sinks every
   *  older stamp one layer down, and the older lessons still count by
   *  their neuron's EMA (the static tail store keeps them from ever
   *  cliffing away, neurons.ts). */
  private demoEpoch = 0;
  /** The classroom's dice: one stream shared by every repeat, so attempt
   *  two explores differently from attempt one. */
  private readonly rng = new NumpyPcg64(RNG0_STATE, RNG0_INC);

  constructor(school: FlightSchool, policy: ServoPolicy, options: {
    lessons?: readonly string[];
    bar?: number;
    maxAttempts?: number;
    loop?: boolean;
  } = {}) {
    this.school = school;
    this.policy = policy;
    this.bar = options.bar ?? DEFAULT_PASS_SCORE;
    this.maxAttempts = options.maxAttempts ?? GRADUATION_ATTEMPTS;
    /** The endless classroom: the curriculum wraps instead of graduating,
     *  `done` never comes. runToCompletion()/teach() must not be used
     *  with it — they would not end. */
    this.loop = options.loop ?? false;
    this.names = options.lessons ?? [...school.lessons.keys()];
    for (const name of this.names) {
      if (!school.lessons.has(name)) {
        throw new Error(`unknown lesson: ${name}`);
      }
      this.counts.set(name, 0);
      this.attempts.set(name, 0);
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

  /** Skip to the next maneuver NOW (the next-maneuver button): the
   *  current flight is dropped ungraded — whatever the demonstration
   *  already stamped stays in the student's memory — and the next
   *  lesson's demonstration begins. Never a stop: in loop mode the
   *  curriculum wraps, bounded mode it may end the course. */
  skipLesson(): void {
    this.pending = null;
    this.advanceLesson();
  }

  /** Jump to the chosen maneuver NOW (the maneuver dropdown): the current
   *  flight is dropped ungraded and the named lesson's demonstration
   *  begins — the SAME student, same policy, same neurons, every weight
   *  carried over untouched. Only a fresh session (page reload, the
   *  restart button) ever starts a new student. */
  switchLesson(name: string): void {
    const index = this.names.indexOf(name);
    if (index < 0) {
      throw new Error(`unknown lesson: ${name}`);
    }
    this.pending = null;
    this.index = index;
    this.stage = "demo";
  }

  /** The lesson the next flight will be about (for narration before it starts). */
  get currentLessonName(): string | null {
    return this.stage === "done" ? null : this.names[this.index] ?? null;
  }

  private absorbPending(): void {
    const pending = this.pending;
    if (pending === null) {
      return;
    }
    this.pending = null;
    const name = pending.lesson.name;
    if (pending.stage === "demo") {
      this.counts.set(name, this.counts.get(name)!
                      + (pending.flight as DemoFlight).stamped);
      this.stage = "repeat";
      return;
    }
    // The student's repeat: grade it against the maneuver's goals.
    const repeat = pending.flight as ExamFlight;
    const score = repeat.goalScore;
    this.scores.set(name, score);
    this.goalReports.set(name, repeat.goalVerdicts());
    if (score >= this.bar) {
      this.passed.push(name);
      this.advanceLesson();
      return;
    }
    this.rounds += 1;                     // back to step 1
    if (!this.loop && this.attempts.get(name)! >= this.maxAttempts) {
      this.failed.push(name);
      this.advanceLesson();
    } else {
      this.stage = "demo";
    }
  }

  private advanceLesson(): void {
    this.index += 1;
    this.stage = "demo";
    if (this.index === this.names.length) {
      if (this.loop) {
        this.wrapLap();
      } else {
        this.finish();
      }
    }
  }

  /** The endless classroom wraps: the curriculum starts over and school
   *  keeps going — same student, memory carrying over, one lap deeper. */
  private wrapLap(): void {
    const graduate = new ServoPolicy();
    graduate.restore(this.policy.snapshot());
    this.graduate = graduate;
    this.school.graduate = graduate;
    this.laps += 1;
    this.index = 0;
  }

  private advance(): SchoolFlight | null {
    if (this.stage === "done") {
      return null;
    }
    const name = this.names[this.index];
    const lesson = this.school.lessons.get(name)!;
    if (this.stage === "demo") {
      const attempt = this.attempts.get(name)! + 1;
      this.attempts.set(name, attempt);
      // Every demonstration lands in the school's NEXT epoch: the
      // student's memory is time-ordered, and lessons learned earlier
      // sink deeper — but their EMA keeps them counting.
      return this.hand(new DemoFlight(this.school, lesson, this.policy,
                                      this.demoEpoch++),
                       lesson, "demo", attempt);
    }
    return this.hand(new ExamFlight(this.school, lesson, this.policy, this.rng),
                     lesson, "repeat", this.attempts.get(name)!);
  }

  private hand(flight: DemoFlight | ExamFlight, lesson: Lesson,
               stage: SessionStage, attempt: number): SchoolFlight {
    this.pending = { flight, lesson, stage, attempt };
    return this.pending;
  }

  private finish(): void {
    // The graduate the flock copies is the classroom's result: the
    // demonstration reflexes plus whatever the loop re-taught.
    const graduate = new ServoPolicy();
    graduate.restore(this.policy.snapshot());
    this.graduate = graduate;
    this.school.graduate = graduate;
    this.stage = "done";
  }
}
