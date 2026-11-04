/**
 * THE ATTACK MISSION — one drone's share of the swarm's sortie
 * (learning/mission.ts).
 *
 * Drones do not appear from nowhere: every one of them LIFTS OFF FROM
 * THE RUNWAY, two abreast in a pair — the lane split into two halves,
 * each seat at the centre of its own half — the pairs five seconds
 * apart at the same threshold, pair after pair, forever, no waves.
 * From there the mission is one chain of the flight school's own
 * proven elements, MIRRORED BY FLANK (the pair's left seat is the
 * left flank's column, the right seat the right flank's):
 *
 *   roll -> rotate -> the straight climb out of ground effect
 *   [takeoff]
 *   the split ON THE CLIMB: a turning departure onto the flank
 *   heading — the left column HSI-45°, the right column HSI+45°
 *                                                            [split]
 *   the climb-out IN THE FLANK, straight to 3000 ft — the level
 *   moment                                            [flank]
 *   THE ATTACK: the left flank's steep descending turn RIGHT
 *   (HSI+90°), the right flank's steep descending turn LEFT
 *   (HSI-90°), both down to 500 ft — the tank in the area below
 *                                                            [attack]
 *   the rally: the left flank's climbing turn LEFT (HSI-45°),
 *   the right flank's climbing turn RIGHT (HSI+45°), both up to
 *   1500 ft                                                 [climb]
 *   pitch up, spin OWN side (left flank left, right flank right),
 *   and RECOVER                                              [spin]
 *   fly level — and from here THE WILL DECIDES: the drone's own
 *   brain takes the controls in free flight                  [will]
 *
 * The chain flies through the PILOT LAYER exactly like the classroom:
 * the mission is the machine's TRUE element cursor (the same
 * PhaseCursor, the same physics-driven advance, the same runway pin
 * via stepMachine), and the machine's own controller seats whoever
 * holds the levers — the instructor flying the true element, the AI
 * flying its memory, or the pair of them (the AI suggesting and
 * learning while the instructor flies). No mode logic lives here.
 *
 * THE WILL PHASE: when the chain completes, the drone does NOT loop —
 * its will takes over. The pilot becomes the AI in FREE FLIGHT (the
 * same shape as the swarm's LiveEpisode: a null cursor, free flight's
 * own goal, the stabilizer's teaching beside it), and from there the
 * drone's life is its own — its tensor brain, trained on the whole
 * sortie's error waves, deciding every next step until it crashes or
 * leaves the area. The repetition of the scenario is the CONVEYOR:
 * pair after pair keeps lifting off from the runway.
 *
 * The SINGULARITY TENSOR's super-goal (learning/singularity.ts) can
 * pull a drone that has fallen behind FORWARD to the formation's
 * current leg — never backward, only at a leg's safe re-entry point
 * (which is what the MISSION TOKENS below name), and never on a
 * will-phase drone (its will has it, beyond the formation's claim).
 */
import { Vec3 } from "../core/vec.js";
import { World } from "../core/world.js";
import { NumpyPcg64 } from "../core/rng.js";
import { Aircraft, HandDecisions } from "../physics/aircraft.js";
import { RigidBody } from "../physics/rigid-body.js";
import {
  FlightInstruments, DenseReward, EpisodeOutcome, classifyOutcome,
} from "./instruments.js";
import {
  ACTION_HOLD, GOAL_FREE_FLIGHT, type TensorStateView,
} from "./servo-tensor.js";
import { ControlMode } from "./guardian.js";
import {
  AiPilot, InstructorPilot, type PilotSighting,
} from "./pilot.js";
import { AircraftController, controllerFor } from "./aircraft-controller.js";
import { springStickTowardNeutral, TRIM_COMMAND } from "./flight.js";
import {
  SYLLABUS, LEVEL_OFF_LEAD_S, PhaseCursor, stepMachine, startCommandsFor,
  DECISION_EVERY, FlightSchool, FlightElement, wrapPi,
  type MachineHost, type ElementLaw, type PhaseReadings, type ElementName,
  type RudderLaw,
  RunwayRollElement, RotateElement, ClimbElement, LevelOffElement,
  SettleElement, RollInElement,
  StallEntryElement, StallWaitElement, DropWingElement, SpinElement,
  OppositeRudderElement, DiveElement,
} from "./school.js";

/**
 * THE MISSION TOKENS — the named legs of the sortie. The tokens are
 * FLANK-AGNOSTIC: the left and right columns fly the same legs with
 * mirrored directions, so the whole formation is always on the same
 * token when it flies in lockstep — which is what makes a token the
 * singularity's super-goal unit. The WILL token is the mission's
 * retirement: a drone flying it is beyond the formation's claim.
 */
export type MissionToken =
  | "takeoff"
  | "split"
  | "flank"
  | "attack"
  | "climb"
  | "spin"
  | "will";

/** The canonical leg order (plurality ties resolve to the earlier). */
export const MISSION_TOKENS: readonly MissionToken[] = [
  "takeoff", "split", "flank", "attack", "climb", "spin", "will",
];

/**
 * The mission's higher-level goal — the intention tensor's goal axis
 * bin. Follow-instructor (0) and free flight (1) are reserved; the
 * classroom's curriculum ends at 21; the attack mission owns 22.
 */
export const GOAL_DRONE_ATTACK = 22;

/** The mission's operating area: the swarm may fly this far from the
 *  origin before it has left the area (out of bounds ends a drone).
 *  The pattern's own geometry sets this: the flank climb-out is ~7 km
 *  of diagonal travel, and the attack glide from 3000 ft down to
 *  500 ft spans kilometres more — the area must hold the whole sortie
 *  in any direction. */
export const MISSION_AREA_M = 9000.0;

const FT = 1.0 / 3.28084;
/** The mission altitude — the level moment in the flank. */
const MISSION_ALT_M = 3000.0 * FT;
/** The attack run's floor: the steep descending turn ends here. */
const ATTACK_ALT_M = 500.0 * FT;
/** The rally altitude: the climbing turn and the spin happen here. */
const RALLY_ALT_M = 1500.0 * FT;
/** The split: the columns peel apart onto HSI ±45°. */
const SPLIT_TURN = Math.PI / 4.0;
/** The attack: the flanks converge onto HSI ∓90°. */
const ATTACK_TURN = Math.PI / 2.0;
/** The rally: from HSI ∓90° back onto the flank's own HSI ±45° —
 * a 135° climbing turn (the roll-out's capture eats the attack arc's
 * overshoot, so the amount is nominal). */
const RALLY_TURN = (3.0 * Math.PI) / 4.0;
/** The level moment in the flank, ticks (8 s at 50 Hz). */
const FLANK_HOLD_TICKS = 400;
/** The level flight before the will takes the controls (8 s). */
const WILL_HOLD_TICKS = 400;
/** The sink through the attack turn AND the glide down to the attack
 *  floor, m/s — the school's own steep-descending regime (the 60°-bank
 *  idle turn holds exactly this). */
const ATTACK_VS_MPS = -6.0;

/**
 * THE SPLIT CLIMB — the straight climb after the rotate, before the
 * columns peel apart: the instructor's own departure (wings level,
 * 15° pitch, full power) up to the split altitude. A turning
 * departure off the runway is flown from here up, wings loaded and
 * speed built — never in ground effect.
 */
const SPLIT_CLIMB_ALT_M = 800.0 * FT;

/**
 * A held-bank turn through an ARBITRARY heading change — the mission's
 * legs are 45° and 90°, not the school's 180°. The school's roll-in
 * carries the heading accumulator in; this holds the bank; the
 * heading-hold roll-out (RollOutByElement) lands on exactly the amount.
 */
class TurnByElement extends FlightElement {
  readonly name: ElementName = "turn";
  readonly stamp = true;

  constructor(bank: number, private readonly amount: number,
              speedTarget: number, throttle: number, rudder: RudderLaw,
              private readonly leadOut: number) {
    super({ bankTarget: bank, carryTurn: true, vsTarget: 0.0,
            speedTarget, rudder, throttle });
  }

  done(r: PhaseReadings): boolean {
    return Math.abs(r.turned) >= this.amount - this.leadOut;
  }
}

/** The same arbitrary-amount turn, held together with a steady
 *  descent — the attack's steep descending turn. */
class DescendingTurnByElement extends FlightElement {
  readonly name: ElementName = "descending turn";
  readonly stamp = true;

  constructor(bank: number, private readonly amount: number,
              vsTarget: number, speedTarget: number, throttle: number,
              rudder: RudderLaw, private readonly leadOut: number) {
    super({ bankTarget: bank, carryTurn: true, vsTarget,
            speedTarget, rudder, throttle });
  }

  done(r: PhaseReadings): boolean {
    return Math.abs(r.turned) >= this.amount - this.leadOut;
  }
}

/** The same arbitrary-amount turn, held together with a climb — the
 *  split's turning departure and the rally's climbing turn home. */
class ClimbingTurnByElement extends FlightElement {
  readonly name: ElementName = "climbing turn";
  readonly stamp = true;

  constructor(bank: number, private readonly amount: number,
              vsTarget: number, speedTarget: number, throttle: number,
              rudder: RudderLaw, private readonly leadOut: number,
              pitchTarget?: number) {
    super({ bankTarget: bank, carryTurn: true, vsTarget,
            speedTarget, rudder, throttle, pitchTarget });
  }

  done(r: PhaseReadings): boolean {
    return Math.abs(r.turned) >= this.amount - this.leadOut;
  }
}

/**
 * The heading-hold roll-out onto an ARBITRARY amount — the base class's
 * law lands on the law's own turnAmount (the school's roll-outs keep
 * the standard 180°), so the split, the attack, and the rally each
 * finish on exactly their heading.
 */
class RollOutByElement extends FlightElement {
  readonly name: ElementName = "roll out";
  readonly stamp = true;
  /** The landing amount's MAGNITUDE — the direction is the cursor's
   *  turnDirection (set from the turn the drone actually flew). */
  private readonly amount: number;

  constructor(amount: number, speedTarget: number,
              throttle: number, vsTarget: number = 0.0,
              pitchTarget?: number) {
    super({ headingHold: true, carryTurn: true,
            turnAmount: Math.abs(amount),
            vsTarget, speedTarget, rudder: 0.0, throttle, pitchTarget });
    this.amount = Math.abs(amount);
  }

  done(r: PhaseReadings): boolean {
    return Math.abs(wrapPi(this.amount - Math.abs(r.turned))) <= 0.014
           && Math.abs(r.bank) <= 0.022;
  }
}

/**
 * Level flight held for a fixed stretch — the moment in the flank at
 * 3000 ft, and the level flight before the will takes the controls.
 */
class LevelHoldElement extends FlightElement {
  readonly name: ElementName = "cruise";
  readonly stamp = true;

  constructor(private readonly ticks: number) {
    super({ vsTarget: 0.0, bankTarget: 0.0,
            speedTarget: SYLLABUS.cruiseSpeedMps,
            rudder: 0.0, throttle: SYLLABUS.cruiseThrottle });
  }

  done(r: PhaseReadings): boolean {
    return r.ticks >= this.ticks && Math.abs(r.vsMps) < 0.5;
  }
}

/**
 * The attack's roll-out: WINGS LEVEL, no heading capture — the steep
 * descending turn's own arc finishes the attack (the 90° lands with
 * the natural roll-out overshoot, which the next leg takes as its
 * HSI reference). A precise heading capture here hunts through the
 * low-energy mush after the 60°-bank idle turn and can trap the
 * sortie forever; the wings just come level and the run continues.
 */
class LevelWingsElement extends FlightElement {
  readonly name: ElementName = "roll out";
  readonly stamp = true;

  constructor(vsTarget: number, speedTarget: number, throttle: number) {
    super({ bankTarget: 0.0, vsTarget, speedTarget, rudder: 0.0,
            throttle });
  }

  done(r: PhaseReadings): boolean {
    return r.ticks >= 100 && Math.abs(r.bank) <= 0.05;
  }
}

/**
 * The attack glide: a held-sink descent (the vertical speed on the
 * elevator, the speed on the power) down to the attack floor — the run
 * on the tank below, finished at exactly 500 ft.
 */
class DescendToElement extends FlightElement {
  readonly name: ElementName = "descend";
  readonly stamp = true;

  constructor(private readonly toAltitudeM: number, vsTarget: number,
              speedTarget: number, throttle: number) {
    super({ vsTarget, bankTarget: 0.0, speedTarget,
            rudder: 0.0, throttle });
  }

  done(r: PhaseReadings): boolean {
    return r.altitudeM <= this.toAltitudeM + LEVEL_OFF_LEAD_S * (-r.vsMps)
           && r.vsMps <= -0.2;
  }
}

/**
 * THE FLANK HEADING — the drone's HSI ±45° home. Marked continuously
 * while the drone flies its flank's level leg (MissionFlight's own
 * step, so the mark is kept however the levers are held) and captured
 * by the rally's roll-out: the climbing turn ends back on the flank
 * heading, however far the attack's natural arc overshot past
 * HSI ∓90°.
 */
export class HeadingMark {
  value = 0.0;
}

/**
 * The rally's roll-out: a heading CAPTURE onto the flank heading —
 * the remaining turn closes from wherever the attack's arc left the
 * column, wings level on the flank's own HSI ±45°.
 */
class RollOutToHeadingElement extends FlightElement {
  readonly name: ElementName = "roll out";
  readonly stamp = true;

  constructor(mark: HeadingMark, vsTarget: number, speedTarget: number,
              throttle: number, pitchTarget?: number) {
    super({ headingTo: () => mark.value, vsTarget, speedTarget,
            rudder: 0.0, throttle, pitchTarget });
  }

  done(r: PhaseReadings): boolean {
    const target = this.law.headingTo?.() ?? r.heading;
    return r.ticks >= 100
           && Math.abs(wrapPi(target - r.heading)) <= 0.02
           && Math.abs(r.bank) <= 0.05;
  }
}

/**
 * The legs of the mission, in order, with the phase index each leg
 * ENTERS at. An entry index is a SAFE RE-ENTRY: every leg starts from
 * wings-level flight, so the singularity may pull a fallen-behind
 * drone forward onto the formation's current leg by re-seating its
 * cursor there — never backward, never mid-turn.
 */
export const MISSION_LEGS: readonly {
  readonly token: MissionToken;
  readonly entryIndex: number;
}[] = [
  { token: "takeoff", entryIndex: 0 },
  { token: "split", entryIndex: 3 },
  { token: "flank", entryIndex: 6 },
  { token: "attack", entryIndex: 10 },
  { token: "climb", entryIndex: 14 },
  { token: "spin", entryIndex: 19 },
  { token: "will", entryIndex: 27 },
];

/** The phase index each token's leg enters at (MISSION_LEGS as a map). */
export const MISSION_ENTRIES: Readonly<Record<MissionToken, number>> = {
  takeoff: 0,
  split: 3,
  flank: 6,
  attack: 10,
  climb: 14,
  spin: 19,
  will: 27,
};

/**
 * The mission chain of one drone, mirrored by its flank side
 * (-1 left column, +1 right column), with the token of every leg —
 * index-aligned with the phase array. The headings are HSI-relative
 * and symmetric: the left column flies -45 / +90 / -45 and the right
 * column +45 / -90 / +45 — both flanks converge on the tank at 500 ft
 * and peel back out onto their own flank heading.
 */
export function missionPlan(flankSign: number,
                            flankMark: HeadingMark): {
  phases: readonly FlightElement[];
  tokens: readonly MissionToken[];
} {
  const S = SYLLABUS;
  const phases: FlightElement[] = [
    // [takeoff] the runway: the instructor's own departure — roll,
    // rotate, the straight climb out of ground effect.
    new RunwayRollElement(),                          // 0
    new RotateElement(),                              // 1
    new ClimbElement(SPLIT_CLIMB_ALT_M),              // 2
    // [split] the columns peel apart ON THE CLIMB: a turning
    // departure onto the flank heading (left column HSI-45, right
    // column HSI+45), the climb never stopping.
    new RollInElement(flankSign * S.bankTurn, S.climbSpeedMps,
                      1.0, 0.0, 0.0, S.pitchClimb),   // 3
    new ClimbingTurnByElement(flankSign * S.bankTurn,
                              flankSign * SPLIT_TURN, 4.0,
                              S.climbSpeedMps, 1.0, 0.0, S.turnLeadOut,
                              S.pitchClimb),          // 4
    new RollOutByElement(flankSign * SPLIT_TURN,
                         S.climbSpeedMps, 1.0, 0.0, S.pitchClimb),   // 5
    // [flank] the climb-out IN THE FLANK: straight down the flank
    // heading to 3000 ft, level off, settle — the moment in the
    // flank.
    new ClimbElement(MISSION_ALT_M),                  // 6
    new LevelOffElement(MISSION_ALT_M, "3000 ft"),    // 7
    new SettleElement(),                              // 8
    new LevelHoldElement(FLANK_HOLD_TICKS),           // 9
    // [attack] into the other flank's side: the left column turns
    // RIGHT (HSI+90), the right column turns LEFT (HSI-90) — the
    // steep descending run on the tank in the area below, to 500 ft.
    new RollInElement(-flankSign * S.bankSteep, S.cruiseSpeedMps,
                      S.idleThrottle, "top", ATTACK_VS_MPS),   // 10
    new DescendingTurnByElement(-flankSign * S.bankSteep,
                                -flankSign * ATTACK_TURN, ATTACK_VS_MPS,
                                S.cruiseSpeedMps, S.idleThrottle, "top",
                                S.steepLeadOut),      // 11
    new LevelWingsElement(ATTACK_VS_MPS, S.cruiseSpeedMps,
                          S.descendThrottle),         // 12
    new DescendToElement(ATTACK_ALT_M, ATTACK_VS_MPS,
                         S.cruiseSpeedMps, S.descendThrottle),  // 13
    // [climb] the rally: the left column's climbing turn LEFT
    // (HSI-45), the right column's climbing turn RIGHT (HSI+45),
    // both up to 1500 ft.
    new RollInElement(flankSign * S.bankTurn, S.climbSpeedMps, 1.0,
                      0.0, 0.0, S.pitchClimb),        // 14
    new ClimbingTurnByElement(flankSign * S.bankTurn,
                              flankSign * RALLY_TURN, 4.0,
                              S.climbSpeedMps, 1.0, 0.0, S.turnLeadOut,
                              S.pitchClimb),          // 15
    new RollOutToHeadingElement(flankMark, 0.0, S.climbSpeedMps, 1.0,
                                S.pitchClimb),        // 16
    new ClimbElement(RALLY_ALT_M),                    // 17
    new LevelOffElement(RALLY_ALT_M, "1500 ft"),      // 18
    // [spin] pitch up, spin OWN side (left column left, right column
    // right), and recover.
    new StallEntryElement(),                          // 19
    new StallWaitElement(),                           // 20
    new DropWingElement(flankSign),                   // 21
    new SpinElement(flankSign),                       // 22
    new OppositeRudderElement(),                      // 23
    new DiveElement(),                                // 24
    new ClimbElement(RALLY_ALT_M),                    // 25
    new LevelOffElement(RALLY_ALT_M, "1500 ft"),     // 26
    // [will] fly level — and from here the drone's will decides.
    new LevelHoldElement(WILL_HOLD_TICKS),            // 27
  ];
  const tokens: MissionToken[] = new Array<MissionToken>(phases.length);
  for (let leg = 0; leg < MISSION_LEGS.length; leg += 1) {
    const from = MISSION_LEGS[leg].entryIndex;
    const to = leg + 1 < MISSION_LEGS.length
      ? MISSION_LEGS[leg + 1].entryIndex : phases.length;
    for (let i = from; i < to; i += 1) {
      tokens[i] = MISSION_LEGS[leg].token;
    }
  }
  return { phases, tokens };
}

export interface MissionOptions {
  /** The drone's flank: -1 the left column, +1 the right column —
   *  it mirrors the split, the attack turn, the rally, and the spin. */
  readonly flankSign: number;
  /** The drone's runway seat: its pair's position on the runway. */
  readonly startPosition: Vec3;
  /** Exploration override (the swarm runs at 5 %). */
  readonly epsilonOverride?: number | null;
  /** Whose hands fly the drone from the runway (default: the PAIR —
   *  the instructor flies, the AI suggests and learns). */
  readonly mode?: ControlMode;
  /** How far from the origin the mission's area reaches. */
  readonly worldHalfExtent?: number;
}

/**
 * ONE DRONE'S SORTIE — the machine flying the attack mission. The
 * same sealed Aircraft the classroom flies, advanced by the same
 * stepMachine (physics, gear, the runway pin, the cursor); the same
 * pilot layer at the levers, seated through the machine's own
 * controller (the mode IS the controller); the drone's own tensor
 * brain suggesting and learning (the instructor's own element
 * observation is the error wave's source — no twin in the swarm, the
 * instructor flies THIS machine). The mission ends in the WILL
 * PHASE: the chain complete, the drone's own brain takes the
 * controls in free flight, and its life ends only physically — a
 * crash, or leaving the area.
 */
export class MissionFlight implements MachineHost {
  readonly aircraft: Aircraft;
  readonly world: World;
  readonly dt = 1.0 / 50.0;
  readonly flankSign: number;

  private readonly phases: readonly FlightElement[];
  private readonly tokens: readonly MissionToken[];
  private cursor: PhaseCursor;

  private readonly instruments: FlightInstruments;
  private readonly epoch: number;
  private readonly dense: DenseReward;
  private pendingReward = 0.0;
  private readonly decisionEvery = DECISION_EVERY;
  private readonly steps = Infinity;
  private commands = Float64Array.from(startCommandsFor("runway"));
  private readonly candidates: readonly number[];

  /** THE PILOT LAYER — same as the classroom: the AI suggests and
   *  learns; the instructor flies the TRUE element; the controller
   *  IS the mode (all mode logic lives in the AircraftController
   *  classes, never here). */
  private readonly ai: AiPilot;
  private readonly instructor: InstructorPilot;
  /** The WILL PHASE's free-flight instructor: the stabilizer whose
   *  observation keeps teaching the AI through its own flight (the
   *  swarm's LiveEpisode law). */
  private readonly freeInstructor = new InstructorPilot(TRIM_COMMAND, true);
  private controller: AircraftController;
  private mode: ControlMode;
  /** The flank heading — marked every tick of the flank's level leg
   *  (the rally's roll-out captures back onto it). */
  private readonly flankMark = new HeadingMark();

  /** THE WILL PHASE — the chain is complete and the drone's own brain
   *  has the controls (free flight; the formation's claim is over). */
  private willPhase = false;

  private outcome: string | null = null;
  tick = 0;

  /**
   * THE DRONE'S PILOT — the swarm builds the brain, the mission flies
   * it: the pilot-layer's whole configuration (the mission's decision
   * cadence, the student's stick spring) named in one place. The
   * constructor draws the network's seeds from the seed stream, so
   * the swarm hands the drone its own dice and this factory.
   */
  static pilot(seedRng: NumpyPcg64): AiPilot {
    return new AiPilot({ seedRng, decisionEvery: DECISION_EVERY,
                         spring: springStickTowardNeutral });
  }

  constructor(ai: AiPilot, instruments: FlightInstruments,
              epoch: number, options: MissionOptions) {
    this.instruments = instruments;
    this.epoch = epoch;
    this.flankSign = options.flankSign;
    this.world = new World({ halfExtent: options.worldHalfExtent
      ?? MISSION_AREA_M });
    // THE RUNWAY SEAT: parked on the gear at 100 ft, wings level, still
    // — the takeoff roll is the mission's first element.
    this.aircraft = new Aircraft(new RigidBody(options.startPosition,
      new Vec3(0.0, 0.0, 0.0)));
    const plan = missionPlan(options.flankSign, this.flankMark);
    this.phases = plan.phases;
    this.tokens = plan.tokens;
    this.cursor = new PhaseCursor(this.phases, this.aircraft.body);
    this.candidates = this.phases
      .filter((phase) => phase.stamp)
      .map((phase) => phase.elementId);
    this.ai = ai;
    // A fresh sortie over the same memory: the runtime resets (the
    // recurrence starts clean at the takeoff roll), the memory persists.
    ai.beginFlight(this.phases[0].elementId);
    this.instructor = new InstructorPilot(startCommandsFor("runway"), false);
    this.mode = options.mode ?? "both-suggest";
    this.controller = controllerFor(this.aircraft, this.mode, this.ai,
                                    this.instructor);
    this.dense = new DenseReward(instruments, 5.0);
    this.dense.reset(this.aircraft.body);
    if (options.epsilonOverride !== undefined
        && options.epsilonOverride !== null) {
      ai.setExploration(options.epsilonOverride);
    }
  }

  /** Whose hands fly this drone (the controls dropdown's sticky
   *  choice, per drone). */
  get controlsMode(): ControlMode {
    return this.mode;
  }

  /** Swap the whole controller — the mode IS the controller. A
   *  mid-mission swap re-seats the instructor's hand from the drone's
   *  CURRENT lever targets; the cursor and the brain run on untouched.
   *  In the will phase the swap is the swarm's free-flight law: the
   *  free-flight instructor (the stabilizer, firm hand, captured
   *  altitude) flies or watches beside the AI. */
  setControlsMode(mode: ControlMode): void {
    if (mode === this.mode) {
      return;
    }
    this.mode = mode;
    const instructor = this.willPhase
      ? this.freeInstructor : this.instructor;
    this.controller = controllerFor(this.aircraft, mode, this.ai,
      instructor,
      { commands: this.commands, body: this.aircraft.body });
  }

  /** Whether the instructor's hand is on the levers right now (the
   *  status line and the panel care). */
  get instructorAtControls(): boolean {
    return this.controller.heldByInstructor;
  }

  /** The hand's decisions for the panel — the classroom's contract. */
  get handInputs(): HandDecisions {
    return this.controller.heldByInstructor
      ? { choices: null, fresh: false }
      : {
        choices: this.ai.choices,
        fresh: this.tick % this.decisionEvery < this.decisionEvery / 2,
      };
  }

  /** THE AI STATE LOG's subject: the drone's tensor network snapshot. */
  stateView(): TensorStateView {
    return this.ai.stateView;
  }

  /** How the sortie ended (null while it flies). */
  get episodeOutcome(): string | null {
    return this.outcome;
  }

  /** Ground contact (the machine-tick's ground judgment, the same law
   *  the classroom's FlightSchool applies). */
  grounded(body: RigidBody): boolean {
    return body.position.z
      <= this.world.terrainHeight(body.position.x, body.position.y) + 1.0;
  }

  /** THE MISSION TOKEN the drone is flying right now — the leg of the
   *  sortie its cursor is in (the singularity's super-goal unit); the
   *  will phase reports its own token, beyond the formation's claim. */
  get missionToken(): MissionToken {
    return this.willPhase ? "will"
      : this.tokens[Math.min(this.cursor.index, this.tokens.length - 1)];
  }

  /** The mission's leg boundary the cursor sits at, or null mid-leg —
   *  only a boundary is a safe place for the super-goal to move it. */
  private get atLegEntry(): number | null {
    const index = this.cursor.index;
    return MISSION_LEGS.some((leg) => leg.entryIndex === index)
      ? index : null;
  }

  /**
   * THE SINGULARITY'S PULL — the super-goal's claim on this drone. A
   * drone that has fallen behind the formation (its own hands lost it
   * — the AI's, or a guardian's recovery) is moved FORWARD onto the
   * formation's current leg, but only and exactly when it is sitting
   * at a leg boundary (every leg starts from wings-level flight), and
   * never backward, and never back onto the runway: a young drone
   * flies its whole takeoff first. A will-phase drone is beyond the
   * claim — its will has it.
   */
  conform(superGoal: MissionToken): void {
    if (this.outcome !== null || this.willPhase) {
      return;
    }
    const mine = this.missionToken;
    if (mine === "takeoff" || superGoal === "takeoff") {
      return;                 // the runway is flown to the end, always
    }
    const entry = this.atLegEntry;
    if (entry === null) {
      return;                 // mid-leg: the leg is flown to the end
    }
    const target = MISSION_ENTRIES[superGoal];
    if (target <= entry) {
      return;                 // forward only — ahead drones fly on
    }
    this.cursor = new PhaseCursor(this.phases, this.aircraft.body, target);
  }

  /**
   * THE WILL — the mission's last handover: the chain is complete, the
   * drone has flown its level moment, and its OWN brain takes the
   * controls (free flight: the AI's own hands, the stabilizer teaching
   * beside it). From here the drone's life is its own.
   */
  private enterWill(): void {
    this.willPhase = true;
    this.mode = "ai";         // the will has the controls
    this.freeInstructor.reseat(this.commands, this.aircraft.body);
    this.controller = controllerFor(this.aircraft, this.mode, this.ai,
                                    this.freeInstructor);
  }

  /** The chain-phase sighting: the machine's TRUE element cursor, the
   *  mission's goal, and the instructor's own live observation as the
   *  error wave's teaching source. */
  private missionSighting(): PilotSighting {
    return {
      dt: this.dt,
      tick: this.tick,
      body: this.aircraft.body,
      groundM: this.world.terrainHeight(this.aircraft.body.position.x,
                                        this.aircraft.body.position.y),
      cursor: this.cursor,
      sense: this.instruments.senseReading(this.aircraft.body,
        this.aircraft.surfacePositions),
      teaching: this.teaching(),
      epoch: this.epoch,
      goalId: GOAL_DRONE_ATTACK,
      candidates: this.candidates,
    };
  }

  /** The will-phase sighting: FREE FLIGHT, the swarm's LiveEpisode
   *  shape — a null cursor, free flight's own goal, and the
   *  stabilizer's observation teaching beside the drone's own hands. */
  private willSighting(): PilotSighting {
    const groundM = this.world.terrainHeight(this.aircraft.body.position.x,
                                             this.aircraft.body.position.y);
    return {
      dt: this.dt,
      tick: this.tick,
      body: this.aircraft.body,
      groundM,
      cursor: null,
      sense: this.instruments.senseReading(this.aircraft.body,
        this.aircraft.surfacePositions),
      teaching: this.freeInstructor.stabilizerTeaching(this.aircraft.body,
                                                        groundM),
      epoch: this.epoch,
      goalId: GOAL_FREE_FLIGHT,
      candidates: null,
    };
  }

  /** One 50 Hz tick of the sortie: the pilot flies, the machine
   *  consumes the inputs through its own input method, and the
   *  drone's life ends only physically. */
  step(): boolean {
    if (this.outcome !== null) {
      return false;
    }
    if (this.tick % this.decisionEvery === 0) {
      this.ai.credit(this.pendingReward, this.epoch);
      this.pendingReward = 0.0;
    }
    const sighting = this.willPhase
      ? this.willSighting() : this.missionSighting();
    this.commands = this.controller.step(sighting, this.commands);
    if (this.willPhase) {
      // FREE FLIGHT — the will's own hands: no chain to advance, the
      // machine consumes the inputs and the outcome is the drone's own.
      this.aircraft.command(this.commands, this.dt);
      this.aircraft.step(this.dt, this.aircraft.body.position.z
        - this.world.terrainHeight(this.aircraft.body.position.x,
                                    this.aircraft.body.position.y));
      this.tick += 1;
      this.commands = this.controller.relax(this.commands, this.dt);
      this.pendingReward += this.dense.tick(this.aircraft.body) + 0.0005;
      const outcome = classifyOutcome(this.aircraft.body, this.world,
                                      this.tick * this.dt);
      if (outcome !== null) {
        this.finish(outcome);
        return false;
      }
      return true;
    }
    // THE MISSION CHAIN: command, physics, the gear, the runway pin,
    // the cursor's physics-driven advance — the classroom's own step.
    const flying = stepMachine(this, this.aircraft, this.cursor,
                               this.commands);
    // THE FLANK HEADING — kept fresh every tick of the flank's level
    // leg: the rally's roll-out captures back onto it.
    if (this.missionToken === "flank") {
      const [forward] = this.aircraft.body.bodyAxes;
      this.flankMark.value = Math.atan2(forward.y, forward.x);
    }
    this.tick += 1;
    this.commands = this.controller.relax(this.commands, this.dt);
    this.pendingReward += this.dense.tick(this.aircraft.body) + 0.0005;
    if (!flying) {
      if (this.cursor.finished) {
        this.enterWill();     // the chain is complete: the will decides
      } else {
        this.finish(EpisodeOutcome.CRASHED);
        return false;
      }
    }
    if (!this.world.isInsideBounds(this.aircraft.body.position)) {
      this.finish(EpisodeOutcome.OUT_OF_BOUNDS);
      return false;
    }
    if (this.tick >= this.steps) {
      this.finish(EpisodeOutcome.ALIVE);
      return false;
    }
    return true;
  }

  /**
   * The instructor's own observation of the machine's TRUE element —
   * the error wave's teaching source in the swarm: the element the
   * cursor is in and the lever stops its own law is driving toward.
   * (The classroom's twin observes an ideal path; the swarm has no
   * ideal path — the instructor flies THIS machine, and its live
   * element law IS the lesson.)
   */
  private teaching(): { elementId: number; actions: Int32Array } {
    const raw = this.cursor.phaseTargets(this.aircraft.body);
    let actions = FlightSchool.movingLevels(raw,
      this.aircraft.surfacePositions);
    if (!actions.some((a) => a !== ACTION_HOLD)) {
      actions = FlightSchool.heldLevels(this.aircraft.surfacePositions);
    }
    return { elementId: this.cursor.current.elementId, actions };
  }

  /** The sortie's epilogue: the final reinforcement, once. */
  private finish(outcome: string): void {
    this.outcome = outcome;
    this.ai.settle(outcome, this.epoch);
  }
}
