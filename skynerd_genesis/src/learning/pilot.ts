/**
 * THE PILOTS — who is at the controls of one machine.
 *
 * The machine (physics/aircraft.ts) is SEALED: a pilot never touches
 * its components. The only path to its levers is the machine's
 * control-input method (Aircraft.command, fed once per tick by the
 * flight that owns the machine); a pilot only READS the machine —
 * through the sighting the flight hands it — and produces the next
 * tick's control inputs. All mode logic lives HERE, outside the
 * aircraft:
 *
 *   InstructorPilot          — the demonstration hand: the phase
 *                              law's targets of the machine's TRUE
 *                              element, eased (the classroom), or the
 *                              stabilizer (the swarm's free flight);
 *   AIPilot                  — the tensor brain: sense, decide,
 *                              apply, relax — the AI's own hands;
 *   AIGuidedInstructorPilot  — (both a) the AI suggests and learns
 *                              continuously, the instructor flies;
 *   AIPilotWithInstructor    — (both b) the AI flies, the guardian
 *                              watches; danger takes the levers (a
 *                              guided flight while recovering), a
 *                              sustained safe stretch hands them back.
 */
import { RigidBody } from "../physics/rigid-body.js";
import {
  stabilizeTargets, isDangerous, RECOVERY_HOLD_S,
} from "./guardian.js";
import { levelOf } from "./instruments.js";
import { ELEMENT_FREE_FLIGHT } from "./servo-tensor.js";
import type {
  NetworkDecision, ServoPolicy, TensorNetwork, TensorStateView,
} from "./servo-tensor.js";
import type { NumpyPcg64 } from "../core/rng.js";
import type { PhaseCursor } from "./school.js";
import type { SenseReading } from "./instruments.js";

/**
 * The instructor's hand: eases commanded deflections toward the raw
 * phase targets at bounded rates. All inputs on the flight controls
 * are smooth — the yoke never jumps, even when a phase changes the
 * law underneath.
 *
 * Per-channel easing rates: the instructor demonstrates at the
 * STUDENT'S hand speed — one ACTION_DELTA per decision tick window
 * (0.05 rad per 0.2 s = 0.25/s). A faster hand carves a pull the
 * student's ±0.05/decision vocabulary cannot track: the demo's
 * stamps land on a path through (reading, yoke) space the student
 * physically cannot follow, and every replay diverges from the
 * lesson it was shown. The instructor shows the maneuver the way a
 * student's hands can fly it.
 */
export class InstructorHand {
  readonly commands: Float64Array;
  /** The DEMO hand moves at the student's own speed; the RECOVERY
   *  hand (the swarm's stabilizer) moves four times firmer — taking
   *  over a dive at demo speed is arriving late with good manners. */
  private static readonly DEMO_RATES = [0.25, 0.25, 0.25, 0.25];
  private static readonly FIRM_RATES = [1.0, 1.0, 1.0, 1.0];
  private readonly rates: readonly number[];

  constructor(commands: ArrayLike<number>, rates?: readonly number[]) {
    this.commands = Float64Array.from(commands);
    this.rates = rates ?? InstructorHand.DEMO_RATES;
  }

  /** The swarm's recovery hand. */
  static firm(): readonly number[] {
    return InstructorHand.FIRM_RATES;
  }

  /** One tick of easing toward the raw targets. */
  easeToward(targets: ArrayLike<number>, dt: number): Float64Array {
    for (let channel = 0; channel < 4; channel++) {
      const rate = this.rates[channel] * dt;
      const delta = Math.max(-rate,
        Math.min(rate, targets[channel] - this.commands[channel]));
      this.commands[channel] += delta;
    }
    return this.commands;
  }
}

/** One pilot's read of the machine it flies, this tick. Everything a
 *  pilot may know: the machine's sealed state, its TRUE element
 *  cursor, the AI's sensed panels, the teaching observation beside
 *  it, and the classroom clocks. */
export interface PilotSighting {
  readonly dt: number;
  readonly tick: number;
  readonly body: RigidBody;
  /** The terrain's height under the machine, m (the stabilizer's
   *  clearance floor — the ground is the pilot's business). */
  readonly groundM: number;
  /** The machine's TRUE element cursor — the classroom's phase chain.
   *  Null in free flight: the swarm has no lesson. */
  readonly cursor: PhaseCursor | null;
  /** The AI's sensed panels this tick (the tensor network's input). */
  readonly sense: SenseReading;
  /** The instructor's observation beside the machine — the error
   *  wave's teaching source; null when nobody teaches. */
  readonly teaching: { elementId: number; actions: Int32Array } | null;
  /** The classroom epoch (the teach pulses' clock). */
  readonly epoch: number;
  /** The goal being realized (the lesson's, or free flight's own). */
  readonly goalId: number;
  /** The maneuver's taught elements (the transition candidates). */
  readonly candidates: readonly number[] | null;
}

/** A pilot: the producer of one machine's control inputs. */
export interface Pilot {
  /** The control inputs for this tick — the ONLY thing that reaches
   *  the machine's levers (the flight feeds them to the machine's
   *  own input method). */
  step(sighting: PilotSighting, commands: Float64Array): Float64Array;
  /** The hands-off reflex after the physics tick: the instructor's
   *  hand holds its positions; the AI's stick springs. */
  relax(commands: Float64Array, dt: number): Float64Array;
  /** The instructor's hand is physically on the levers right now. */
  readonly heldByInstructor: boolean;
}

/**
 * THE INSTRUCTOR — the demonstration hand. Flies the machine's TRUE
 * element: the phase law of the element the machine is actually in
 * (the classroom — never a speculative intent), eased at the
 * student's hand speed; in free flight, the stabilizer holds the
 * captured altitude, wings level, and steers home before the world's
 * edge.
 */
export class InstructorPilot implements Pilot {
  private hand: InstructorHand;
  private readonly freeFlight: boolean;
  private capturedAltM = 0.0;

  constructor(startCommands: ArrayLike<number>, freeFlight: boolean) {
    this.hand = new InstructorHand(startCommands,
      freeFlight ? InstructorHand.firm() : undefined);
    this.freeFlight = freeFlight;
  }

  /** Takeover-ready: the hand re-seats from the machine's CURRENT
   *  lever targets, and the free-flight stabilizer captures the
   *  altitude where the machine is now. */
  reseat(commands: ArrayLike<number>, body: RigidBody): void {
    this.hand = new InstructorHand(commands,
      this.freeFlight ? InstructorHand.firm() : undefined);
    if (this.freeFlight) {
      this.capturedAltM = body.position.z;
    }
  }

  /** The instructor's raw targets this tick. */
  private targets(body: RigidBody, cursor: PhaseCursor | null,
                  groundM: number): Float64Array {
    return cursor !== null
      ? cursor.phaseTargets(body)
      : stabilizeTargets(
          Math.max(this.capturedAltM, groundM + InstructorPilot.CLEARANCE_M),
          body);
  }

  /** The free-flight clearance: the stabilizer never chases its
   *  captured altitude below this many metres over the terrain. */
  private static readonly CLEARANCE_M = 30.0;   // ~100 ft AGL

  step(sighting: PilotSighting, _commands: Float64Array): Float64Array {
    return Float64Array.from(
      this.hand.easeToward(
        this.targets(sighting.body, sighting.cursor, sighting.groundM),
        sighting.dt));
  }

  relax(commands: Float64Array, _dt: number): Float64Array {
    return commands;              // the hand holds its own positions
  }

  get heldByInstructor(): boolean {
    return true;
  }

  /** The swarm's teaching observation: the stabilizer's own targets
   *  as lever levels (the error wave's source in free flight). Null
   *  in the classroom — the twin observes there. */
  stabilizerTeaching(body: RigidBody, groundM: number): {
      elementId: number; actions: Int32Array } | null {
    if (!this.freeFlight) {
      return null;
    }
    const targets = stabilizeTargets(
      Math.max(this.capturedAltM, groundM + InstructorPilot.CLEARANCE_M),
      body);
    return {
      elementId: ELEMENT_FREE_FLIGHT,
      actions: new Int32Array([levelOf(0, targets[0]),
        levelOf(1, targets[1]), levelOf(2, targets[2]),
        levelOf(3, targets[3])]),
    };
  }
}

/**
 * THE AI — the tensor brain at the controls. One decision window at
 * a time: sense, decide, apply — then the hands-off reflex between
 * windows. In the guided modes the same object is the SUGGESTION
 * engine (it always decides; the instructor's hands may fly) and
 * the error wave's student (learnFromInstructor).
 */
export class AIPilot implements Pilot {
  /** The last decision window's decision (null before the first). */
  lastDecision: NetworkDecision | null = null;
  /** The AI's last per-channel choices (the panel's AERT strip). */
  lastChoices: Int32Array | null = null;

  constructor(private readonly network: TensorNetwork,
              private readonly policy: ServoPolicy,
              private readonly rng: NumpyPcg64,
              private readonly decisionEvery: number,
              private readonly spring: (commands: Float64Array,
                                        dt: number) => Float64Array) {}

  /** THE AI STATE LOG's subject: the brain's live snapshot. */
  get stateView(): TensorStateView {
    return this.network.stateView();
  }

  /** Whether the decision window is open this tick. */
  windowOpen(tick: number): boolean {
    return tick % this.decisionEvery === 0;
  }

  /** One decision window: sense, decide (the suggestion; the state
   *  log's subject; the element the AI claims). Between windows this
   *  is a no-op. */
  decideAt(sighting: PilotSighting): void {
    if (!this.windowOpen(sighting.tick)) {
      return;
    }
    this.lastDecision = this.network.decide(sighting.goalId,
      sighting.sense, sighting.candidates, this.rng);
    this.lastChoices = this.lastDecision.actions;
  }

  /** The AI's hands: drive the levers toward the decision's stops. */
  applyTo(commands: Float64Array): Float64Array {
    const decision = this.lastDecision;
    if (decision !== null && decision.acted) {
      return this.policy.commandFrom(commands, decision.actions);
    }
    return commands;
  }

  /** The guided mode's learning: the instructor's element and levers
   *  against the AI's own decision — the error wave, every window. */
  learnFromInstructor(sighting: PilotSighting): void {
    if (!this.windowOpen(sighting.tick) || sighting.teaching === null) {
      return;
    }
    this.network.teach(sighting.teaching.elementId,
                       sighting.teaching.actions, sighting.epoch);
  }

  step(sighting: PilotSighting, commands: Float64Array): Float64Array {
    // One decision, one lever step per window — never per tick.
    if (!this.windowOpen(sighting.tick)) {
      return commands;
    }
    this.decideAt(sighting);
    return this.applyTo(commands);
  }

  relax(commands: Float64Array, dt: number): Float64Array {
    return this.spring(commands, dt);
  }

  get heldByInstructor(): boolean {
    return false;
  }
}

/**
 * (BOTH a) THE GUIDED FLIGHT — the AI suggests, the instructor flies.
 * The AI decides every window (its suggestion; its state log), the
 * instructor's hand flies the machine's true element, and the error
 * wave teaches the AI continuously: its suggestion against the
 * instructor's levers, its fired classes against the teacher's.
 */
export class AIGuidedInstructorPilot implements Pilot {
  constructor(private readonly ai: AIPilot,
              private readonly instructor: InstructorPilot) {}

  step(sighting: PilotSighting, commands: Float64Array): Float64Array {
    this.ai.decideAt(sighting);             // the suggestion + the log
    this.ai.learnFromInstructor(sighting); // the error wave, every window
    return this.instructor.step(sighting, commands);
  }

  relax(commands: Float64Array, dt: number): Float64Array {
    return this.instructor.relax(commands, dt);   // the hand holds
  }

  get heldByInstructor(): boolean {
    return true;
  }
}

/** The recovery window: decision windows to stay safe before the
 *  guarded flight hands the levers back (both hosts decide at 5 Hz). */
const RECOVERY_WINDOWS = Math.round(RECOVERY_HOLD_S * 5.0);

/**
 * (BOTH b) THE GUARDED FLIGHT — the AI flies, the instructor watches.
 * A dangerous state takes the levers (a guided flight while the
 * instructor recovers the machine — the AI suggesting and learning
 * through the recovery), and a sustained safe stretch hands them
 * back.
 */
export class AIPilotWithInstructor implements Pilot {
  private takeover = false;
  private safeWindows = 0;

  constructor(private readonly ai: AIPilot,
              private readonly instructor: InstructorPilot) {}

  step(sighting: PilotSighting, commands: Float64Array): Float64Array {
    this.ai.decideAt(sighting);   // the AI always decides (log + suggestion)
    if (!this.takeover) {
      const applied = this.ai.windowOpen(sighting.tick)
        ? this.ai.applyTo(commands) : commands;
      if (this.ai.windowOpen(sighting.tick)
          && isDangerous(sighting.body)) {
        this.takeover = true;               // the guardian takes the levers
        this.safeWindows = 0;
        this.instructor.reseat(applied, sighting.body);
      }
      return applied;
    }
    // Taken over: the instructor flies, the AI suggests and learns
    // through the recovery.
    this.ai.learnFromInstructor(sighting);
    const held = this.instructor.step(sighting, commands);
    if (this.ai.windowOpen(sighting.tick)) {
      this.safeWindows = isDangerous(sighting.body)
        ? 0 : this.safeWindows + 1;
      if (this.safeWindows >= RECOVERY_WINDOWS) {
        this.takeover = false;              // handed back to the AI
        this.safeWindows = 0;
      }
    }
    return held;
  }

  relax(commands: Float64Array, dt: number): Float64Array {
    return this.takeover
      ? this.instructor.relax(commands, dt)
      : this.ai.relax(commands, dt);
  }

  get heldByInstructor(): boolean {
    return this.takeover;
  }
}
