/**
 * THE AIRCRAFT CONTROLLERS — who is at the controls, as the machine's
 * own plug (learning/aircraft-controller.ts).
 *
 * Every Aircraft OWNS its controller (Aircraft.attachController — the
 * machine knows who flies it), and every controller OWNS its pilots:
 *
 *   AircraftController          — the abstract socket: one machine,
 *                                 step / relax / heldByInstructor;
 *   SoloController              — ONE pilot at the controls: the
 *                                 instructor's hand (Solo:
 *                                 Instructor) or the AI's own brain
 *                                 (Solo: AI);
 *   PairController              — THE PAIR: the AI beside the
 *                                 instructor. Suggest (Pair:
 *                                 Instructor): the AI decides and
 *                                 learns every window, the
 *                                 instructor's hand flies. Guarded
 *                                 (Pair: AI): the AI flies, the
 *                                 instructor watches — a dangerous
 *                                 state takes the levers (a suggest
 *                                 flight while the instructor
 *                                 recovers the machine), a
 *                                 sustained safe stretch hands them
 *                                 back.
 *
 * The flights hold a controller; the controller holds the pilots; the
 * pilots produce the control inputs; the machine consumes them
 * through its single input method. The controls dropdown names a
 * controller (controllerFor): a mid-air swap re-seats the
 * instructor's hand from the machine's CURRENT lever targets and
 * swaps the whole controller — the machine, its cursor, and the
 * brain run on untouched.
 */
import type { RigidBody } from "../physics/rigid-body.js";
import type { Aircraft } from "../physics/aircraft.js";
import {
  isDangerous, RECOVERY_HOLD_S, type ControlMode,
} from "./guardian.js";
import type { Pilot, PilotSighting } from "./pilot.js";
import { InstructorPilot, AiPilot } from "./pilot.js";

/**
 * THE SOCKET — the machine's own controller. Subclasses decide who
 * flies; the machine, the flights, and the UI ask through this face.
 */
export abstract class AircraftController {
  protected constructor(readonly aircraft: Aircraft) {
    aircraft.attachController(this);   // the machine knows who flies it
  }

  /** One tick at the controls: the pilots' inputs for this tick. */
  abstract step(sighting: PilotSighting,
                commands: Float64Array): Float64Array;

  /** The hands-off reflex after the physics tick. */
  abstract relax(commands: Float64Array, dt: number): Float64Array;

  /** The instructor's hand is physically on the levers right now. */
  abstract get heldByInstructor(): boolean;

  /** The AI at (or beside) the controls, if there is one. */
  abstract get ai(): AiPilot | null;

  /** The instructor's hand, if there is one. */
  abstract get instructor(): InstructorPilot | null;
}

/** ONE PILOT AT THE CONTROLS — Solo: Instructor or Solo: AI. */
export class SoloController extends AircraftController {
  constructor(aircraft: Aircraft, private readonly pilot: Pilot) {
    super(aircraft);
    if (pilot instanceof AiPilot) {
      pilot.markHands(true);   // the AI's own hands fly
    }
  }

  step(sighting: PilotSighting, commands: Float64Array): Float64Array {
    return this.pilot.step(sighting, commands);
  }

  relax(commands: Float64Array, dt: number): Float64Array {
    return this.pilot.relax(commands, dt);
  }

  get heldByInstructor(): boolean {
    return this.pilot.heldByInstructor;
  }

  get ai(): AiPilot | null {
    return this.pilot instanceof AiPilot ? this.pilot : null;
  }

  get instructor(): InstructorPilot | null {
    return this.pilot instanceof InstructorPilot
      ? this.pilot : null;
  }
}

/**
 * THE PAIR — the AI and the instructor flying together. In the
 * suggest pairing the instructor's hands fly and the AI's decisions
 * are its suggestions (it decides and learns every window, but earns
 * no flight credit — the instructor's stick is the truth); in the
 * guarded pairing the AI's own hands fly until the instructor must
 * take over, and its decisions earn credit exactly while it flies.
 */
export class PairController extends AircraftController {
  private takeover = false;
  private safeWindows = 0;

  constructor(aircraft: Aircraft, private readonly pairAi: AiPilot,
              private readonly pairInstructor: InstructorPilot,
              private readonly guarded: boolean) {
    super(aircraft);
    // the guarded pairing starts with the AI's own hands on; the
    // suggest pairing's AI never earns flight credit
    this.pairAi.markHands(this.guarded);
  }

  step(sighting: PilotSighting, commands: Float64Array): Float64Array {
    if (!this.guarded) {
      // SUGGEST: the AI decides (the suggestion + the state log) and
      // learns from the instructor's hands, every window; the
      // instructor's hand flies the machine's true element.
      this.pairAi.markHands(false);
      this.pairAi.decide(sighting);
      this.pairAi.learnFromInstructor(sighting);
      return this.pairInstructor.step(sighting, commands);
    }
    // GUARDED: the AI flies; the instructor watches.
    this.pairAi.decide(sighting);   // the AI always decides (log + flight)
    if (!this.takeover) {
      this.pairAi.markHands(true);
      const applied = this.pairAi.windowOpen(sighting.tick)
        ? this.pairAi.applyTo(commands) : commands;
      if (this.pairAi.windowOpen(sighting.tick)
          && isDangerous(sighting.body)) {
        this.takeover = true;               // the guardian takes the levers
        this.pairAi.markHands(false);
        this.safeWindows = 0;
        this.pairInstructor.reseat(applied, sighting.body);
      }
      return applied;
    }
    // Taken over: the instructor flies, the AI suggests and learns
    // through the recovery.
    this.pairAi.markHands(false);
    this.pairAi.learnFromInstructor(sighting);
    const held = this.pairInstructor.step(sighting, commands);
    if (this.pairAi.windowOpen(sighting.tick)) {
      this.safeWindows = isDangerous(sighting.body)
        ? 0 : this.safeWindows + 1;
      if (this.safeWindows >= PairController.RECOVERY_WINDOWS) {
        this.takeover = false;              // handed back to the AI
        this.safeWindows = 0;
        this.pairAi.markHands(true);
      }
    }
    return held;
  }

  relax(commands: Float64Array, dt: number): Float64Array {
    return !this.guarded || this.takeover
      ? this.pairInstructor.relax(commands, dt)
      : this.pairAi.relax(commands, dt);
  }

  get heldByInstructor(): boolean {
    return !this.guarded || this.takeover;
  }

  get ai(): AiPilot | null {
    return this.pairAi;
  }

  get instructor(): InstructorPilot | null {
    return this.pairInstructor;
  }

  /** The recovery window: decision windows to stay safe before the
   *  guarded pairing hands the levers back. */
  private static readonly RECOVERY_WINDOWS =
    Math.round(RECOVERY_HOLD_S * 5.0);
}

/**
 * THE CONTROLS DROPDOWN — the controls mode names the controller.
 * The instructor's hand is re-seated from the machine's current lever
 * targets before the swap, so whichever way the mode changes the
 * yoke never jumps.
 */
export function controllerFor(aircraft: Aircraft, mode: ControlMode,
                              ai: AiPilot,
                              instructor: InstructorPilot,
                              reseat?: {
    commands: Float64Array; body: RigidBody }): AircraftController {
  if (reseat !== undefined) {
    instructor.reseat(reseat.commands, reseat.body);
  }
  switch (mode) {
    case "ai": return new SoloController(aircraft, ai);
    case "instructor": return new SoloController(aircraft, instructor);
    case "both-suggest":
      return new PairController(aircraft, ai, instructor, false);
    case "both-guarded":
      return new PairController(aircraft, ai, instructor, true);
  }
}
