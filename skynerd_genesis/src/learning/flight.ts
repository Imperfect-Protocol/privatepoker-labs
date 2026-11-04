/**
 * One life: a drone's flight episode. Ported 1:1 from
 * `skynerd/learning/flight.py` (Genesis Milestone 3, will-less path).
 *
 * The will-aware goal hierarchy is a later milestone; this port covers the
 * servo-policy path: each 0.2 s decision window the policy acts from the
 * instrument state key, the dense reward accumulates improvement credit,
 * and the episode ends physically (crash / stall / out of bounds) or at the
 * 20 s horizon.
 */
import { Vec3 } from "../core/vec.js";
import { World } from "../core/world.js";
import { NumpyPcg64 } from "../core/rng.js";
import { RigidBody } from "../physics/rigid-body.js";
import { THROTTLE } from "../physics/surfaces.js";
import { Aircraft, HandDecisions } from "../physics/aircraft.js";
import { Quaternion } from "../core/quaternion.js";
import {
  DenseReward, EpisodeOutcome, FlightInstruments, classifyOutcome,
} from "./instruments.js";
import { levelOf } from "./instruments.js";
import { ControlMode } from "./guardian.js";
import {
  AiPilot, InstructorPilot, type AiBrainState, type PilotSighting,
} from "./pilot.js";
import {
  AircraftController, controllerFor,
} from "./aircraft-controller.js";
import {
  ELEMENT_FREE_FLIGHT, GOAL_FREE_FLIGHT, type TensorStateView,
} from "./servo-tensor.js";

export const TRIM_SPEED = 38.0;       // m/s — 85 mph, the C172 economy cruise
export const HORIZON = 20.0;
const STICK_SPRING_TAU = 1.5;    // s: hands off, the stick springs back toward neutral
const DT = 1.0 / 50.0;           // the episode's physics clock, 50 Hz
export const TRIM_COMMAND = [0.0, 0.02485, 0.0, 0.8];   // trimmed for level cruise: 85 mph at 2300 RPM

/**
 * The stick spring: a held command decays toward trim when not actively
 * pushed. Real airframes self-center; without this, every random explore
 * push accumulates forever and no memory can save the drone.
 */
export function springTowardNeutral(commands: Float64Array, dt: number): Float64Array {
  const out = new Float64Array(4);
  for (let channel = 0; channel < 4; channel++) {
    out[channel] = commands[channel]
      + (TRIM_COMMAND[channel] - commands[channel]) * (dt / STICK_SPRING_TAU);
  }
  return out;
}

/**
 * The student's stick spring: the three STICK channels self-center
 * toward trim, but the THROTTLE is a friction lever — it stays where
 * the student last set it. A spring-loaded throttle silently pulls
 * power off a hands-off student mid-climb, and no memory can hold a
 * setting the airframe keeps undoing. The swarm's episodes keep the
 * four-channel spring (their exploration law was tuned on it); the
 * classroom's hands-off reflex is the physical one.
 */
export function springStickTowardNeutral(commands: Float64Array,
                                         dt: number): Float64Array {
  const out = new Float64Array(4);
  for (let channel = 0; channel < 4; channel++) {
    out[channel] = channel === THROTTLE
      ? commands[channel]   // friction holds the lever
      : commands[channel]
        + (TRIM_COMMAND[channel] - commands[channel]) * (dt / STICK_SPRING_TAU);
  }
  return out;
}

export interface EpisodeResult {
  survival: number;
  outcome: string;
  finalStateKey: number[];
  finalCommands: Float64Array;
}

export interface EpisodeOptions {
  startStalled?: boolean;
  startSpiral?: boolean;
  startPosition?: Vec3;
  epsilonOverride?: number | null;
  /** An existing machine to fly (its body must already be positioned);
   *  by default the episode builds its own. */
  aircraft?: Aircraft;
  /** THE DRONE'S OWN PILOT to fly this life with (the carpet's drones:
   *  the flock's living shard IS the flying brain, its learning stays
   *  in the shard and its death scream carries it); by default the
   *  episode grows its own from the brain token. */
  pilot?: AiPilot;
  /** The episode's horizon, seconds — how long the DRONE SIDE of the
   *  library flies before calling the episode survived. Infinity means
   *  NO expiry: the flight lives until it crashes, stalls, or leaves
   *  the world. (Default: HORIZON, the old 20 s.) */
  horizon?: number;
}

export class LiveEpisode {
  /** The machine flying this episode — ONE Aircraft class, the same one
   *  the classroom flies. The episode is only the drone's life in it. */
  readonly aircraft: Aircraft;
  private readonly world = new World({ halfExtent: 800.0 });
  private readonly dense: DenseReward;
  /** The controller's command state — the hand at the controls. */
  private commands: Float64Array = Float64Array.from(TRIM_COMMAND);
  private readonly instruments: FlightInstruments;
  private readonly epoch: number;
  private pendingReward = 0.0;
  private readonly decisionEvery = 10;
  private readonly steps: number;
  tick = 0;
  private outcome: string | null = null;
  /** THE AI — THE DRONE'S ENCAPSULATED TENSOR BRAIN: every tensor
   *  lives inside it (the memory, the pulsating network, the error
   *  neurons); the episode never sees one. Free flight has no
   *  maneuver chain; the sensors' codes and the skill's coherence
   *  are the servo's whole input. */
  readonly ai: AiPilot;
  private readonly instructor: InstructorPilot;
  /** THE CONTROLLER at the machine's controls (the mode IS the
   *  controller: all mode logic lives in the AircraftController
   *  classes; the episode never branches on it). */
  private controller: AircraftController;
  private mode: ControlMode = "ai";

  constructor(brain: AiBrainState | null, instruments: FlightInstruments,
              epoch: number, rng: NumpyPcg64,
              options: EpisodeOptions = {}) {
    this.instruments = instruments;
    this.epoch = epoch;
    const body = options.aircraft !== undefined
      ? options.aircraft.body
      : new RigidBody(
          options.startPosition ?? new Vec3(0.0, 0.0, 50.0),
          new Vec3(TRIM_SPEED, 0.0, 0.0));
    if (options.startStalled) {
      body.velocity = new Vec3(5.0, 0.0, -3.5);
    }
    if (options.startSpiral) {
      // The launch lottery's worst ticket: overbanked in a speed dive.
      // Wings-level trim cannot save this; only remembered recovery can.
      const side = rng.uniform() < 0.5 ? 1.0 : -1.0;
      const bank = side * (60.0 + 15.0 * rng.uniform()) * Math.PI / 180.0;
      body.attitude = Quaternion.fromEuler(0.0, 0.0, bank);
      body.velocity = new Vec3(14.0, 0.0, -4.0);
    }
    this.aircraft = options.aircraft ?? new Aircraft(body);
    // THE BRAIN — constructed exactly where the network used to draw
    // (the frozen stream: the spiral dice, then the network's seeds).
    // A drone that brings its own pilot flies it (a fresh life over
    // the same memory); otherwise the episode grows its own from the
    // brain token.
    this.ai = options.pilot ?? new AiPilot({ seedRng: rng,
      decisionEvery: this.decisionEvery, spring: springTowardNeutral,
      startElementId: ELEMENT_FREE_FLIGHT });
    if (options.pilot === undefined && brain !== null) {
      this.ai.inherit(brain);
    }
    this.ai.beginFlight(ELEMENT_FREE_FLIGHT);
    if (options.epsilonOverride !== undefined
        && options.epsilonOverride !== null) {
      this.ai.setExploration(options.epsilonOverride);
    }
    this.instructor = new InstructorPilot(TRIM_COMMAND, true);
    this.controller = controllerFor(this.aircraft, this.mode, this.ai,
                                    this.instructor);
    this.dense = new DenseReward(instruments, 5.0);
    this.dense.reset(this.aircraft.body);
    this.steps = Math.floor((options.horizon ?? HORIZON) / DT);
  }

  /** Survival so far, seconds (the episode's length once it has ended). */
  get survival(): number {
    return this.tick * DT;
  }

  /** How the episode ended (null while it is still flying). */
  get episodeOutcome(): string | null {
    return this.outcome;
  }

  /** The hand's current commands — the controller's own write state.
   *  Reading the machine is a different path: the aircraft's own
   *  getters (handCommands, surfacePositions, the instruments) are the
   *  sealed truth of what the machine holds and does. */
  get handCommands(): Float64Array {
    return this.commands;
  }

  /** The drone's decisions for the panel — the same contract as the
   *  classroom's ClassroomFlight.handInputs. */
  get handInputs(): HandDecisions {
    return this.controller.heldByInstructor
      ? { choices: null, fresh: false }
      : {
        choices: this.ai.choices,
        fresh: this.tick % this.decisionEvery < this.decisionEvery / 2,
      };
  }

  /** THE UI's STATE VIEW — the AI state log's snapshot of this
   *  drone's tensor network. */
  stateView(): TensorStateView {
    return this.ai.stateView;
  }

  /** THE CONTROLS DROPDOWN — this drone's sticky mode. */
  get controlsMode(): ControlMode {
    return this.mode;
  }

  /** Swap the whole pilot — the mode IS the pilot object. A mid-air
   *  swap re-seats the instructor's hand from the drone's CURRENT
   *  lever targets, and the stabilizer captures the altitude where
   *  the drone is now. */
  setControlsMode(mode: ControlMode): void {
    if (mode === this.mode) {
      return;
    }
    this.mode = mode;
    this.controller = controllerFor(this.aircraft, mode, this.ai,
      this.instructor,
      { commands: this.commands, body: this.aircraft.body });
  }

  /** Whether the instructor currently holds the levers (the status
   *  line and the panel care). */
  get instructorAtControls(): boolean {
    return this.controller.heldByInstructor;
  }

  /**
   * One 50 Hz tick of the drone's life. The call order is the frozen
   * protocol exactly (same RNG stream, same reinforcement), so stepping
   * live in the browser and running headless are the same flight.
   * Returns false once the episode has ended.
   */
  step(): boolean {
    if (this.tick % this.decisionEvery === 0) {
      this.ai.credit(this.pendingReward, this.epoch);
      this.pendingReward = 0.0;
    }
    // THE PILOT AT THE CONTROLS: the mode IS the pilot object — the
    // episode never branches on it. The pilot reads the sighting (the
    // drone's own state, its free-flight null cursor, the stabilizer's
    // teaching beside it) and produces this tick's control inputs.
    const sighting: PilotSighting = {
      dt: DT,
      tick: this.tick,
      body: this.aircraft.body,
      groundM: this.world.terrainHeight(this.aircraft.body.position.x,
                                        this.aircraft.body.position.y),
      cursor: null,
      sense: this.instruments.senseReading(this.aircraft.body,
                                           this.aircraft.surfacePositions),
      teaching: this.instructor.stabilizerTeaching(this.aircraft.body,
        this.world.terrainHeight(this.aircraft.body.position.x,
                                  this.aircraft.body.position.y)),
      epoch: this.epoch,
      goalId: GOAL_FREE_FLIGHT,
      candidates: null,
    };
    this.commands = this.controller.step(sighting, this.commands);
    // The AI's own fired decisions earn credit — the eligibility's
    // bookkeeping lives inside the brain now (the controllers tell it
    // whose hands fly).
    this.aircraft.command(this.commands, DT);
    this.aircraft.step(DT, this.aircraft.body.position.z
      - this.world.terrainHeight(this.aircraft.body.position.x,
                                 this.aircraft.body.position.y));
    // The hands-off reflex: the AI's stick springs; the instructor's
    // hand holds its own positions.
    this.commands = this.controller.relax(this.commands, DT);
    this.pendingReward += this.dense.tick(this.aircraft.body) + 0.0005;
    const outcome = classifyOutcome(this.aircraft.body, this.world,
                                    (this.tick + 1) * DT);
    this.tick += 1;
    if (outcome !== null || this.tick >= this.steps) {
      this.finish(outcome);
      return false;
    }
    return true;
  }

  /** The episode's epilogue: the final reinforcement, exactly once. */
  private finish(outcome: string | null): void {
    this.outcome = outcome === null ? EpisodeOutcome.ALIVE : outcome;
    this.ai.settle(outcome, this.epoch);
  }
}

/** Run one episode headless (the library fast path). */
export function flyEpisode(brain: AiBrainState | null,
                           instruments: FlightInstruments,
                           epoch: number, rng: NumpyPcg64,
                           options: EpisodeOptions = {}): EpisodeResult {
  const episode = new LiveEpisode(brain, instruments, epoch, rng, options);
  while (episode.step()) { /* fly to completion */ }
  // The record is the MACHINE's, read through its sealed getters.
  return {
    survival: episode.survival,
    outcome: episode.episodeOutcome!,
    finalStateKey: instruments.stateKey(episode.aircraft.body,
                                        episode.aircraft.surfacePositions[THROTTLE]),
    finalCommands: episode.aircraft.handCommands,
  };
}
