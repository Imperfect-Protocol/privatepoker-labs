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
import { ServoPolicy, TensorNetwork, GOAL_FREE_FLIGHT,
         ELEMENT_FREE_FLIGHT } from "./servo-tensor.js";

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
  private commands = Float64Array.from(TRIM_COMMAND);
  private readonly policy: ServoPolicy;
  private readonly instruments: FlightInstruments;
  private readonly epoch: number;
  private readonly rng: NumpyPcg64;
  /** The drone's tensor network: the seven tensors as dispatcher
   *  nodes on the shared queue — stun on input, fire on relaxation,
   *  message the law's thresholded projection. Free flight has no
   *  maneuver chain; the sensors' codes and the skill's coherence
   *  are the servo's whole input. */
  private readonly network: TensorNetwork;
  private readonly recent: Array<[readonly number[], Int32Array]> = [];
  private pendingReward = 0.0;
  private readonly decisionEvery = 10;
  private readonly steps: number;
  tick = 0;
  private outcome: string | null = null;
  private lastChoices: Int32Array | null = null;

  constructor(policy: ServoPolicy, instruments: FlightInstruments,
              epoch: number, rng: NumpyPcg64,
              options: EpisodeOptions = {}) {
    this.policy = policy;
    this.instruments = instruments;
    this.epoch = epoch;
    this.rng = rng;
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
    this.network = new TensorNetwork(policy, ELEMENT_FREE_FLIGHT, rng);
    this.dense = new DenseReward(instruments, 5.0);
    this.dense.reset(this.aircraft.body);
    this.steps = Math.floor((options.horizon ?? HORIZON) / DT);
    if (options.epsilonOverride !== undefined
        && options.epsilonOverride !== null) {
      policy.epsilon = options.epsilonOverride;
    }
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
    return {
      choices: this.lastChoices,
      fresh: this.tick % this.decisionEvery < this.decisionEvery / 2,
    };
  }

  /**
   * One 50 Hz tick of the drone's life. The call order is the frozen
   * protocol exactly (same RNG stream, same reinforcement), so stepping
   * live in the browser and running headless are the same flight.
   * Returns false once the episode has ended.
   */
  step(): boolean {
    if (this.tick % this.decisionEvery === 0) {
      if (this.recent.length > 0) {
        this.policy.reinforce(this.recent, this.pendingReward, this.epoch);
      }
      this.pendingReward = 0.0;
      // The drone reads the MACHINE, not its own hand: the panel comes
      // from the aircraft's current instruments (the body's physical
      // truth and the surface positions the actuators actually hold) —
      // the same reading the classroom's students get.
      const sense = this.instruments.senseReading(this.aircraft.body,
                                                  this.aircraft.surfacePositions);
      const decision = this.network.decide(GOAL_FREE_FLIGHT, sense,
                                            null, this.rng);
      this.commands = this.policy.commandFrom(this.commands, decision.actions);
      this.lastChoices = decision.actions;
      // The reinforcement key is the servo tensor's observation prefix:
      // the skill's forward. Only FIRED decisions earn credit — a hold
      // (no news from the skill) has no key to credit.
      if (decision.acted) {
        this.recent.push([decision.key, decision.actions]);
        if (this.recent.length > 10) {
          this.recent.shift();
        }
      }
    }
    this.aircraft.command(this.commands, DT);
    this.aircraft.step(DT, this.aircraft.body.position.z
      - this.world.terrainHeight(this.aircraft.body.position.x,
                                 this.aircraft.body.position.y));
    this.commands = springTowardNeutral(this.commands, DT);
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
    this.outcome = outcome;
    if (outcome === EpisodeOutcome.CRASHED || outcome === EpisodeOutcome.STALLED) {
      this.policy.reinforce(this.recent, -2.0, this.epoch);
    } else if (outcome === null) {
      this.outcome = EpisodeOutcome.ALIVE;
      this.policy.reinforce(this.recent, 0.5, this.epoch);
    }
  }
}

/** Run one episode headless (the library fast path). */
export function flyEpisode(policy: ServoPolicy, instruments: FlightInstruments,
                           epoch: number, rng: NumpyPcg64,
                           options: EpisodeOptions = {}): EpisodeResult {
  const episode = new LiveEpisode(policy, instruments, epoch, rng, options);
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
