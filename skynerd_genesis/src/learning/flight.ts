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
import { SurfacesState } from "../physics/surfaces.js";
import { Quaternion } from "../core/quaternion.js";
import {
  DenseReward, EpisodeOutcome, FlightInstruments, classifyOutcome,
} from "./instruments.js";
import { ServoPolicy } from "./servo-tensor.js";

export const TRIM_SPEED = 10.5;
export const HORIZON = 20.0;
const STICK_SPRING_TAU = 1.5;    // s: hands off, the stick springs back toward neutral
export const TRIM_COMMAND = [0.0, 0.0, 0.0, 0.15];

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
}

export function flyEpisode(policy: ServoPolicy, instruments: FlightInstruments,
                           epoch: number, rng: NumpyPcg64,
                           options: EpisodeOptions = {}): EpisodeResult {
  const world = new World({ halfExtent: 800.0 });
  const body = new RigidBody(
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
  const servos = new SurfacesState();
  const dt = 1.0 / 50.0;
  const dense = new DenseReward(instruments, 5.0);
  dense.reset(body);
  let commands = Float64Array.from(TRIM_COMMAND);
  const recent: Array<[number[], Int32Array]> = [];
  let pendingReward = 0.0;
  const decisionEvery = 10;
  const steps = Math.floor(HORIZON / dt);
  let outcome: string | null = null;
  let tick = 0;
  if (options.epsilonOverride !== undefined && options.epsilonOverride !== null) {
    policy.epsilon = options.epsilonOverride;
  }
  for (let t = 0; t < steps; t++) {
    tick = t;
    if (tick % decisionEvery === 0) {
      const timeEpoch = tick + epoch * 100000;
      if (recent.length > 0) {
        policy.reinforce(recent, pendingReward, timeEpoch);
      }
      pendingReward = 0.0;
      const stateKey = instruments.stateKey(body, commands[3]);
      const actions = policy.choose(stateKey, rng);
      commands = policy.commandFrom(commands, actions);
      recent.push([stateKey, actions]);
      if (recent.length > 10) {
        recent.shift();
      }
    }
    servos.command(commands, dt);
    body.step(servos, dt);
    commands = springTowardNeutral(commands, dt);
    pendingReward += dense.tick(body) + 0.0005;
    outcome = classifyOutcome(body, world, (tick + 1) * dt);
    if (outcome !== null) {
      break;
    }
  }
  const timeEpoch = tick + epoch * 100000;
  if (outcome === EpisodeOutcome.CRASHED || outcome === EpisodeOutcome.STALLED) {
    policy.reinforce(recent, -2.0, timeEpoch);
    return {
      survival: (tick + 1) * dt,
      outcome,
      finalStateKey: instruments.stateKey(body, commands[3]),
      finalCommands: Float64Array.from(commands),
    };
  }
  if (outcome === null) {
    outcome = EpisodeOutcome.ALIVE;
    policy.reinforce(recent, 0.5, timeEpoch);
  }
  return {
    survival: (tick + 1) * dt,
    outcome,
    finalStateKey: instruments.stateKey(body, commands[3]),
    finalCommands: Float64Array.from(commands),
  };
}
