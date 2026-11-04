/**
 * Episode gates for the FIVE-NEURON brain (the pulsating intention
 * system). The frozen Python episode reference
 * (reference/c172-episodes-frozen.json) predates the neuron redesign —
 * its sense keys, tensor shape and choice sequence belong to the old
 * flat servo tensor, so its exact value gates are retired here
 * (re-freeze with tools/freeze-episodes.mjs once the design settles).
 *
 * What this test still gates, exactly:
 * - REPLAY DETERMINISM: the same episode, twice, from the same policy
 *   snapshot and the same rng state, is bit-exact — trajectories,
 *   choices, outcomes.
 * - flyEpisode PARITY: the real API reproduces the manual replay.
 * - NEURON MEMORY: the episodes actually teach the servo neuron, and a
 *   snapshot survives a round trip.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { Vec3 } from "../src/core/vec.js";
import { World } from "../src/core/world.js";
import { NumpyPcg64 } from "../src/core/rng.js";
import { RigidBody } from "../src/physics/rigid-body.js";
import { SurfacesState } from "../src/physics/surfaces.js";
import { GRAVITY, RHO_SEA_LEVEL } from "../src/physics/aero.js";
import {
  DenseReward, EpisodeOutcome, FlightInstruments, classifyOutcome,
} from "../src/learning/instruments.js";
import {
  ServoPolicy, TensorNetwork, TensorContext, ACTION_HOLD, GOAL_FREE_FLIGHT,
  ELEMENT_FREE_FLIGHT,
} from "../src/learning/servo-tensor.js";
import {
  TRIM_COMMAND, TRIM_SPEED, flyEpisode, springTowardNeutral,
} from "../src/learning/flight.js";
import { check, finish, section } from "./harness.js";

const refPath = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "reference", "c172-episodes-frozen.json");
const ref = JSON.parse(readFileSync(refPath, "utf8"));

const DT = 1.0 / 50.0;

const instruments = new FlightInstruments(TRIM_SPEED);
const policy = new ServoPolicy();

/** One episode replay against the brain, mirroring flyEpisode exactly
 *  (agl-aware step, terminal credit at the episode's epoch layer). */
function replayEpisode(pol: ServoPolicy, epoch: number, rng: NumpyPcg64,
                       epsilonOverride: number | null,
                       sample: boolean) {
  const world = new World({ halfExtent: 800.0 });
  const body = new RigidBody(new Vec3(0.0, 0.0, 50.0), new Vec3(TRIM_SPEED, 0.0, 0.0));
  const servos = new SurfacesState();
  const dense = new DenseReward(instruments, 5.0);
  dense.reset(body);
  const network = new TensorNetwork(pol, ELEMENT_FREE_FLIGHT, rng);
  let commands = Float64Array.from(TRIM_COMMAND);
  const recent: Array<[readonly number[], Int32Array]> = [];
  let pending = 0.0;
  const choices: number[] = [];
  const samples: number[][] = [];
  if (epsilonOverride !== null) {
    pol.epsilon = epsilonOverride;
  }
  let outcome: string | null = null;
  let tick = 0;
  for (let t = 0; t < 1000; t++) {
    tick = t;
    if (tick % 10 === 0) {
      if (recent.length > 0) {
        pol.reinforce(recent, pending, epoch);
      }
      pending = 0.0;
      // The drone reads the MACHINE: the panel comes from the body's
      // physical truth and the surface positions the actuators
      // actually hold (LiveEpisode's law).
      const sense = instruments.senseReading(body, servos.positions);
      const decision = network.decide(GOAL_FREE_FLIGHT, sense, null, rng);
      const actions = decision.actions;
      commands = pol.commandFrom(commands, actions);
      recent.push([decision.key, actions]);
      if (recent.length > 10) {
        recent.shift();
      }
      choices.push(choiceOf(actions));
    }
    servos.command(commands, DT);
    const diag = body.step(servos, DT, GRAVITY, RHO_SEA_LEVEL,
                           body.position.z - world.terrainHeight(body.position.x, body.position.y));
    commands = springTowardNeutral(commands, DT);
    pending += dense.tick(body) + 0.0005;
    if (sample && tick % 5 === 0) {
      const q = body.attitude;
      samples.push([body.position.x, body.position.y, body.position.z,
                    body.velocity.x, body.velocity.y, body.velocity.z,
                    q.w, q.x, q.y, q.z,
                    body.rateP, body.rateQ, body.rateR,
                    servos.positions[0], servos.positions[1],
                    servos.positions[2], servos.positions[3],
                    diag.airspeed, diag.alpha, diag.beta, diag.qDyn,
                    diag.lift, diag.drag, diag.thrust]);
    }
    outcome = classifyOutcome(body, world, (tick + 1) * DT);
    if (outcome !== null) {
      break;
    }
  }
  if (outcome === EpisodeOutcome.CRASHED || outcome === EpisodeOutcome.STALLED) {
    pol.reinforce(recent, -2.0, epoch);
  } else {
    pol.reinforce(recent, 0.5, epoch);
  }
  return {
    outcome: outcome ?? EpisodeOutcome.ALIVE,
    survival: (tick + 1) * DT,
    choices,
    samples,
  };
}

/** The tracer's choice encoding: 0 = hold-all, else the option index
 *  (each lever's stops in channel order). */
function choiceOf(actions: Int32Array): number {
  const base = [1, 10, 21, 30];
  for (let channel = 0; channel < 4; channel++) {
    if (actions[channel] !== ACTION_HOLD) {
      return base[channel] + actions[channel];
    }
  }
  return 0;
}

ref.episodes.forEach((ep: any) => {
  section(`episode ${ep.epoch} seed ${1007 + ep.epoch} (neuron brain)`);
  {
    const snapshot = policy.snapshot();
    const savedEpsilon = policy.epsilon;
    const rngA = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
    const runA = replayEpisode(policy, ep.epoch, rngA, ep.epsilon_override, true);
    // Replay determinism: same snapshot, same dice -> bit-exact flight.
    policy.restore(snapshot);
    policy.epsilon = savedEpsilon;
    const rngB = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
    const runB = replayEpisode(policy, ep.epoch, rngB, ep.epsilon_override, true);
    check("outcome identical across replays", runA.outcome === runB.outcome,
          `got ${runA.outcome} vs ${runB.outcome}`);
    check("survival identical across replays",
          runA.survival === runB.survival,
          `got ${runA.survival} vs ${runB.survival}`);
    check("choices identical across replays",
          JSON.stringify(runA.choices) === JSON.stringify(runB.choices));
    check("trajectory identical across replays",
          JSON.stringify(runA.samples) === JSON.stringify(runB.samples));
    check("the episode actually flew",
          runA.choices.length > 0 && runA.samples.length > 0);
    void ep.outcome;   // the frozen Python values are retired (see header)
    void ep.survival;
  }
});

// flyEpisode parity: the real API reproduces the manual replay's outcome
// and survival on a fresh brain with the same dice.
{
  section("flyEpisode API parity (episode 0)");
  const ep = ref.episodes[0];
  const fresh = new ServoPolicy();
  const rng = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
  const result = flyEpisode(fresh.snapshot(), instruments, ep.epoch, rng);
  const manual = new ServoPolicy();
  const rngManual = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
  const run = replayEpisode(manual, ep.epoch, rngManual, ep.epsilon_override, false);
  check("outcome matches the manual replay",
        result.outcome === run.outcome,
        `got ${result.outcome} want ${run.outcome}`);
  check("survival matches the manual replay",
        Math.abs(result.survival - run.survival) <= 1e-12,
        `got ${result.survival} want ${run.survival}`);
  check("final state key exists",
        result.finalStateKey.length === 5);
}

// The neuron memory after the episodes.
{
  section("neuron memory after the episodes");
  check("the controls tensor learned",
        policy.servoTensor.storedCombinations > 0,
        `got ${policy.servoTensor.storedCombinations} keys`);
  check("the brain holds all five tensors",
        policy.storedCombinations >= policy.servoTensor.storedCombinations
        && policy.goalTensor.storedCombinations === 0,   // free flight files no goal map
        `total ${policy.storedCombinations}`);
  const snap = policy.snapshot();
  const keysBefore = policy.storedCombinations;
  policy.imprint(99, 99, { location: [0, 0], energy: [0, 0, 0, 0, 0],
                           flight: [0, 0, 0, 0, 0], controls: [4, 5, 4, 6] },
                 [3, 3, 3, 5], new TensorContext(0), 1.0, 12345);
  check("imprint fires into the neurons",
        policy.storedCombinations > keysBefore,
        `got ${policy.storedCombinations} want > ${keysBefore}`);
  policy.restore(snap);
  check("snapshot round-trips the brain",
        policy.storedCombinations === keysBefore,
        `got ${policy.storedCombinations} want ${keysBefore}`);
}

finish("episode.test");
