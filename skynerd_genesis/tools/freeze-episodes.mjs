/**
 * Freeze the C172 learning episodes: re-record the Milestone 3 acceptance
 * sequence (three lives of one learned policy, blank start) on the corrected
 * airframe — the 1111 kg battle-C172, the body-rate quaternion propagation
 * and the ground-effect glue (agl-aware step). The scenario INPUTS (rng
 * state/inc, epsilon override, epochs) are the frozen Python seeds; the
 * outcomes, choices, trajectories and the learned brain are re-recorded
 * TS-side. The replay mirrors test/episode.test.ts exactly (including the
 * agl-aware body.step), so the frozen file is what that test gates against.
 *
 * Run the whole sequence TWICE, assert bit-exact JSON, then write
 * reference/c172-episodes-frozen.json. The Python flight-ref.json stays on
 * disk as the historical record for the old 1.5 kg airframe.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { Vec3 } from "../build/src/core/vec.js";
import { World } from "../build/src/core/world.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { RigidBody } from "../build/src/physics/rigid-body.js";
import { SurfacesState } from "../build/src/physics/surfaces.js";
import { GRAVITY, RHO_SEA_LEVEL } from "../build/src/physics/aero.js";
import {
  DenseReward, EpisodeOutcome, FlightInstruments, classifyOutcome,
} from "../build/src/learning/instruments.js";
import {
  ServoPolicy, HOLD, PUSH, GOAL_FREE_FLIGHT, ELEMENT_FREE_FLIGHT,
} from "../build/src/learning/servo-tensor.js";
import {
  TRIM_COMMAND, TRIM_SPEED, flyEpisode, springTowardNeutral,
} from "../build/src/learning/flight.js";

const seedPath = join(dirname(fileURLToPath(import.meta.url)),
                      "..", "reference", "flight-ref.json");
const seedDoc = JSON.parse(readFileSync(seedPath, "utf8"));

const DT = 1.0 / 50.0;

/** The choice encoding shared by the test and the Python tracer:
 *  0 = hold-all, else 1 + surface*2 + push. */
function choiceOf(actions) {
  for (let surface = 0; surface < 4; surface++) {
    if (actions[surface] !== HOLD) {
      return 1 + surface * 2 + (actions[surface] === PUSH ? 1 : 0);
    }
  }
  return 0;
}

function snapshotPolicy(p) {
  return { brain: p.snapshot(), epsilon: p.epsilon };
}

function restorePolicy(p, snap) {
  p.restore(snap.brain);
  p.epsilon = snap.epsilon;
}

/** One episode's main replay, mirroring test/episode.test.ts exactly. */
function replayEpisode(policy, instruments, ep) {
  const rng = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
  const world = new World({ halfExtent: 800.0 });
  const body = new RigidBody(new Vec3(0.0, 0.0, 50.0), new Vec3(TRIM_SPEED, 0.0, 0.0));
  const servos = new SurfacesState();
  const dense = new DenseReward(instruments, 5.0);
  dense.reset(body);
  let commands = Float64Array.from(TRIM_COMMAND);
  const recent = [];
  let pending = 0.0;
  const choices = [];
  let outcome = null;
  if (ep.epsilon_override !== null) {
    policy.epsilon = ep.epsilon_override;
  }
  let tick = 0;
  for (let t = 0; t < 1000; t++) {
    tick = t;
    if (tick % 10 === 0) {
      if (recent.length > 0) {
        policy.reinforce(recent, pending, ep.epoch);
      }
      pending = 0.0;
      const sense = instruments.senseReading(body, commands);
      const actions = policy.choose(ELEMENT_FREE_FLIGHT, sense, rng);
      commands = policy.commandFrom(commands, actions);
      recent.push([[ELEMENT_FREE_FLIGHT, ...sense.flight, ...sense.controls],
                   actions]);
      if (recent.length > 10) {
        recent.shift();
      }
      choices.push(choiceOf(actions));
    }
    servos.command(commands, DT);
    body.step(servos, DT, GRAVITY, RHO_SEA_LEVEL,
              body.position.z - world.terrainHeight(body.position.x, body.position.y));
    commands = springTowardNeutral(commands, DT);
    pending += dense.tick(body) + 0.0005;
    outcome = classifyOutcome(body, world, (tick + 1) * DT);
    if (outcome !== null) {
      break;
    }
  }
  if (outcome === EpisodeOutcome.CRASHED || outcome === EpisodeOutcome.STALLED) {
    policy.reinforce(recent, -2.0, ep.epoch);
  } else {
    policy.reinforce(recent, 0.5, ep.epoch);
  }
  if (outcome === null) {
    outcome = EpisodeOutcome.ALIVE;
  }
  return {
    outcome,
    survival: (tick + 1) * DT,
    choices,
    finalStateKey: instruments.stateKey(body, commands[3]),
  };
}

/** The second identical replay with sampling enabled (same trick as the
 *  Python tracer and the test). */
function traceSamples(pol, inst, ep) {
  const rng = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
  const world = new World({ halfExtent: 800.0 });
  const body = new RigidBody(new Vec3(0.0, 0.0, 50.0), new Vec3(TRIM_SPEED, 0.0, 0.0));
  const servos = new SurfacesState();
  const dense = new DenseReward(inst, 5.0);
  dense.reset(body);
  let commands = Float64Array.from(TRIM_COMMAND);
  const recent = [];
  let pending = 0.0;
  const samples = [];
  if (ep.epsilon_override !== null) {
    pol.epsilon = ep.epsilon_override;
  }
  let outcome = null;
  let tick = 0;
  for (let t = 0; t < 1000; t++) {
    tick = t;
    if (tick % 10 === 0) {
      if (recent.length > 0) {
        pol.reinforce(recent, pending, ep.epoch);
      }
      pending = 0.0;
      const sense = inst.senseReading(body, commands);
      const actions = pol.choose(ELEMENT_FREE_FLIGHT, sense, rng);
      commands = pol.commandFrom(commands, actions);
      recent.push([[ELEMENT_FREE_FLIGHT, ...sense.flight, ...sense.controls],
                   actions]);
      if (recent.length > 10) {
        recent.shift();
      }
    }
    servos.command(commands, DT);
    const diag = body.step(servos, DT, GRAVITY, RHO_SEA_LEVEL,
                          body.position.z - world.terrainHeight(body.position.x, body.position.y));
    commands = springTowardNeutral(commands, DT);
    pending += dense.tick(body) + 0.0005;
    if (tick % 5 === 0) {
      const q = body.attitude;
      samples.push({
        pos: [body.position.x, body.position.y, body.position.z],
        vel: [body.velocity.x, body.velocity.y, body.velocity.z],
        quat: [q.w, q.x, q.y, q.z],
        rates: [body.rateP, body.rateQ, body.rateR],
        surfaces: Array.from(servos.positions),
        diag: [diag.airspeed, diag.alpha, diag.beta, diag.qDyn, diag.lift,
               diag.drag, diag.thrust],
      });
    }
    outcome = classifyOutcome(body, world, (tick + 1) * DT);
    if (outcome !== null) {
      break;
    }
  }
  if (outcome === EpisodeOutcome.CRASHED || outcome === EpisodeOutcome.STALLED) {
    pol.reinforce(recent, -2.0, ep.epoch);
  } else {
    pol.reinforce(recent, 0.5, ep.epoch);
  }
  return samples;
}


/** The whole acceptance sequence: 3 episodes on one shared policy, the
 *  flyEpisode cross-check, and the learned brain afterwards. */
function runAll() {
  const instruments = new FlightInstruments(TRIM_SPEED);
  const policy = new ServoPolicy();
  const episodes = [];
  for (const ep of seedDoc.episodes) {
    const snapshot = snapshotPolicy(policy);
    const main = replayEpisode(policy, instruments, ep);
    restorePolicy(policy, snapshot);
    const samples = traceSamples(policy, instruments, ep);
    episodes.push({
      epoch: ep.epoch,
      rng_state: ep.rng_state,
      rng_inc: ep.rng_inc,
      epsilon_override: ep.epsilon_override,
      outcome: main.outcome,
      survival: main.survival,
      choices: main.choices,
      final_state_key: main.finalStateKey,
      samples,
    });
  }
  // The real flyEpisode API on a fresh policy must reproduce episode 0.
  const ep0 = seedDoc.episodes[0];
  const fresh = new ServoPolicy();
  const flyRng = new NumpyPcg64(BigInt(ep0.rng_state), BigInt(ep0.rng_inc));
  const fly = flyEpisode(fresh, instruments, ep0.epoch, flyRng);
  const flyEpisode0 = {
    outcome: fly.outcome,
    survival: fly.survival,
    final_state_key: fly.finalStateKey,
  };
  const consistent = flyEpisode0.outcome === episodes[0].outcome
    && flyEpisode0.survival === episodes[0].survival
    && JSON.stringify(flyEpisode0.final_state_key) === JSON.stringify(episodes[0].final_state_key);
  if (!consistent) {
    console.error("INCONSISTENT: flyEpisode does not reproduce the replayed episode 0");
    process.exit(1);
  }
  const firstEntries = [];
  for (const [index, slots] of policy.servoNeuron.values) {
    if (firstEntries.length >= 5) {
      break;
    }
    firstEntries.push({
      key: index,
      values: Array.from(slots),
      epochs: Array.from(policy.servoNeuron.epochs.get(index)),
    });
  }
  return {
    episodes,
    fly_episode_0: flyEpisode0,
    brain_after: {
      goal_keys: policy.goalNeuron.storedCombinations,
      flight_keys: policy.flightNeuron.storedCombinations,
      engine_keys: policy.engineNeuron.storedCombinations,
      controls_keys: policy.controlsNeuron.storedCombinations,
      servo_keys: policy.servoNeuron.storedCombinations,
      first_entries: firstEntries,
    },
  };
}

const a = runAll();
const b = runAll();
if (JSON.stringify(a) !== JSON.stringify(b)) {
  console.error("NOT DETERMINISTIC: two full runs differ");
  process.exit(1);
}
for (const ep of a.episodes) {
  console.log(`episode ${ep.epoch} (eps ${ep.epsilon_override ?? "default"}): ` +
              `${ep.outcome}, survival ${ep.survival}, ${ep.choices.length} choices, ` +
              `${ep.samples.length} samples ✓`);
}
console.log(`brain after: ${a.brain_after.servo_keys} servo keys, ` +
    `${a.brain_after.goal_keys} goal, ${a.brain_after.flight_keys} flight, ` +
    `${a.brain_after.engine_keys} engine, ${a.brain_after.controls_keys} controls`);

const doc = {
  schema: "c172-episodes-frozen/1",
  note: "TS-frozen learning-episode reference: three lives of one shared " +
        "policy on the 1111 kg battle-C172 airframe (body-rate quaternion " +
        "propagation, ground-effect glue, agl-aware step). The scenario " +
        "seeds (rng state/inc, epochs, epsilon overrides) are the frozen " +
        "Python inputs; outcomes, choices, trajectories and the learned " +
        "brain supersedes the Python flight-ref.json, which is the " +
        "historical record of the old 1.5 kg airframe.",
  ...a,
};
writeFileSync(new URL("../reference/c172-episodes-frozen.json", import.meta.url),
              JSON.stringify(doc, null, 1) + "\n");
console.log("wrote reference/c172-episodes-frozen.json (bit-exact across two runs)");
