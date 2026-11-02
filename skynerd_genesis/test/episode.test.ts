/**
 * Full episode replay vs the Python reference (reference/flight-ref.json):
 * three lives of one learned policy, blank start. This is the Milestone 3
 * acceptance gate: comparable trajectories and outcomes under controlled
 * seeds — here "comparable" means exact.
 *
 * The replay below mirrors the Python tracer in golden/flight_ref.py (which
 * asserts its own faithfulness against the real fly_episode). Exact gates:
 * outcome, survival tick, every decision choice, final state key, and the
 * learned tensor afterwards. Trajectory samples at 1e-6 (libm ulps).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { Vec3 } from "../src/core/vec.js";
import { World } from "../src/core/world.js";
import { NumpyPcg64 } from "../src/core/rng.js";
import { RigidBody } from "../src/physics/rigid-body.js";
import { SurfacesState } from "../src/physics/surfaces.js";
import {
  DenseReward, EpisodeOutcome, FlightInstruments, classifyOutcome,
} from "../src/learning/instruments.js";
import { ServoPolicy, HOLD, PUSH } from "../src/learning/servo-tensor.js";
import {
  TRIM_COMMAND, TRIM_SPEED, flyEpisode, springTowardNeutral,
} from "../src/learning/flight.js";
import { check, finish, section } from "./harness.js";

const refPath = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "reference", "flight-ref.json");
const ref = JSON.parse(readFileSync(refPath, "utf8"));

const DT = 1.0 / 50.0;
const TRAJ_TOL = 1e-6;

function close(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol + 1e-12 * Math.abs(b);
}

/** The Python tracer's choice encoding: 0 = hold-all, else 1 + surface*2 + push. */
function choiceOf(actions: Int32Array): number {
  for (let surface = 0; surface < 4; surface++) {
    if (actions[surface] !== HOLD) {
      return 1 + surface * 2 + (actions[surface] === PUSH ? 1 : 0);
    }
  }
  return 0;
}

const instruments = new FlightInstruments(TRIM_SPEED);
const policy = new ServoPolicy();

interface PolicySnapshot {
  values: Map<number, Float32Array>;
  epochs: Map<number, Int32Array>;
  rewardTotal: number;
  epsilon: number;
}

function snapshotPolicy(p: ServoPolicy): PolicySnapshot {
  return {
    values: new Map([...p.tensor.values].map(([k, v]) => [k, Float32Array.from(v)])),
    epochs: new Map([...p.tensor.epochs].map(([k, e]) => [k, Int32Array.from(e)])),
    rewardTotal: p.tensor.rewardTotal,
    epsilon: p.epsilon,
  };
}

function restorePolicy(p: ServoPolicy, snap: PolicySnapshot): void {
  p.tensor.values.clear();
  for (const [k, v] of snap.values) {
    p.tensor.values.set(k, v);
  }
  p.tensor.epochs.clear();
  for (const [k, e] of snap.epochs) {
    p.tensor.epochs.set(k, e);
  }
  p.tensor.rewardTotal = snap.rewardTotal;
  p.epsilon = snap.epsilon;
}

ref.episodes.forEach((ep: any) => {
  section(`episode ${ep.epoch} seed ${1007 + ep.epoch}`);
  {
    const snapshot = snapshotPolicy(policy);
    const rng = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
    const world = new World({ halfExtent: 800.0 });
    const body = new RigidBody(new Vec3(0.0, 0.0, 50.0), new Vec3(TRIM_SPEED, 0.0, 0.0));
    const servos = new SurfacesState();
    const dense = new DenseReward(instruments, 5.0);
    dense.reset(body);
    let commands = Float64Array.from(TRIM_COMMAND);
    const recent: Array<[number[], Int32Array]> = [];
    let pending = 0.0;
    const choices: number[] = [];
    let outcome: string | null = null;
    if (ep.epsilon_override !== null) {
      policy.epsilon = ep.epsilon_override;
    }
    let tick = 0;
    for (let t = 0; t < 1000; t++) {
      tick = t;
      if (tick % 10 === 0) {
        const timeEpoch = tick + ep.epoch * 100000;
        if (recent.length > 0) {
          policy.reinforce(recent, pending, timeEpoch);
        }
        pending = 0.0;
        const stateKey = instruments.stateKey(body, commands[3]);
        const actions = policy.choose(stateKey, rng);
        commands = policy.commandFrom(commands, actions);
        recent.push([stateKey, actions]);
        if (recent.length > 10) {
          recent.shift();
        }
        choices.push(choiceOf(actions));
      }
      servos.command(commands, DT);
      body.step(servos, DT);
      commands = springTowardNeutral(commands, DT);
      pending += dense.tick(body) + 0.0005;
      outcome = classifyOutcome(body, world, (tick + 1) * DT);
      if (outcome !== null) {
        break;
      }
    }
    // Terminal credit, exactly like the real fly_episode: the traced replay
    // drives the same shared policy the Python reference drove.
    const timeEpoch = tick + ep.epoch * 100000;
    if (outcome === EpisodeOutcome.CRASHED || outcome === EpisodeOutcome.STALLED) {
      policy.reinforce(recent, -2.0, timeEpoch);
    } else {
      policy.reinforce(recent, 0.5, timeEpoch);
    }
    if (outcome === null) {
      outcome = EpisodeOutcome.ALIVE;
    }

    check("outcome exact", outcome === ep.outcome, `got ${outcome} want ${ep.outcome}`);
    check("survival exact", close((tick + 1) * DT, ep.survival, 1e-12),
          `got ${(tick + 1) * DT} want ${ep.survival}`);
    check("decision count", choices.length === ep.choices.length,
          `got ${choices.length} want ${ep.choices.length}`);
    let choicesOk = true;
    for (let i = 0; i < Math.min(choices.length, ep.choices.length); i++) {
      if (choices[i] !== ep.choices[i]) {
        choicesOk = false;
        check(`choice[${i}]`, false, `got ${choices[i]} want ${ep.choices[i]}`);
        break;
      }
    }
    check("every decision choice exact", choicesOk);

    // Trajectory: restore the pre-episode policy state and replay identically
    // with sampling enabled (same trick as the Python tracer).
    restorePolicy(policy, snapshot);
    const rng2 = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
    const samples = traceSamples(policy, instruments, ep.epoch, rng2,
                                 ep.epsilon_override);
    let samplesOk = true;
    const failures: string[] = [];
    for (let i = 0; i < Math.min(samples.length, ep.samples.length); i++) {
      const got = samples[i];
      const want = ep.samples[i];
      const fields: Array<[readonly number[], readonly number[], string]> = [
        [got.pos, want.pos, "pos"], [got.vel, want.vel, "vel"],
        [got.quat, want.quat, "quat"], [got.rates, want.rates, "rates"],
        [got.surfaces, want.surfaces, "surfaces"], [got.diag, want.diag, "diag"],
      ];
      for (const [g, w, name] of fields) {
        for (let j = 0; j < w.length; j++) {
          if (!close(g[j], w[j], TRAJ_TOL)) {
            samplesOk = false;
            failures.push(`sample[${i}].${name}[${j}]: got ${g[j]} want ${w[j]}`);
          }
        }
      }
    }
    check("trajectory samples match", samplesOk, failures.slice(0, 2).join("; "));
  }
});

/** Second identical replay that records trajectory samples. */
function traceSamples(pol: ServoPolicy, inst: FlightInstruments, epoch: number,
                      rng: NumpyPcg64, epsilonOverride: number | null) {
  const world = new World({ halfExtent: 800.0 });
  const body = new RigidBody(new Vec3(0.0, 0.0, 50.0), new Vec3(TRIM_SPEED, 0.0, 0.0));
  const servos = new SurfacesState();
  const dense = new DenseReward(inst, 5.0);
  dense.reset(body);
  let commands = Float64Array.from(TRIM_COMMAND);
  const recent: Array<[number[], Int32Array]> = [];
  let pending = 0.0;
  const samples: Array<{ pos: number[]; vel: number[]; quat: number[]; rates: number[];
                         surfaces: number[]; diag: number[] }> = [];
  if (epsilonOverride !== null) {
    pol.epsilon = epsilonOverride;
  }
  let outcome: string | null = null;
  let tick = 0;
  for (let t = 0; t < 1000; t++) {
    tick = t;
    if (tick % 10 === 0) {
      const timeEpoch = tick + epoch * 100000;
      if (recent.length > 0) {
        pol.reinforce(recent, pending, timeEpoch);
      }
      pending = 0.0;
      const stateKey = inst.stateKey(body, commands[3]);
      const actions = pol.choose(stateKey, rng);
      commands = pol.commandFrom(commands, actions);
      recent.push([stateKey, actions]);
      if (recent.length > 10) {
        recent.shift();
      }
    }
    servos.command(commands, DT);
    const diag = body.step(servos, DT);
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
  // Same terminal credit as the main replay so the shared policy's state
  // stays identical to the Python sequence.
  const timeEpoch = tick + epoch * 100000;
  if (outcome === EpisodeOutcome.CRASHED || outcome === EpisodeOutcome.STALLED) {
    pol.reinforce(recent, -2.0, timeEpoch);
  } else {
    pol.reinforce(recent, 0.5, timeEpoch);
  }
  return samples;
}

// The real flyEpisode API on a fresh policy must reproduce episode 0 exactly.
{
  section("flyEpisode API vs golden episode 0");
  const ep = ref.episodes[0];
  const fresh = new ServoPolicy();
  const rng = new NumpyPcg64(BigInt(ep.rng_state), BigInt(ep.rng_inc));
  const result = flyEpisode(fresh, instruments, ep.epoch, rng);
  check("outcome", result.outcome === ep.outcome,
        `got ${result.outcome} want ${ep.outcome}`);
  check("survival", close(result.survival, ep.survival, 1e-12),
        `got ${result.survival} want ${ep.survival}`);
  check("final state key",
        result.finalStateKey.every((v, i) => v === ep.final_state_key[i]));
}

// Tensor comparison after the three episodes.
{
  section("learned tensor after three episodes");
  const want = ref.tensor_after;
  check("key count", policy.tensor.storedCombinations === want.keys,
        `got ${policy.tensor.storedCombinations} want ${want.keys}`);
  check("reward total", close(policy.tensor.rewardTotal, want.reward_total, 1e-9),
        `got ${policy.tensor.rewardTotal} want ${want.reward_total}`);
  const shape = [8, 8, 8, 8, 8, 8, 4];
  const strides = [8 * 8 * 8 * 8 * 8 * 4, 8 * 8 * 8 * 8 * 4, 8 * 8 * 8 * 4,
                   8 * 8 * 4, 8 * 4, 4, 1];
  const indexOf = (parts: readonly number[]) =>
    parts.reduce((acc, p, i) => acc + p * strides[i], 0);
  let entriesOk = true;
  const firstEntries = want.first_entries as Array<{ key: number[]; values: number[]; epochs: number[] }>;
  for (const entry of firstEntries) {
    const index = indexOf(entry.key);
    const slots = policy.tensor.values.get(index);
    const slotEpochs = policy.tensor.epochs.get(index);
    if (slots === undefined || slotEpochs === undefined) {
      entriesOk = false;
      check(`entry ${entry.key} present`, false, "missing in TS tensor");
      break;
    }
    for (let s = 0; s < 8; s++) {
      if (slots[s] !== entry.values[s] || slotEpochs[s] !== entry.epochs[s]) {
        entriesOk = false;
        check(`entry ${entry.key} slot ${s}`, false,
              `got ${slots[s]}@${slotEpochs[s]} want ${entry.values[s]}@${entry.epochs[s]}`);
        break;
      }
    }
    if (!entriesOk) break;
  }
  check("first entries exact (float32 slots + epochs)", entriesOk);
  void shape;
}

finish("episode.test");
