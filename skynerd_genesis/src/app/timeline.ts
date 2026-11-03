/**
 * Wave recording for the browser app: the traced twin of flyEpisode +
 * WaveRunner.runWave. Mirrors the exact code path (same RNG consumption,
 * same reinforcement, same flock bookkeeping) while sampling every physics
 * tick for playback. Validated bit-exact against the Python golden baseline
 * by test/app-trace.test.ts — the waves replay Python's numbers exactly.
 */
import { Vec3 } from "../core/vec.js";
import { World } from "../core/world.js";
import { NumpyPcg64 } from "../core/rng.js";
import { RigidBody } from "../physics/rigid-body.js";
import { SurfacesState } from "../physics/surfaces.js";
import { GRAVITY, RHO_SEA_LEVEL } from "../physics/aero.js";
import {
  DenseReward, EpisodeOutcome, FlightInstruments, classifyOutcome,
} from "../learning/instruments.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import { Flock } from "../learning/flock.js";
import {
  TRIM_COMMAND, TRIM_SPEED, HORIZON, springTowardNeutral,
} from "../learning/flight.js";

export interface DroneSample {
  x: number;
  y: number;
  z: number;
  fwdX: number;
  fwdY: number;
  pitch: number;    // rad, positive nose up
  bank: number;     // rad, positive right wing down
  speed: number;    // m/s
  vs: number;       // m/s, vertical speed
  velX: number;     // velocity components, m/s (drive the flight path marker)
  velY: number;
  throttle: number; // 0..1 commanded
  ail: number;      // aileron position, rad (physical, slewed)
  elev: number;     // elevator position, rad
  rud: number;      // rudder position, rad
}

export interface EpisodeTrace {
  samples: DroneSample[];
  survival: number;
  outcome: string;
  deathX: number;
  deathY: number;
}

export interface WaveTrace {
  wave: number;
  episodes: EpisodeTrace[];
  survivals: number[];
  meanSurvival: number;
  deaths: number;
  flockSize: number;
}

export interface FlockStats {
  births: number;
  obituariesHeard: number;
}

export interface SeedRef {
  state: string;   // PCG64 state, decimal string (from reference/rng-ref.json)
  inc: string;
}

const DT = 1.0 / 50.0;

/** Traced twin of flyEpisode (will-less path, epsilon override). */
function tracedEpisode(policy: ServoPolicy, instruments: FlightInstruments,
                       epoch: number, rng: NumpyPcg64, startPosition: Vec3,
                       epsilon: number): EpisodeTrace {
  const world = new World({ halfExtent: 800.0 });
  const body = new RigidBody(startPosition, new Vec3(TRIM_SPEED, 0.0, 0.0));
  const servos = new SurfacesState();
  const dense = new DenseReward(instruments, 5.0);
  dense.reset(body);
  let commands = Float64Array.from(TRIM_COMMAND);
  const recent: Array<[number[], Int32Array]> = [];
  let pending = 0.0;
  const samples: DroneSample[] = [];
  const steps = Math.floor(HORIZON / DT);
  let outcome: string | null = null;
  let tick = 0;
  policy.epsilon = epsilon;
  for (let t = 0; t < steps; t++) {
    tick = t;
    if (tick % 10 === 0) {
      const timeEpoch = tick + epoch * 100000;
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
    }
    servos.command(commands, DT);
    body.step(servos, DT, GRAVITY, RHO_SEA_LEVEL,
              body.position.z - world.terrainHeight(body.position.x, body.position.y));
    commands = springTowardNeutral(commands, DT);
    pending += dense.tick(body) + 0.0005;
    const [fwd] = body.bodyAxes;
    samples.push({
      x: body.position.x, y: body.position.y, z: body.position.z,
      fwdX: fwd.x, fwdY: fwd.y,
      pitch: body.deckPitch,
      bank: body.bank,
      speed: body.velocity.length(),
      vs: body.velocity.z,
      velX: body.velocity.x,
      velY: body.velocity.y,
      throttle: commands[3],
      ail: servos.positions[0],
      elev: servos.positions[1],
      rud: servos.positions[2],
    });
    outcome = classifyOutcome(body, world, (tick + 1) * DT);
    if (outcome !== null) {
      break;
    }
  }
  const timeEpoch = tick + epoch * 100000;
  if (outcome === EpisodeOutcome.CRASHED || outcome === EpisodeOutcome.STALLED) {
    policy.reinforce(recent, -2.0, timeEpoch);
  } else if (outcome === null) {
    outcome = EpisodeOutcome.ALIVE;
    policy.reinforce(recent, 0.5, timeEpoch);
  }
  const last = samples[samples.length - 1];
  return {
    samples,
    survival: (tick + 1) * DT,
    outcome,
    deathX: outcome === EpisodeOutcome.ALIVE ? NaN : last.x,
    deathY: outcome === EpisodeOutcome.ALIVE ? NaN : last.y,
  };
}

/** Traced twin of WaveRunner.runWave (hazard rate 0, as in run_waves). */
function tracedWave(flock: Flock, instruments: FlightInstruments, waveIndex: number,
                    rng: NumpyPcg64, waveSize: number, rowSpacing: number,
                    epsilon: number): WaveTrace {
  const newborns: ServoPolicy[] = [];
  for (let i = 0; i < waveSize; i++) {
    newborns.push(new ServoPolicy());
  }
  for (const policy of newborns) {
    flock.seedNewborn(policy, rng);
  }
  for (const policy of newborns) {
    flock.enroll(policy);
  }
  const episodes: EpisodeTrace[] = [];
  const survivals: number[] = [];
  let deaths = 0;
  for (let slot = 0; slot < newborns.length; slot++) {
    const policy = newborns[slot];
    const position = new Vec3(0.0, (slot - waveSize / 2.0) * rowSpacing, 50.0);
    const hazard = rng.uniform() < 0.0;   // drawn even at rate 0, like the Python
    const trace = tracedEpisode(policy, instruments, waveIndex, rng, position,
                                epsilon);
    episodes.push(trace);
    survivals.push(trace.survival);
    const died = trace.outcome !== EpisodeOutcome.ALIVE;
    deaths += died ? 1 : 0;
    if (died) {
      flock.reap(policy);
      flock.broadcastObituary(policy, rng);
    }
    void hazard;
  }
  return {
    wave: waveIndex,
    episodes,
    survivals,
    meanSurvival: survivals.reduce((a, b) => a + b, 0.0) / survivals.length,
    deaths,
    flockSize: flock.size,
  };
}

/** Run a whole recorded wave sequence from a frozen PCG64 seed reference. */
export function runTracedWaves(seedRef: SeedRef, waves: number,
                               waveSize = 8, epsilon = 0.05,
                               rowSpacing = 40.0):
    { traces: WaveTrace[]; flock: FlockStats } {
  const rng = new NumpyPcg64(BigInt(seedRef.state), BigInt(seedRef.inc));
  const flock = new Flock();
  const instruments = new FlightInstruments(TRIM_SPEED);
  const traces: WaveTrace[] = [];
  for (let wave = 0; wave < waves; wave++) {
    traces.push(tracedWave(flock, instruments, wave, rng, waveSize,
                           rowSpacing, epsilon));
  }
  return {
    traces,
    flock: { births: flock.births, obituariesHeard: flock.obituariesHeard },
  };
}
