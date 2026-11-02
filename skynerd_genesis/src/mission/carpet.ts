/**
 * Carpet waves: the swarm's generations and its distributed inheritance.
 * Ported 1:1 from `skynerd/mission/carpet.py` (Genesis Milestone 4,
 * will-less path: no school, no will factory).
 *
 * Each wave spawns drones assembled as mosaics of the living flock, they fly
 * their episode, survivors keep their memory shard and stay enrolled, and the
 * dying scream their tensor to whoever hears it. There is no station: the
 * brain lives only in the drones, it thins with every death, and a total
 * wipe means total amnesia. First wave dies mostly blind; later waves are
 * born knowing what the living still remember.
 */
import { Vec3 } from "../core/vec.js";
import { NumpyPcg64 } from "../core/rng.js";
import { Flock } from "../learning/flock.js";
import { FlightInstruments, EpisodeOutcome } from "../learning/instruments.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import { flyEpisode } from "../learning/flight.js";

export interface WaveMetrics {
  wave: number;
  meanSurvival: number;
  bestSurvival: number;
  deaths: number;
  epsilon: number;
  survivals: number[];
  spiralSurvivals: number[];
  flockSize: number;
}

export interface WaveRunnerOptions {
  waveSize?: number;
  rowSpacing?: number;
  spawnRadius?: number;
  hazardRate?: number;
}

export class WaveRunner {
  readonly flock: Flock;
  readonly instruments: FlightInstruments;
  readonly waveSize: number;
  /** The carpet must fit inside the sky: shrink row spacing for wide waves. */
  readonly rowSpacing: number;
  readonly spawnRadius: number;
  readonly hazardRate: number;
  history: WaveMetrics[] = [];          // per-wave metric records

  constructor(flock: Flock, instruments: FlightInstruments,
              options: WaveRunnerOptions = {}) {
    this.flock = flock;
    this.instruments = instruments;
    this.waveSize = options.waveSize ?? 8;
    this.spawnRadius = options.spawnRadius ?? 700.0;
    this.rowSpacing = Math.min(options.rowSpacing ?? 40.0,
                               (2.0 * this.spawnRadius) / this.waveSize);
    this.hazardRate = options.hazardRate ?? 0.0;
  }

  runWave(waveIndex: number, rng: NumpyPcg64, epsilon: number): WaveMetrics {
    const survivals: number[] = [];
    const spiralSurvivals: number[] = [];
    let deaths = 0;
    // A carpet launches together: the whole wave is born at once, each
    // newborn a mosaic of what the PREVIOUS waves' survivors remember, then
    // joins the living. Within the wave, knowledge moves only by screams.
    const newborns: ServoPolicy[] = [];
    for (let i = 0; i < this.waveSize; i++) {
      newborns.push(new ServoPolicy());
    }
    for (const policy of newborns) {
      this.flock.seedNewborn(policy, rng);
    }
    for (const policy of newborns) {
      this.flock.enroll(policy);
    }
    for (let slot = 0; slot < newborns.length; slot++) {
      const policy = newborns[slot];
      const position = new Vec3(0.0, (slot - this.waveSize / 2.0) * this.rowSpacing, 50.0);
      const hazard = rng.uniform() < this.hazardRate;
      const result = flyEpisode(policy, this.instruments, waveIndex, rng, {
        startPosition: position,
        epsilonOverride: epsilon,
        startSpiral: hazard,
      });
      survivals.push(result.survival);
      if (hazard) {
        spiralSurvivals.push(result.survival);
      }
      const died = result.outcome === EpisodeOutcome.CRASHED
        || result.outcome === EpisodeOutcome.STALLED
        || result.outcome === EpisodeOutcome.OUT_OF_BOUNDS;
      deaths += died ? 1 : 0;
      if (died) {
        // The dying drone's shard leaves the brain, then it screams its
        // tensor to whoever is still in the sky to hear it.
        this.flock.reap(policy);
        this.flock.broadcastObituary(policy, rng);
      }
      // A survivor keeps its shard; it stays enrolled.
    }
    const metrics: WaveMetrics = {
      wave: waveIndex,
      meanSurvival: survivals.reduce((a, b) => a + b, 0.0) / survivals.length,
      bestSurvival: Math.max(...survivals),
      deaths,
      epsilon,
      survivals,
      spiralSurvivals,
      flockSize: this.flock.size,
    };
    this.history.push(metrics);
    return metrics;
  }
}
