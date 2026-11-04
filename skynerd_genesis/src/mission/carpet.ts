/**
 * Carpet waves: the swarm's generations and its distributed inheritance.
 * Ported 1:1 from `skynerd/mission/carpet.py` (Genesis Milestone 4,
 * will-less path: no school, no will factory).
 *
 * Each wave spawns drones assembled as mosaics of the living flock, they fly
 * their episode, survivors keep their memory shard and stay enrolled, and
 * the dying scream their tensor to whoever hears it. There is no station: the
 * brain lives only in the drones, it thins with every death, and a total
 * wipe means total amnesia. First wave dies mostly blind; later waves are
 * born knowing what the living still remember.
 *
 * LiveWave is the wave as a LIVE, steppable run — one 50 Hz tick per
 * step, the exact frozen call order (same RNG stream, same screams), so
 * driving it headless and flying it tick by tick in the browser are the
 * same flight. WaveRunner.runWave is the headless wrapper.
 */
import { Vec3 } from "../core/vec.js";
import { NumpyPcg64 } from "../core/rng.js";
import { RigidBody } from "../physics/rigid-body.js";
import { Aircraft } from "../physics/aircraft.js";
import { Flock } from "../learning/flock.js";
import { FlightInstruments, EpisodeOutcome } from "../learning/instruments.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import { LiveEpisode, TRIM_SPEED } from "../learning/flight.js";

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

/** One drone of a wave: its brain and its ONE machine (physics/aircraft.ts). */
export interface WaveDrone {
  readonly policy: ServoPolicy;
  readonly aircraft: Aircraft;
  /** The drone's life — null until its slot launches. */
  flight: LiveEpisode | null;
  /** The launch lottery's ticket for this drone. */
  hazard: boolean;
  /** How its life ended (null while queued or flying). */
  outcome: string | null;
}

export interface LiveWaveOptions {
  waveSize: number;
  rowSpacing: number;
  hazardRate: number;
  epsilon: number;
}

/** One wave as a live, steppable run — see the file header. */
export class LiveWave {
  readonly drones: WaveDrone[] = [];
  readonly survivals: number[] = [];
  readonly spiralSurvivals: number[] = [];
  private deaths = 0;
  /** The slot currently flying (=== drones.length once the wave is over). */
  private slot = 0;

  constructor(private readonly flock: Flock,
              private readonly instruments: FlightInstruments,
              private readonly waveIndex: number,
              private readonly rng: NumpyPcg64,
              private readonly options: LiveWaveOptions) {
    // A carpet launches together: the whole wave is born at once, each
    // newborn a mosaic of what the PREVIOUS waves' survivors remember,
    // then joins the living. Within the wave, knowledge moves only by
    // screams.
    const newborns: ServoPolicy[] = [];
    for (let i = 0; i < this.options.waveSize; i++) {
      newborns.push(new ServoPolicy());
    }
    for (const policy of newborns) {
      this.flock.seedNewborn(policy, rng);
    }
    for (const policy of newborns) {
      this.flock.enroll(policy);
    }
    for (let slot = 0; slot < newborns.length; slot++) {
      const position = new Vec3(0.0,
        (slot - this.options.waveSize / 2.0) * this.options.rowSpacing, 50.0);
      this.drones.push({
        policy: newborns[slot],
        aircraft: new Aircraft(new RigidBody(position,
                                             new Vec3(TRIM_SPEED, 0.0, 0.0))),
        flight: null,
        hazard: false,
        outcome: null,
      });
    }
    this.launch(0);
  }

  get deathsCount(): number {
    return this.deaths;
  }

  get finished(): boolean {
    return this.slot >= this.drones.length;
  }

  /** The drone flying right now (null once the wave is over). */
  get flying(): WaveDrone | null {
    return this.drones[this.slot] ?? null;
  }

  /**
   * One 50 Hz tick of the drone currently flying. Returns false once the
   * whole wave has flown.
   */
  step(): boolean {
    if (this.finished) {
      return false;
    }
    const drone = this.drones[this.slot];
    if (drone.flight!.step()) {
      return true;
    }
    // Its life ended: record it, and the dying scream to whoever hears.
    drone.outcome = drone.flight!.episodeOutcome;
    this.survivals.push(drone.flight!.survival);
    if (drone.hazard) {
      this.spiralSurvivals.push(drone.flight!.survival);
    }
    const died = drone.outcome === EpisodeOutcome.CRASHED
      || drone.outcome === EpisodeOutcome.STALLED
      || drone.outcome === EpisodeOutcome.OUT_OF_BOUNDS;
    if (died) {
      // The dying drone's shard leaves the brain, then it screams its
      // tensor to whoever is still in the sky to hear it.
      this.deaths += 1;
      this.flock.reap(drone.policy);
      this.flock.broadcastObituary(drone.policy, this.rng);
    }
    // A survivor keeps its shard; it stays enrolled.
    this.slot += 1;
    if (this.finished) {
      return false;
    }
    this.launch(this.slot);
    return true;
  }

  /** Launch one drone's life: the launch lottery, then its episode. */
  private launch(slot: number): void {
    const drone = this.drones[slot];
    drone.hazard = this.rng.uniform() < this.options.hazardRate;
    drone.flight = new LiveEpisode(drone.policy, this.instruments,
                                   this.waveIndex, this.rng, {
      aircraft: drone.aircraft,
      epsilonOverride: this.options.epsilon,
      startSpiral: drone.hazard,
    });
  }
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

  /** Fly one whole wave headless (the library fast path). */
  runWave(waveIndex: number, rng: NumpyPcg64, epsilon: number): WaveMetrics {
    const wave = new LiveWave(this.flock, this.instruments, waveIndex, rng, {
      waveSize: this.waveSize,
      rowSpacing: this.rowSpacing,
      hazardRate: this.hazardRate,
      epsilon,
    });
    while (wave.step()) { /* the wave flies */ }
    const metrics: WaveMetrics = {
      wave: waveIndex,
      meanSurvival: wave.survivals.reduce((a, b) => a + b, 0.0)
                     / wave.survivals.length,
      bestSurvival: Math.max(...wave.survivals),
      deaths: wave.deathsCount,
      epsilon,
      survivals: wave.survivals,
      spiralSurvivals: wave.spiralSurvivals,
      flockSize: this.flock.size,
    };
    this.history.push(metrics);
    return metrics;
  }
}
