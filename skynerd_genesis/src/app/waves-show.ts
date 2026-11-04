/**
 * The drone attack waves, LIVE: the swarm as a conveyor.
 *
 * No waiting, no playback: the first wave flies while the second wave
 * lifts off right behind it at a distance, and the next behind that —
 * wave after wave, forever, every tick computed as you watch it. A
 * wave is many drones in a line; the line behind the previous wave's
 * line. Drones of neighbouring waves are CONNECTED: every drone links
 * to every nearby drone — same wave or the wave behind — the swarm's
 * mesh, drawn live on the sky. When one dies its memory shard leaves
 * the flock and it screams its tensor to whoever is still connected
 * enough to hear.
 *
 * Every drone flies the ONE Aircraft class (physics/aircraft.ts) — the
 * same machine the flight school flies. The scene holds every drone
 * alive right now plus the fresh crash sites; the aircraft dropdown
 * lists them all and the panel shows the selected one.
 */
import { NumpyPcg64 } from "../core/rng.js";
import { Vec3 } from "../core/vec.js";
import { Flock } from "../learning/flock.js";
import { FlightInstruments, EpisodeOutcome } from "../learning/instruments.js";
import { TRIM_SPEED, LiveEpisode } from "../learning/flight.js";
import { ServoPolicy, type BrainSnapshot } from "../learning/servo-tensor.js";
import { Scene, type SceneAircraft } from "./scene.js";

/** The carpet's shape: 8 drones per line, 5 % exploration, no hazard. */
export const WAVE_SIZE = 8;
/** The sky holds at most this many living drones: a new wave launches
 *  only as many as fit (40 alive — nothing lifts off; 39 — one does). */
export const DRONE_CAP = 40;
export const WAVE_EPSILON = 0.05;
const WAVE_ROW_SPACING = 40.0;
/** A new wave lifts off this often — the DISTANCE between waves, not a
 *  wait: the previous wave is still flying when the next one launches
 *  right behind it. */
export const WAVE_SPAWN_EVERY = 4.0;
/** Drones connect to every other drone within this range, metres —
 *  same wave or neighbouring waves, the swarm's mesh. */
export const LINK_RANGE = 180.0;
const TRAIL_CAP = 1200;
const DT = 1.0 / 50.0;

/** One drone of the conveyor: its generation, seat, brain, machine, life. */
interface ConveyorDrone {
  readonly wave: number;
  readonly slot: number;
  readonly policy: ServoPolicy;
  readonly flight: LiveEpisode;
  readonly entry: SceneAircraft;
}

/** The swarm's connections right now: every live pair within range. */
export interface SwarmLink {
  a: SceneAircraft;
  b: SceneAircraft;
}

/** The drone attack waves, live — see the file header. */
export class WavesShow {
  readonly scene = new Scene();
  private readonly flock = new Flock();
  private readonly instruments = new FlightInstruments(TRIM_SPEED);
  /** The swarm's master dice: flock inheritance, and each new drone's
   *  own stream drawn from it at spawn. */
  private readonly rng: NumpyPcg64;
  /** The boot-trained brain: every newborn drone is transplanted with
   *  these tensors before the flock's living mosaic teaches on top. */
  private readonly brainSnapshot: BrainSnapshot | null;
  private simTime = 0.0;
  private nextWaveAt = 0.0;
  /** The next wave's number (0-based; the first wave is 0). */
  private nextWave = 0;
  private readonly drones: ConveyorDrone[] = [];

  constructor(seed: { state: string; inc: string },
              brain?: ServoPolicy) {
    this.rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    this.brainSnapshot = brain === undefined ? null : brain.snapshot();
    this.launchWave();
    this.nextWaveAt = WAVE_SPAWN_EVERY;
  }

  get waveCount(): number {
    return this.nextWave;
  }

  get simSeconds(): number {
    return this.simTime;
  }

  /** Drones still flying right now. */
  get aliveCount(): number {
    return this.drones.filter((d) => d.flight.episodeOutcome === null).length;
  }

  /** The living flock — the swarm's distributed brain. */
  get flockSize(): number {
    return this.flock.size;
  }

  get births(): number {
    return this.flock.births;
  }
  get obituaries(): number {
    return this.flock.obituariesHeard;
  }

  /** The swarm's mesh: every live pair within LINK_RANGE. */
  links(): SwarmLink[] {
    const links: SwarmLink[] = [];
    for (let i = 0; i < this.drones.length; i++) {
      const a = this.drones[i];
      if (a.flight.episodeOutcome !== null) {
        continue;
      }
      for (let j = i + 1; j < this.drones.length; j++) {
        const b = this.drones[j];
        if (b.flight.episodeOutcome !== null) {
          continue;
        }
        if (withinRange(a.entry.aircraft.body.position,
                        b.entry.aircraft.body.position, LINK_RANGE)) {
          links.push({ a: a.entry, b: b.entry });
        }
      }
    }
    return links;
  }

  /**
   * Advance the swarm by one 50 Hz tick: every drone's life steps at
   * once, the dead scream once, and the next wave lifts off right
   * behind the last at its distance — no waiting.
   */
  step(): void {
    this.simTime += DT;
    for (const drone of this.drones) {
      if (drone.flight.episodeOutcome !== null) {
        continue;                          // already ended: a crash site
      }
      if (drone.flight.step()) {
        this.sampleDrone(drone);
      } else {
        this.droneDied(drone);
      }
    }
    this.pruneOldCrashSites();
    // The cap: a wave lifts off only while the sky has room — 40 alive
    // means nothing new launches; the moment one dies, room opens and
    // the next wave launches (as many slots as fit, no waiting).
    if (this.simTime >= this.nextWaveAt && this.aliveCount < DRONE_CAP) {
      this.launchWave();
      this.nextWaveAt += WAVE_SPAWN_EVERY;
    }
  }
  /** One wave lifts off: a line of new drones — assembled as mosaics of
   *  what the living still remember — right behind the previous wave.
   *  Only as many slots as the sky's cap has room for. */
  private launchWave(): void {
    const wave = this.nextWave++;
    const slots = Math.min(WAVE_SIZE, DRONE_CAP - this.aliveCount);
    for (let slot = 0; slot < slots; slot++) {
      const policy = new ServoPolicy();
      // THE BRAIN TRANSPLANT: the newborn receives the boot-trained
      // tensors first, then the flock's living mosaic teaches on top —
      // every drone starts from what the flight school learned.
      if (this.brainSnapshot !== null) {
        policy.restore(this.brainSnapshot);
      }
      this.flock.seedNewborn(policy, this.rng);
      this.flock.enroll(policy);
      // The drone's own dice, drawn from the swarm's master stream: its
      // exploration is its own, deterministic and replayable.
      const state = this.rng.draw64();
      const inc = this.rng.draw64();
      const position = new Vec3(0.0,
        (slot - WAVE_SIZE / 2.0) * WAVE_ROW_SPACING, 50.0);
      const flight = new LiveEpisode(policy, this.instruments, wave,
                                     new NumpyPcg64(state, inc), {
        startPosition: position,
        epsilonOverride: WAVE_EPSILON,
        horizon: Infinity,   // drones NEVER expire: they fly until they crash
      });
      const entry = this.scene.add({
        label: `Wave ${wave + 1} \u00b7 Drone ${slot + 1}`,
        aircraft: flight.aircraft,
        hand: { choices: null, fresh: false },
        trail: [{ x: position.x, y: position.y, z: position.z }],
        flying: true,
        dead: false,
      });
      this.drones.push({ wave, slot, policy, flight, entry });
    }
  }

  /** A drone's life ended: its shard leaves the brain and it screams
   *  its tensor to whoever is still connected to hear it. */
  private droneDied(drone: ConveyorDrone): void {
    const outcome = drone.flight.episodeOutcome;
    drone.entry.flying = false;
    drone.entry.dead = outcome !== null && died(outcome);
    if (drone.entry.dead) {
      this.flock.reap(drone.policy);
      this.flock.broadcastObituary(drone.policy, this.rng);
    }
  }

  /** Keep the sky honest: drones whose life ended (crashed or survived
   *  their episode) leave the scene once their wave is two behind — the
   *  living fly on, and the scene shows the world as it is now. Their
   *  memory shard lives on in the flock, whatever it was. */
  private pruneOldCrashSites(): void {
    const oldestKept = this.nextWave - 2;
    for (let i = this.drones.length - 1; i >= 0; i--) {
      const drone = this.drones[i];
      if (drone.wave < oldestKept && drone.flight.episodeOutcome !== null) {
        this.scene.remove(drone.entry);
        this.drones.splice(i, 1);
      }
    }
  }
  /** The flying drone's truth onto its scene entry. */
  private sampleDrone(drone: ConveyorDrone): void {
    drone.entry.hand = drone.flight.handInputs;
    const position = drone.entry.aircraft.body.position;
    drone.entry.trail.push({ x: position.x, y: position.y, z: position.z });
    if (drone.entry.trail.length > TRAIL_CAP) {
      drone.entry.trail.shift();
    }
  }
}

/** The outcomes that end a drone dead. */
function died(outcome: string): boolean {
  return outcome === EpisodeOutcome.CRASHED
    || outcome === EpisodeOutcome.STALLED
    || outcome === EpisodeOutcome.OUT_OF_BOUNDS;
}

/** Two drones are connected when they sit within the range. */
function withinRange(a: Vec3, b: Vec3, range: number): boolean {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz <= range * range;
}
