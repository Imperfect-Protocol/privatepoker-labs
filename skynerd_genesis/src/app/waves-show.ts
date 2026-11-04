/**
 * The drone attack, LIVE: the swarm's sortie as a conveyor of PAIRS.
 *
 * Drones do NOT appear from nowhere: every PAIR LIFTS OFF FROM THE
 * RUNWAY — two drones abreast — and the next pair rolls five seconds
 * behind it, pair after pair, forever, no waves. Each drone flies the
 * attack mission (learning/mission.ts): the orbit climb to 3000 ft,
 * the split onto HSI ±45° (the left seat the left flank, the right
 * seat the right flank), the level moment in the flank, the ATTACK —
 * the converging steep descending turns down to 500 ft onto the tank
 * below — the rally's climbing turns up to 1500 ft, the spin own side
 * and recover, and then THE WILL DECIDES: the drone's own brain takes
 * the controls in free flight for whatever its tensor memory wants
 * next. The repetition of the scenario is the conveyor itself.
 *
 * The flock has a SINGULARITY BRAIN: besides each drone's own
 * tensors, the swarm maintains one shared SINGULARITY TENSOR
 * (learning/singularity.ts), updated every round from every drone any
 * other drone can reach within two hops of the live mesh, and shared
 * back — and its SUPER-GOAL (the mission leg the plurality flies)
 * keeps the whole carpet in formation: a drone that falls behind is
 * pulled forward onto the formation's leg at the next safe boundary.
 * When one dies its memory shard leaves the flock and it screams its
 * tensor to whoever is still connected enough to hear.
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
import { TRIM_SPEED } from "../learning/flight.js";
import { MissionFlight, MISSION_AREA_M, type MissionToken } from "../learning/mission.js";
import { SingularityTensor, type SingularityMember } from "../learning/singularity.js";
import { SYLLABUS } from "../learning/school.js";
import { ServoPolicy, type BrainSnapshot, type TensorStateView } from "../learning/servo-tensor.js";
import { Scene, type SceneAircraft } from "./scene.js";
import {
  hudDelta, readoutOf, meanReadoutOf, type HudDelta,
} from "./hud-delta.js";
import type { ControlMode } from "../learning/guardian.js";

/** The sky holds at most this many living drones: a new pair lifts off
 * only when the sky has room for both of its seats (40 alive —
 * nothing lifts off; 39 — the pair waits for a drone to die). */
export const DRONE_CAP = 40;
export const PAIR_EPSILON = 0.05;
/** The two seats of one pair, metres apart on the runway. */
const PAIR_GAP = 40.0;
/** Successive pairs in trail down the runway, metres — the pairs take
 *  off one behind the other, five seconds apart. */
const PAIR_TRAIL = 90.0;
/** The pair launch cadence, seconds — the first pair lifts off at
 *  once, the next rolls five seconds behind it, pair after pair. */
export const PAIR_LAUNCH_EVERY = 5.0;
/** The SingularityTensor's update round, ticks (1 s): the swarm speaks
 *  into the shared vessel, the vessel shares back, and the super-goal
 *  pulls whoever fell behind. */
const SINGULARITY_ROUND_TICKS = 50;
/** Drones connect to every other drone within this range, metres —
 *  the swarm's mesh, every nearby pair. */
export const LINK_RANGE = 180.0;
const TRAIL_CAP = 1200;
const DT = 1.0 / 50.0;

/** One drone of the conveyor: its pair, seat, brain, machine, life. */
interface ConveyorDrone {
  readonly pair: number;
  readonly policy: ServoPolicy;
  readonly flight: MissionFlight;
  readonly entry: SceneAircraft;
}

/** The swarm's connections right now: every live pair within range. */
export interface SwarmLink {
  a: SceneAircraft;
  b: SceneAircraft;
}

/** The drone attack, live — see the file header. */
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
  private nextPairAt = 0.0;
  /** The next pair's number (0-based; the first pair is 0). */
  private nextPair = 0;
  private readonly drones: ConveyorDrone[] = [];
  /** THE SINGULARITY BRAIN — the one shared tensor the flock maintains
   *  together, and its super-goal over the formation. */
  private readonly singularity = new SingularityTensor();
  private roundTick = 0;
  /** Whose hands fly every NEW drone from the runway — the scenario's
   *  default is the PAIR: the instructor flies, the AI suggests and
   *  learns (the controls dropdown overrides it per drone, sticky). */
  readonly defaultControlsMode: ControlMode = "both-suggest";

  constructor(seed: { state: string; inc: string },
              brain?: ServoPolicy) {
    this.rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    this.brainSnapshot = brain === undefined ? null : brain.snapshot();
    this.launchPair();
    this.nextPairAt = PAIR_LAUNCH_EVERY;
  }

  /** How many pairs have lifted off so far. */
  get pairCount(): number {
    return this.nextPair;
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

  /** THE SUPER-GOAL — the mission leg the formation's plurality flies
   *  (the singularity's shared intent, the UI's status line). */
  get singularityGoal(): MissionToken {
    return this.singularity.superGoal;
  }

  /** The singularity's completed update rounds. */
  get singularityRounds(): number {
    return this.singularity.rounds;
  }

  /** THE AI STATE LOG's subject: the selected drone's tensor network
   *  view (whichever aircraft the panel shows; null while none fly). */
  get selectedStateView(): TensorStateView | null {
    const current = this.scene.current;
    if (current === null) {
      return null;
    }
    const drone = this.drones.find((d) => d.entry === current
                                    && d.flight.episodeOutcome === null);
    return drone === undefined ? null : drone.flight.stateView();
  }

  /** THE CONTROLS DROPDOWN — whose hands fly the SELECTED drone. The
   *  choice is STICKY per drone: switching the aircraft dropdown to
   *  another drone leaves this one flying under whoever was chosen
   *  for it (and its own mode shows when it is selected again). */
  setControlsMode(mode: ControlMode): void {
    const current = this.scene.current;
    if (current === null) {
      return;
    }
    const drone = this.drones.find((d) => d.entry === current
                                    && d.flight.episodeOutcome === null);
    if (drone !== undefined) {
      drone.flight.setControlsMode(mode);
    }
  }

  /** The selected drone's controls mode (null when it is not a flying
   *  drone — the dropdown shows nothing to change). */
  get selectedControlsMode(): ControlMode | null {
    const current = this.scene.current;
    if (current === null) {
      return null;
    }
    const drone = this.drones.find((d) => d.entry === current
                                    && d.flight.episodeOutcome === null);
    return drone === undefined ? null : drone.flight.controlsMode;
  }

  /** Whether the instructor's hand is on the selected drone's levers
   *  right now (a guarded takeover included) — the status line. */
  get selectedInstructorAtControls(): boolean {
    const current = this.scene.current;
    if (current === null) {
      return false;
    }
    const drone = this.drones.find((d) => d.entry === current
                                    && d.flight.episodeOutcome === null);
    return drone !== undefined && drone.flight.instructorAtControls;
  }

  /** The AI log's delta column: the selected drone against the MEAN of
   *  the drone carpet (the swarm's centre of mass, in pilot's units;
   *  empty when the selected aircraft is not a flying drone). */
  get carpetDelta(): HudDelta | null {
    const current = this.scene.current;
    if (current === null) {
      return null;
    }
    const drone = this.drones.find((d) => d.entry === current
                                    && d.flight.episodeOutcome === null);
    if (drone === undefined || this.aliveCount < 2) {
      return null;
    }
    const carpet = meanReadoutOf(this.drones
      .filter((d) => d.flight.episodeOutcome === null)
      .map((d) => d.entry.aircraft));
    if (carpet === null) {
      return null;
    }
    return hudDelta(readoutOf(drone.entry.aircraft), carpet);
  }

  /** The wiki tab's scenario description (the waves). */
  get wiki(): {
    title: string; goal: string;
    elements: readonly string[]; goals: readonly string[];
  } {
    return {
      title: "drone attack waves",
      goal: "pairs lift off from the runway five seconds apart, fly " +
            "the attack mission — the split onto the flanks, the level " +
            "moment at 3000 ft, the converging steep descending attack " +
            "turns down to 500 ft, the rally's climbing turns up to " +
            "1500 ft, the spin own side and recover — and then the " +
            "will decides: the drone's own brain takes the controls in " +
            "free flight, in formation with the flock's " +
            "SingularityTensor: the one shared brain, updated from " +
            "every neighbour and every neighbour's neighbour",
      elements: ["roll", "rotate", "climb", "level off",
                "roll in", "turn (the split onto the flank)",
                "cruise (the level moment in the flank)",
                "roll in", "descending turn (the attack)",
                "descend (down to 500 ft)",
                "climbing turn (the rally)", "climb (to 1500 ft)",
                "stall entry", "drop wing", "spin",
                "opposite rudder", "dive",
                "cruise (level, then the will decides)"],
      goals: ["take off from the runway in pairs, five seconds apart",
              "climb to 3000 ft and level off",
              "split: the left column onto HSI-45, the right onto HSI+45",
              "fly level at 3000 ft in the flank",
              "attack: the left flank's steep descending turn right, " +
              "the right flank's steep descending turn left, down to " +
              "500 ft",
              "rally: the climbing turns onto HSI ±45 up to 1500 ft",
              "spin own side and recover",
              "fly level, and let the will decide the next steps",
              "act in formation with the singularity's super-goal",
              "keep the swarm's memory alive (the obituaries heard)"],
    };
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
   * once, the dead scream once, and the next pair rolls onto the
   * runway five seconds behind the last — the conveyor never stops.
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
    // THE SINGULARITY'S ROUND: the flock speaks into the shared tensor,
    // the shared tensor answers, and the super-goal pulls whoever has
    // fallen behind onto the formation's leg.
    this.roundTick += 1;
    if (this.roundTick >= SINGULARITY_ROUND_TICKS) {
      this.roundTick = 0;
      this.runSingularityRound();
    }
    // The cap: a pair lifts off only while the sky has room for BOTH
    // of its seats — 40 alive means nothing new rolls; the moment two
    // drones die, room opens and the next pair launches.
    if (this.simTime >= this.nextPairAt && this.aliveCount + 2 <= DRONE_CAP) {
      this.launchPair();
      this.nextPairAt += PAIR_LAUNCH_EVERY;
    }
  }

  /**
   * ONE SINGULARITY ROUND: the live members and their mesh go in, the
   * shared vessel is maintained from everyone another member can reach
   * within two hops, and the super-goal that comes out is conformed to
   * by every drone of the formation.
   */
  private runSingularityRound(): void {
    const live = this.drones
      .filter((d) => d.flight.episodeOutcome === null);
    if (live.length === 0) {
      return;
    }
    const mesh: Array<[number, number]> = [];
    for (let i = 0; i < live.length; i++) {
      for (let j = i + 1; j < live.length; j++) {
        if (withinRange(live[i].flight.aircraft.body.position,
                        live[j].flight.aircraft.body.position,
                        LINK_RANGE)) {
          mesh.push([i, j]);
        }
      }
    }
    const members: SingularityMember[] = live
      .map((d) => ({ policy: d.policy, token: d.flight.missionToken }));
    this.singularity.updateRound(members, mesh, this.rng);
    for (const drone of live) {
      drone.flight.conform(this.singularity.superGoal);
    }
  }

  /**
   * ONE PAIR LIFTS OFF FROM THE RUNWAY: two drones abreast — the left
   * seat the left flank's column, the right seat the right flank's —
   * in trail behind the pair before, assembled as mosaics of what the
   * living still remember, and each one flying the attack mission from
   * its own takeoff roll. The conveyor never stops: pair after pair,
   * five seconds apart, as long as the sky has room for both seats.
   */
  private launchPair(): void {
    const pair = this.nextPair++;
    for (const flankSign of [-1.0, 1.0]) {
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
      // THE PAIR SEAT: abreast within the pair, in trail down the
      // runway — parked on the gear at 100 ft, waiting for the roll.
      const position = new Vec3(-pair * PAIR_TRAIL,
        flankSign * PAIR_GAP / 2.0, SYLLABUS.runwayM);
      const flight = new MissionFlight(policy, this.instruments, pair,
                                       new NumpyPcg64(state, inc), {
        flankSign,
        startPosition: position,
        epsilonOverride: PAIR_EPSILON,
        mode: this.defaultControlsMode,
        worldHalfExtent: MISSION_AREA_M,
      });
      const entry = this.scene.add({
        label: `Pair ${pair + 1} \u00b7 ${flankSign < 0.0 ? "left" : "right"}`,
        aircraft: flight.aircraft,
        hand: { choices: null, fresh: false },
        trail: [{ x: position.x, y: position.y, z: position.z }],
        flying: true,
        dead: false,
      });
      this.drones.push({ pair, policy, flight, entry });
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

  /** Keep the sky honest: drones whose life ended (crashed or flew out
   *  of the area) leave the scene once their pair is a few pairs behind
   *  the newest — the living fly on, and the scene shows the world as
   *  it is now. Their memory shard lives on in the flock, whatever it
   *  was. */
  private pruneOldCrashSites(): void {
    const oldestKept = this.nextPair - 6;
    for (let i = this.drones.length - 1; i >= 0; i--) {
      const drone = this.drones[i];
      if (drone.pair < oldestKept && drone.flight.episodeOutcome !== null) {
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
