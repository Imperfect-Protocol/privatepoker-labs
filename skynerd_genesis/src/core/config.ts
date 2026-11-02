/**
 * Central configuration for the Skynerd simulator. Ported from
 * `skynerd/config.py` (Genesis Milestone 1).
 *
 * One frozen object owns every knob. Modules receive the config; they do not
 * read module-level globals, so experiments can run several configs in one
 * process.
 *
 * Pipeline switches mirror the p2p_nerual_spatial ablation flags:
 * `--no-learning`, `--no-noise`, `--no-haar`, `--no-gaussian`.
 */

export interface Config {
  // --- world and randomness -------------------------------------------------
  readonly seed: number;
  readonly episodeSeconds: number;      // hard stop for one drone episode
  readonly worldHalfExtent: number;     // metres; flat square world, centred at origin

  // --- clocks ---------------------------------------------------------------
  // The event loop (comms, slow tensors) and the physics tick are separate
  // clocks. Physics MUST be a fixed dt; servo dynamics are meaningless
  // at event-loop timescales.
  readonly physicsHz: number;           // fixed 6-DOF tick
  readonly motionHz: number;            // neighbour-set recomputation / swarm motion
  readonly visionHz: number;            // radar raycast + SwarmEye ring-buffer push

  // --- radio ---------------------------------------------------------------
  readonly radioRange: number;          // metres, hard link boundary
  readonly radioDropExponent: number;   // drop probability grows with (dist/range)^exp
  readonly radioDelay: number;          // seconds, constant simulated air delay

  // --- stun / coalescing / propagation (ported semantics) -------------------
  readonly initialTtl: number;
  readonly stunK: number;               // sigmoid steepness in stun delay
  readonly stunMidpoint: number;        // signals in cluster at the sigmoid midpoint
  readonly beaconInterval: number;

  // --- learning tensors -----------------------------------------------------
  readonly angleBins: number;
  readonly hopBins: number;
  readonly timeDepth: number;
  readonly hebbianBaseline: number;
  readonly hebbianMaxWeight: number;
  readonly hebbianReward: number;       // slow success reward (route + tactics)
  readonly beaconReward: number;
  readonly deathRewardScale: number;    // anti-Hebbian arm multiplier
  readonly mirrorHopDecay: number;      // witnessed-reward decay per relay hop
  readonly stationMergeRate: number;    // slow consolidation against forgetting

  // --- servo learning -------------------------------------------------------
  readonly servoStateBins: number;
  readonly servoSurfaces: number;       // aileron, elevator, rudder, throttle
  readonly servoRewardScale: number;    // dense per-tick reward scale

  // --- vision ----------------------------------------------------------------
  readonly eyeBins: number;             // azimuth x elevation x range
  readonly eyeMaxSources: number;       // Source(M) cap on the glued SwarmEye
  readonly eyeRangeMax: number;         // metres; range bins are log-spaced to here

  // --- wavelet / chaos pipeline ----------------------------------------------
  readonly noiseStrength: number;
  readonly gaussianWidth: number;
  readonly messageThreshold: number;
  readonly nodeThreshold: number;

  // --- ablation switches -------------------------------------------------------
  readonly noLearning: boolean;
  readonly noNoise: boolean;
  readonly noHaar: boolean;
  readonly noGaussian: boolean;
  readonly noCompression: boolean;
  readonly noTtl: boolean;
  readonly noTwist: boolean;

  // --- waves -------------------------------------------------------------------
  readonly waveSize: number;            // drones per row; carpet rows are wave_size wide
  readonly waveRows: number;            // rows per wave (the "carpet")
  readonly waveGap: number;             // metres between carpet rows
  readonly rowSpacing: number;          // metres between drones in a row
}

/** The Python defaults, frozen. `config.replace({...})` derives variants. */
export const DEFAULT_CONFIG: Config = Object.freeze({
  seed: 42,
  episodeSeconds: 60.0,
  worldHalfExtent: 1000.0,

  physicsHz: 50.0,
  motionHz: 10.0,
  visionHz: 2.0,

  radioRange: 300.0,
  radioDropExponent: 2.0,
  radioDelay: 0.02,

  initialTtl: 8,
  stunK: 0.1,
  stunMidpoint: 20.0,
  beaconInterval: 1.0,

  angleBins: 8,
  hopBins: 8,
  timeDepth: 8,
  hebbianBaseline: 0.1,
  hebbianMaxWeight: 5.0,
  hebbianReward: 0.35,
  beaconReward: 0.12,
  deathRewardScale: 1.5,
  mirrorHopDecay: 0.5,
  stationMergeRate: 0.01,

  servoStateBins: 8,
  servoSurfaces: 4,
  servoRewardScale: 0.1,

  eyeBins: 8,
  eyeMaxSources: 8,
  eyeRangeMax: 500.0,

  noiseStrength: 0.00001,
  gaussianWidth: 0.30,
  messageThreshold: 0.006,
  nodeThreshold: 0.00005,

  noLearning: false,
  noNoise: false,
  noHaar: false,
  noGaussian: false,
  noCompression: false,
  noTtl: false,
  noTwist: false,

  waveSize: 8,
  waveRows: 3,
  waveGap: 60.0,
  rowSpacing: 40.0,
});

/** Quantities computed once from a Config. Owned by the config, not scattered. */
export interface Derived {
  readonly config: Config;
  readonly physicsDt: number;
  readonly motionDt: number;
  readonly visionDt: number;
}

export function derived(config: Config = DEFAULT_CONFIG): Derived {
  return Object.freeze({
    config,
    physicsDt: 1.0 / config.physicsHz,
    motionDt: 1.0 / config.motionHz,
    visionDt: 1.0 / config.visionHz,
  });
}

/** Derive a variant config (the `dataclasses.replace` analogue). */
export function replace(config: Config, changes: Partial<Config>): Config {
  return Object.freeze({ ...config, ...changes });
}

/**
 * The clock contract for the whole simulation (see golden-baseline.json).
 * NOTE: the Python `fly_episode` currently hardcodes dt = 1/50 and horizon =
 * 20 s inside flight.py; the Config values agree with it today, and the port
 * scheduler must treat these as the agreed clocks either way.
 */
export const CLOCKS = Object.freeze({
  physicsHz: 50.0,
  physicsDt: 0.02,
  decisionWindowS: 0.2,   // every 10 physics ticks
  motionHz: 10.0,
  visionHz: 2.0,
  episodeHorizonS: 20.0,
});
