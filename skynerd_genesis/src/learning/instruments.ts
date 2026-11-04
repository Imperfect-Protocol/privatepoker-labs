/**
 * THE PANEL — the instrument quantization the tensors read.
 *
 * Every gauge is binned at the coarseness of a real needle, in the
 * units a pilot reads them: feet, feet-per-minute, mph, degrees. The
 * binning is the sensor quantization; the situation tensors (servo
 * tensor) encode these binned panels into SITUATION cells.
 *
 * Egocentric error channels ([-1,1] error norms) stay for the DENSE
 * REWARD and the outcome classifier — the tensors read the quantized
 * panel, the reward reads the errors.
 */
import { Vec3 } from "../core/vec.js";
import { World } from "../core/world.js";
import { RigidBody } from "../physics/rigid-body.js";
import {
  AILERON, ELEVATOR, RUDDER, THROTTLE,
} from "../physics/surfaces.js";

export const ERROR_BINS = 8;
const MPH_PER_MPS = 2.23694;
const FPM_PER_MPS = 196.85;    // m/s of climb -> ft/min
const FT_PER_M = 3.28084;
const DEGREES_PER_RAD = 180.0 / Math.PI;

/** Bin a value in [-1, 1] into [0, bins-1]; clamps at the ends. */
export function binError(value: number, bins: number = ERROR_BINS): number {
  const clamped = Math.max(-1.0, Math.min(1.0, value));
  const shifted = (clamped + 1.0) / 2.0;
  const index = Math.floor(shifted * bins);
  return Math.min(bins - 1, Math.max(0, index));
}

/**
 * A quantizer for a GAUGE: the lower edges of every bin above the
 * first, in the gauge's own unit. A value below the first edge lands
 * in bin 0; a value at or above the last edge lands in the final
 * "and beyond" bin.
 */
export class GaugeBins {
  readonly count: number;
  private readonly edges: readonly number[];

  constructor(count: number, edges: readonly number[]) {
    this.count = count;
    this.edges = edges;
    if (edges.length !== count - 1) {
      throw new Error(`gauge with ${count} bins needs ${count - 1} edges`);
    }
  }

  /** Which bin a reading falls into. */
  bin(value: number): number {
    let index = 0;
    while (index < this.edges.length && value >= this.edges[index]) {
      index += 1;
    }
    return index;
  }
}

/**
 * THE PANEL'S GAUGES — the quantization tables, in the units the panel
 * shows them. The circle gauges (pitch, bank, HSI) bin the full
 * 0-345+ degree card; VSI bins the SIGNED fpm scale (climbs and
 * descents both fly); ALT is feet above the ground.
 */
const VSI_GAUGE = new GaugeBins(16, [-8000, -6000, -4000, -3000, -2000,
  -1000, 0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000]);
const ASI_GAUGE = new GaugeBins(8, [40, 50, 60, 70, 80, 90, 120]);
const ALT_GAUGE = new GaugeBins(8, [500, 1000, 2000, 3000, 4000, 5000, 6000]);
const RPM_GAUGE = new GaugeBins(8, [950, 1200, 1450, 1700, 1950, 2200, 2450]);
/** Turn coordinator: standard-rate marks from full-left to full-right,
 *  the reading normalized to [-1, 1] (±1 = ±3x standard rate). */
const TURN_GAUGE = new GaugeBins(7, [-0.75, -0.5, -0.25, 0, 0.25, 0.5]);
/** The panel gauge counts, named for the situation tensors that read them. */
export const LOCATION_SHAPE = [8, 24] as const;          // alt, hsi
export const ENERGY_SHAPE = [8, 8, 16, 24, 8] as const;  // alt, asi, vsi, pitch, rpm
export const FLIGHT_SHAPE = [16, 8, 24, 24, 7] as const; // vsi, asi, pitch, bank, turn

/**
 * THE COMMAND VOCABULARY — each lever's stops, the positions a hand
 * can command it to AND the coarseness the ControlsTensor reads the
 * levers at:
 *  - aileron and rudder: 9 uniform stops (-100%, -75%, -50%, -25%, 0,
 *    +25%, +50%, +75%, +100% of travel);
 *  - elevator: 11 stops on a VERNIER scale — fine near the center
 *    (0, ±6.25%, ±12.5%, ±25% of travel) and coarse at the extremes
 *    (±50%, ±100%). The C172's whole gentle envelope lives within
 *    ±25% of travel (the trim is 0.025 rad, a turn's back-pressure
 *    0.043, a climb 0.09 — uniform quarter-travel stops cannot tell
 *    them apart), while the stall entry needs the full ±100% pull;
 *  - throttle: 7 stops (0, ~16%, ~33%, ~48%, ~66%, ~82%, 100%).
 */
const AILERON_STOPS: readonly number[] = stops(9);
const RUDDER_STOPS: readonly number[] = stops(9);
const ELEVATOR_STOPS: readonly number[] = [
  -0.35, -0.175, -0.0875, -0.04375, -0.021875,
  0.0,
  0.021875, 0.04375, 0.0875, 0.175, 0.35,
];
const THROTTLE_STOPS: readonly number[] = stops(7, 0.0, 1.0);

/** Uniform stops across a lever's travel. */
function stops(count: number, low = -0.35, high = 0.35): number[] {
  const table: number[] = [];
  for (let level = 0; level < count; level++) {
    table.push(low + ((high - low) * level) / (count - 1));
  }
  return table;
}

const CHANNEL_STOPS: readonly (readonly number[])[] = [
  AILERON_STOPS, ELEVATOR_STOPS, RUDDER_STOPS, THROTTLE_STOPS,
];

/** The controls panel's gauge counts: the levers at their stops. */
export const CONTROLS_SHAPE: readonly number[] =
  CHANNEL_STOPS.map((table) => table.length);

/** How many stops each lever has. */
export function stopCount(channel: number): number {
  return CHANNEL_STOPS[channel].length;
}

/** The physical value of a command level: the stop it names. */
export function levelValue(channel: number, level: number): number {
  return CHANNEL_STOPS[channel][level];
}

/** The quantized level a lever position reads at: nearest stop. */
export function levelOf(channel: number, position: number): number {
  const table = CHANNEL_STOPS[channel];
  let best = 0;
  let bestDistance = Math.abs(position - table[0]);
  for (let level = 1; level < table.length; level++) {
    const distance = Math.abs(position - table[level]);
    if (distance < bestDistance) {
      best = level;
      bestDistance = distance;
    }
  }
  return best;
}

export const SITUATION_AXIS = 64;      // the situation bus width
export const SITUATION_CELL_AXIS = 32; // the energy tensor's own situation width

// ===========================================================================
// THE SITUATION CLASSES — each sensor tensor's output vocabulary. The
// classes are NAMED: what the tensor gossips is not an opaque code but
// a classification the rest of the brain can reason with. The teacher
// functions label the binned panel deterministically (the instructor's
// own instruments); the sensor tensors STAMP the panel->class map into
// their memory as the demonstration flies, and recall it (fuzzily,
// with the teacher as the never-seen-before fallback) at runtime.
// ===========================================================================

/** LOCATION: where the plane is in space — altitude and heading,
 *  independent of any required action. The class is the packed panel
 *  itself: 8 altitude bands x 24 heading bands ("cruising at 3000 ft,
 *  heading west"). */
export const LOC_CLASS_AXIS = 8 * 24;
export function classifyLocation(altBin: number, hsiBin: number): number {
  return altBin * 24 + hsiBin;
}

/** ENERGY: the plane's energy state — for deciding whether recovery
 *  steps are required, whether the target altitude is achieved, whether
 *  to climb or descend. */
export const ENERGY_CLASS_AXIS = 16;
export const ENERGY_CLASSES: readonly string[] = [
  "level cruise",                        // 0
  "climbing",                           // 1
  "climbing, losing speed",              // 2
  "climbing fast",                       // 3
  "descending",                         // 4
  "descending slow",                    // 5
  "accelerating descent (gaining speed, losing altitude)",  // 6
  "slow, high angle of attack (not enough speed)",          // 7
  "incipient stall",                    // 8
  "developed stall (falling)",          // 9
  "fast and level",                     // 10
  "high-power climb",                   // 11
  "low-power descent",                  // 12
  "reserved", "reserved", "reserved",   // 13-15
];
/** The energy teacher: [alt, asi, vsi, pitch, rpm] bins -> the class.
 *  Pitch is the 24-bin circle (12 = level, >=13 nose up, <=10 nose
 *  down); VSI is the signed 16-bin scale (7 = 0-1000 fpm up, <=5 more
 *  than 1000 fpm down); ASI bins: <=2 below ~60 mph (the stall
 *  region), 5 = the 80-90 mph cruise band. */
export function classifyEnergy(_alt: number, asi: number, vsi: number,
                               pitch: number, rpm: number): number {
  if (asi <= 2) {                        // below ~60 mph: the stall region
    return (vsi <= 5 || pitch <= 10) ? 9 : 8;
  }
  if (asi <= 4 && pitch >= 13) {
    return 7;                            // slow, nose up: high AoA
  }
  if (vsi >= 8) {                        // > 1000 fpm up
    return asi <= 5 ? 2 : 3;             // losing speed, or climbing fast
  }
  if (vsi === 7 && pitch >= 14) {
    return rpm >= 6 ? 11 : 1;            // high-power climb, or climbing
  }
  if (vsi <= 5) {                       // > 1000 fpm down
    return asi >= 6 ? 6                 // accelerating descent
         : rpm <= 2 ? 12                // low-power descent
         : 4;                          // descending
  }
  if (vsi === 6 && pitch >= 12) {
    return 5;                            // descending slow, nose held up
  }
  if (vsi <= 6 && pitch <= 11) {
    return 4;                           // sinking, nose low
  }
  if (asi >= 6) {
    return 10;                          // fast and level
  }
  return 0;                             // level cruise
}

/** FLIGHT: the fine-grained attitude classes — what the airframe is
 *  doing right now, so the servo can pick the correction. */
export const FLIGHT_CLASS_AXIS = 16;
export const FLIGHT_CLASSES: readonly string[] = [
  "wings level, coordinated",            // 0
  "banking left",                        // 1
  "banking right",                       // 2
  "yawing right, need left rudder",      // 3
  "yawing left, need right rudder",      // 4
  "banking left, losing altitude",       // 5
  "banking right, losing altitude",      // 6
  "banking left, climbing",              // 7
  "banking right, climbing",             // 8
  "overbanked left",                     // 9
  "overbanked right",                    // 10
  "nose high",                           // 11
  "nose low",                            // 12
  "reserved", "reserved", "reserved",    // 13-15
];
/** The flight teacher: [vsi, asi, pitch, bank, turn] bins -> the
 *  class. Bank is the 24-bin circle (12 = level); the turn
 *  coordinator is 7 bins (3 = zero, <=1 left marks, >=5 right). */
export function classifyFlight(vsi: number, _asi: number, pitch: number,
                                bank: number, turn: number): number {
  if (bank <= 9) {
    return 9;                            // overbanked left (< -45 deg)
  }
  if (bank >= 15) {
    return 10;                           // overbanked right (> +45 deg)
  }
  if (bank <= 11) {                      // left bank, 15-45 deg
    if (vsi >= 8) { return 7; }          // banking left, climbing
    if (vsi <= 5) { return 5; }          // banking left, losing altitude
    return 1;
  }
  if (bank >= 13) {                      // right bank
    if (vsi >= 8) { return 8; }
    if (vsi <= 5) { return 6; }
    return 2;
  }
  // wings near level: the yaw detection — the ball is off with no bank
  if (turn <= 1) {
    return 4;                            // yawing left: need right rudder
  }
  if (turn >= 5) {
    return 3;                            // yawing right: need left rudder
  }
  if (pitch >= 15 && vsi >= 9) {
    return 11;                           // nose high (climbing hard)
  }
  if (pitch <= 9 && vsi <= 4) {
    return 12;                           // nose low (diving)
  }
  if (pitch >= 16) {
    return 11;                           // nose high
  }
  if (pitch <= 8) {
    return 12;                           // nose low
  }
  return 0;
}

/** CONTROLS: what the hand is holding, in pilot's terms. The class
 *  packs each lever's coarse state: elevator (pushing down / near
 *  trim / pulling up / pulling up high) x aileron (left / neutral /
 *  right) x rudder (left / neutral / right) x throttle (idle / low /
 *  cruise / high / full). */
export const CONTROLS_CLASS_AXIS = 4 * 3 * 3 * 5;
/** The controls teacher: [ail, elev, rud, thr] stop levels -> the
 *  packed class. */
export function classifyControls(ail: number, elev: number, rud: number,
                                 thr: number): number {
  const elevCoarse = elev <= 3 ? 0 : elev <= 7 ? 1 : elev <= 9 ? 2 : 3;
  const ailCoarse = ail <= 2 ? 0 : ail <= 5 ? 1 : 2;
  const rudCoarse = rud <= 2 ? 0 : rud <= 5 ? 1 : 2;
  const thrCoarse = thr <= 1 ? 0 : thr <= 3 ? 1 : thr === 4 ? 2
                  : thr === 5 ? 3 : 4;
  return elevCoarse * 45 + ailCoarse * 15 + rudCoarse * 5 + thrCoarse;
}

// --- THE CLASS LABELS — each class vocabulary, in pilot's words (the
// AI state log's columns read these) ---------------------------------------

/** The altitude bands' names, bin -> band text. */
const ALT_BANDS: readonly string[] = [
  "under 500 ft", "500-1000 ft", "1000-2000 ft", "2000-3000 ft",
  "3000-4000 ft", "4000-5000 ft", "5000-6000 ft", "above 6000 ft",
];

/** A location class, decoded: "3000-4000 ft · heading 15°L". The
 *  heading is the SIGNED 15-degree circle (bin 12 = on course). */
export function labelLocationClass(cls: number): string {
  const altBin = Math.floor(cls / 24);
  const hsiBin = cls % 24;
  const off = (hsiBin - 12) * 15;   // signed degrees off course
  const heading = off === 0 ? "on course"
    : `${Math.abs(off)}\u00b0${off < 0 ? "L" : "R"}`;
  return `${ALT_BANDS[Math.min(altBin, 7)]} \u00b7 ${heading}`;
}

/** An energy class, by name (the 16-class vocabulary). */
export function labelEnergyClass(cls: number): string {
  return ENERGY_CLASSES[Math.min(Math.max(cls, 0), 15)];
}

/** A flight class, by name (the 16-class vocabulary). */
export function labelFlightClass(cls: number): string {
  return FLIGHT_CLASSES[Math.min(Math.max(cls, 0), 15)];
}

const ELEV_COARSE_NAMES: readonly string[] =
  ["pushing down", "near trim", "pulling up", "pulling up high"];
const AIL_COARSE_NAMES: readonly string[] =
  ["ailerons left low", "ailerons neutral", "ailerons right low"];
const RUD_COARSE_NAMES: readonly string[] =
  ["rudder left", "rudder neutral", "rudder right"];
const THR_COARSE_NAMES: readonly string[] =
  ["throttle idle", "throttle low", "throttle cruise", "throttle high",
   "throttle full"];

/** A controls class, decoded: "pulling up high · ailerons left low ·
 *  rudder neutral · throttle cruise". */
export function labelControlsClass(cls: number): string {
  const elevCoarse = Math.floor(cls / 45);
  const ailCoarse = Math.floor(cls / 15) % 3;
  const rudCoarse = Math.floor(cls / 5) % 3;
  const thrCoarse = cls % 5;
  return `${ELEV_COARSE_NAMES[Math.min(elevCoarse, 3)]}` +
    ` \u00b7 ${AIL_COARSE_NAMES[ailCoarse]}` +
    ` \u00b7 ${RUD_COARSE_NAMES[rudCoarse]}` +
    ` \u00b7 ${THR_COARSE_NAMES[thrCoarse]}`;
}

/**
 * A signed angle on the 24-bin 15-degree circle: -180 deg is bin 0,
 * level is bin 12, +180 is bin 23. The SAME 0-345+ quantization the
 * card shows, read the honest way — the wrap sits at straight-inverted
 * (never flown), NOT at level flight, so a fluttering needle moves
 * between ADJACENT bins the fuzzy recall can bridge.
 */
function circleBin(radians: number): number {
  const degrees = radians * DEGREES_PER_RAD;
  return Math.min(23, Math.max(0, Math.floor((degrees + 180.0) / 15.0)));
}

/** The turn coordinator reading: the compass turn rate against the
 *  standard rate (3 deg/s), clamped to the instrument's ±3x marks. */
function turnCoordinatorReading(body: RigidBody): number {
  // Aviation yaw rate: psi_dot = (q*sin(phi) + r*cos(phi)) / cos(theta).
  const numerator = body.rateQ * Math.sin(body.bank)
                  + body.rateR * Math.cos(body.bank);
  const cosTheta = Math.max(0.2, Math.cos(body.deckPitch));
  const standardRate = Math.PI / 60.0;    // 3 deg/s in rad/s
  return Math.max(-1.0, Math.min(1.0, (numerator / cosTheta)
    / (3.0 * standardRate)));
}

/** The panel as the situation tensors read it: four sub-panels, every
 *  gauge binned by its own quantization table.
 *
 *   location  alt, hsi                — where the flight is, which way home
 *   energy    alt, asi, vsi,
 *             pitch, rpm              — the energy state, engine included
 *   flight    vsi, asi, pitch,
 *             bank, turn              — what the airframe is doing right now
 *   controls  ail, elev, rud,
 *             thr                    — the levers as they actually stand
 */
export interface SenseReading {
  /** [alt, hsi] — the location panel. */
  readonly location: number[];
  /** [alt, asi, vsi, pitch, rpm] — the energy panel. */
  readonly energy: number[];
  /** [vsi, asi, pitch, bank, turn] — the flight panel. */
  readonly flight: number[];
  /** [ail, elev, rud, thr] — the controls panel, at the stops. */
  readonly controls: number[];
}

/** Turns a RigidBody into binned panel keys and error norms. */
export class FlightInstruments {
  readonly trimSpeed: number;
  readonly speedBand: number;
  readonly vsBand: number;

  constructor(trimSpeed: number, speedBand = 6.0, vsBand = 4.0) {
    this.trimSpeed = trimSpeed;
    this.speedBand = speedBand;
    this.vsBand = vsBand;
  }

  /** (vsErr, bankErr, speedErr), each normalized to [-1, 1]. */
  errors(body: RigidBody): Vec3 {
    const vsErr = Math.max(-1.0, Math.min(1.0, body.velocity.z / this.vsBand));
    const bank = body.bank;
    const bankErr = Math.max(-1.0, Math.min(1.0, bank / (Math.PI / 4.0)));
    const speed = body.velocity.length();
    const speedErr = Math.max(-1.0, Math.min(1.0, (speed - this.trimSpeed) / this.speedBand));
    return new Vec3(vsErr, bankErr, speedErr);
  }

  /**
   * HSI: signed bearing delta to target, normalized [-1, 1] by 90 deg.
   * Positive delta: the target bearing lies to the LEFT (math convention,
   * +Z up): turning left increases bearing. The HSI needle, not a GPS.
   */
  headingErr(body: RigidBody, targetBearing: number): number {
    const [forward] = body.bodyAxes;
    const bearing = Math.atan2(forward.y, forward.x);
    const twoPi = 2.0 * Math.PI;
    const delta = ((((targetBearing - bearing + Math.PI) % twoPi) + twoPi) % twoPi) - Math.PI;
    return Math.max(-1.0, Math.min(1.0, delta / (Math.PI / 2.0)));
  }

  /**
   * Legacy 5-bin key (vs, bank, asi, rpm, hsi) — the free-flight episode's
   * recorded final state. The intention tensor keys on sense() instead.
   */
  stateKey(body: RigidBody, throttlePosition = 0.15,
           targetBearing: number | null = null): number[] {
    const errors = this.errors(body);
    const heading = targetBearing === null ? 0.0 : this.headingErr(body, targetBearing);
    return [
      binError(errors.x),
      binError(errors.y),
      binError(errors.z),
      binError(Math.max(-1.0, Math.min(1.0, throttlePosition * 2.0 - 1.0))),
      binError(heading),
    ];
  }

  /**
   * The student's full panel reading — every gauge quantized by its own
   * table, grouped into the sub-panels the situation tensors encode.
   * `controls` is the actual surface/throttle state of the airframe (the
   * engine gauge reads the throttle lever), so demonstration and repeat
   * read the same panel.
   */
  senseReading(body: RigidBody, controls: ArrayLike<number>,
               targetBearing: number | null = null): SenseReading {
    const [ailLevel, elevLevel, rudLevel, thrLevel] = [
      levelOf(AILERON, controls[AILERON]),
      levelOf(ELEVATOR, controls[ELEVATOR]),
      levelOf(RUDDER, controls[RUDDER]),
      levelOf(THROTTLE, controls[THROTTLE]),
    ];
    const altFt = body.position.z * FT_PER_M;
    const asiMph = body.velocity.length() * MPH_PER_MPS;
    const vsiFpm = body.velocity.z * FPM_PER_MPS;
    const heading = targetBearing === null ? 0.0 : this.headingErr(body, targetBearing);
    // The engine gauge: RPM idles at 700 and redlines at 2700.
    const rpm = 700.0 + 2000.0 * controls[THROTTLE];
    return {
      location: [
        ALT_GAUGE.bin(altFt),
        circleBin(heading * (Math.PI / 2.0)),
      ],
      energy: [
        ALT_GAUGE.bin(altFt),
        ASI_GAUGE.bin(asiMph),
        VSI_GAUGE.bin(vsiFpm),
        circleBin(body.deckPitch),
        RPM_GAUGE.bin(rpm),
      ],
      flight: [
        VSI_GAUGE.bin(vsiFpm),
        ASI_GAUGE.bin(asiMph),
        circleBin(body.deckPitch),
        circleBin(body.bank),
        TURN_GAUGE.bin(turnCoordinatorReading(body)),
      ],
      controls: [ailLevel, elevLevel, rudLevel, thrLevel],
    };
  }

  /** Weighted error norm: lower is better flight. */
  penalty(body: RigidBody): number {
    const errors = this.errors(body);
    return Math.abs(errors.x) + 0.5 * Math.abs(errors.y) + 0.3 * Math.abs(errors.z);
  }
}

/**
 * Baseline-centered per-tick reward around the weighted error norm.
 *
 * Raw improvement is mean-positive while gliding toward trim, which floods
 * every action with credit. Subtracting a running mean (EMA) makes average
 * flight worth zero: only better-than-recent-average control earns positive
 * weight, and only worse-than-average earns negative.
 */
export class DenseReward {
  private previousPenalty: number | null = null;
  private baseline = 0.0;

  constructor(readonly instruments: FlightInstruments, readonly scale = 5.0,
              readonly baselineAlpha = 0.01) {}

  reset(body: RigidBody): void {
    this.previousPenalty = this.instruments.penalty(body);
    this.baseline = 0.0;
  }

  tick(body: RigidBody): number {
    const penalty = this.instruments.penalty(body);
    if (this.previousPenalty === null) {
      this.previousPenalty = penalty;
    }
    const improvement = this.previousPenalty - penalty;
    this.previousPenalty = penalty;
    const centered = improvement - this.baseline;
    this.baseline += this.baselineAlpha * (improvement - this.baseline);
    return centered * this.scale;
  }
}

/** How a flight ended. Named states, always. */
export const EpisodeOutcome = {
  ALIVE: "alive",            // reached the time horizon still flying
  CRASHED: "crashed",        // ground contact
  STALLED: "stalled",        // too slow for too long
  OUT_OF_BOUNDS: "out_of_bounds",
} as const;
export type EpisodeOutcomeName = typeof EpisodeOutcome[keyof typeof EpisodeOutcome];

/**
 * None while flying; an EpisodeOutcome name when the episode ends.
 * (The Python signature takes an unused `elapsed` — kept for parity.)
 */
export function classifyOutcome(body: RigidBody, world: World, _elapsed: number,
                                stallSpeed = 4.0, slowLimit = 1.5): string | null {
  const ground = world.terrainHeight(body.position.x, body.position.y);
  if (body.position.z <= ground + 1.0) {
    return EpisodeOutcome.CRASHED;
  }
  if (body.velocity.length() < stallSpeed) {
    body.slowSeconds += 0.02;
    if (body.slowSeconds >= slowLimit) {
      return EpisodeOutcome.STALLED;
    }
  } else {
    body.slowSeconds = 0.0;
  }
  if (!world.isInsideBounds(body.position)) {
    return EpisodeOutcome.OUT_OF_BOUNDS;
  }
  return null;
}
