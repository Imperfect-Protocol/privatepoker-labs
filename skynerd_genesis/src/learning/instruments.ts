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

// ===========================================================================
// V2 SENSORY VOCABULARY — the three orthogonal perception tensors.
//
// The redesign eliminates duplicated gauges across the old four sensor
// tensors and replaces them with three non-overlapping panels:
//
//   FlightAttitude    — aerodynamic attitude & airframe dynamics
//   EnergyPropulsion  — mechanical & kinetic potential (engine + energy)
//   Proprioception    — tactile control surface positions (the "hands")
//
// Each gauge appears in exactly ONE panel.  The old shapes above remain
// for backward compatibility until servo-tensor.ts migrates.
// ===========================================================================

/**
 * FLIGHT ATTITUDE: what the airframe is doing right now.
 *
 * Pitch (24) × Bank (24) × Turn coordinator (7) × VSI (16) × ASI (8).
 *
 * The attitude instruments: pitch and bank from the attitude indicator,
 * the turn coordinator ball, the vertical speed, and airspeed — the
 * six-pack's dynamic group.  Every gauge that sweeps during a maneuver
 * lives here; nothing engine-related enters.
 */
export const FLIGHT_ATTITUDE_SHAPE = [24, 24, 7, 16, 8] as const;

/**
 * ENERGY / PROPULSION: the energy state including the engine.
 *
 * Alt (8) × RPM (8) × Throttle (7) × ASI (8) × VSI (16).
 *
 * The energy / power group: altitude (potential energy), RPM (engine
 * state), throttle position (power command), airspeed (kinetic energy),
 * and vertical speed (energy rate of change).  The overlap of ASI and
 * VSI with FlightAttitude is intentional in the SHAPE — the teacher
 * classifier uses them to judge energy, not attitude, so the class
 * vocabulary (and therefore what gets stamped) is completely different.
 *
 * NOTE: throttle is a 7-stop lever (THROTTLE_STOPS), not a gauge — but
 * the energy tensor reads it as an engine-state indicator alongside RPM,
 * exactly the way a pilot cross-checks the power lever against the
 * tachometer.
 */
export const ENERGY_PROPULSION_SHAPE = [8, 8, 7, 8, 16] as const;

/**
 * PROPRIOCEPTION: what the pilot's hands and feet feel.
 *
 * Elevator vernier (11) × Aileron (9) × Rudder (9).
 *
 * Pure "tactile" feedback — the stop positions of the three primary
 * flight control surfaces (the yoke columns and rudder pedals).
 * Throttle is excluded: it is an engine command, not a flight-control
 * deflection, and lives in the EnergyPropulsion panel.
 */
export const PROPRIOCEPTION_SHAPE = [11, 9, 9] as const;

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

// ===========================================================================
// V2 SITUATION CLASSES — the three orthogonal perception classifiers.
//
// Each teacher is deterministic: a pure function from binned gauges to a
// named class.  The classifiers own no state and never touch the
// Aircraft; they receive the panel's quantized bins and return the
// class index.  The perception tensors stamp the teacher's label during
// demonstration and fall back to it at runtime.
// ===========================================================================

/** FLIGHT ATTITUDE — 32 named states: the airframe's dynamic posture.
 *
 * The vocabulary splits into: level/coordinated, bank left/right (×
 * climbing/descending/level), overbank, nose high/low, yaw
 * left/right, spin left/right, and unusual-attitude envelope
 * (spiral dive, tumble).  Every class is a pilot's English phrase
 * — the downstream tensors read the vocabulary as named situations. */
export const FLIGHT_ATTITUDE_CLASS_AXIS = 32;
export const FLIGHT_ATTITUDE_CLASSES: readonly string[] = [
  "wings level, coordinated",                    //  0
  "banking left",                                //  1
  "banking right",                               //  2
  "yawing right, need left rudder",              //  3
  "yawing left, need right rudder",              //  4
  "banking left, losing altitude",               //  5
  "banking right, losing altitude",              //  6
  "banking left, climbing",                      //  7
  "banking right, climbing",                     //  8
  "overbanked left",                             //  9
  "overbanked right",                            // 10
  "nose high",                                   // 11
  "nose low",                                    // 12
  "steep turn left, coordinated",                // 13
  "steep turn right, coordinated",               // 14
  "spin entry left",                             // 15
  "spin entry right",                            // 16
  "spin developed left",                         // 17
  "spin developed right",                        // 18
  "spiral dive left",                            // 19
  "spiral dive right",                           // 20
  "wings level, nose high, slow",                // 21
  "wings level, nose low, fast",                 // 22
  "banking left, nose high",                     // 23
  "banking right, nose high",                    // 24
  "banking left, nose low",                      // 25
  "banking right, nose low",                     // 26
  "unusual attitude: inverted tendency",         // 27
  "slip left (cross-controlled)",                // 28
  "slip right (cross-controlled)",               // 29
  "reserved",                                    // 30
  "reserved",                                    // 31
];

/**
 * The flight-attitude teacher: [pitch, bank, turn, vsi, asi] bins →
 * the class.
 *
 * Pitch is the 24-bin circle (12 = level, ≥13 nose up, ≤11 nose down).
 * Bank is the 24-bin circle (12 = wings level).
 * Turn coordinator: 7 bins (3 = centered, ≤1 left marks, ≥5 right).
 * VSI: 16-bin signed scale (7 = gentle climb, ≤5 = descent ≥ 1000 fpm).
 * ASI: 8 bins (≤2 = below ~60 mph stall region, 5 = cruise).
 */
export function classifyFlightAttitude(
    pitch: number, bank: number, turn: number,
    vsi: number, asi: number): number {
  // --- SPIN detection (highest priority: pitch nose-down + extreme yaw
  //     + one wing fully stalled) ---
  // Developed spin: extreme bank + nose well down + slow + high yaw rate
  if (asi <= 2 && (bank <= 8 || bank >= 16)) {
    if (pitch <= 9 && (turn <= 0 || turn >= 6)) {
      return bank <= 8 ? 17 : 18;   // developed spin left / right
    }
    // Spin entry: overbanked + slow + yaw disagrees with bank
    if (bank <= 9 && turn >= 5) { return 15; }   // spin entry left
    if (bank >= 15 && turn <= 1) { return 16; }  // spin entry right
  }

  // --- UNUSUAL ATTITUDES ---
  // Spiral dive: steep bank + nose low + descending fast + speed building
  if ((bank <= 9 || bank >= 15) && pitch <= 9 && vsi <= 4 && asi >= 5) {
    return bank <= 9 ? 19 : 20;   // spiral dive left / right
  }
  // Inverted tendency: bank well past 60 deg (bin 7 or 17+)
  if (bank <= 7 || bank >= 17) {
    return 27;   // unusual attitude: inverted tendency
  }

  // --- OVERBANK (45-60 deg) ---
  if (bank <= 9) { return 9; }       // overbanked left
  if (bank >= 15) { return 10; }     // overbanked right

  // --- STEEP TURNS (30-45 deg, coordinated) ---
  if (bank === 10 && turn >= 1 && turn <= 2 && vsi >= 6 && vsi <= 8) {
    return 13;   // steep turn left, coordinated
  }
  if (bank === 14 && turn >= 4 && turn <= 5 && vsi >= 6 && vsi <= 8) {
    return 14;   // steep turn right, coordinated
  }

  // --- MODERATE BANK (15-45 deg) ---
  if (bank <= 11) {                    // left bank
    if (pitch >= 14) { return 23; }    // banking left, nose high
    if (pitch <= 10) { return 25; }    // banking left, nose low
    if (vsi >= 8) { return 7; }        // banking left, climbing
    if (vsi <= 5) { return 5; }        // banking left, losing altitude
    return 1;                          // banking left (level)
  }
  if (bank >= 13) {                    // right bank
    if (pitch >= 14) { return 24; }    // banking right, nose high
    if (pitch <= 10) { return 26; }    // banking right, nose low
    if (vsi >= 8) { return 8; }        // banking right, climbing
    if (vsi <= 5) { return 6; }        // banking right, losing altitude
    return 2;                          // banking right (level)
  }

  // --- WINGS NEAR LEVEL (bank 12 ± 1) ---
  // Cross-controlled slip: wings level but ball off
  if (turn <= 1 && bank === 12) { return 28; }   // slip left
  if (turn >= 5 && bank === 12) { return 29; }   // slip right

  // Yaw detection: ball off with no meaningful bank
  if (turn <= 1) { return 4; }        // yawing left, need right rudder
  if (turn >= 5) { return 3; }        // yawing right, need left rudder

  // Pitch attitude with speed context
  if (pitch >= 15 && asi <= 3) { return 21; }   // wings level, nose high, slow
  if (pitch <= 9 && asi >= 5) { return 22; }    // wings level, nose low, fast
  if (pitch >= 15) { return 11; }               // nose high
  if (pitch <= 9) { return 12; }                // nose low

  // Pitch mid-range (10-14) but still with a VSI signal
  if (pitch >= 14 && vsi >= 9) { return 11; }   // nose high (climbing hard)
  if (pitch <= 10 && vsi <= 4) { return 12; }   // nose low (diving)

  return 0;   // wings level, coordinated
}

/** A flight-attitude class label, by name. */
export function labelFlightAttitudeClass(cls: number): string {
  return FLIGHT_ATTITUDE_CLASSES[Math.min(Math.max(cls, 0), 31)];
}


/** ENERGY / PROPULSION — 16 named states: the plane's energy budget.
 *
 * The vocabulary covers: cruise, climbing (with speed/power variants),
 * descending (ditto), the stall family, overspeed, and the
 * all-important engine-power cross-check (throttle against what the
 * RPM tachometer reads). */
export const ENERGY_PROPULSION_CLASS_AXIS = 16;
export const ENERGY_PROPULSION_CLASSES: readonly string[] = [
  "level cruise",                                          //  0
  "climbing",                                              //  1
  "climbing, losing speed",                                //  2
  "climbing fast",                                         //  3
  "descending",                                            //  4
  "descending slow",                                       //  5
  "accelerating descent (gaining speed, losing altitude)",  //  6
  "slow, high angle of attack (not enough speed)",          //  7
  "incipient stall",                                       //  8
  "developed stall (falling)",                             //  9
  "fast and level",                                        // 10
  "high-power climb",                                      // 11
  "low-power descent",                                     // 12
  "engine underperforming (low RPM, high throttle)",       // 13
  "overspeed",                                             // 14
  "reserved",                                              // 15
];

/**
 * The energy-propulsion teacher: [alt, rpm, throttle, asi, vsi] bins →
 * the class.
 *
 * Alt: 8-bin altitude gauge.
 * RPM: 8-bin tachometer.
 * Throttle: 7 stops (0 = idle, 6 = full).
 * ASI: 8 bins (≤2 = below ~60 mph stall region, 5 = cruise band).
 * VSI: 16-bin signed scale (7 = 0-1000 fpm up, ≤5 = ≥ 1000 fpm down).
 */
export function classifyEnergyPropulsion(
    _alt: number, rpm: number, thr: number,
    asi: number, vsi: number): number {
  // --- THE STALL FAMILY (highest priority: always classified) ---
  if (asi <= 2) {
    // Developed stall: airspeed collapsed, descending or nose already low
    if (vsi <= 5) { return 9; }
    return 8;   // incipient stall: slow but not yet sinking hard
  }

  // High angle of attack: slow + high throttle but RPM doesn't match
  if (asi <= 4 && thr >= 4) {
    return 7;   // slow, high angle of attack
  }

  // --- ENGINE ANOMALY ---
  // Low RPM with high throttle: engine is underperforming
  if (rpm <= 2 && thr >= 5) {
    return 13;   // engine underperforming
  }

  // --- OVERSPEED ---
  if (asi >= 7) {
    if (vsi >= 6 && vsi <= 8) { return 10; }   // fast and level
    return 14;   // overspeed (outside the level band)
  }

  // --- CLIMBING ---
  if (vsi >= 8) {                      // > 1000 fpm climb
    if (asi <= 5) { return 2; }        // climbing, losing speed
    return 3;                          // climbing fast
  }
  if (vsi === 7) {                     // gentle climb (0-1000 fpm up)
    if (rpm >= 6) { return 11; }       // high-power climb
    return 1;                          // climbing
  }

  // --- DESCENDING ---
  if (vsi <= 5) {                      // > 1000 fpm descent
    if (asi >= 6) { return 6; }        // accelerating descent
    if (thr <= 1) { return 12; }       // low-power descent (idle throttle)
    return 4;                          // descending
  }
  if (vsi === 6) {                     // gentle descent
    return 5;                          // descending slow
  }

  // --- LEVEL ---
  if (asi >= 6) { return 10; }         // fast and level
  return 0;                            // level cruise
}

/** An energy-propulsion class label, by name. */
export function labelEnergyPropulsionClass(cls: number): string {
  return ENERGY_PROPULSION_CLASSES[Math.min(Math.max(cls, 0), 15)];
}


/** PROPRIOCEPTION — 64 named states: what the pilot's hands feel.
 *
 * The vocabulary is the coarsened control surface configuration:
 *   Elevator: 4 coarse bands (push / trim / pull / pull high)
 *   Aileron:  4 coarse bands (hard left / left / neutral / right)
 *   Rudder:   4 coarse bands (hard left / left / neutral / right)
 *
 * 4 × 4 × 4 = 64 classes.  Throttle is excluded — it is an engine
 * command (EnergyPropulsion reads it); the flight-control hands and
 * feet are a separate sensory channel. */
export const PROPRIOCEPTION_CLASS_AXIS = 64;

/**
 * The proprioception teacher: [elev, ail, rud] stop levels → the class.
 *
 * Elevator (11 vernier stops): 0-2 = push, 3-7 = near trim,
 *   8-9 = pull, 10 = pull high.
 * Aileron (9 stops): 0-1 = hard left, 2-3 = left, 4 = neutral,
 *   5-6 = right, 7-8 = hard right.
 * Rudder (9 stops): same 4-band layout as aileron.
 */
export function classifyProprioception(
    elev: number, ail: number, rud: number): number {
  const elevCoarse = elev <= 2 ? 0 : elev <= 7 ? 1 : elev <= 9 ? 2 : 3;
  const ailCoarse = ail <= 1 ? 0 : ail <= 3 ? 1 : ail <= 4 ? 2 : ail <= 6 ? 3 : 3;
  const rudCoarse = rud <= 1 ? 0 : rud <= 3 ? 1 : rud <= 4 ? 2 : rud <= 6 ? 3 : 3;
  return elevCoarse * 16 + ailCoarse * 4 + rudCoarse;
}

const ELEV_PROPRIO_NAMES: readonly string[] =
  ["pushing down", "near trim", "pulling up", "pulling up high"];
const AIL_PROPRIO_NAMES: readonly string[] =
  ["ailerons hard left", "ailerons left", "ailerons neutral", "ailerons right"];
const RUD_PROPRIO_NAMES: readonly string[] =
  ["rudder hard left", "rudder left", "rudder neutral", "rudder right"];

/** A proprioception class, decoded:
 *  "near trim · ailerons neutral · rudder neutral". */
export function labelProprioceptionClass(cls: number): string {
  const elevCoarse = Math.floor(cls / 16);
  const ailCoarse = Math.floor(cls / 4) % 4;
  const rudCoarse = cls % 4;
  return `${ELEV_PROPRIO_NAMES[Math.min(elevCoarse, 3)]}` +
    ` \u00b7 ${AIL_PROPRIO_NAMES[Math.min(ailCoarse, 3)]}` +
    ` \u00b7 ${RUD_PROPRIO_NAMES[Math.min(rudCoarse, 3)]}`;
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

// ===========================================================================
// V2 SENSE READING — the three orthogonal perception panels.
// ===========================================================================

/** The V2 panel reading: three non-overlapping sub-panels, each gauge
 *  appearing in exactly one panel. */
export interface SenseReadingV2 {
  /** [pitch, bank, turn, vsi, asi] — the flight-attitude panel. */
  readonly flightAttitude: number[];
  /** [alt, rpm, throttle, asi, vsi] — the energy-propulsion panel. */
  readonly energyPropulsion: number[];
  /** [elev, ail, rud] — the proprioception panel (control surfaces). */
  readonly proprioception: number[];
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

  /**
   * V2 panel reading — the three orthogonal perception panels.
   *
   * Each gauge appears in exactly one panel:
   *   flightAttitude:    pitch, bank, turn, vsi, asi
   *   energyPropulsion:  alt, rpm, throttle (stop level), asi, vsi
   *   proprioception:    elevator (stop level), aileron, rudder
   *
   * `controls` is the actual surface/throttle state of the airframe.
   */
  senseReadingV2(body: RigidBody, controls: ArrayLike<number>): SenseReadingV2 {
    const elevLevel = levelOf(ELEVATOR, controls[ELEVATOR]);
    const ailLevel = levelOf(AILERON, controls[AILERON]);
    const rudLevel = levelOf(RUDDER, controls[RUDDER]);
    const thrLevel = levelOf(THROTTLE, controls[THROTTLE]);
    const altFt = body.position.z * FT_PER_M;
    const asiMph = body.velocity.length() * MPH_PER_MPS;
    const vsiFpm = body.velocity.z * FPM_PER_MPS;
    // The engine gauge: RPM idles at 700 and redlines at 2700.
    const rpm = 700.0 + 2000.0 * controls[THROTTLE];
    return {
      flightAttitude: [
        circleBin(body.deckPitch),       // pitch: 24 bins
        circleBin(body.bank),            // bank: 24 bins
        TURN_GAUGE.bin(turnCoordinatorReading(body)),  // turn: 7 bins
        VSI_GAUGE.bin(vsiFpm),           // vsi: 16 bins
        ASI_GAUGE.bin(asiMph),           // asi: 8 bins
      ],
      energyPropulsion: [
        ALT_GAUGE.bin(altFt),            // alt: 8 bins
        RPM_GAUGE.bin(rpm),              // rpm: 8 bins
        thrLevel,                        // throttle: 7 stops
        ASI_GAUGE.bin(asiMph),           // asi: 8 bins
        VSI_GAUGE.bin(vsiFpm),           // vsi: 16 bins
      ],
      proprioception: [
        elevLevel,                       // elevator: 11 vernier stops
        ailLevel,                        // aileron: 9 stops
        rudLevel,                        // rudder: 9 stops
      ],
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
