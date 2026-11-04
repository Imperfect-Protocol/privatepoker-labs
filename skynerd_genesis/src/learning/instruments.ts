/**
 * Flight instruments and the dense per-tick reward signal.
 * Ported 1:1 from `skynerd/learning/reward.py` (Genesis Milestone 3).
 *
 * Egocentric error signals only — no GPS, no absolute position. The same
 * philosophy as the twisted embeddings: what a drone feels, not where it is.
 *
 * Three normalized error channels in [-1, 1], each binned into 8:
 * - vertical-speed error: climbing or sinking relative to level flight
 * - bank error: wing-level error (rolled right is positive)
 * - airspeed error: relative to the airframe's trim speed
 *
 * The dense reward is error *improvement*: positive when the weighted error
 * norm shrinks. Crash detection is physical: ground contact or loss of
 * control ends the episode.
 */
import { Vec3 } from "../core/vec.js";
import { World } from "../core/world.js";
import { RigidBody } from "../physics/rigid-body.js";
import {
  AILERON, ELEVATOR, RUDDER, THROTTLE, SURFACE_LIMITS,
} from "../physics/surfaces.js";

export const ERROR_BINS = 8;

/** Bin a value in [-1, 1] into [0, bins-1]; clamps at the ends. */
export function binError(value: number, bins: number = ERROR_BINS): number {
  const clamped = Math.max(-1.0, Math.min(1.0, value));
  const shifted = (clamped + 1.0) / 2.0;           // 0..1
  const index = Math.floor(shifted * bins);
  return Math.min(bins - 1, Math.max(0, index));
}

/** Bin a value in [0, span] into [0, bins-1]; clamps at the ends. */
function binSpan(value: number, span: number, bins: number): number {
  const index = Math.floor((value / span) * bins);
  return Math.min(bins - 1, Math.max(0, index));
}

/**
 * THE SENSE AXES — split by the neuron that reads them. The full panel
 * the student reads is three sub-panels plus the engine gauge, and the
 * control positions as they stand:
 *
 *   flight   vs, bank, asi, hsi, pitch, alt — the flight instruments
 *            (VSI, bank, airspeed, heading needle, attitude, altimeter)
 *   engine   rpm — the engine gauge (the throttle position)
 *   controls ail, elev, rud, thr — the yoke and pedals AS THEY STAND,
 *            at the coarseness of a hand: the demo's held pull and the
 *            student's chase of it must land in the same bin, or the
 *            student flies blind past the target
 */
export const FLIGHT_SENSE_SHAPE = [8, 8, 32, 8, 16, 64] as const;
export const ENGINE_SENSE_SHAPE = [8] as const;
export const CONTROLS_SENSE_SHAPE = [4, 4, 4, 4] as const;

const ASI_SPAN_MPS = 64.0;
const PITCH_SPAN_RAD = 0.6;
const ALT_SPAN_M = 1024.0;
// The pitch axis reads at ATTITUDE-HOLDING resolution: a swept maneuver
// distinguishes its target attitude from a balloon past it, so the
// student's runaway and the instructor's hold must not share a fuzzy
// neighborhood. 16 bins over the span: ~2.2° per bin.

/** The panel as the neurons read it: the three sub-panels, binned. */
export interface SenseReading {
  /** [vs, bank, asi, hsi, pitch, alt] — the flight instruments. */
  readonly flight: number[];
  /** [rpm] — the engine gauge. */
  readonly engine: number[];
  /** [aileron, elevator, rudder, throttle] — the controls as they stand. */
  readonly controls: number[];
}

/** Turns a RigidBody into binned flight-state keys and error norms. */
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
   * The student's full sense reading — the SENSE_SHAPE axes: flight and
   * engine instruments, attitude, altitude, and the control positions as
   * they stand. This is the observation the intention tensor keys on
   * (goal and element axes go in front of it). `controls` is the ACTUAL
   * surface/throttle state of the airframe (what the hand or the student
   * has eased to), so demonstration and repeat read the same panel.
   */
  /**
   * The panel as the neurons read it — the three sub-panels, binned.
   * `controls` is the ACTUAL surface/throttle state of the airframe
   * (what the hand or the student has eased to), so demonstration and
   * repeat read the same panel.
   */
  senseReading(body: RigidBody, controls: ArrayLike<number>,
               targetBearing: number | null = null): SenseReading {
    const errors = this.errors(body);
    const heading = targetBearing === null ? 0.0 : this.headingErr(body, targetBearing);
    const [ailLow, ailHigh] = SURFACE_LIMITS[AILERON];
    const [elevLow, elevHigh] = SURFACE_LIMITS[ELEVATOR];
    const [rudLow, rudHigh] = SURFACE_LIMITS[RUDDER];
    const rpm = binError(Math.max(-1.0, Math.min(1.0, controls[THROTTLE] * 2.0 - 1.0)));
    return {
      flight: [
        binError(errors.x),                              // vs
        binError(errors.y),                              // bank
        binSpan(body.velocity.length(), ASI_SPAN_MPS, 32),  // asi
        binError(heading),                               // hsi
        binError(body.deckPitch / PITCH_SPAN_RAD, 16),  // pitch
        binSpan(body.position.z, ALT_SPAN_M, 64),        // alt
      ],
      engine: [rpm],
      controls: [
        binSpan(controls[AILERON] - ailLow, ailHigh - ailLow, 4),    // aileron
        binSpan(controls[ELEVATOR] - elevLow, elevHigh - elevLow, 4),  // elevator
        binSpan(controls[RUDDER] - rudLow, rudHigh - rudLow, 4),     // rudder
        // The throttle lever at the hand's coarseness: full throttle
        // and the cruise trim must share a bin, or the stick spring's
        // drift between them orphans the action memory.
        Math.min(3, Math.floor(controls[THROTTLE] * 4)),             // throttle
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
