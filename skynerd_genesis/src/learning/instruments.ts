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

export const ERROR_BINS = 8;

/** Bin a value in [-1, 1] into [0, bins-1]; clamps at the ends. */
export function binError(value: number, bins: number = ERROR_BINS): number {
  const clamped = Math.max(-1.0, Math.min(1.0, value));
  const shifted = (clamped + 1.0) / 2.0;           // 0..1
  const index = Math.floor(shifted * bins);
  return Math.min(bins - 1, Math.max(0, index));
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
    const [, right] = body.bodyAxes;
    // Rolled right => right wing down => right.z < 0 => bank positive.
    const bank = -Math.asin(Math.max(-1.0, Math.min(1.0, right.z)));
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
   * (vs, bank, asi, rpm, hsi) bins — VSI, bank, ASI, engine, HSI.
   * Without a bearing target the HSI reads centered (the drone is
   * on-course by definition).
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
