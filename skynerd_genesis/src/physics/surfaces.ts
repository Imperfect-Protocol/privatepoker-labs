/**
 * Control surfaces: four channels, rate-limited slew toward commanded targets.
 * Ported 1:1 from `skynerd/physics/surfaces.py` (Genesis Milestone 3).
 *
 * The brain commands surface targets; physics obeys with real actuator
 * limits. Deflections teleport nowhere: a servo moves at a bounded slew rate
 * toward its target and clamps at its mechanical limits. These are airframe
 * properties, not control logic.
 */

export const AILERON = 0;
export const ELEVATOR = 1;
export const RUDDER = 2;
export const THROTTLE = 3;
export const CHANNEL_NAMES = ["aileron", "elevator", "rudder", "throttle"] as const;

/** Mechanical limits per channel: [min, max], radians for the three surfaces, [0, 1] for throttle. */
export const SURFACE_LIMITS: readonly (readonly [number, number])[] = [
  [-0.35, 0.35],
  [-0.35, 0.35],
  [-0.35, 0.35],
  [0.0, 1.0],
];

/** Slew rate: rad/s for surfaces, full range per second for throttle. */
export const SURFACE_SLEW_RATE = 6.0;

/** Physical surface positions. Owned by the airframe, updated per tick. */
export class SurfacesState {
  readonly positions = new Float64Array(4);

  /** Slew toward targets, rate-limited and clamped. Deterministic. */
  command(targets: ArrayLike<number>, dt: number): void {
    if (dt <= 0.0) {
      throw new Error("dt must be positive");
    }
    for (let channel = 0; channel < 4; channel++) {
      const [low, high] = SURFACE_LIMITS[channel];
      const target = Math.min(high, Math.max(low, targets[channel]));
      const delta = target - this.positions[channel];
      const maxStep = SURFACE_SLEW_RATE * dt;
      const step = Math.max(-maxStep, Math.min(maxStep, delta));
      this.positions[channel] += step;
    }
  }

  reset(): void {
    this.positions.fill(0.0);
  }
}
