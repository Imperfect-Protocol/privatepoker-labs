"""Flight instruments and the dense per-tick reward signal.

Egocentric error signals only — no GPS, no absolute position. The same
philosophy as the twisted embeddings: what a drone feels, not where it is.

Three normalized error channels in [-1, 1], each binned into 8:

- vertical-speed error: climbing or sinking relative to level flight
- bank error: wing-level error (rolled right is positive)
- airspeed error: relative to the airframe's trim speed

The dense reward is error *improvement*: positive when the weighted error
norm shrinks. This gives the servo tensor immediate credit at the timescale
where Hebbian correlation actually converges. Crash detection is physical:
ground contact or loss of control ends the episode.
"""
from __future__ import annotations

import math

from skynerd.physics.rigid_body import RigidBody
from skynerd.vec import Vec3

ERROR_BINS = 8


def bin_error(value: float, bins: int = ERROR_BINS) -> int:
    """Bin a value in [-1, 1] into [0, bins-1]; clamps at the ends."""
    clamped = max(-1.0, min(1.0, value))
    shifted = (clamped + 1.0) / 2.0          # 0..1
    index = int(shifted * bins)
    return min(bins - 1, max(0, index))


class FlightInstruments:
    """Turns a RigidBody into binned flight-state keys and error norms."""

    def __init__(self, trim_speed: float, speed_band: float = 6.0,
                 vs_band: float = 4.0) -> None:
        self.trim_speed = float(trim_speed)
        self.speed_band = float(speed_band)
        self.vs_band = float(vs_band)

    def errors(self, body: RigidBody) -> Vec3:
        """(vs_err, bank_err, speed_err), each normalized to [-1, 1]."""
        vs_err = max(-1.0, min(1.0, body.velocity.z / self.vs_band))
        forward, right, up = body.body_axes
        # Rolled right => right wing down => right.z < 0 => bank positive.
        bank = -math.asin(max(-1.0, min(1.0, right.z)))
        bank_err = max(-1.0, min(1.0, bank / (math.pi / 4.0)))
        speed = body.velocity.length()
        speed_err = max(-1.0, min(1.0, (speed - self.trim_speed) / self.speed_band))
        return Vec3(vs_err, bank_err, speed_err)

    def heading_err(self, body: RigidBody, target_bearing: float) -> float:
        """HSI: signed bearing delta to target, normalized [-1, 1] by 90 deg.

        Positive delta: the target bearing lies to the LEFT (math convention,
        +Z up): turning left increases bearing. The HSI needle, not a GPS.
        """
        forward, _, _ = body.body_axes
        bearing = math.atan2(forward.y, forward.x)
        delta = (target_bearing - bearing + math.pi) % (2.0 * math.pi) - math.pi
        return max(-1.0, min(1.0, delta / (math.pi / 2.0)))

    def errors4(self, body: RigidBody, target_bearing: float) -> tuple:
        """(vs, bank, speed, heading) errors — the full panel with the HSI."""
        errors = self.errors(body)
        return (errors.x, errors.y, errors.z,
                self.heading_err(body, target_bearing))

    def state_key(self, body: RigidBody,
                  throttle_position: float = 0.15,
                  target_bearing: float | None = None) -> tuple:
        """(vs, bank, asi, rpm, hsi) bins — VSI, bank, ASI, engine, HSI.

        Without a bearing target the HSI reads centered (the drone is
        on-course by definition).
        """
        errors = self.errors(body)
        heading = 0.0 if target_bearing is None else self.heading_err(body, target_bearing)
        return (bin_error(errors.x), bin_error(errors.y), bin_error(errors.z),
                bin_error(max(-1.0, min(1.0, throttle_position * 2.0 - 1.0))),
                bin_error(heading))

    def penalty(self, body: RigidBody) -> float:
        """Weighted error norm: lower is better flight."""
        errors = self.errors(body)
        return abs(errors.x) + 0.5 * abs(errors.y) + 0.3 * abs(errors.z)


class DenseReward:
    """Baseline-centered per-tick reward around the weighted error norm.

    Raw improvement is mean-positive while gliding toward trim, which floods
    every action with credit. Subtracting a running mean (EMA) makes average
    flight worth zero: only better-than-recent-average control earns positive
    weight, and only worse-than-average earns negative.
    """

    def __init__(self, instruments: FlightInstruments, scale: float = 5.0,
                 baseline_alpha: float = 0.01) -> None:
        self.instruments = instruments
        self.scale = float(scale)
        self.baseline_alpha = float(baseline_alpha)
        self._previous_penalty = None
        self._baseline = 0.0

    def reset(self, body: RigidBody) -> None:
        self._previous_penalty = self.instruments.penalty(body)
        self._baseline = 0.0

    def tick(self, body: RigidBody) -> float:
        penalty = self.instruments.penalty(body)
        if self._previous_penalty is None:
            self._previous_penalty = penalty
        improvement = self._previous_penalty - penalty
        self._previous_penalty = penalty
        centered = improvement - self._baseline
        self._baseline += self.baseline_alpha * (improvement - self._baseline)
        return centered * self.scale


class EpisodeOutcome:
    """How a flight ended. Named states, always."""

    ALIVE = "alive"        # reached the time horizon still flying
    CRASHED = "crashed"    # ground contact
    STALLED = "stalled"    # too slow for too long
    OUT_OF_BOUNDS = "out_of_bounds"


def classify_outcome(body: RigidBody, world, elapsed: float, *,
                     stall_speed: float = 4.0,
                     slow_limit: float = 1.5) -> str | None:
    """None while flying; an EpisodeOutcome constant when the episode ends."""
    ground = world.terrain_height(body.position.x, body.position.y)
    if body.position.z <= ground + 1.0:
        return EpisodeOutcome.CRASHED
    if body.velocity.length() < stall_speed:
        body.slow_seconds = getattr(body, "slow_seconds", 0.0) + 0.02
        if body.slow_seconds >= slow_limit:
            return EpisodeOutcome.STALLED
    else:
        body.slow_seconds = 0.0
    if not world.is_inside_bounds(body.position):
        return EpisodeOutcome.OUT_OF_BOUNDS
    return None
