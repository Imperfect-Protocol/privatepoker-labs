"""Radio link model: range, distance-based drop, constant delay.

Physics-derived topology: two drones are neighbours iff in range. Link
quality degrades with distance — no free lunch at the edge of range.
Deterministic given the rng; the sim, not this class, owns randomness.
"""
from __future__ import annotations

import math


class RadioLink:
    def __init__(self, *, radio_range: float = 300.0,
                 drop_exponent: float = 2.0,
                 max_drop: float = 0.5) -> None:
        self.radio_range = float(radio_range)
        self.drop_exponent = float(drop_exponent)
        self.max_drop = float(max_drop)

    def in_range(self, distance: float) -> bool:
        return distance <= self.radio_range

    def drop_probability(self, distance: float) -> float:
        """0 at the origin of range, rising to max_drop at the edge."""
        if not self.in_range(distance):
            return 1.0
        t = distance / self.radio_range
        return self.max_drop * (t ** self.drop_exponent)

    def roll_delivery(self, distance: float, uniform_sample: float) -> bool:
        """Deliver iff in range and the sample beats the drop probability."""
        return self.in_range(distance) and uniform_sample >= self.drop_probability(distance)

    def propagation_delay(self, distance: float) -> float:
        return 0.02 + distance / 3.0e8 * 1e3  # 20 ms base; light-speed term is decorative
