"""The tank: a slow, aggressive ground target.

State machine, explicitly: PATROL -> ENGAGING (sees a drone in range) ->
FIRING (cooldown elapsed) -> back to ENGAGING. It tracks the nearest drone
in sensor range and rotates its turret; if a drone holds still in front of
it too long, that drone eats a hit. Dumb on purpose: the interesting
intelligence in this project is the swarm's, not the tank's.
"""
from __future__ import annotations

import math

from skynet.vec import Vec3


class TankState:
    PATROL = "patrol"
    ENGAGING = "engaging"
    FIRING = "firing"


class Tank:
    SENSOR_RANGE = 150.0
    FIRE_COOLDOWN = 2.0
    TURRET_TRACK_RATE = 0.8    # rad/s

    def __init__(self, position: Vec3) -> None:
        self.position = position
        self.state = TankState.PATROL
        self.turret_azimuth = 0.0
        self.cooldown = 0.0
        self.shots_fired = 0
        self.hits = 0
        self.health = 100.0
        self.destroyed = False

    def step(self, dt: float, drones: list) -> Vec3 | None:
        """Advance one physics tick. Returns a shot Vec3 position if it fires."""
        if self.destroyed:
            return None
        self.cooldown = max(0.0, self.cooldown - dt)
        visible = [(drone_id, position) for drone_id, position in drones
                   if position.distance_to(self.position) <= self.SENSOR_RANGE]
        if not visible:
            self.state = TankState.PATROL
            return None
        nearest_id, nearest_position = min(
            visible, key=lambda item: item[1].distance_to(self.position))
        self.state = TankState.ENGAGING
        bearing = math.atan2(nearest_position.y - self.position.y,
                             nearest_position.x - self.position.x)
        # Track the nearest drone with a bounded slew.
        delta = (bearing - self.turret_azimuth + math.pi) % (2 * math.pi) - math.pi
        step = max(-self.TURRET_TRACK_RATE * dt, min(self.TURRET_TRACK_RATE * dt, delta))
        self.turret_azimuth += step
        if self.cooldown <= 0.0 and abs(delta) < 0.15:
            self.cooldown = self.FIRE_COOLDOWN
            self.shots_fired += 1
            self.state = TankState.FIRING
            return nearest_position
        return None

    def take_hit(self, damage: float) -> bool:
        """Apply damage. Returns True if this hit destroyed the tank."""
        if self.destroyed:
            return False
        self.health -= damage
        if self.health <= 0.0:
            self.destroyed = True
            self.state = "destroyed"
            return True
        return False
