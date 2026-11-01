"""Tests for skynet.physics. Run: python tests/test_physics.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynet.physics.aero import AeroTables
from skynet.physics.rigid_body import RigidBody
from skynet.physics.surfaces import (SurfacesState, SURFACE_LIMITS,
                                     SURFACE_SLEW_RATE)
from skynet.physics.world import World, Hill, Box
from skynet.vec import Vec3

DT = 0.02


def cruise_body(y=0.0, z=100.0, speed=18.0, yaw=0.0):
    return RigidBody(Vec3(0.0, y, z), Vec3(speed, 0.0, 0.0), yaw=yaw)


def run(body: RigidBody, seconds: float, targets=(0.0, 0.0, 0.0, 0.0)) -> RigidBody:
    servos = SurfacesState()
    steps = int(round(seconds / DT))
    for _ in range(steps):
        servos.command(targets, DT)
        body.step(servos, DT)
    return body


def test_servo_slew_and_limits():
    servos = SurfacesState()
    servos.command((1.0, -1.0, 0.5, 1.0), DT)   # big targets
    # One tick at 50 Hz: max travel is slew_rate * dt
    max_step = SURFACE_SLEW_RATE * DT
    assert abs(servos.positions[0] - max_step) < 1e-12
    assert abs(servos.positions[1] + max_step) < 1e-12
    # Throttle slews by the same rate per second toward [0, 1]
    assert abs(servos.positions[3] - max_step) < 1e-12
    for _ in range(10000):
        servos.command((10.0, -10.0, 10.0, 10.0), DT)
    for channel in range(4):
        low, high = SURFACE_LIMITS[channel]
        assert low - 1e-12 <= servos.positions[channel] <= high + 1e-12
    assert abs(servos.positions[0] - SURFACE_LIMITS[0][1]) < 1e-12


def test_quaternion_yaw_integration():
    body = cruise_body()
    body.rate_r = math.pi / 4.0   # nose right, per our convention
    attitude = body.attitude
    for _ in range(50):            # 1 second
        attitude = attitude.integrate_body_rates(0.0, 0.0, body.rate_r, DT)
    forward, _, _ = attitude.body_axes()
    expected_yaw = math.pi / 4.0
    # Explicit Euler over 50 steps carries O(dt) truncation error.
    assert abs(forward.x - math.cos(expected_yaw)) < 1e-3
    assert abs(forward.y - math.sin(expected_yaw)) < 1e-3
    assert abs(forward.z) < 1e-9


def test_free_fall():
    body = RigidBody(Vec3(0.0, 0.0, 100.0), Vec3(0.0, 0.0, 0.0))
    run(body, 1.0)
    # Falls slower than vacuum: belly-first, alpha ~ 90 deg, drag is real.
    assert 94.0 < body.position.z < 96.5      # vacuum would be 95.1
    assert -9.9 < body.velocity.z < -4.0


def test_glide():
    """Launched at trim speed with neutral surfaces: sinks slowly, stays intact."""
    body = cruise_body(z=100.0, speed=11.0)
    run(body, 5.0)
    assert 0.0 < body.position.z < 100.0          # sank, did not climb for free
    assert body.position.z > 70.0                 # glide ratio better than ~2:1
    speed = body.velocity.length()
    assert 6.0 < speed < 18.0                     # energy-conserving envelope
    assert all(math.isfinite(v) for v in (body.velocity.x, body.velocity.y, body.velocity.z))


def test_powered_flight_stays_bounded():
    """Full throttle, neutral surfaces: climbs (thrust >> drag), stays bounded."""
    body = cruise_body(z=50.0, speed=15.0)
    run(body, 30.0, targets=(0.0, 0.0, 0.0, 1.0))
    speed = body.velocity.length()
    assert speed < 40.0
    assert body.position.z > 50.0                 # thrust converts to climb, not magic
    assert body.position.z < 400.0                # bounded, no divergence
    for rate in (body.rate_p, body.rate_q, body.rate_r):
        assert abs(rate) < 20.0
    assert all(math.isfinite(v) for v in (body.velocity.x, body.velocity.y, body.velocity.z))


def test_elevator_pitch_response():
    neutral = cruise_body(speed=18.0)
    pulling = cruise_body(speed=18.0)
    run(neutral, 1.0)
    run(pulling, 1.0, targets=(0.0, 0.35, 0.0, 0.0))
    # Positive elevator = nose up: forward vector gains +Z vs neutral.
    f_neutral, _, _ = neutral.body_axes
    f_pull, _, _ = pulling.body_axes
    assert f_pull.z > f_neutral.z + 0.05
    assert abs(f_neutral.z) < 0.2                 # neutral stays roughly level


def test_aileron_roll_response():
    neutral = cruise_body(speed=18.0)
    rolling = cruise_body(speed=18.0)
    run(neutral, 1.0)
    run(rolling, 1.0, targets=(0.35, 0.0, 0.0, 0.0))
    # Positive aileron = roll right = right wing down; the canopy (up vector)
    # leans toward +Y (right) vs neutral.
    _, _, up_neutral = neutral.body_axes
    _, _, up_roll = rolling.body_axes
    assert up_roll.y > up_neutral.y + 0.05


def test_rudder_yaw_response():
    neutral = cruise_body(speed=18.0)
    turning = cruise_body(speed=18.0)
    run(neutral, 1.0)
    run(turning, 1.0, targets=(0.0, 0.0, 0.35, 0.0))
    f_neutral, _, _ = neutral.body_axes
    f_turn, _, _ = turning.body_axes
    # Positive rudder = nose right (+Y component grows vs neutral).
    assert f_turn.y > f_neutral.y + 0.02


def test_determinism():
    a = run(cruise_body(), 3.0, targets=(0.1, 0.05, 0.0, 0.6))
    b = run(cruise_body(), 3.0, targets=(0.1, 0.05, 0.0, 0.6))
    assert a.position == b.position
    assert a.attitude.body_axes() == b.attitude.body_axes()


def test_terrain_height():
    world = World(hills=(Hill(100.0, 0.0, 30.0, 40.0),))
    assert world.terrain_height(0.0, 0.0) < 1.5    # hill tail at distance
    assert abs(world.terrain_height(100.0, 0.0) - 30.0) < 1e-9
    assert 0.0 < world.terrain_height(100.0, 80.0) < 30.0
    assert World().terrain_height(500.0, -500.0) == 0.0


def test_raycast_terrain():
    world = World(hills=(Hill(100.0, 0.0, 30.0, 40.0),))
    ground_at_origin = world.terrain_height(0.0, 0.0)   # ~1.32: hill tail
    # Downward ray: hits local ground height exactly.
    assert abs(world.raycast(Vec3(0.0, 0.0, 10.0), Vec3(0.0, 0.0, -1.0), 100.0)
               - (10.0 - ground_at_origin)) < 0.1
    # Horizontal ray into the hill: hits before x=200, on the surface.
    hit = world.raycast(Vec3(0.0, 0.0, 10.0), Vec3(1.0, 0.0, 0.0), 500.0)
    assert math.isfinite(hit) and hit < 200.0
    hit_point = Vec3(0.0, 0.0, 10.0).add(Vec3(1.0, 0.0, 0.0).normalized().scale(hit))
    surface = world.terrain_height(hit_point.x, hit_point.y)
    assert abs(hit_point.z - surface) < 0.5
    # Upward ray never hits terrain.
    assert world.raycast(Vec3(0.0, 0.0, 10.0), Vec3(0.0, 0.0, 1.0), 500.0) == math.inf
    # Max distance respected: 20 m stops well before the slope reaches z=10.
    assert world.raycast(Vec3(0.0, 0.0, 10.0), Vec3(1.0, 0.0, 0.0), 20.0) == math.inf


def test_raycast_boxes():
    world = World(boxes=(Box(Vec3(50.0, -5.0, 0.0), Vec3(60.0, 5.0, 20.0)),))
    assert abs(world.raycast(Vec3(0.0, 0.0, 10.0), Vec3(1.0, 0.0, 0.0), 500.0) - 50.0) < 1e-6
    # Misses past the box edge.
    assert world.raycast(Vec3(0.0, 50.0, 10.0), Vec3(1.0, 0.0, 0.0), 500.0) == math.inf
    # Nearest hit wins over terrain: the ray at z=15 clears the hill's slope
    # (which only reaches z=15 at x≈52.9) but not the box (z spans 0..20).
    world = World(boxes=(Box(Vec3(50.0, -5.0, 0.0), Vec3(60.0, 5.0, 20.0)),),
                  hills=(Hill(100.0, 0.0, 30.0, 40.0),))
    assert abs(world.raycast(Vec3(0.0, 0.0, 15.0), Vec3(1.0, 0.0, 0.0), 500.0) - 50.0) < 1e-6
    # And from lower, the hill genuinely occludes the box first.
    assert abs(world.raycast(Vec3(0.0, 0.0, 10.0), Vec3(1.0, 0.0, 0.0), 500.0) - 40.7) < 0.5


def test_stall_exists():
    """A drone pitched up hard at low speed loses lift and sinks fast."""
    body = RigidBody(Vec3(0.0, 0.0, 100.0), Vec3(8.0, 0.0, 0.0))
    run(body, 4.0, targets=(0.0, 0.35, 0.0, 0.2))
    # Deep stall: it is falling, not flying.
    assert body.velocity.z < -5.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
