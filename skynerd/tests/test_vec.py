"""Tests for skynerd.vec and skynerd.config. Run: python tests/test_vec.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynerd.config import Config
from skynerd.vec import (Vec3, Heading, RangeBins, azimuth_radians, azimuth_bin,
                        elevation_bin, elevation_radians, fuzzy_bin_weights)


def test_vector_basics():
    a = Vec3(1.0, 2.0, 3.0)
    b = Vec3.of([4.0, 5.0, 6.0])
    assert a.add(b) == Vec3(5.0, 7.0, 9.0)
    assert a.sub(b) == Vec3(-3.0, -3.0, -3.0)
    assert a.scale(2.0) == Vec3(2.0, 4.0, 6.0)
    assert abs(a.dot(b) - 32.0) < 1e-12
    c = Vec3(1.0, 0.0, 0.0).cross(Vec3(0.0, 1.0, 0.0))
    assert c == Vec3(0.0, 0.0, 1.0)
    assert abs(a.length() - math.sqrt(14.0)) < 1e-12
    assert Vec3(0.0, 0.0, 0.0).normalized() == Vec3(0.0, 0.0, 0.0)
    assert a.distance_to(b) == Vec3(3.0, 3.0, 3.0).length()


def test_rotation_and_heading():
    east = Vec3(1.0, 0.0, 0.0)
    north = east.rotated_about_z(math.pi / 2.0)
    assert abs(north.x) < 1e-12 and abs(north.y - 1.0) < 1e-12

    h = Heading.from_yaw_pitch(0.0, 0.0)
    assert abs(h.forward.x - 1.0) < 1e-12
    assert abs(h.right.y + 1.0) < 1e-12  # facing +X, right hand is -Y
    assert abs(h.up.z - 1.0) < 1e-12

    h = Heading.from_yaw_pitch(math.pi / 2.0, 0.0)  # facing +Y
    assert abs(h.forward.y - 1.0) < 1e-12
    assert abs(h.right.x - 1.0) < 1e-12  # right of +Y is +X

    # orthonormality
    for heading in [Heading.from_yaw_pitch(0.7, 0.2), Heading.from_yaw_pitch(-2.0, -0.4)]:
        assert abs(heading.forward.dot(heading.right)) < 1e-12
        assert abs(heading.forward.dot(heading.up)) < 1e-12
        assert abs(heading.right.dot(heading.up)) < 1e-12
        assert abs(heading.forward.length() - 1.0) < 1e-12


def test_azimuth_elevation_bins():
    # +X is azimuth bin 0 (with bins=8, bin width 45 deg, bin 0 centred on +X)
    assert azimuth_bin(Vec3(1.0, 0.0, 0.0), 8) == 0
    assert azimuth_bin(Vec3(0.0, 1.0, 0.0), 8) == 2   # +Y = 90 deg = bin 2
    assert azimuth_bin(Vec3(-1.0, 0.0, 0.0), 8) == 4  # -X = 180 deg = bin 4
    assert azimuth_bin(Vec3(0.0, -1.0, 0.0), 8) == 6  # -Y = 270 deg = bin 6

    # elevation: horizontal is mid bins, up is top bin, down is bin 0
    assert elevation_bin(Vec3(1.0, 0.0, 0.0), 8) == 4
    assert elevation_bin(Vec3(0.0, 0.0, 100.0), 8) == 7
    assert elevation_bin(Vec3(0.0, 0.0, -100.0), 8) == 0

    # round trip: bin -> representative direction -> same bin
    for i in range(8):
        yaw = i * (2.0 * math.pi / 8.0)
        offset = Vec3(math.cos(yaw), math.sin(yaw), 0.0)
        assert azimuth_bin(offset, 8) == i


def test_fuzzy_bin_weights():
    weights = fuzzy_bin_weights(0.0, 2.0 * math.pi, 8)
    total = sum(w for _, w in weights)
    assert abs(total - 1.0) < 1e-12
    # angle 0 is the exact centre of bin 0 -> all weight there
    assert dict(weights) == {0: 1.0, 1: 0.0}

    # angle at the bin 0 / bin 1 boundary splits half/half
    weights = fuzzy_bin_weights(math.pi / 8.0, 2.0 * math.pi, 8)
    d = dict(weights)
    assert abs(d[0] - 0.5) < 1e-12 and abs(d[1] - 0.5) < 1e-12

    # angle at three-quarters into bin 0 -> 75/25 over bins 0 and 1
    weights = fuzzy_bin_weights(math.pi / 16.0, 2.0 * math.pi, 8)
    d = dict(weights)
    assert abs(d[0] - 0.75) < 1e-12 and abs(d[1] - 0.25) < 1e-12

    # dominant bin always agrees with azimuth_bin's hard assignment
    for i in range(8):
        yaw = i * (2.0 * math.pi / 8.0) + 0.1  # slightly off-centre
        offset = Vec3(math.cos(yaw), math.sin(yaw), 0.0)
        dominant = max(fuzzy_bin_weights(azimuth_radians(offset), 2.0 * math.pi, 8),
                       key=lambda p: p[1])[0]
        assert dominant == azimuth_bin(offset, 8)
    # note: fuzzy_bin_weights is circular (azimuth). Elevation is clamped, not
    # circular (zenith is not adjacent to nadir), so elevation uses elevation_bin
    # directly; no fuzzy variant for it in v1.


def test_range_bins():
    bins = RangeBins(max_range=500.0, bins=8, min_range=1.0)
    assert bins.bin_of(1.0) == 0
    assert bins.bin_of(500.0) == 7
    assert bins.bin_of(10000.0) == 7     # clamped
    assert bins.bin_of(0.1) == 0         # clamped
    # monotonic
    distances = [2.0, 5.0, 10.0, 50.0, 100.0, 200.0, 400.0]
    indices = [bins.bin_of(d) for d in distances]
    assert indices == sorted(indices)
    # centers are geometric, inside their bins, increasing
    centers = [bins.bin_center(i) for i in range(8)]
    assert centers == sorted(centers)
    for i in range(8):
        assert bins.bin_of(centers[i]) == i


def test_config():
    cfg = Config()
    derived = cfg.derived()
    assert abs(derived.physics_dt - 0.02) < 1e-12
    assert abs(derived.motion_dt - 0.1) < 1e-12
    # frozen dataclass: replace makes a copy, original untouched
    faster = cfg.replace(physics_hz=100.0)
    assert abs(faster.derived().physics_dt - 0.01) < 1e-12
    assert abs(cfg.derived().physics_dt - 0.02) < 1e-12
    # seeded rng is reproducible
    assert cfg.seeded_rng().uniform() == cfg.seeded_rng().uniform()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
