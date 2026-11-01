"""Tests for skynerd vision. Run: python tests/test_vision.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynerd.config import Config
from skynerd.physics.rigid_body import RigidBody
from skynerd.physics.world import Box, World
from skynerd.tensors.swarm_eye import (EyeCapture, VisionMemory,
                                      transform_into_my_frame)
from skynerd.vec import Vec3
from skynerd.vision.raycaster import RadarEye

CONFIG = Config()
WORLD = World(boxes=(Box(Vec3(100.0, -5.0, 0.0), Vec3(110.0, 5.0, 20.0)),))


def flying(yaw: float = 0.0, pos: Vec3 | None = None) -> RigidBody:
    return RigidBody(pos or Vec3(0.0, 0.0, 30.0), Vec3(10.5, 0.0, 0.0), yaw=yaw)


def test_eye_sees_box_dead_ahead():
    eye = RadarEye(CONFIG, WORLD).sweep(flying())
    # Box spans 100-110m ahead: azimuth centre bin, mid range bins.
    assert eye.sum() > 0
    centre_az = eye[4]          # azimuth index 4 = +X with span 2pi/8 from -pi
    assert centre_az.sum() > 0
    hit_bins = np.argwhere(centre_az > 0)
    ranges = {rb for _, rb in hit_bins}
    # Box at 100-110 m sits in the upper-middle log range bins (1..500 m span).
    assert all(3 <= rb <= 7 for rb in ranges), ranges


def test_eye_empty_sky_is_zero():
    # Flat ground still exists: down-looking rays hit it 30 m below.
    eye = RadarEye(CONFIG, World()).sweep(flying())
    above_horizon = eye[:, 4:, :]               # elevation axis is 1
    assert above_horizon.sum() == 0
    # The nadir cone sees the ground.
    assert eye[:, 0, :].sum() > 0


def test_eye_occlusion_is_real():
    # A tall wall at ~50 m and a big hill behind it: the wall must shadow
    # the hill in the forward cone — no far hits may appear there.
    world = World(boxes=(Box(Vec3(50.0, -2.0, 0.0), Vec3(55.0, 2.0, 60.0)),),
                  hills=(__import__("skynerd.physics.world", fromlist=["Hill"])
                         .Hill(400.0, 0.0, 80.0, 60.0),))
    eye = RadarEye(CONFIG, world).sweep(flying())
    forward_cone = eye[4]                       # azimuth bin centred on +X
    assert forward_cone.sum() > 0               # the wall is seen
    # Every forward-cone hit is the wall (near range bins), never the hill.
    assert forward_cone[:, 6:].sum() == 0
    # Sanity: without the wall the hill would be visible far away.
    eye_open = RadarEye(CONFIG, World(hills=world.hills)).sweep(flying())
    assert eye_open[4][:, 6:].sum() > 0


def test_transform_rotates_azimuth():
    eye = np.zeros((8, 8, 8), dtype=np.uint8)
    eye[4, 4, 5] = 7              # dead ahead in the source's frame
    capture = EyeCapture(source_address=b"x" * 20, eye=eye,
                         heading_azimuth=np.pi / 2.0,   # source faced +Y
                         position=Vec3(100.0, 0.0, 30.0), captured_at=1.0)
    # I face +X: the source's "ahead" is to my right -> shift to azimuth bin 6.
    transformed = transform_into_my_frame(capture, Vec3(0.0, 0.0, 30.0), 0.0)
    assert transformed[4, 4, 5].sum() == 0
    assert transformed[6, 4, 5] == 7
    # Same heading: identity.
    same = transform_into_my_frame(capture, Vec3(0.0, 0.0, 30.0), np.pi / 2.0)
    assert same[4, 4, 5] == 7


def test_memory_ring_and_cap():
    memory = VisionMemory(CONFIG)
    for tick in range(12):
        memory.push(EyeCapture(source_address=b"a" * 20,
                               eye=np.full((8, 8, 8), tick, dtype=np.uint8),
                               heading_azimuth=0.0,
                               position=Vec3(0.0, 0.0, 30.0), captured_at=tick))
    history = memory.ring[b"a" * 20]
    assert len(history) == 8                    # ring capped at time_depth
    assert history[-1].captured_at == 11
    assert history[0].captured_at == 4          # oldest four evicted


def test_memory_source_cap_evicts_stalest():
    memory = VisionMemory(CONFIG)
    for source in range(CONFIG.eye_max_sources + 2):
        memory.push(EyeCapture(source_address=bytes([source]) * 20,
                               eye=np.zeros((8, 8, 8), dtype=np.uint8),
                               heading_azimuth=0.0,
                               position=Vec3(float(source), 0.0, 30.0),
                               captured_at=float(source)))
    assert len(memory.ring) == CONFIG.eye_max_sources
    assert b"\x00" * 20 not in memory.ring      # stalest evicted


def test_glued_tensor_shape_and_content():
    memory = VisionMemory(CONFIG)
    own = np.zeros((8, 8, 8), dtype=np.uint8)
    own[4, 4, 3] = 1
    memory.push(EyeCapture(source_address=b"a" * 20, eye=own,
                           heading_azimuth=0.0,
                           position=Vec3(0.0, 0.0, 30.0), captured_at=0.0))
    theirs = np.zeros((8, 8, 8), dtype=np.uint8)
    theirs[4, 4, 5] = 2
    memory.push(EyeCapture(source_address=b"b" * 20, eye=theirs,
                           heading_azimuth=np.pi / 2.0,
                           position=Vec3(100.0, 0.0, 30.0), captured_at=0.5))
    glued = memory.glue(Vec3(0.0, 0.0, 30.0), 0.0)
    assert glued.shape == (2, 8, 8, 8, 8)       # M padded to 2
    # Self slice untransformed, newest time slot 0.
    assert glued[0, 4, 4, 3, 0] == 1
    # Received slice re-framed: source faced +Y, I face +X -> bin 6.
    assert glued[1, 6, 4, 5, 0] == 2
    # Provenance rows align with the Source axis.
    addresses = [c.source_address for c in memory.sources()]
    assert addresses == [b"a" * 20, b"b" * 20]
    # Time slots beyond the newest are zero (no invented history).
    assert glued[..., 1:].sum() == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
