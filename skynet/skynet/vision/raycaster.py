"""Radar eye: a spherical raycast sweep mapped to a regular 8x8x8 tensor.

No perspective projection, no sim-cheating: from the drone's position, one
ray per (azimuth, elevation) cone marches the world and returns the first
hit distance — real occlusion, drones cannot see through hills or boxes.
The hit distance bins into log-spaced range bins (same RangeBins as vec).
A cone with no hit within eye_max_range contributes zero.

Output is the capture-frame EyeTensor: Azimuth(8) x Elevation(8) x Range(8),
uint8 occupancy counts. Fuzzy values, exact pose metadata travels beside it.
"""
from __future__ import annotations

import numpy as np

from skynet.config import Config
from skynet.physics.rigid_body import RigidBody
from skynet.physics.world import World
from skynet.vec import Heading, RangeBins, Vec3, azimuth_bin, elevation_bin


class RadarEye:
    def __init__(self, config: Config, world: World) -> None:
        self.world = world
        self.bins = config.eye_bins
        self.range_max = config.eye_range_max
        self.range_bins = RangeBins(max_range=self.range_max, bins=self.bins,
                                    min_range=1.0)

    def sweep(self, body: RigidBody) -> np.ndarray:
        """One capture: cast bins x bins rays along the body heading frame."""
        forward, right, up = body.body_axes
        eye = np.zeros((self.bins, self.bins, self.bins), dtype=np.uint8)
        span = 2.0 * np.pi / self.bins
        half_span = np.pi / self.bins
        for az_index in range(self.bins):
            # Azimuth bin centres relative to body forward (+X body).
            az = az_index * span - np.pi
            for el_index in range(self.bins):
                el = (el_index + 0.5) * (np.pi / self.bins) - np.pi / 2.0
                direction = (forward.scale(np.cos(az) * np.cos(el))
                             .add(right.scale(np.sin(az) * np.cos(el)))
                             .add(up.scale(np.sin(el))))
                hit = self.world.raycast(body.position, direction, self.range_max)
                if hit < self.range_max:
                    eye[az_index, el_index, self.range_bins.bin_of(hit)] += 1
        return eye
