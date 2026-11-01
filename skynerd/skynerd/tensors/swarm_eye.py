"""SwarmEye: per-drone visual memory and the glued decision tensor.

Each drone keeps a ring of its own last 8 eye captures plus the last capture
from every source it has heard from (capped at eye_max_sources). Received
eye-tensors travel in their capture frame with exact pose metadata; the
deterministic frame transform runs locally at fusion time — one resample per
consumer, never per hop.

The glued tensor is the derived decision input:

    Source(M) x Azimuth(8) x Elevation(8) x Range(8) x Time(8)

M is padded to a power of two; empty source slots are all-zero (the tensor's
own occupancy mask). Provenance stays exact in `sources`: (address, pose,
capture_time) rows in the same order as the Source axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skynerd.config import Config
from skynerd.vec import Vec3, azimuth_bin


@dataclass(frozen=True)
class EyeCapture:
    """One eye tensor plus its exact capture metadata. Fuzzy voxels, exact pose."""
    source_address: bytes
    eye: np.ndarray            # (8, 8, 8) uint8, capture frame
    heading_azimuth: float     # world-frame azimuth of the capture heading
    position: Vec3
    captured_at: float


def transform_into_my_frame(capture: EyeCapture, my_position: Vec3,
                            my_heading_azimuth: float, bins: int = 8) -> np.ndarray:
    """Deterministic re-frame of a received eye tensor.

    v1 transform: rotate the azimuth axis by the bearing difference between
    the source's capture heading and my current heading (parallax from the
    source offset is a refinement, documented as out of scope for v1).
    Pure math over (tensor, pose): same inputs, same output, everywhere.
    """
    delta = capture.heading_azimuth - my_heading_azimuth
    if delta == 0.0:
        return capture.eye.copy()
    bin_shift = azimuth_bin(Vec3(np.cos(delta), np.sin(delta), 0.0), bins)
    shifted = np.zeros_like(capture.eye)
    for az in range(bins):
        shifted[(az + bin_shift) % bins] = capture.eye[az]
    return shifted


class VisionMemory:
    """Owns self captures + received captures; builds the glued SwarmEye."""

    def __init__(self, config: Config) -> None:
        self.time_depth = config.time_depth
        self.max_sources = config.eye_max_sources
        self.ring: dict[bytes, list] = {}       # source_address -> [EyeCapture]
        self.cap = config.eye_bins

    def push(self, capture: EyeCapture) -> None:
        """Insert a capture; oldest slot evicted, stalest source evicted at cap."""
        history = self.ring.setdefault(capture.source_address, [])
        history.append(capture)
        if len(history) > self.time_depth:
            history.pop(0)
        if len(self.ring) > self.max_sources:
            oldest_source = min(self.ring, key=lambda a: self.ring[a][-1].captured_at)
            if oldest_source != capture.source_address or len(self.ring) > self.max_sources:
                del self.ring[oldest_source]

    def sources(self) -> list:
        """Provenance rows: latest capture per source, in Source-axis order."""
        return [history[-1] for history in self.ring.values()]

    def glue(self, my_position: Vec3, my_heading_azimuth: float) -> np.ndarray:
        """Build the padded decision tensor. Shape M x 8 x 8 x 8 x 8."""
        captures = self.sources()
        m_real = len(captures)
        m_padded = 1
        while m_padded < max(m_real, 1):
            m_padded *= 2
        glued = np.zeros((m_padded, self.cap, self.cap, self.cap, self.time_depth),
                         dtype=np.float32)
        for slot, capture in enumerate(captures):
            frame = (transform_into_my_frame(capture, my_position,
                                             my_heading_azimuth, self.cap)
                     if capture.position.distance_to(my_position) > 1e-6
                     else capture.eye)
            glued[slot, ..., 0] = frame        # newest time slot is index 0
        return glued
