"""3D vector math, heading frames, and binning for skynerd.

Pure numpy-backed math, no simulator dependencies. Everything the vision,
physics, and network layers need to talk about direction:

- ``Vec3``: small immutable 3D vector.
- Heading frame: a body frame built from a forward vector (right/up derived,
  roll-free v1 — good enough for bank-to-turn fixed-wing).
- Angle binning: azimuth bins around the horizontal plane (bin 0 = +X forward,
  positive clockwise seen from above) and elevation bins from -90..+90 deg.
  Range bins are log-spaced so nearby objects get fine resolution — the radio
  can only resolve distance coarsely far away anyway.
- Fuzzy bin weights: a direction never belongs to exactly one bin; it is a
  normalized distribution over the two nearest bins. Same philosophy as the
  p2p_nerual_spatial embeddings: fuzzy hints, exact metadata.

Angles are radians everywhere in code; bins are the only integer view.
"""
from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np


class Vec3:
    """Immutable 3D vector. Construct via Vec3(x, y, z) or Vec3.of(array)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @staticmethod
    def of(array) -> "Vec3":
        a = np.asarray(array, dtype=float).reshape(3)
        return Vec3(a[0], a[1], a[2])

    @staticmethod
    def zero() -> "Vec3":
        return Vec3(0.0, 0.0, 0.0)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def add(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> "Vec3":
        return Vec3(self.x * factor, self.y * factor, self.z * factor)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def distance_to(self, other: "Vec3") -> float:
        return self.sub(other).length()

    def normalized(self) -> "Vec3":
        n = self.length()
        if n == 0.0:
            return Vec3(0.0, 0.0, 0.0)
        return self.scale(1.0 / n)

    def rotated_about_z(self, radians: float) -> "Vec3":
        c, s = math.cos(radians), math.sin(radians)
        return Vec3(self.x * c - self.y * s, self.x * s + self.y * c, self.z)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)


class Heading:
    """A roll-free body frame: forward, right, up derived from a forward vector.

    Up is world +Z as long as forward is not vertical; a fixed-wing swarm in
    bank-to-turn flight never needs roll in its perception frame for v1.
    """
    __slots__ = ("forward", "right", "up")

    def __init__(self, forward: Vec3) -> None:
        f = forward.normalized()
        if f.length() == 0.0 or abs(f.z) > 0.999:
            f = Vec3(1.0, 0.0, 0.0)  # degenerate: face +X
        world_up = Vec3(0.0, 0.0, 1.0)
        right = f.cross(world_up).normalized()
        if right.length() == 0.0:
            right = Vec3(0.0, 1.0, 0.0)
        up = right.cross(f).normalized()
        self.forward = f
        self.right = right
        self.up = up

    @classmethod
    def from_yaw_pitch(cls, yaw: float, pitch: float) -> "Heading":
        """yaw: rotation about +Z from +X; pitch: positive nose-up."""
        cp = math.cos(pitch)
        forward = Vec3(cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch))
        return cls(forward)


def azimuth_radians(offset: Vec3) -> float:
    """Signed azimuth of offset in [-pi, pi); 0 = +X, positive clockwise from above."""
    return math.atan2(offset.y, offset.x)


def elevation_radians(offset: Vec3) -> float:
    """Signed elevation of offset in [-pi/2, pi/2]; 0 = horizontal, positive up."""
    horizontal = math.hypot(offset.x, offset.y)
    return math.atan2(offset.z, horizontal)


def azimuth_bin(offset: Vec3, bins: int) -> int:
    """Bin 0 centred on +X, positive clockwise from above."""
    angle = azimuth_radians(offset)
    shifted = (angle + math.pi / bins) % (2.0 * math.pi)
    return int(shifted // (2.0 * math.pi / bins)) % bins


def elevation_bin(offset: Vec3, bins: int) -> int:
    """Bins span -pi/2..+pi/2; bin 0 is the lowest (nadir), bins-1 zenith."""
    angle = elevation_radians(offset)
    shifted = (angle + math.pi / 2.0) / math.pi  # 0..1
    return min(bins - 1, max(0, int(shifted * bins)))


def fuzzy_bin_weights(angle: float, span: float, bins: int) -> List[Tuple[int, float]]:
    """Normalized weights over the two bins nearest to ``angle``.

    ``span`` is the total angular width the bins divide (2*pi for azimuth,
    pi for elevation). Bin centres sit at multiples of the bin width, so bin 0
    is centred on angle 0 — the same convention as ``azimuth_bin``. Returns
    [(bin, weight), ...] with weights summing to 1. A direction is a fuzzy
    hint, never a hard assignment.
    """
    width = span / bins
    x = (angle % span) / width                  # bin centres at integers
    nearest = math.floor(x + 0.5)               # matches azimuth_bin's rounding
    t = x - nearest                             # in [-0.5, 0.5)
    w_near = 1.0 - abs(t)
    other = (nearest - 1) % bins if t < 0.0 else (nearest + 1) % bins
    return [(nearest % bins, w_near), (other, abs(t))]


class RangeBins:
    """Log-spaced range bins over (min_range, max_range]. Bin 0 is the closest."""
    def __init__(self, max_range: float, bins: int, min_range: float = 1.0) -> None:
        if min_range <= 0.0 or max_range <= min_range:
            raise ValueError("range bins need 0 < min_range < max_range")
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.bins = int(bins)
        self._log_min = math.log(min_range)
        self._log_max = math.log(max_range)

    def bin_of(self, distance: float) -> int:
        d = min(max(distance, self.min_range), self.max_range)
        t = (math.log(d) - self._log_min) / (self._log_max - self._log_min)
        return min(self.bins - 1, int(t * self.bins))

    def bin_center(self, index: int) -> float:
        """Geometric centre distance of a bin (the representative range)."""
        t = (index + 0.5) / self.bins
        return math.exp(self._log_min + t * (self._log_max - self._log_min))


class Quaternion:
    """Unit quaternion for attitude. Body convention: X forward, Y right, Z up.

    Positive rotation about +Y pitches the nose DOWN (right-handed axes with
    Z up), positive about +Z yaws the nose right (+X toward +Y), positive
    about +X rolls left. Control surfaces are defined in the learner-friendly
    convention (positive elevator = nose up) and the physics layer translates.
    """
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def normalized(self) -> "Quaternion":
        n = math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if n == 0.0:
            return Quaternion()
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def multiply(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate(self, vector: Vec3) -> Vec3:
        """Rotate a world vector by this attitude (body-to-world for attitudes)."""
        q = self
        u = Vec3(q.x, q.y, q.z)
        t = u.cross(vector).scale(2.0)
        return vector.add(t.scale(q.w)).add(u.cross(t))

    @classmethod
    def from_axis_angle(cls, axis: Vec3, radians: float) -> "Quaternion":
        a = axis.normalized()
        half = radians / 2.0
        s = math.sin(half)
        return cls(math.cos(half), a.x * s, a.y * s, a.z * s)

    @classmethod
    def from_euler(cls, yaw: float, pitch: float, roll: float) -> "Quaternion":
        """ZYX Euler composition: yaw about +Z, then pitch about +Y, then roll about +X."""
        return (cls.from_axis_angle(Vec3(0.0, 0.0, 1.0), yaw)
                .multiply(cls.from_axis_angle(Vec3(0.0, 1.0, 0.0), pitch))
                .multiply(cls.from_axis_angle(Vec3(1.0, 0.0, 0.0), roll)))

    def body_axes(self) -> Tuple[Vec3, Vec3, Vec3]:
        """(forward, right, up): this attitude applied to +X, +Y, +Z."""
        return (self.rotate(Vec3(1.0, 0.0, 0.0)),
                self.rotate(Vec3(0.0, 1.0, 0.0)),
                self.rotate(Vec3(0.0, 0.0, 1.0)))

    def integrate_body_rates(self, p: float, q: float, r: float, dt: float) -> "Quaternion":
        """Advance attitude by body angular rates (rad/s) over dt; renormalizes."""
        # q_dot = 0.5 * q ⊗ (0, omega_body); explicit Euler is fine at small dt.
        half = 0.5 * dt
        dq = Quaternion(
            -half * (p * self.x + q * self.y + r * self.z),
            half * (p * self.w + q * self.z - r * self.y),
            half * (q * self.w + r * self.x - p * self.z),
            half * (r * self.w + p * self.y - q * self.x),
        )
        return Quaternion(self.w + dq.w, self.x + dq.x, self.y + dq.y, self.z + dq.z).normalized()
