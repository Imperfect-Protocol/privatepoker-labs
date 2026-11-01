"""The world: gravity, air, terrain, and raycastable occluders.

The physics layer needs only gravity and air density. The vision layer needs
raycast: real occlusion, no sim-cheating — a ray stops at the first hit, so
drones cannot see through hills or buildings. Terrain is flat ground plus
optional smooth hills (sum of Gaussians); occluders are axis-aligned boxes
(buildings, walls).

The mission layer (tanks, station, carpet) is built elsewhere; this module is
pure geometry and environment constants.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from skynet.physics.aero import GRAVITY, RHO_SEA_LEVEL
from skynet.vec import Vec3

_EARTH_EPSILON = 1e-9


@dataclass(frozen=True)
class Hill:
    center_x: float
    center_y: float
    height: float
    radius: float   # Gaussian sigma, metres


@dataclass(frozen=True)
class Box:
    """Axis-aligned box occluder."""
    min_corner: Vec3
    max_corner: Vec3


class World:
    def __init__(self, *, hills: tuple = (), boxes: tuple = (),
                 gravity: Vec3 = GRAVITY, rho: float = RHO_SEA_LEVEL,
                 half_extent: float = 1000.0) -> None:
        self.hills = tuple(hills)
        self.boxes = tuple(boxes)
        self.gravity = gravity
        self.rho = rho
        self.half_extent = half_extent

    def terrain_height(self, x: float, y: float) -> float:
        height = 0.0
        for hill in self.hills:
            d2 = (x - hill.center_x) ** 2 + (y - hill.center_y) ** 2
            height += hill.height * math.exp(-d2 / (2.0 * hill.radius * hill.radius))
        return height

    def is_inside_bounds(self, position: Vec3) -> bool:
        return (abs(position.x) <= self.half_extent
                and abs(position.y) <= self.half_extent
                and position.z >= -10.0)

    def raycast(self, origin: Vec3, direction: Vec3,
                max_distance: float) -> float:
        """Distance to first hit (terrain or box), or math.inf if clear.

        ``direction`` need not be normalized. Real occlusion: first hit wins.
        """
        direction = direction.normalized()
        if direction.length() == 0.0:
            return math.inf
        hit_box = self._raycast_boxes(origin, direction, max_distance)
        hit_terrain = self._raycast_terrain(origin, direction,
                                            min(max_distance, hit_box))
        return min(hit_box, hit_terrain)

    def _raycast_boxes(self, origin: Vec3, direction: Vec3,
                       max_distance: float) -> float:
        """Nearest slab-intersection distance across all boxes, or inf."""
        best = math.inf
        for box in self.boxes:
            t = _ray_box(origin, direction, box)
            if t < best:
                best = t
        return best

    def _raycast_terrain(self, origin: Vec3, direction: Vec3,
                         max_distance: float) -> float:
        """March along the ray, bisect when it dips below terrain."""
        if origin.z < self.terrain_height(origin.x, origin.y):
            return 0.0
        flat_shortcut = (not self.hills and direction.z >= 0.0)
        if flat_shortcut:
            return math.inf  # upward ray over flat ground never hits
        step = max(1.0, max_distance / 256.0)
        travelled = 0.0
        previous_above = True
        while travelled < max_distance:
            step = min(step, max_distance - travelled)
            travelled += step
            point = origin.add(direction.scale(travelled))
            above = point.z >= self.terrain_height(point.x, point.y)
            if previous_above and not above:
                return _bisect_terrain(self, origin, direction,
                                       travelled - step, travelled)
            previous_above = above
        return math.inf


def _ray_box(origin: Vec3, direction: Vec3, box: Box) -> float:
    """Slab method. Returns hit distance or math.inf."""
    t_min, t_max = 0.0, math.inf
    for o, d, lo, hi in ((origin.x, direction.x, box.min_corner.x, box.max_corner.x),
                         (origin.y, direction.y, box.min_corner.y, box.max_corner.y),
                         (origin.z, direction.z, box.min_corner.z, box.max_corner.z)):
        if abs(d) < _EARTH_EPSILON:
            if o < lo or o > hi:
                return math.inf
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return math.inf
    if t_max < 0.0:
        return math.inf
    return max(t_min, 0.0)


def _bisect_terrain(world: World, origin: Vec3, direction: Vec3,
                    low: float, high: float, iterations: int = 12) -> float:
    """Bisect the crossing between ``low`` (above) and ``high`` (below)."""
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        point = origin.add(direction.scale(middle))
        if point.z >= world.terrain_height(point.x, point.y):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
