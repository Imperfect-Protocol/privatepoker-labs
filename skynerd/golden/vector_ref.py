"""Reference vectors for the TypeScript port (Genesis Milestone 1).

Computes a fixed set of Vec3 / Heading / Quaternion / binning results with
the Python implementation and writes them as JSON. The TypeScript test suite
compares its own outputs against this document within tolerance.

Usage (from privatepoker-labs/skynerd):
    python golden/vector_ref.py --out ../skynerd_genesis/reference/vector-ref.json
"""
from __future__ import annotations

import argparse
import sys
import json
import math
import platform
from pathlib import Path

SKYNERD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYNERD))

import numpy as np

from skynerd.vec import (Vec3, Heading, Quaternion, RangeBins,
                         azimuth_radians, elevation_radians,
                         azimuth_bin, elevation_bin, fuzzy_bin_weights)
from skynerd.physics.world import World, Hill, Box
from skynerd.config import Config


def vec(v: Vec3) -> list:
    return [v.x, v.y, v.z]


def quat(q: Quaternion) -> list:
    return [q.w, q.x, q.y, q.z]


def main() -> int:
    parser = argparse.ArgumentParser(description="emit vector reference JSON")
    parser.add_argument("--out", default="../skynerd_genesis/reference/vector-ref.json")
    args = parser.parse_args()
    out: dict = {"schema": "skynerd-vector-ref/1",
                 "python": platform.python_version(),
                 "numpy": np.__version__}

    # --- Vec3 arithmetic -------------------------------------------------
    a = Vec3(1.5, -2.0, 3.25)
    b = Vec3(-0.5, 4.0, 1.0)
    out["vec3"] = {
        "a": vec(a), "b": vec(b),
        "add": vec(a.add(b)), "sub": vec(a.sub(b)),
        "scale": vec(a.scale(2.5)),
        "dot": a.dot(b),
        "cross": vec(a.cross(b)),
        "length": a.length(),
        "distance": a.distance_to(b),
        "normalized": vec(a.normalized()),
        "normalized_zero": vec(Vec3.zero().normalized()),
        "rotated_about_z": vec(a.rotated_about_z(0.7)),
    }

    # --- Heading ----------------------------------------------------------
    cases = [Vec3(1.0, 0.0, 0.0), Vec3(0.3, 0.4, 0.1), Vec3(0.0, -2.0, 0.5),
             Vec3(0.0, 0.0, 5.0), Vec3(0.0, 0.0, 0.0)]
    out["heading"] = [
        {"forward": vec(h.forward), "right": vec(h.right), "up": vec(h.up)}
        for h in (Heading(c) for c in cases)]

    # --- angles and bins ---------------------------------------------------
    offsets = [Vec3(1.0, 0.0, 0.0), Vec3(1.0, 1.0, 0.0), Vec3(-1.0, 0.2, 0.3),
               Vec3(0.0, -1.0, -0.2), Vec3(0.5, 0.5, 2.0)]
    out["angles"] = [{"azimuth": azimuth_radians(o), "elevation": elevation_radians(o),
                      "az_bin8": azimuth_bin(o, 8), "el_bin8": elevation_bin(o, 8)}
                     for o in offsets]
    out["fuzzy_azimuth"] = [
        {"angle": azimuth_radians(o), "weights": [[i, w] for i, w in
                                                  fuzzy_bin_weights(azimuth_radians(o), 2 * math.pi, 8)]}
        for o in offsets]
    out["fuzzy_elevation"] = [
        {"angle": elevation_radians(o), "weights": [[i, w] for i, w in
                                                    fuzzy_bin_weights(elevation_radians(o), math.pi, 8)]}
        for o in offsets]

    # --- range bins ---------------------------------------------------------
    rb = RangeBins(max_range=500.0, bins=8, min_range=1.0)
    distances = [0.5, 1.0, 2.0, 10.0, 55.0, 120.0, 499.0, 1000.0]
    out["range_bins"] = {"max_range": 500.0, "bins": 8, "min_range": 1.0,
                         "bin_of": [rb.bin_of(d) for d in distances],
                         "centers": [rb.bin_center(i) for i in range(8)]}

    # --- quaternion -----------------------------------------------------------
    q = Quaternion.from_euler(0.5, -0.2, 0.3)
    p = Quaternion.from_axis_angle(Vec3(0.0, 0.0, 1.0), 0.4)
    v = Vec3(1.0, 2.0, 3.0)
    out["quaternion"] = {
        "from_euler": [q.w, q.x, q.y, q.z],
        "multiply": quat(q.multiply(p)),
        "rotate": vec(q.rotate(v)),
        "body_axes": [vec(ax) for ax in q.body_axes()],
        "from_axis_angle": quat(p),
    }
    rates = [(0.1, -0.05, 0.02), (-0.3, 0.2, 0.15)]
    out["integrate_body_rates"] = [
        {"rates": list(r), "dt": 0.02,
         "q": quat(q.integrate_body_rates(*r, 0.02))}
        for r in rates]

    # --- world coordinates ------------------------------------------------------
    world = World(hills=(Hill(120.0, -80.0, 60.0, 90.0), Hill(-300.0, 200.0, 25.0, 140.0)),
                  boxes=(Box(Vec3(50.0, 60.0, 0.0), Vec3(80.0, 90.0, 30.0)),),
                  half_extent=800.0)
    points = [Vec3(0.0, 0.0, 50.0), Vec3(120.0, -80.0, 10.0), Vec3(120.0, -80.0, 80.0),
              Vec3(900.0, 0.0, 50.0), Vec3(65.0, 75.0, 15.0)]
    out["world"] = {
        "gravity": vec(world.gravity), "rho": world.rho, "half_extent": world.half_extent,
        "terrain_height": [world.terrain_height(pt.x, pt.y) for pt in points],
        "inside_bounds": [world.is_inside_bounds(pt) for pt in points],
    }

    # --- clocks (recomputed here; capture.py records them in the golden run) ----
    config = Config()
    out["clocks"] = {"physics_hz": config.physics_hz, "motion_hz": config.motion_hz,
                     "vision_hz": config.vision_hz,
                     "physics_dt": 1.0 / config.physics_hz}

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
