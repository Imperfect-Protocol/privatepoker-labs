"""Compare two golden-run documents.

Baseline acceptance for the Genesis port: the TypeScript implementation must
reproduce these numbers within tolerance. Two Python captures of the same
commit must match EXACTLY (determinism proof); port comparisons use --tol.

Usage (from privatepoker-labs/skynerd):
    python golden/compare.py golden/baseline.json golden/replay.json [--tol 1e-9]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Fields that legitimately differ between captures (provenance, not behaviour).
VOLATILE = {"captured_at", "git_dirty"}


def walk(a, b, path="$", tol=1e-9, diffs=None):
    if diffs is None:
        diffs = []
    if isinstance(a, dict):
        if not isinstance(b, dict):
            diffs.append(f"{path}: object vs {type(b).__name__}")
            return diffs
        for key in sorted(set(a) | set(b)):
            if key in VOLATILE:
                continue
            if key not in a:
                diffs.append(f"{path}.{key}: missing in A")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in B")
            else:
                walk(a[key], b[key], f"{path}.{key}", tol, diffs)
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", tol, diffs)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            diffs.append(f"{path}: {a} vs {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=tol):
            diffs.append(f"{path}: {a} vs {b}")
    elif a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="compare golden runs")
    parser.add_argument("baseline")
    parser.add_argument("replay")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="absolute tolerance (default 0 = exact)")
    args = parser.parse_args()
    a = json.loads(Path(args.baseline).read_text())
    b = json.loads(Path(args.replay).read_text())
    diffs = walk(a, b, tol=args.tol)
    if diffs:
        print(f"DIFFER ({len(diffs)}):")
        for d in diffs[:40]:
            print(f"  {d}")
        return 1
    print(f"MATCH: {args.replay} reproduces {args.baseline} exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
