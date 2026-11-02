"""RNG reference streams for the TypeScript port (Genesis Milestone 3).

The servo policy draws rng.uniform() and rng.integers(9) from numpy's
default_rng (PCG64). Exact episode replay across implementations requires
reproducing that stream, so the TS port emulates the generator (see
src/core/rng.ts) and this module freezes the ground truth:

- PCG64 state/increment for each seed (128-bit, decimal strings for BigInt)
- long uniform() and int9() streams
- an interleaved battery that exercises the 32-bit half-cache carry-over

Usage (from privatepoker-labs/skynerd):
    python golden/rng_ref.py --out ../skynerd_genesis/reference/rng-ref.json
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

STREAMS = 400
INT9_DRAWS = 600
INTERLEAVE_PATTERN = "uiiuui" * 30


def capture_seed(seed: int) -> dict:
    state = np.random.default_rng(seed).bit_generator.state
    record = {
        "seed": seed,
        # 128-bit integers exceed JSON numbers: ship as decimal strings.
        "state": str(state["state"]["state"]),
        "inc": str(state["state"]["inc"]),
    }
    r = np.random.default_rng(seed)
    record["uniform"] = [float(r.uniform()) for _ in range(STREAMS)]
    r = np.random.default_rng(seed)
    record["int9"] = [int(r.integers(9)) for _ in range(INT9_DRAWS)]
    r = np.random.default_rng(seed)
    values = []
    for op in INTERLEAVE_PATTERN:
        values.append(float(r.uniform()) if op == "u" else int(r.integers(9)))
    record["interleave"] = {"pattern": INTERLEAVE_PATTERN, "values": values}
    r = np.random.default_rng(seed)
    p8 = [int(v) for v in r.permutation(8)]
    r = np.random.default_rng(seed)
    p16 = [int(v) for v in r.permutation(16)]
    record["permutation"] = {"n8": p8, "n16": p16}
    r = np.random.default_rng(seed)
    choices = []
    for n, k in ((8, 3), (12, 5), (300, 7), (64, 12), (5, 5), (9, 9)):
        choices.append({"n": n, "k": k, "values": [int(v) for v in r.choice(n, size=k, replace=False)]})
    record["choice"] = choices
    # Mixed battery: choice, permutation, uniform, int9 in sequence with
    # shared 32-bit half-cache state.
    r = np.random.default_rng(seed)
    mixed = []
    for op in ("c", "p", "u", "i", "c"):
        if op == "c":
            mixed.append([int(v) for v in r.choice(40, size=6, replace=False)])
        elif op == "p":
            mixed.append([int(v) for v in r.permutation(6)])
        elif op == "u":
            mixed.append(float(r.uniform()))
        else:
            mixed.append(int(r.integers(9)))
    record["mixed"] = {"ops": ["choice40x6", "perm6", "uniform", "int9", "choice40x6"],
                       "values": mixed}
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="emit RNG reference JSON")
    parser.add_argument("--out", default="../skynerd_genesis/reference/rng-ref.json")
    args = parser.parse_args()
    doc = {
        "schema": "skynerd-rng-ref/1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seeds": [capture_seed(42), capture_seed(43), capture_seed(1007)],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
