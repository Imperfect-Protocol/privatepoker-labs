"""Wave report: the numbers that matter, in the house report style."""
from __future__ import annotations

import statistics


def print_wave_report(runner, *, title: str = "SKYNET WAVE REPORT") -> None:
    history = runner.history
    if not history:
        print("no waves run")
        return
    print("=" * 78)
    print(f"{title}")
    print("=" * 78)
    medians = [statistics.median(w["survivals"]) for w in history]
    print("wave | drones | deaths | median | mean   | best  | full-horizon survivors")
    print("-----+--------+--------+--------+--------+-------+-----------------------")
    for wave, metrics in enumerate(history):
        survivors = sum(1 for s in metrics["survivals"] if s >= 19.9)
        print(f"{wave:4d} | {len(metrics['survivals']):6d} | "
              f"{metrics['deaths']:6d} | {medians[wave]:6.2f} | "
              f"{metrics['mean_survival']:6.2f} | {metrics['best_survival']:5.2f} | "
              f"{survivors}")
    print("-" * 78)
    early = statistics.median(medians[:2]) if len(medians) >= 2 else medians[0]
    late = statistics.median(medians[-2:]) if len(medians) >= 2 else medians[0]
    print(f"inheritance trend (first-two -> last-two wave medians): "
          f"{early:.2f}s -> {late:.2f}s")
    flock = runner.flock
    print(f"living memory shards: {flock.size} drones carry the brain "
          f"({flock.births} births, {flock.obituaries_heard} obituaries heard)")
    print(f"best single life: {max(w['best_survival'] for w in history):.2f}s")
    print("=" * 78)
