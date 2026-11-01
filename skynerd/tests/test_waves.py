"""Wave inheritance without a station: the brain lives in the drones.
Run: python tests/test_waves.py"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from skynerd.config import Config
from skynerd.learning.flight import TRIM_SPEED
from skynerd.learning.flock import Flock
from skynerd.learning.reward import FlightInstruments
from skynerd.learning.servo_tensor import ServoPolicy
from skynerd.mission.carpet import WaveRunner

WAVES = 12
WAVE_SIZE = 32
EPSILON = 0.05   # low noise: exploration tweaks, it does not murder
HAZARDS = 0.5    # half of all launches enter a spiral dive: the launch lottery


def fly_swarm(flock: Flock, seed: int) -> tuple:
    rng = Config().replace(seed=seed).seeded_rng()
    runner = WaveRunner(flock, FlightInstruments(trim_speed=TRIM_SPEED),
                        wave_size=WAVE_SIZE, hazard_rate=HAZARDS)
    for wave in range(WAVES):
        metrics = runner.run_wave(wave, rng, epsilon=EPSILON)
        print(f"  wave {metrics['wave']:2d}: mean {metrics['mean_survival']:5.2f}s  "
              f"deaths {metrics['deaths']:2d}/{WAVE_SIZE}  "
              f"flock {metrics['flock_size']:2d}")
    return runner


def main():
    print("CONTROL: drones never teach or hear screams (no inheritance).")
    control = fly_swarm(Flock(donor_count=0, witness_prob=0.0), seed=7)
    print("\nTAUGHT: mosaics of the living, death screams, veteran shards.")
    taught = fly_swarm(Flock(), seed=7)

    # The honest measure: the launch lottery. This airframe self-rights in
    # calm air, so a null policy is safe there — memory shows its value in
    # the spiral dives, where only remembered recovery saves the drone.
    control_spirals = [t for w in control.history[-6:]
                       for t in w["spiral_survivals"]]
    taught_spirals = [t for w in taught.history[-6:]
                      for t in w["spiral_survivals"]]
    control_rescue = statistics.median(control_spirals)
    taught_rescue = statistics.median(taught_spirals)
    print(f"\nspiral-dive rescue (late waves, median): "
          f"control {control_rescue:.2f}s vs taught {taught_rescue:.2f}s")
    assert taught_rescue > control_rescue, (
        "distributed inheritance does not carry recovery knowledge")
    print("DISTRIBUTED-INHERITANCE MILESTONE PASSED: the taught flock "
          "recovers the launch lottery's spiral dives better than drones "
          "that never share memory.")

    # Total wipe = total amnesia: a newborn into an empty flock is blank.
    orphan = ServoPolicy()
    Flock().seed_newborn(orphan, np.random.default_rng(1))
    assert not orphan.tensor.values, "orphan drone inherited from nobody"
    print("AMNESIA CHECK PASSED: wipe the flock and the brain dies with it.")

    # No two births alike: the same flock raises different newborns.
    rng = np.random.default_rng(2)
    a, b = ServoPolicy(), ServoPolicy()
    flock = taught.flock
    flock.seed_newborn(a, rng)
    flock.seed_newborn(b, rng)
    assert a.tensor.values != b.tensor.values, "newborns were identical clones"
    print("MOSAIC CHECK PASSED: every birth is a different mix of the living.")


if __name__ == "__main__":
    main()
