"""skynet runner: waves, the cull, and the self-test.

Examples:
    python sim_runner.py --self-test
    python sim_runner.py --waves 10 --wave-size 16
    python sim_runner.py --waves 30 --wave-size 32 --epsilon 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from skynet.config import Config
from skynet.learning.flight import TRIM_SPEED
from skynet.learning.reward import FlightInstruments
from skynet.learning.school import FlightSchool, DEFAULT_PASS_SCORE
from skynet.learning.servo_tensor import ServoPolicy
from skynet.learning.flock import Flock
from skynet.learning.will import WillTensor
from skynet.metrics import print_wave_report
from skynet.mission.carpet import WaveRunner


def run_waves(config: Config, waves: int, wave_size: int, epsilon: float,
              seed: int, *, school_enabled: bool = False,
              will_enabled: bool = False,
              min_score_to_pass_school: float = DEFAULT_PASS_SCORE) -> int:
    rng = config.replace(seed=seed).seeded_rng()
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    school = FlightSchool(instruments) if school_enabled else None

    def will_factory(source_rng: np.random.Generator) -> WillTensor:
        return WillTensor(np.random.default_rng(source_rng.integers(2**63)))

    flock = Flock()
    runner = WaveRunner(flock, instruments, wave_size=wave_size, school=school,
                        will_factory=will_factory if will_enabled else None)
    if school is not None:
        # The classroom runs ONCE: one reference graduate, whose weights
        # every newborn copies at birth. One report card, one graduate.
        print("FLIGHT SCHOOL REPORT CARD (the reference graduate)")
        school.teach(ServoPolicy(), bar=min_score_to_pass_school)
    for wave in range(waves):
        metrics = runner.run_wave(wave, rng, epsilon=epsilon)
        print(f"wave {wave:3d}: mean {metrics['mean_survival']:5.2f}s  "
              f"best {metrics['best_survival']:5.2f}s  deaths {metrics['deaths']}/{wave_size}")
    print_wave_report(runner)
    return 0


def run_self_test() -> int:
    import subprocess
    here = Path(__file__).resolve().parents[1]
    suites = ["test_vec", "test_wavelet", "test_sparse_hebbian", "test_physics",
              "test_net", "test_vision", "test_tactics", "test_will",
              "test_school", "test_learn_to_fly", "test_mirror_learning",
              "test_waves"]
    failures = 0
    for suite in suites:
        result = subprocess.run([sys.executable, str(here / "tests" / f"{suite}.py")],
                                capture_output=True, text=True)
        status = "PASS" if result.returncode == 0 else "FAIL"
        if result.returncode != 0:
            failures += 1
            print(f"{status} {suite}\n{result.stdout}\n{result.stderr}")
        else:
            print(f"{status} {suite}  ({result.stdout.strip().splitlines()[-1]})")
    print(f"\n{'ALL SUITES PASSED' if failures == 0 else f'{failures} SUITES FAILED'}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="skynet: fully Hebbian drone swarm")
    parser.add_argument("--self-test", action="store_true",
                        help="run all test suites and exit")
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--wave-size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="exploration rate for every drone")
    parser.add_argument("--school", action="store_true",
                        help="send every newborn to flight school before launch")
    parser.add_argument("--will", action="store_true",
                        help="give every drone an emergent will (needs --school to thrive)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score-to-pass-school", type=float,
                        default=DEFAULT_PASS_SCORE,
                        help="lowest lesson score (0-6) the reference graduate "
                             "must earn before the class is done; weaker lessons "
                             "are drilled until they pass (default 4 = good)")
    parser.add_argument("--merge-rate", type=float, default=0.2)
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    config = Config()
    return run_waves(config, args.waves, args.wave_size, args.epsilon, args.seed,
                     school_enabled=args.school, will_enabled=args.will,
                     min_score_to_pass_school=args.min_score_to_pass_school)


if __name__ == "__main__":
    raise SystemExit(main())
