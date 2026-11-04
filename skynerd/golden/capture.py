"""Golden-run capture: freeze the Python Skynerd as the behavioural reference.

Milestone 0 of the Genesis design brief. Runs deterministic scenarios with
fixed seeds, records their metrics into one JSON document. Run it twice and
compare with compare.py: an honest baseline must reproduce itself exactly.

Usage (from privatepoker-labs/skynerd):
    python golden/capture.py --out golden/baseline.json
    python golden/capture.py --out golden/replay.json
    python golden/compare.py golden/baseline.json golden/replay.json

The TypeScript port (skynerd_genesis/) validates itself against the
baseline.json produced here.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

SKYNERD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYNERD))

import numpy as np

from skynerd.config import Config
from skynerd.learning.flight import TRIM_SPEED, HORIZON, fly_episode
from skynerd.learning.flock import Flock
from skynerd.learning.reward import FlightInstruments
from skynerd.learning.school import FlightSchool
from skynerd.learning.servo_tensor import ServoPolicy
from skynerd.learning.will import WillTensor
from skynerd.mission.carpet import WaveRunner


# --- scenarios -------------------------------------------------------------
# Every scenario seeds its own rng. No scenario touches another's state.

def scenario_learn_to_fly() -> dict:
    """Blank servo policy, one episode per seed: the raw survival baseline."""
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    survivals, outcomes = [], []
    for episode in range(30):
        policy = ServoPolicy()
        rng = np.random.default_rng(1000 + episode)
        seconds, outcome, _, _ = fly_episode(policy, instruments, episode, rng)
        survivals.append(seconds)
        outcomes.append(outcome)
    return {"episodes": 30, "survivals": survivals, "outcomes": outcomes,
            "mean_survival": float(np.mean(survivals))}


def scenario_waves() -> dict:
    """Carpet waves with distributed inheritance (no school, no will)."""
    config = Config().replace(seed=42)
    rng = config.seeded_rng()
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    runner = WaveRunner(Flock(), instruments, wave_size=8)
    waves = []
    for wave in range(5):
        metrics = runner.run_wave(wave, rng, epsilon=0.05)
        waves.append({"wave": wave,
                      "survivals": metrics["survivals"],
                      "deaths": metrics["deaths"],
                      "flock_size": metrics["flock_size"]})
    return {"waves": 5, "wave_size": 8, "epsilon": 0.05, "per_wave": waves}


def scenario_school() -> dict:
    """Flight school: the reference curriculum and its graded report card."""
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    school = FlightSchool(instruments)
    policy = ServoPolicy()
    counts = school.teach(policy, bar=4.0)
    lessons = [{"lesson": name,
                "score": round(school.student_quality(policy, school.lessons[name]), 6),
                "stamps": counts[name]}
               for name in school.lessons]
    return {"lesson_count": len(lessons), "lessons": lessons,
            "graduate_keys": len(school.graduate.tensor.values)}


def scenario_will_flight() -> dict:
    """One willed life: the goal hierarchy in flight."""
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    rng = np.random.default_rng(11)
    will = WillTensor(np.random.default_rng(rng.integers(2**63)))
    policy = ServoPolicy()
    seconds, outcome, _, _ = fly_episode(policy, instruments, 0, rng, will=will)
    return {"survival": seconds, "outcome": outcome}


SCENARIOS = {
    "learn_to_fly": scenario_learn_to_fly,
    "waves": scenario_waves,
    "school": scenario_school,
    "will_flight": scenario_will_flight,
}


# --- document ---------------------------------------------------------------

def environment() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=SKYNERD, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def clock_rates(config: Config) -> dict:
    """The agreed clocks. fly_episode currently hardcodes dt and horizon in
    flight.py; the Config values are recorded for the TypeScript port's
    scheduler even where the Python sim does not yet read them."""
    return {
        "physics_hz": config.physics_hz,
        "physics_dt": 1.0 / config.physics_hz,
        "physics_dt_effective": 1.0 / 50.0,   # flight.py hardcodes 1/50
        "motion_hz": config.motion_hz,
        "vision_hz": config.vision_hz,
        "episode_horizon_s": HORIZON,          # flight.py hardcodes 20.0
        "decision_window_s": 0.2,              # every 10 physics ticks
    }


def capture() -> dict:
    config = Config()
    return {
        "schema": "skynerd-golden/1",
        "environment": environment(),
        "clocks": clock_rates(config),
        "config": asdict(config),
        "scenarios": {name: fn() for name, fn in SCENARIOS.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="capture golden runs")
    parser.add_argument("--out", default="golden/baseline.json")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all",
                        help="capture one scenario only")
    args = parser.parse_args()
    doc = capture()
    if args.scenario != "all":
        doc["scenarios"] = {args.scenario: doc["scenarios"][args.scenario]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"captured {out} ({out.stat().st_size} bytes)")
    for name, data in doc["scenarios"].items():
        summary = {k: v for k, v in data.items()
                   if k in ("mean_survival", "episodes", "lesson_count",
                            "survival", "outcome", "waves")}
        print(f"  {name}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
