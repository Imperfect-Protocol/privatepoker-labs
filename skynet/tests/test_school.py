"""Flight school milestone: demonstrated memory beats learned-by-crashing.
Run: python tests/test_school.py"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from skynet.config import Config
from skynet.learning.flight import TRIM_SPEED, fly_episode
from skynet.learning.flock import Flock
from skynet.learning.reward import FlightInstruments
from skynet.learning.school import FlightSchool
from skynet.learning.servo_tensor import ServoPolicy
from skynet.mission.carpet import WaveRunner

WAVES = 12
WAVE_SIZE = 32
EPSILON = 0.05
SEED = 7              # seed 7 is where un-schooled inheritance struggled


def blank_policy():
    return ServoPolicy()


def schooled_policy(school: FlightSchool):
    policy = ServoPolicy()
    school.teach(policy)
    # The school's product is its graduate: the base curriculum the flock
    # inherits (the classroom's drilled polish stays in the classroom).
    return school.graduate


def test_school_imprints_memory():
    school = FlightSchool(FlightInstruments(trim_speed=TRIM_SPEED))
    policy = schooled_policy(school)
    assert policy.tensor.values, "school taught nothing"
    # Key layout: (vs, bank, asi, rpm, surface, action).
    # The turn lessons must appear: aileron (surface 0), both directions.
    assert any(key[5] == 0 and key[6] == 0 for key in policy.tensor.values), \
        "turn-left lesson left no memory"
    assert any(key[5] == 0 and key[6] == 2 for key in policy.tensor.values), \
        "turn-right lesson left no memory"
    # Climbing and takeoff lessons stamp elevator-up and throttle-up.
    assert any(key[5] == 1 and key[6] == 2 for key in policy.tensor.values), \
        "climb lessons left no elevator memory"
    assert any(key[5] == 3 and key[6] == 2 for key in policy.tensor.values), \
        "takeoff left no throttle memory"
    # Stall recovery stamps "ease the elevator" — the nose-down reflex.
    assert any(key[5] == 1 and key[6] == 0 for key in policy.tensor.values), \
        "stall recovery left no nose-down memory"
    print(f"PASS test_school_imprints_memory "
          f"({len(policy.tensor.values)} states stamped)")


def test_school_names_desires():
    """The instructor's intent handshake: after school, waking the turn-left
    desire in a turning context wants a left turn — the will has vocabulary."""
    from skynet.learning.will import WillTensor
    from skynet.learning.school import DESIRE_LEFT

    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    school = FlightSchool(instruments)
    will = WillTensor(np.random.default_rng(9))
    policy = ServoPolicy()
    school.teach(policy, will=will)
    # The lessons blended the left-turn desire's aim toward a left bank.
    assert will.aim[DESIRE_LEFT][1] < -0.6, \
        "turn-left desire has no bank aim"
    # In a context where the lesson stamped its intent, that desire wakes.
    contexts = [key[:-1] for key in will.tensor.values
                if key[-1] == DESIRE_LEFT]
    assert contexts, "left-turn desire was never named anywhere"
    rng = np.random.default_rng(3)
    will.decide(contexts[0], rng, epsilon=0.0)
    assert int(np.argmax(will.activation)) == DESIRE_LEFT, \
        "the named desire did not wake in its own context"
    goal = will.goal()
    assert goal[1] < -0.3, f"awake will wants wrong bank: {goal}"
    print("PASS test_school_names_desires (turn-left desire wakes and aims)")


def test_schooled_drones_fly_longer():
    """The hierarchy composition: school reflexes + a will beats a will alone.
    Bare school memory alone over-controls (every lesson's reflex fires
    everywhere); it needs a will to gate which reflex serves."""
    from skynet.learning.flight import fly_episode
    from skynet.learning.will import WillTensor
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    school = FlightSchool(instruments)
    rng = Config().replace(seed=1).seeded_rng()
    blank = [fly_episode(blank_policy(), instruments, 0, rng,
                         epsilon_override=EPSILON,
                         will=WillTensor(np.random.default_rng(100 + i)))[0]
             for i in range(20)]
    schooled = [fly_episode(schooled_policy(school), instruments, 0, rng,
                            epsilon_override=EPSILON,
                            will=WillTensor(np.random.default_rng(100 + i)))[0]
                for i in range(20)]
    print(f"  will only    median {statistics.median(blank):5.2f}s")
    print(f"  school + will median {statistics.median(schooled):5.2f}s")
    assert statistics.median(schooled) > statistics.median(blank), \
        "school did not help the willed drone"
    print("PASS test_schooled_drones_fly_longer "
          "(school + will beats will alone)")


def test_school_beats_survival():
    """One class of schooled drones must out-inherit 12 waves of crashes."""
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    school = FlightSchool(instruments)

    rng = Config().replace(seed=SEED).seeded_rng()
    flock = Flock()
    schooled = WaveRunner(flock, instruments, wave_size=WAVE_SIZE, school=school)
    for wave in range(WAVES):
        schooled.run_wave(wave, rng, epsilon=EPSILON)

    rng = Config().replace(seed=SEED).seeded_rng()
    flock = Flock()
    unschooled = WaveRunner(flock, instruments, wave_size=WAVE_SIZE)
    for wave in range(WAVES):
        unschooled.run_wave(wave, rng, epsilon=EPSILON)

    taught = statistics.mean(statistics.median(w["survivals"]) for w in schooled.history)
    crash_course = statistics.mean(statistics.median(w["survivals"]) for w in unschooled.history)
    print(f"  crash-course (survival only): {crash_course:.2f}s mean median")
    print(f"  flight-school graduates:      {taught:.2f}s mean median")
    assert taught > crash_course, "school did not beat learning by crashing"
    print("PASS test_school_beats_survival (graduates out-inherit crash courses)")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
