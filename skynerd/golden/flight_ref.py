"""Flight reference: fixed-command physics runs and full episodes (Milestone 3).

Records what the TypeScript port must reproduce:

1. Physics trajectories with FIXED surface commands (no policy, no RNG):
   the deterministic core — rigid body, aero, servo slew. Exact match
   expected.
2. Three episodes of one learned policy (ServoPolicy, blank start):
   initial RNG state (PCG64), every decision's choice, sampled trajectory,
   outcome, survival, final state, and the learned tensor afterwards.
   Exact episode replay exercises the RNG emulation, the epsilon-greedy
   policy, Hebbian reward/score with float32 tensor storage, the dense
   reward, instruments, and outcome classification.

Usage (from privatepoker-labs/skynerd):
    python golden/flight_ref.py --out ../skynerd_genesis/reference/flight-ref.json
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import sys
SKYNERD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYNERD))

import numpy as np

from skynerd.learning.flight import fly_episode
from skynerd.learning.reward import (DenseReward, EpisodeOutcome,
                                     FlightInstruments, classify_outcome)
from skynerd.learning.servo_tensor import ServoPolicy
from skynerd.physics.rigid_body import RigidBody
from skynerd.physics.surfaces import SurfacesState
from skynerd.physics.world import World
from skynerd.vec import Vec3

DT = 1.0 / 50.0
TICKS = 500                 # 10 s per physics run
SAMPLE_EVERY = 5            # sample every 5 ticks (0.1 s)
EPISODE_TICKS = 1000        # 20 s horizon, as in fly_episode


def sample(body: RigidBody, servos: SurfacesState, diag) -> dict:
    q = body.attitude
    return {
        "pos": [body.position.x, body.position.y, body.position.z],
        "vel": [body.velocity.x, body.velocity.y, body.velocity.z],
        "quat": [q.w, q.x, q.y, q.z],
        "rates": [body.rate_p, body.rate_q, body.rate_r],
        "surfaces": [float(p) for p in servos.positions],
        "diag": [diag.airspeed, diag.alpha, diag.beta, diag.q_dyn,
                 diag.lift, diag.drag, diag.thrust],
    }


def physics_run(name: str, commands, *, start=None, stalled: bool = False) -> dict:
    world = World(half_extent=800.0)
    body = RigidBody(*(start if start else (Vec3(0.0, 0.0, 50.0), Vec3(10.5, 0.0, 0.0))))
    if stalled:
        body.velocity = Vec3(5.0, 0.0, -3.5)
    servos = SurfacesState()
    samples = []
    for tick in range(TICKS):
        servos.command(commands, DT)
        diag = body.step(servos, DT)
        if tick % SAMPLE_EVERY == 0:
            samples.append(sample(body, servos, diag))
    return {"name": name, "commands": [float(c) for c in commands],
            "ticks": TICKS, "dt": DT, "samples": samples}


def episode_run(policy: ServoPolicy, instruments, epoch: int, seed: int,
                *, epsilon_override=None) -> dict:
    rng = np.random.default_rng(seed)
    state = rng.bit_generator.state
    # Re-seed a fresh generator: the recording below re-runs the episode to
    # log decisions; the flight itself must see an untouched stream.
    record = {
        "epoch": epoch,
        "epsilon_override": epsilon_override,
        "rng_state": str(state["state"]["state"]),
        "rng_inc": str(state["state"]["inc"]),
    }
    seconds, outcome, final_key, final_commands = fly_episode(
        policy, instruments, epoch, rng, epsilon_override=epsilon_override)
    record.update({
        "survival": seconds,
        "outcome": outcome,
        "final_state_key": list(final_key),
        "final_commands": [float(c) for c in final_commands],
    })
    return record


def episode_with_trace(policy: ServoPolicy, instruments, epoch: int, seed: int,
                       *, epsilon_override=None) -> dict:
    """Like episode_run, but replays with a tracer to log decisions/samples."""
    # Snapshot the pre-episode policy state: the tracer must fly the episode
    # exactly as the real one did, from the same starting memory.
    pre_values = {k: v.copy() for k, v in policy.tensor.values.items()}
    pre_epochs = {k: e.copy() for k, e in policy.tensor.epochs.items()}
    pre_reward_total = policy.tensor.reward_total
    pre_epsilon = policy.epsilon
    record = episode_run(policy, instruments, epoch, seed,
                        epsilon_override=epsilon_override)

    # Decision log + trajectory: instrument the same code path by running the
    # episode with an identical policy copy and identical rng seed.
    policy_copy = ServoPolicy()
    policy_copy.epsilon = pre_epsilon
    policy_copy.tensor.values = pre_values
    policy_copy.tensor.epochs = pre_epochs
    policy_copy.tensor.reward_total = pre_reward_total
    rng = np.random.default_rng(seed)
    world = World(half_extent=800.0)
    body = RigidBody(Vec3(0.0, 0.0, 50.0), Vec3(10.5, 0.0, 0.0))
    servos = SurfacesState()
    from skynerd.learning.reward import DenseReward
    dense = DenseReward(instruments, scale=5.0)
    dense.reset(body)
    commands = np.array([0.0, 0.0, 0.0, 0.15])
    recent: list = []
    pending = 0.0
    choices = []
    samples = []
    if epsilon_override is not None:
        policy_copy.epsilon = epsilon_override
    outcome = None
    for tick in range(EPISODE_TICKS):
        if tick % 10 == 0:
            time_epoch = tick + epoch * 100000
            if recent:
                policy_copy.reinforce(recent, pending, time_epoch=time_epoch)
            pending = 0.0
            state_key = instruments.state_key(body, commands[3])
            actions = policy_copy.choose(state_key, rng)
            commands = policy_copy.command_from(commands, actions)
            recent.append((state_key, actions))
            if len(recent) > 10:
                recent.pop(0)
            changed = [(s, int(actions[s])) for s in range(4) if actions[s] != 1]
            choices.append(0 if not changed else
                           1 + changed[0][0] * 2 + (changed[0][1] == 2))
        servos.command(commands, DT)
        diag = body.step(servos, DT)
        commands = commands + (np.array([0.0, 0.0, 0.0, 0.15]) - commands) * (DT / 1.5)
        pending += dense.tick(body) + 0.0005
        if tick % SAMPLE_EVERY == 0:
            samples.append(sample(body, servos, diag))
        outcome = classify_outcome(body, world, (tick + 1) * DT)
        if outcome is not None:
            break
    # sanity: the traced replay must end the same way as the real episode
    traced_outcome = outcome if outcome is not None else EpisodeOutcome.ALIVE
    assert traced_outcome == record["outcome"], (traced_outcome, record["outcome"])
    assert abs((tick + 1) * DT - record["survival"]) < 1e-12
    record["choices"] = choices
    record["samples"] = samples
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="emit flight reference JSON")
    parser.add_argument("--out", default="../skynerd_genesis/reference/flight-ref.json")
    args = parser.parse_args()

    doc = {
        "schema": "skynerd-flight-ref/1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "physics_runs": [
            physics_run("trim_cruise", (0.0, 0.0, 0.0, 0.15)),
            physics_run("full_throttle_climb", (0.0, 0.05, 0.0, 1.0)),
            physics_run("stick_forward_stall", (0.0, 0.3, 0.0, 0.0)),
            physics_run("rolling_pull", (0.35, -0.1, 0.35, 0.4)),
            physics_run("knife_edge_slip", (0.0, 0.0, 0.35, 0.5),
                        start=(Vec3(0.0, 0.0, 300.0), Vec3(15.0, 0.0, 0.0))),
            physics_run("stalled_drop", (0.0, 0.0, 0.0, 0.0), stalled=True),
        ],
        "episodes": [],
    }

    instruments = FlightInstruments(trim_speed=10.5)
    policy = ServoPolicy()
    doc["episodes"].append(episode_with_trace(policy, instruments, 0, 1007))
    doc["episodes"].append(episode_with_trace(policy, instruments, 1, 1008))
    doc["episodes"].append(episode_with_trace(policy, instruments, 2, 1009,
                                              epsilon_override=0.0))
    doc["tensor_after"] = {
        "keys": len(policy.tensor.values),
        "reward_total": policy.tensor.reward_total,
        "first_entries": [
            {"key": list(k),
             "values": [float(v) for v in policy.tensor.values[k]],
             "epochs": [int(e) for e in policy.tensor.epochs[k]]}
            for k in list(policy.tensor.values)[:5]
        ],
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    for ep in doc["episodes"]:
        print(f"  episode {ep['epoch']}: survival {ep['survival']:.2f}s "
              f"outcome {ep['outcome']} decisions {len(ep['choices'])}")
    print(f"  tensor: {doc['tensor_after']['keys']} keys, "
          f"reward_total {doc['tensor_after']['reward_total']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
