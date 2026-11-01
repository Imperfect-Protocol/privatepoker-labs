"""The flight episode: one drone, one life, the shared training loop.

Extracted from the learn-to-fly milestone so the wave runner and the tests
fly the exact same loop. A drone starts at trim (hardware calibration),
decides at 5 Hz, earns baseline-centered dense reward with a faint alive
bonus, and its decisions are reinforced once per decision; deaths and
survival stamp their terminal credit.
"""
from __future__ import annotations

import math

import numpy as np

from skynet.learning.reward import (DenseReward, EpisodeOutcome, FlightInstruments,
                                    classify_outcome)
from skynet.learning.servo_tensor import ServoPolicy
from skynet.learning.will import WillTensor, goal_reward
from skynet.physics.rigid_body import RigidBody
from skynet.physics.surfaces import SurfacesState
from skynet.physics.world import World
from skynet.vec import Vec3

TRIM_SPEED = 10.5
TRIM_COMMAND = np.array([0.0, 0.0, 0.0, 0.15])
HORIZON = 20.0
STICK_SPRING_TAU = 1.5   # s: hands off, the stick springs back toward neutral


def spring_toward_neutral(commands: np.ndarray, dt: float) -> np.ndarray:
    """The stick spring: a held command decays toward trim when not actively
    pushed. Real airframes self-center; without this, every random explore
    push accumulates forever and no memory can save the drone."""
    return commands + (TRIM_COMMAND - commands) * (dt / STICK_SPRING_TAU)


def fly_episode(policy: ServoPolicy, instruments: FlightInstruments, epoch: int,
                rng: np.random.Generator, *, start_stalled: bool = False,
                start_spiral: bool = False,
                start_position: Vec3 | None = None,
                epsilon_override: float | None = None,
                will: WillTensor | None = None) -> tuple:
    """One life. Returns (survival_seconds, outcome, final_state_key, final_commands).

    With a will, the goal hierarchy takes over: each decision window the
    will wakes its desires and hands down a goal (instrument-error targets),
    the servo tensor earns credit for progress toward THAT goal (no longer
    level flight), and the window's outcome flows back up through will.learn
    — desires that pay off grow, desires that hurt shrink.
    """
    world = World(half_extent=800.0)
    body = RigidBody(start_position or Vec3(0.0, 0.0, 50.0),
                     Vec3(TRIM_SPEED, 0.0, 0.0))
    if start_stalled:
        body.velocity = Vec3(5.0, 0.0, -3.5)
    if start_spiral:
        # The launch lottery's worst ticket: overbanked in a speed dive.
        # Wings-level trim cannot save this; only remembered recovery can.
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        bank = side * (60.0 + 15.0 * rng.uniform()) * math.pi / 180.0
        body.attitude = body.attitude.from_euler(0.0, 0.0, bank)
        body.velocity = Vec3(14.0, 0.0, -4.0)
    servos = SurfacesState()
    dt = 1.0 / 50.0
    dense = None if will is not None else DenseReward(instruments, scale=5.0)
    if dense is not None:
        dense.reset(body)
    commands = TRIM_COMMAND.copy()
    recent: list = []
    pending_reward = 0.0
    # The will's side of the ledger: outcome of the current window, the
    # error motion it caused, and progress toward the current goal.
    window_reward = 0.0
    window_errors = instruments.errors(body).as_array() + (0.0,)
    goal = None
    previous_progress = None
    windows_until_goal = 0
    decision_every = 10
    steps = int(HORIZON / dt)
    outcome = None
    tick = 0
    if epsilon_override is not None:
        policy.epsilon = epsilon_override
    for tick in range(steps):
        if tick % decision_every == 0:
            time_epoch = tick + epoch * 100000
            if recent:
                policy.reinforce(recent, pending_reward, time_epoch=time_epoch)
            if will is not None:
                context = instruments.state_key(body, commands[3])
                panel = (instruments.errors4(body, will.target_bearing)
                         if will.target_bearing is not None
                         else instruments.errors(body).as_array() + (0.0,))
                error_delta = tuple(np.asarray(panel) - np.asarray(window_errors))
                will.learn(context, window_reward, error_delta,
                           time_epoch=time_epoch)
                window_reward = 0.0
                window_errors = panel
                previous_progress = None
                windows_until_goal -= 1
                if windows_until_goal <= 0:
                    # Hold a goal for a full second before re-willing: a
                    # desire lived for 0.2s is a twitch, not an intent.
                    goal = will.decide(context, rng, epsilon=0.05)
                    windows_until_goal = 5
            pending_reward = 0.0
            state_key = instruments.state_key(body, commands[3])
            actions = policy.choose(state_key, rng)
            commands = policy.command_from(commands, actions)
            recent.append((state_key, actions))
            if len(recent) > 10:
                recent.pop(0)
        servos.command(commands, dt)
        body.step(servos, dt)
        commands = spring_toward_neutral(commands, dt)
        if will is None:
            pending_reward += dense.tick(body) + 0.0005
        else:
            panel = (instruments.errors4(body, will.target_bearing)
                     if will.target_bearing is not None
                     else instruments.errors(body).as_array() + (0.0,))
            reward, previous_progress = goal_reward(
                panel, goal, previous_progress, scale=5.0)
            pending_reward += reward + 0.0005
            window_reward += reward
        outcome = classify_outcome(body, world, (tick + 1) * dt)
        if outcome is not None:
            break
    time_epoch = tick + epoch * 100000
    if outcome in (EpisodeOutcome.CRASHED, EpisodeOutcome.STALLED):
        policy.reinforce(recent, -2.0, time_epoch=time_epoch)
        if will is not None:
            will.learn(instruments.state_key(body, commands[3]), -2.0,
                       tuple(np.asarray(instruments.errors(body).as_array()) + (0.0,)
                             - np.asarray(window_errors)),
                       time_epoch=time_epoch)
        return (tick + 1) * dt, outcome, instruments.state_key(body), commands.copy()
    if outcome is None:
        outcome = EpisodeOutcome.ALIVE
        policy.reinforce(recent, 0.5, time_epoch=time_epoch)
        if will is not None:
            will.learn(instruments.state_key(body, commands[3]), 0.5,
                       tuple(np.asarray(instruments.errors(body).as_array()) + (0.0,)
                             - np.asarray(window_errors)),
                       time_epoch=time_epoch)
    return (tick + 1) * dt, outcome, instruments.state_key(body), commands.copy()
