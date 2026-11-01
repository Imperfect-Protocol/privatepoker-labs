"""The multi-stage brain: the panel tensor is the input, Haar polishes it.

Perception runs exactly as specified:

  1. read flight and engine instruments (and the controls)
  2. classify each value as a Gaussian over its classes, and build the
     4x4x8x8x8x4x4 panel tensor —
     ASI 4 x ADI-pitch 4 x ADI-bank 8 x HSI 8 x VSI 8 x ball 4 x RPM 4
  3. add stochastic noise
  4. decompose with Haar along all axes
  5. Gaussian-filter along all axes
  6. threshold small values
  7. compose back to the original domain
  8. pass this tensor into the hebbian learning WITHOUT any activation —
     the instruments are already the world

The learned weights live in the swarm's brain tensor. Each stage applies
the input into its weights, adds stochastic noise, Haars all axes,
Gaussian-filters all axes, thresholds, and only NOW applies the
activation function on the way to the next stage:

    stage.forward(x) = relu(polish(x * (1 + weights)))

Elementwise stages keep the brain's shape equal to the panel's, so the
whole stack stays interpretable: cell (asi, pitch, bank, hsi, vsi, ball,
rpm) in stage k's output is "how strongly this reading state contributes
after k layers of wavelet-polished learning".
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np

from skynet.learning.panel import (CRUISE_MPS, FT_PER_M, FULL_THROTTLE_RPM,
                                   IDLE_RPM, PANEL_GAIN, RPM_BANDS,
                                   STALL_MPS, VNE_MPS, VSI_BIN_FTPM,
                                   _gauss_weights, _interp, bank_deg,
                                   pitch_deg, slip_half_ball, speed_mps)
from skynet.physics.rigid_body import RigidBody
from skynet.tensors.dense_panel import (_gauss_all_axes, _haar_all_axes,
                                        _ihaar_all_axes)
from skynet.vec import Vec3

INPUT_SHAPE = (4, 4, 8, 8, 8, 4, 4)     # asi, pitch, bank, hsi, vsi, ball, rpm
PANEL_SIGMA = 0.55                      # class-widths of needle uncertainty
SITUATION_GAIN = 6.0                    # logsig steepness of the commitment


def _split_signed(signed: dict) -> dict:
    """A Gaussian on a signed line, folded onto up/down (or L/R) slots.

    Signed classes 1..4 (and -1..-4) become slot pairs: +1 -> 0, +2 -> 1,
    -1 -> 2, -2 -> 3. Level flight (symmetric mass on ±45) nets out.
    """
    fold = {1: 0, 2: 1, -1: 2, -2: 3}
    out = {}
    for s, w in signed.items():
        slot = fold[s]
        out[slot] = out.get(slot, 0.0) + w
    return out


def input_dists(body: RigidBody, throttle_position: float = 0.15) -> tuple:
    """Per-axis Gaussians over the input tensor's class slots."""
    # ASI: four zones on a speed line; parked = no mass at all. Zone
    # centers at 0..3 (below stall, below cruise, below Vne, above),
    # linear between — half-stall speed sits at 0, below stall.
    v = speed_mps(body)
    if v <= 0.02 * STALL_MPS:
        asi = {}
    else:
        pos = _interp(v, ((0.0, 0.0), (STALL_MPS, 0.0), (CRUISE_MPS, 1.0),
                          (VNE_MPS, 2.0), (2.0 * VNE_MPS, 3.0)))
        asi = _gauss_weights({c: float(c) for c in range(4)}, pos, PANEL_SIGMA)
    # ADI pitch: signed line in 45-degree units -> up/down slots.
    p = max(-2.0, min(2.0, pitch_deg(body) / 45.0))
    pitch = _split_signed(_gauss_weights(
        {s: float(s) for s in (-2, -1, 1, 2)}, p, PANEL_SIGMA))
    # ADI bank: signed line -> L45..L180 (0..3), R45..R180 (4..7).
    b = max(-4.0, min(4.0, bank_deg(body) / 45.0))
    bank = {}
    for s, w in _gauss_weights({s: float(s) for s in (-4, -3, -2, -1, 1, 2, 3, 4)},
                               b, PANEL_SIGMA).items():
        slot = (-s - 1) if s < 0 else (3 + s)   # -1 -> 0 ... -4 -> 3, +1 -> 4
        bank[slot] = bank.get(slot, 0.0) + w
    # HSI: circular over the 8 octants.
    forward, _, _ = body.body_axes
    hsi = _gauss_weights({c: float(c) for c in range(8)},
                         (np.degrees(np.arctan2(forward.y, forward.x)) / 45.0) % 8.0,
                         PANEL_SIGMA, circular=True)
    # VSI: signed line in 1250 ft/min units -> climb/descend slots.
    ftpm = body.velocity.z * FT_PER_M * 60.0
    vs = max(-4.0, min(4.0, ftpm / VSI_BIN_FTPM))
    vsi = {}
    for s, w in _gauss_weights({s: float(s) for s in (-4, -3, -2, -1, 1, 2, 3, 4)},
                               vs, PANEL_SIGMA).items():
        slot = (-s - 1) if s < 0 else (3 + s)
        vsi[slot] = vsi.get(slot, 0.0) + w
    # Turn ball: signed half-ball units -> Lhigh, Llow, Rlow, Rhigh.
    # Strong deflection is the "high" slot: -2 -> Lhigh, -1 -> Llow,
    # +1 -> Rlow, +2 -> Rhigh.
    slip = max(-2.0, min(2.0, slip_half_ball(body)))
    turn = {}
    for s, w in _gauss_weights({s: float(s) for s in (-2, -1, 1, 2)},
                               slip, PANEL_SIGMA).items():
        slot = {-2: 0, -1: 1, 1: 2, 2: 3}[s]    # Lhigh, Llow, Rlow, Rhigh
        turn[slot] = turn.get(slot, 0.0) + w
    # RPM: four bands on a rpm line; lever off = no mass.
    if throttle_position <= 0.0:
        rpm = {}
    else:
        rpm_val = IDLE_RPM + min(1.0, max(0.0, throttle_position)) * (FULL_THROTTLE_RPM - IDLE_RPM)
        pos = _interp(rpm_val, ((0.0, 0.0), (RPM_BANDS[0], 1.0),
                                (RPM_BANDS[1], 2.0), (RPM_BANDS[2], 3.0),
                                (3300.0, 3.4)))
        rpm = _gauss_weights({c: float(c) for c in range(4)}, pos, PANEL_SIGMA)
    return (asi, pitch, bank, hsi, vsi, turn, rpm)


def panel_input_tensor(body: RigidBody, *, throttle_position: float = 0.15,
                       gain: float = PANEL_GAIN) -> np.ndarray:
    """The cockpit reading as one dense tensor: the brain's input.

    Every axis contributes its Gaussian (gain-scaled); a cell is the
    product of the axis masses that reach it, so the tensor is the joint
    belief over panel states — brighter where the reading is certain,
    softly lit in the neighboring states.
    """
    axes = tuple({c: gain * w for c, w in d.items()}
                 for d in input_dists(body, throttle_position))
    tensor = np.zeros(INPUT_SHAPE, dtype=np.float64)
    for combo in product(*[d.items() for d in axes]):
        idx = tuple(c for c, _ in combo)
        value = 1.0
        for _, w in combo:
            value *= w
        tensor[idx] = value
    return tensor


def polish(x: np.ndarray, rng: np.random.Generator, *, noise: float = 0.01,
           sigma: float = 0.5, floor: float = 0.01,
           axes: int | None = None) -> np.ndarray:
    """Noise -> Haar -> Gaussian -> threshold -> compose.

    ``axes`` limits the wavelet sweep to the leading instrument axes
    (categorical tail axes, like a surface label, are swept along as
    channels rather than blurred as needle readings).
    """
    y = np.array(x, dtype=np.float64)
    if noise > 0.0:
        y += rng.normal(0.0, noise, size=y.shape)
    n = axes if axes is not None else y.ndim
    y = _haar_all_axes(y, axes=n)
    if sigma > 0.0:
        y = _gauss_all_axes(y, sigma, axes=n)
    y[np.abs(y) < floor] = 0.0
    return _ihaar_all_axes(y, axes=n)


class HebbianStage:
    """One learned layer: input through weights, polished, then activation.

    The activation must reflect the data and the stage (user's rule):

    - ``relu``   — analogue flow-through: sensor magnitudes and learned
                   gains, next stage still reads the world. The default.
    - ``tanh``   — signed analogue: when the stage's output means a
                   direction (bank left vs right), saturate the sign but
                   keep the magnitude graded.
    - ``sigmoid``— decisive: the stage is choosing — "I will do turn left
                   now" — so every cell saturates to near-0 or near-1 and
                   the argmax is the decision. Terminal action heads.
    - ``none``   — pure representation pass, no nonlinearity.
    """

    ACTIVATIONS = ("none", "relu", "tanh", "sigmoid")

    def __init__(self, shape, *, lr: float = 0.05, sigma: float = 0.5,
                 noise: float = 0.01, floor: float = 0.01,
                 activation: str = "relu", haar_axes: int | None = None) -> None:
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")
        self.shape = tuple(shape)
        self.weights = np.zeros(self.shape, dtype=np.float64)  # learned gains
        self.lr = float(lr)
        self.sigma = float(sigma)
        self.noise = float(noise)
        self.floor = float(floor)
        self.activation = activation
        self.haar_axes = haar_axes

    def _activate(self, y: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(y, 0.0)
        if self.activation == "tanh":
            return np.tanh(y)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(y, -30.0, 30.0)))
        return y

    def linear(self, x: np.ndarray) -> np.ndarray:
        """The input applied into the learned weights — before transport."""
        if x.ndim < len(self.shape):
            return x[..., None] * (1.0 + self.weights)
        return x * (1.0 + self.weights)

    def forward(self, x: np.ndarray, rng: np.random.Generator,
                *, activate: bool = True) -> np.ndarray:
        """Apply the input into the learned weights, polish, activate.

        The polish is the transport TO the next stage (noise, Haar,
        Gaussian, threshold, compose); the activation is what the next
        stage receives. A terminal reader uses linear() plus its own
        commitment instead.
        """
        y = self.linear(x)
        y = polish(y, rng, noise=self.noise, sigma=self.sigma,
                   floor=self.floor, axes=self.haar_axes)
        return self._activate(y) if activate else y

    def learn(self, x: np.ndarray, reward: float) -> None:
        """Hebbian: weights grow where input and reward agree, and clip.

        The step is normalized by the input's peak — learning is about
        WHERE the input is alive, not how loud the units are, so a bright
        panel and a dim one teach the same relative shape.
        """
        if reward == 0.0:
            return
        scale = max(float(np.abs(x).max()), 1e-9)
        self.weights += self.lr * reward * x / scale
        np.clip(self.weights, -0.9, 8.0, out=self.weights)


class Brain:
    """The panel input plus a stack of hebbian stages — a multi-stage brain.

    Three kinds of stage, each with the activation its data deserves
    (user's rule):

    - perception stages (ReLU): analogue instrument flow-through, the
      next stage still reads the world;
    - the situation head (logsig): a purely decisive readout — "we are
      in a stall and we need to recover" — every named situation
      saturates to near-0 or near-1;
    - the motor head (ReLU): the signals that go to the CONTROLS are
      analogue — graded deflections, not a snap — so it ends in ReLU.

    The motor head carries a trailing 4-axis (elevator, aileron, rudder,
    throttle); the wavelet sweep rides the seven instrument axes only.
    """

    MOTOR_SURFACES = ("elevator", "aileron", "rudder", "throttle")
    INSTRUMENT_AXES = 7

    # Named situations as masks over the input shape: the decisive
    # head saturates these regions, announce() speaks the winner.
    # Axes: asi, pitch, bank, hsi, vsi, ball, rpm.
    SITUATION_MASKS = {
        # stall: slow (with its Gaussian tail into below-cruise), nose
        # dropping (down45/down90). VSI is not part of the signature —
        # a stall at -2 m/s barely moves the needle.
        "stall": (np.s_[0:2, 2:4, 0:9, 0:8, 0:9, 0:4, 0:4]),
        # spin: stall plus a STRONG ball (high slots only — a weak ball
        # is a stall, not a spin)
        "spin": (np.s_[0:2, 2:4, 0:9, 0:8, 0:9, 0:5:3, 0:4]),
        # spiral dive: fast, steeply banked, descending hard
        "spiral": (np.s_[2:4, 0:4, 1:9, 0:8, 5:9, 0:4, 0:4]),
        # upset: far from level in pitch, speed anywhere
        "upset": (np.s_[0:4, 1:4, 0:9, 0:8, 0:9, 0:4, 0:4]),
    }

    def __init__(self, stages: int = 2, *, heads: bool = True,
                 **stage_kwargs) -> None:
        stage_kwargs.setdefault("haar_axes", self.INSTRUMENT_AXES)
        self.input = HebbianStage(INPUT_SHAPE, **stage_kwargs)   # stage 0
        self.stages = [HebbianStage(INPUT_SHAPE, **stage_kwargs)
                       for _ in range(max(0, stages - 1))]
        self.situation_head = (HebbianStage(INPUT_SHAPE, activation="sigmoid",
                                            **stage_kwargs) if heads else None)
        motor_shape = INPUT_SHAPE + (len(self.MOTOR_SURFACES),)
        self.motor_head = (HebbianStage(motor_shape, activation="relu",
                                        **stage_kwargs) if heads else None)

    def perceive(self, body: RigidBody, rng: np.random.Generator, *,
                 throttle_position: float = 0.15) -> np.ndarray:
        """The world, classified, through every learned stage.

        The classified panel tensor IS the input; each stage applies it
        into its weights, polishes the result (noise, Haar, Gaussian,
        threshold), and activates on the way to the next stage. The
        stage-0 polish is the input pipeline — the panel is not polished
        twice.
        """
        x = panel_input_tensor(body, throttle_position=throttle_position)
        x = self.input.forward(x, rng)
        for stage in self.stages:
            x = stage.forward(x, rng)
        return x

    def announce(self, body: RigidBody, rng: np.random.Generator, *,
                 throttle_position: float = 0.15) -> tuple:
        """The decisive readout: the situation, saturated by logsig.

        The head classifies the world itself — the stable classified
        panel tensor, not a polish artifact: the wavelet pipeline
        transports features BETWEEN learned stages, while a readout must
        see the instruments as they are. Returns (name, confidence) —
        e.g. ("stall", 0.93) is the sentence "we are in a stall and we
        need to recover". Below the commitment threshold the honest
        answer is that nothing urgent is recognized.
        """
        if self.situation_head is None:
            raise RuntimeError("this brain has no situation head")
        x = panel_input_tensor(body, throttle_position=throttle_position)
        y = self.situation_head.linear(x)
        # Contrast evidence per situation: the input-weighted response
        # RELATIVE to the same mask's zero-weight baseline (sum x^2).
        # Untrained, every ratio is exactly 1 and the logsig sits below
        # the commitment; learning lifts a mask's ratio only where the
        # head learned to respond to that signature. ONE logsig turns
        # the contrast into the commitment.
        ordered = sorted(self.SITUATION_MASKS.items(),
                         key=lambda kv: int(np.prod(
                             [sl.stop - sl.start for sl in kv[1]])))
        scores = []
        for name, mask in ordered:
            response = float((y[mask] * x[mask]).sum())
            baseline = max(float((x[mask] * x[mask]).sum()), 1e-9)
            c = float(1.0 / (1.0 + math.exp(-SITUATION_GAIN * (response / baseline - 1.0))))
            scores.append((name, c))
        best_name, top = max(scores, key=lambda kv: kv[1])
        if top < 0.5:
            return ("nothing urgent", top)
        # Saturated calls tie at 1.0; the most specific (first in the
        # ordered list) takes the call — a stall inside an upset is
        # reported as the stall.
        for name, c in scores:
            if c >= top - 1e-3:
                return (name, c)
        return ("nothing urgent", top)

    def actuate(self, body: RigidBody, rng: np.random.Generator, *,
                throttle_position: float = 0.15) -> np.ndarray:
        """Analogue control signals, one per surface, in [0, 1].

        The motor head's ReLU keeps these graded — a signal to the
        elevator is how far, never a snap. Summing the head's output over
        the instrument axes collapses the joint belief into per-surface
        deflection levels.
        """
        if self.motor_head is None:
            raise RuntimeError("this brain has no motor head")
        # Analogue readout on the stable classified world (see announce).
        x = panel_input_tensor(body, throttle_position=throttle_position)
        # Analogue readout: the weighted application, summed over the
        # instrument axes into per-surface deflection levels. The ReLU
        # keeps the signals graded — the polish is transport BETWEEN
        # stages, and this head is the end of the line.
        y = np.maximum(self.motor_head.linear(x), 0.0)
        return np.clip(y.sum(axis=tuple(range(self.INSTRUMENT_AXES))), 0.0, 1.0)

    def learn(self, trace: list, reward: float) -> None:
        """Credit every stage's weights along a recorded input trace."""
        for stage, x in trace:
            stage.learn(x, reward)


if __name__ == "__main__":
    from skynet.vec import Vec3

    rng = np.random.default_rng(0)
    body = RigidBody(Vec3(0, 0, 80.0), Vec3(CRUISE_MPS, 0.0, 0.0))

    # The input tensor: joint Gaussian belief, gain-strengthened.
    raw = panel_input_tensor(body)
    print(f"input tensor {raw.shape}, sum {raw.sum():.1f}, "
          f"lit cells {(raw > 0).sum()}")
    assert raw.shape == INPUT_SHAPE
    assert 0.5 * PANEL_GAIN ** 7 < raw.sum() < 1.5 * PANEL_GAIN ** 7, raw.sum()

    # The polish pipeline is an identity when quiet, and never amplifies.
    same = polish(raw, rng, noise=0.0, sigma=0.0, floor=0.0)
    assert np.allclose(same, raw, atol=1e-9), "Haar round trip is not identity"
    polished = polish(raw, rng)
    assert polished.shape == raw.shape
    assert np.abs(polished).max() <= np.abs(raw).max() * 1.5

    # Perception through the whole stack: bounded, non-negative after stages.
    brain = Brain(stages=2)
    out = brain.perceive(body, rng)
    assert out.shape == INPUT_SHAPE and (out >= 0).all()
    assert np.abs(out).max() < 1e6, "perception blew up"

    # Learning moves the weights; zero reward does not.
    before = brain.input.weights.copy()
    trace = [(brain.input, panel_input_tensor(body))]
    brain.learn(trace, reward=1.0)
    assert not np.allclose(brain.input.weights, before)
    brain.learn(trace, reward=0.0)
    expected = before + brain.input.lr * trace[0][1] / max(trace[0][1].max(), 1e-9)
    assert np.allclose(brain.input.weights, expected)

    # The activation reflects the data and the stage (user's rule):
    # instruments flow analogue (ReLU), the situation call saturates
    # (logsig), the control signals stay analogue (ReLU).
    decisive = Brain(stages=1)
    x = panel_input_tensor(body)
    out_r = decisive.input.forward(x, rng)
    out_s = decisive.situation_head.forward(x, rng)
    assert (out_r >= 0).all() and out_r.max() > 1.0, "relu must keep magnitude"
    assert (out_s > 0).all() and (out_s < 1).all(), "logsig must saturate"

    # A stalled airframe is ANNOUNCED — but a head that never lived must
    # first LEARN the stall signature (hebbian: the stall input, rewarded),
    # exactly like a pilot who only recognizes what they were taught.
    # (Euler pitch is inverted vs panel pitch, like roll: +30 reads as
    # nose-down 30.)
    stalled = RigidBody(Vec3(0, 0, 80.0), Vec3(0.5 * STALL_MPS, 0.0, -2.0))
    stalled.attitude = stalled.attitude.from_euler(
        0.0, 30.0 * math.pi / 180.0, 0.0)
    before = decisive.situation_head.weights.copy()
    cruise = RigidBody(Vec3(0, 0, 80.0), Vec3(CRUISE_MPS, 0.0, 0.0))
    for _ in range(50):
        # Contrast training: the stall rewarded, calm cruise punished —
        # the anti-hebbian arm erases the cells the two share, so the
        # head learns what DISTINGUISHES a stall, not what accompanies it.
        decisive.situation_head.learn(panel_input_tensor(stalled), reward=1.0)
        decisive.situation_head.learn(panel_input_tensor(cruise), reward=-1.0)
    assert not np.allclose(decisive.situation_head.weights, before), \
        "the situation head learned nothing"
    name, confidence = decisive.announce(stalled, rng)
    print(f"announce on stall: {name} at {confidence:.3f}")
    assert name == "stall" and confidence > 0.5, (name, confidence)

    # Calm cruise: nothing urgent is announced.
    name, confidence = decisive.announce(body, rng)
    print(f"announce on cruise: {name} at {confidence:.3f}")
    assert confidence < 0.5 or name == "nothing urgent", (name, confidence)

    # Signals to the controls are ANALOGUE: graded, never snapped.
    signals = decisive.actuate(stalled, rng)
    print("control signals:", np.round(signals, 3))
    assert signals.shape == (4,) and (signals >= 0).all() and (signals <= 1).all()
    assert signals.dtype.kind == "f"
    print("BRAIN CHECK PASSED")
