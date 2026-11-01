"""The cockpit panel: flight and engine instruments as one-hot vectors.

Every instrument is a small one-hot alphabet; the drone's state word is
the flat concatenation of all of them — 60 binary slots:

  ASI   4   (below stall, below cruise, below Vne, above Vne)
            zero vector (0,0,0,0) only at zero speed (parked)
  ADI   4x8 pitch (up45, up90, down45, down90) x
            bank (L45, L90, L135, L180, R45, R90, R135, R180)
            each zero vector = within 22.5 deg of neutral
  HSI   8   compass octants N NE E SE S SW W NW — always one-hot;
            north itself is (1,0,0,0,0,0,0,0)
  VSI   8   climb 1250/2500/3750/5000 then descend 1250/2500/3750/5000
            ft/min; zero vector = neutral (within 625 ft/min)
  TURN  4   (ball left high, ball left low, ball right low, ball right
            high); zero vector = centered ball
  RPM   4   (<1700, <2300, <2700, >=2700) — always one-hot

For tensor addressing each axis folds into one index: 0 = the zero
vector, otherwise 1 + the active slot (HSI and RPM are always lit, so
their index is just the active slot). The panel key is therefore
(5, 5, 9, 8, 9, 5, 4) — 324,000 panel states, each a readable bit word.

The airframe maps the C172 numbers onto sim units: cruise 80 mph is
10.5 m/s, stall 60 mph is 0.75 cruise, Vne 163 mph is about 2.0 cruise.
"""
from __future__ import annotations

import math

from skynet.physics.rigid_body import RigidBody
from skynet.vec import Vec3

# The airframe's cruise speed (80 mph mapped onto sim units). Kept here
# rather than imported from flight.py — the panel sits below the flight
# loop, and servo_tensor needs the panel without pulling the loop in.
CRUISE_MPS = 10.5
STALL_MPS = 0.75 * CRUISE_MPS          # 60 mph
VNE_MPS = 2.0 * CRUISE_MPS             # construction speed, ~163 mph
FT_PER_M = 3.28084
VSI_BIN_FTPM = 1250.0                  # one VSI slot
VSI_RUNGS = 4                          # climb rungs (and descend rungs)

HSI_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
RPM_BANDS = (1700.0, 2300.0, 2700.0)
IDLE_RPM = 700.0
FULL_THROTTLE_RPM = 2700.0

# Tensor axis sizes: 0 = the instrument's zero vector, else 1 + slot.
# HSI and RPM are always one-hot, so their index is the active slot.
PANEL_AXES = (5, 5, 9, 8, 9, 5, 4)
PANEL_VECTOR_LEN = 4 + 4 + 8 + 8 + 8 + 4 + 4   # 40 one-hot bits


def asi_vec(speed_mps: float) -> tuple:
    if speed_mps <= 1e-6:
        return (0, 0, 0, 0)
    if speed_mps < STALL_MPS:
        return (1, 0, 0, 0)
    if speed_mps < CRUISE_MPS:
        return (0, 1, 0, 0)
    if speed_mps < VNE_MPS:
        return (0, 0, 1, 0)
    return (0, 0, 0, 1)


def pitch_vec(body: RigidBody) -> tuple:
    """(up45, up90, down45, down90); zero within 22.5 deg of level."""
    forward, _, _ = body.body_axes
    deg = math.degrees(math.asin(max(-1.0, min(1.0, forward.z))))
    if abs(deg) < 22.5:
        return (0, 0, 0, 0)
    rung = 1 if abs(deg) >= 67.5 else 0    # 0 = 45, 1 = 90
    if deg > 0:
        return (1, 0, 0, 0) if rung == 0 else (0, 1, 0, 0)
    return (0, 0, 1, 0) if rung == 0 else (0, 0, 0, 1)


def bank_vec(body: RigidBody) -> tuple:
    """(L45, L90, L135, L180, R45, R90, R135, R180); zero near wings level."""
    _, right, _ = body.body_axes
    deg = -math.degrees(math.asin(max(-1.0, min(1.0, right.z))))
    if abs(deg) < 22.5:
        return (0,) * 8
    rung = 0                                # 0=45, 1=90, 2=135, 3=180
    for i, edge in enumerate((67.5, 112.5, 157.5)):
        if abs(deg) >= edge:
            rung = i + 1
    vec = [0] * 8
    vec[rung if deg < 0 else 4 + rung] = 1
    return tuple(vec)


def hsi_vec(body: RigidBody) -> tuple:
    """Always one-hot: the compass octant the nose points at, N first."""
    forward, _, _ = body.body_axes
    bearing = math.atan2(forward.y, forward.x)
    slot = int(round(bearing / (math.pi / 4.0))) % 8
    vec = [0] * 8
    vec[slot] = 1
    return tuple(vec)


def vsi_vec(vz_mps: float) -> tuple:
    """(climb 1250..5000, descend 1250..5000 ft/min); zero = neutral."""
    ftpm = vz_mps * FT_PER_M * 60.0
    if abs(ftpm) < VSI_BIN_FTPM / 2.0:
        return (0,) * 8
    rung = min(VSI_RUNGS - 1, int(abs(ftpm) / VSI_BIN_FTPM))
    vec = [0] * 8
    vec[rung if ftpm > 0 else VSI_RUNGS + rung] = 1
    return tuple(vec)


def turn_vec(body: RigidBody) -> tuple:
    """(ball left high, ball left low, ball right low, ball right high)."""
    forward, right, _ = body.body_axes
    velocity = body.velocity
    slip = velocity.x * right.x + velocity.y * right.y + velocity.z * right.z
    half = 0.25 * CRUISE_MPS                # half-ball sideways drift
    if abs(slip) < 0.15 * half:
        return (0, 0, 0, 0)
    strength = 1 if abs(slip) < half else 0  # 1 = low, 0 = high
    vec = [0, 0, 0, 0]
    vec[0 + strength if slip < 0 else 2 + (1 - strength)] = 1
    return tuple(vec)


def rpm_vec(throttle_position: float) -> tuple:
    """(<1700, <2300, <2700, >=2700) — always one-hot."""
    rpm = IDLE_RPM + max(0.0, min(1.0, throttle_position)) * (FULL_THROTTLE_RPM - IDLE_RPM)
    if throttle_position <= 0.0:
        rpm = 0.0
    vec = [0, 0, 0, 0]
    for i, edge in enumerate(RPM_BANDS):
        if rpm < edge:
            vec[i] = 1
            return tuple(vec)
    return (0, 0, 0, 1)


PANEL_GAIN = 3.0    # scalar strengthening the Gaussian signal into the brain


def speed_mps(body: RigidBody) -> float:
    v = body.velocity
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def pitch_deg(body: RigidBody) -> float:
    forward, _, _ = body.body_axes
    return math.degrees(math.asin(max(-1.0, min(1.0, forward.z))))


def bank_deg(body: RigidBody) -> float:
    """Signed bank, left negative — the school's convention."""
    _, right, _ = body.body_axes
    return -math.degrees(math.asin(max(-1.0, min(1.0, right.z))))


def slip_half_ball(body: RigidBody) -> float:
    """Sideways drift in half-ball units (the turn coordinator's ball)."""
    forward, right, _ = body.body_axes
    velocity = body.velocity
    slip = velocity.x * right.x + velocity.y * right.y + velocity.z * right.z
    return slip / (0.25 * CRUISE_MPS)


def _gauss_weights(positions: dict, mean: float, sigma: float,
                   circular: bool = False) -> dict:
    """A normalized Gaussian over class positions, mean at the reading.

    positions maps class index -> its place on the axis (signed line, or
    0..7 on the compass when circular). The reading's mean sits between
    the classes; each class keeps exp(-0.5 d²/σ²) of the mass, so 80 deg
    of bank is mostly the 90 class with a little 45 — and every axis
    blurs the same way, zones included.
    """
    if circular:
        def dist(c):
            return ((c - mean + 4.0) % 8.0) - 4.0
    else:
        def dist(c):
            return c - mean
    out = {c: math.exp(-0.5 * (dist(c) / sigma) ** 2)
           for c in positions}
    total = sum(out.values())
    return {c: w / total for c, w in out.items() if w > 1e-4}


PANEL_SIGMA = 0.55    # class-widths; the needle's own uncertainty

# Class positions on each axis's signed line, and the fold to tensor index.
_ASI_CLASSES = (1, 2, 3, 4)          # stall, cruise, Vne, above — speed line
_PITCH_FOLD = {0: 0, 1: 1, 2: 2, -1: 3, -2: 4}
_PITCH_CLASSES = tuple(_PITCH_FOLD)
_BANK_FOLD = {-4: 4, -3: 3, -2: 2, -1: 1, 0: 0, 1: 5, 2: 6, 3: 7, 4: 8}
_BANK_CLASSES = tuple(_BANK_FOLD)
_VSI_FOLD = {-4: 8, -3: 7, -2: 6, -1: 5, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
_VSI_CLASSES = tuple(_VSI_FOLD)
_TURN_FOLD = {-2: 1, -1: 2, 0: 0, 1: 3, 2: 4}
_TURN_CLASSES = tuple(_TURN_FOLD)
_RPM_CLASSES = (0, 1, 2, 3)


def _interp(value: float, knots: tuple) -> float:
    """Piecewise-linear map value -> axis position through (v, pos) knots."""
    for (v0, p0), (v1, p1) in zip(knots, knots[1:]):
        if value <= v1:
            return p0 + (p1 - p0) * (value - v0) / (v1 - v0)
    return knots[-1][1]


def asi_dist(speed_mps: float) -> dict:
    """Gaussian over the 4 speed zones (index 0 = parked at zero speed)."""
    if speed_mps <= 0.02 * STALL_MPS:
        return {0: 1.0}
    pos = _interp(speed_mps, ((0.0, 0.5), (STALL_MPS, 1.0),
                              (CRUISE_MPS, 2.0), (VNE_MPS, 3.0),
                              (2.0 * VNE_MPS, 4.0)))
    return _gauss_weights({c: c for c in _ASI_CLASSES}, pos, PANEL_SIGMA)


def pitch_dist(body: RigidBody) -> dict:
    """Gaussian over (neutral, up45, up90, down45, down90), mean at pitch."""
    forward, _, _ = body.body_axes
    deg = math.degrees(math.asin(max(-1.0, min(1.0, forward.z))))
    pos = max(-2.0, min(2.0, deg / 45.0))
    return _gauss_weights({s: _PITCH_FOLD[s] for s in _PITCH_CLASSES},
                          pos, PANEL_SIGMA)


def bank_dist(body: RigidBody) -> dict:
    """Gaussian over the ±45/90/135/180 bank classes, mean at bank."""
    _, right, _ = body.body_axes
    deg = -math.degrees(math.asin(max(-1.0, min(1.0, right.z))))
    pos = max(-4.0, min(4.0, -deg / 45.0))
    return _gauss_weights({s: _BANK_FOLD[s] for s in _BANK_CLASSES},
                          pos, PANEL_SIGMA)


def hsi_dist(body: RigidBody) -> dict:
    """Circular Gaussian over the 8 compass octants, mean at the bearing."""
    forward, _, _ = body.body_axes
    bearing = math.degrees(math.atan2(forward.y, forward.x)) / 45.0
    return _gauss_weights({c: c for c in range(8)}, bearing % 8.0,
                          PANEL_SIGMA, circular=True)


def vsi_dist(vz_mps: float) -> dict:
    """Gaussian over climb/descend 1250 ft/min classes, mean at VSI."""
    ftpm = vz_mps * FT_PER_M * 60.0
    pos = max(-4.0, min(4.0, ftpm / VSI_BIN_FTPM))
    return _gauss_weights({s: _VSI_FOLD[s] for s in _VSI_CLASSES},
                          pos, PANEL_SIGMA)


def turn_dist(body: RigidBody) -> dict:
    """Gaussian over the ball classes (-2..+2 half-ball units), mean at slip."""
    forward, right, _ = body.body_axes
    velocity = body.velocity
    slip = velocity.x * right.x + velocity.y * right.y + velocity.z * right.z
    pos = max(-2.0, min(2.0, slip / (0.25 * CRUISE_MPS)))
    return _gauss_weights({s: _TURN_FOLD[s] for s in _TURN_CLASSES},
                          pos, PANEL_SIGMA)


def rpm_dist(throttle_position: float) -> dict:
    """Gaussian over the RPM bands, mean at the lever's RPM."""
    rpm = IDLE_RPM + max(0.0, min(1.0, throttle_position)) * (FULL_THROTTLE_RPM - IDLE_RPM)
    if throttle_position <= 0.0:
        rpm = 0.0
    pos = _interp(rpm, ((0.0, 0.0), (RPM_BANDS[0], 1.0), (RPM_BANDS[1], 2.0),
                        (RPM_BANDS[2], 3.0), (3300.0, 3.4)))
    return _gauss_weights({c: float(c) for c in _RPM_CLASSES},
                          pos, PANEL_SIGMA)


def panel_dists(body: RigidBody, *, throttle_position: float = 0.15) -> tuple:
    """The cockpit reading as per-axis Gaussian distributions, gain-scaled.

    Each dict maps tensor axis index -> membership weight; the masses sum
    to ~PANEL_GAIN per axis (the scalar the brain pipeline asked for), and
    the needle sits between the marks — the state says so.
    """
    dists = (asi_dist(_speed(body)), pitch_dist(body), bank_dist(body),
             hsi_dist(body), vsi_dist(body.velocity.z),
             turn_dist(body), rpm_dist(throttle_position))
    return tuple({c: PANEL_GAIN * w for c, w in d.items()} for d in dists)


def panel_vector(body: RigidBody, *, throttle_position: float = 0.15) -> tuple:
    """The full cockpit reading as one flat one-hot word, 60 bits."""
    return (asi_vec(_speed(body)) + pitch_vec(body) + bank_vec(body)
            + hsi_vec(body) + vsi_vec(body.velocity.z)
            + turn_vec(body) + rpm_vec(throttle_position))


def panel_key(body: RigidBody, *, throttle_position: float = 0.15) -> tuple:
    """The same reading folded into tensor axis indices.

    (asi, pitch, bank, hsi, vsi, turn, rpm); 0 = that instrument's zero
    vector where one exists, else 1 + the active slot. HSI and RPM are
    always lit, so their index is the active slot (north = 0).
    """
    vec = panel_vector(body, throttle_position=throttle_position)
    cursor = 0
    key = []
    for width, zeroable in ((4, True), (4, True), (8, True), (8, False),
                            (8, True), (4, True), (4, False)):
        slots = vec[cursor:cursor + width]
        cursor += width
        active = [i for i, bit in enumerate(slots) if bit]
        key.append((1 + active[0]) if active and zeroable else (active[0] if active else 0))
    return tuple(key)


def _speed(body: RigidBody) -> float:
    v = body.velocity
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


if __name__ == "__main__":
    body = RigidBody(Vec3(0, 0, 80.0), Vec3(CRUISE_MPS, 0.0, 0.0))
    key = panel_key(body, throttle_position=0.65)
    vec = panel_vector(body, throttle_position=0.65)
    print("level cruise panel key:", key)
    print("level cruise bits:     ", vec)
    assert key == (3, 0, 0, 0, 0, 0, 1), key
    assert len(vec) == PANEL_VECTOR_LEN
    assert sum(vec) == 3, "ASI + HSI + RPM should be lit"

    # 80 deg of bank samples to the 90 slot (nearest of 45/90/135/180).
    body.attitude = body.attitude.from_euler(0.0, 0.0, 80.0 * math.pi / 180.0)
    key = panel_key(body, throttle_position=0.65)
    print("bank 80 deg key:       ", key)
    assert key[2] == 2, key           # 0 neutral, 1-4 left, 5-8 right; 90 = 2
    assert panel_vector(body)[8:16] == (0, 1, 0, 0, 0, 0, 0, 0)

    # Wings level again: crisp neutral.
    body.attitude = body.attitude.from_euler(0.0, 0.0, 0.0)
    body.velocity = Vec3(STALL_MPS * 0.9, 0.0, -6.0)
    key = panel_key(body, throttle_position=0.1)
    print("slow descent key:      ", key)
    assert key[0] == 2, key    # total speed 9.3 m/s: below cruise, above stall
    assert key[1] == 0 and key[2] == 0, key                   # level
    assert key[4] == 5, key                                   # ~ -1200 fpm

    # Every axis is a Gaussian centered at the true value: 80 deg of bank
    # is mostly the 90 class with some 45, and the masses sum to one.
    body.attitude = body.attitude.from_euler(0.0, 0.0, 80.0 * math.pi / 180.0)
    body.velocity = Vec3(CRUISE_MPS, 0.0, 0.0)
    dists = panel_dists(body, throttle_position=0.65)
    for name, d in zip(("asi", "pitch", "bank", "hsi", "vsi", "turn", "rpm"), dists):
        assert abs(sum(d.values()) - PANEL_GAIN) < 0.01, (name, d)
    print("bank 80 deg distribution:", {k: round(v, 3) for k, v in dists[2].items()})
    assert dists[2][2] == max(dists[2].values())          # 90 class dominates
    assert dists[2][1] == sorted(dists[2].values())[-2]   # 45 class second
    # A crisp 90 dominates its class but every class keeps some mass —
    # the Gaussian is over ALL classes, mean at the value.
    body.attitude = body.attitude.from_euler(0.0, 0.0, 90.0 * math.pi / 180.0)
    d = panel_dists(body)[2]
    assert max(d, key=d.get) == 2 and d[2] > d[1], d
    print("PANEL CHECK PASSED")
