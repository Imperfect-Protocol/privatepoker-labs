"""Airframe aerodynamic model: coefficient tables, forces, moments.

A deliberately simple fixed-wing UAV model — qualitatively right, not
CFD-exact. The learned layers never see these coefficients; they see
positions, bearings, and flight-instrument errors. What matters is that the
envelope is honest: lift needs airspeed, drag grows with induced lift, the
airframe is weathervane-stable, controls have authority, and stall exists.

Sign conventions (body frame X forward, Y right, Z up):

- alpha (angle of attack) > 0: relative wind from below the nose.
- beta (sideslip) > 0: relative wind from the right.
- Positive rotation about +Y is nose DOWN; about +Z is nose RIGHT; about +X
  is roll LEFT.
- Surface channels: [aileron, elevator, rudder, throttle], with
  learner-friendly signs: positive aileron = roll right, positive elevator =
  nose up, positive rudder = nose right, throttle in [0, 1]. The translation
  from these friendly signs to the body-frame moment signs happens here.
- Angular damping coefficients are stored positive and applied as decay;
  RigidBody integrates them implicitly so the fixed tick cannot diverge.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from skynet.vec import Vec3

RHO_SEA_LEVEL = 1.225
GRAVITY = Vec3(0.0, 0.0, -9.81)


@dataclass(frozen=True)
class AeroTables:
    """Airframe constants. One instance per airframe design."""
    mass: float = 1.5                 # kg
    wing_area: float = 0.5            # m^2
    wingspan: float = 1.5             # m
    inertia_x: float = 0.3            # kg m^2 (roll)
    inertia_y: float = 1.0            # kg m^2 (pitch)
    inertia_z: float = 1.2            # kg m^2 (yaw)

    # Lift. CL is clipped to [CL_min, CL_max]; beyond alpha_stall the wing is
    # separated and lift stops growing with alpha (hard clip in v1).
    cl0: float = 0.30
    cl_alpha: float = 5.0             # per rad
    alpha_stall: float = 0.35         # rad (~20 deg)
    cl_min: float = -0.6
    cl_max: float = 1.2

    # Drag: parasite + induced + alpha-dependent separation drag.
    cd0: float = 0.035
    k_induced: float = 0.06
    cd_alpha: float = 0.10

    # Side force.
    cy_beta: float = 0.15

    # Pitch: positive Cm is nose DOWN (body +Y moment), which inverts the
    # textbook (z-down) sign. cm_alpha > 0 is pitch-stable here: alpha > 0
    # (nose above the velocity) must produce a nose-down moment.
    # cm_de maps the friendly elevator sign (positive = nose up).
    # Stiffness is kept modest and damping strong so the untrimmed short
    # period is a visible wobble, not a numerical energy pump.
    # cm0 is a slight nose-UP bias so neutral trim is a gentle glide, not a dive.
    cm0: float = -0.01
    cm_alpha: float = 0.35
    cm_de: float = 0.9                # per rad, applied as -cm_de * elevator
    cm_q: float = 12.0                # damping magnitude, applied as decay

    # Roll: positive aileron = roll right = negative moment about +X.
    cl_da: float = 0.15
    cl_p: float = 0.45                # damping magnitude

    # Yaw: positive rudder = nose right = positive moment about +Z.
    cn_dr: float = 0.08
    cn_beta: float = 0.12
    cn_r: float = 0.25                # damping magnitude

    # Propulsion: thrust along body forward, throttle in [0, 1].
    thrust_max: float = 9.0           # N

    @property
    def chord(self) -> float:
        return self.wing_area / self.wingspan


@dataclass(frozen=True)
class AeroDiag:
    """Diagnostics for one integration step (tests, instruments, metrics)."""
    airspeed: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    q_dyn: float = 0.0
    lift_coefficient: float = 0.0
    lift: float = 0.0
    drag: float = 0.0
    thrust: float = 0.0


def _body_airspeed(velocity: Vec3, forward: Vec3, right: Vec3, up: Vec3) -> tuple:
    return (velocity.dot(forward), velocity.dot(right), velocity.dot(up))


def _lift_coefficient(alpha: float, tables: AeroTables) -> float:
    cl = tables.cl0 + tables.cl_alpha * alpha
    return max(tables.cl_min, min(tables.cl_max, cl))


def aero_force(velocity: Vec3, forward: Vec3, right: Vec3, up: Vec3,
               servos, tables: AeroTables, rho: float = RHO_SEA_LEVEL
               ) -> tuple:
    """World-frame aerodynamic + propulsion force. Returns (force, AeroDiag).

    ``servos`` is a SurfacesState (skynet.physics.surfaces) or any object with
    ``positions`` as [aileron, elevator, rudder, throttle].
    """
    speed = velocity.length()
    throttle = float(servos.positions[3])
    thrust_vec = forward.scale(throttle * tables.thrust_max)
    if speed < 0.5:
        # Too slow for meaningful aerodynamics; propulsion and gravity only.
        return thrust_vec, AeroDiag(airspeed=speed, thrust=throttle * tables.thrust_max)

    vx, vy, vz = _body_airspeed(velocity, forward, right, up)
    alpha = math.atan2(-vz, max(vx, 1e-6))
    beta = math.atan2(vy, math.hypot(vx, vz))
    q_dyn = 0.5 * rho * speed * speed

    cl = _lift_coefficient(alpha, tables)
    cd = tables.cd0 + tables.k_induced * cl * cl + tables.cd_alpha * abs(alpha)

    lift = q_dyn * tables.wing_area * cl
    drag = q_dyn * tables.wing_area * cd
    side = q_dyn * tables.wing_area * tables.cy_beta * beta

    vel_unit = velocity.scale(1.0 / speed)
    # Lift direction: body-up projected perpendicular to the velocity,
    # i.e. (vel x up) x vel = up_perp. (Order matters: the reversed cross
    # product yields -up_perp, an anti-lift airframe.)
    lift_dir = vel_unit.cross(up).cross(vel_unit)
    lift_dir = lift_dir.normalized() if lift_dir.length() > 1e-9 else up
    drag_dir = vel_unit.scale(-1.0)
    side_dir = right.scale(-1.0)  # beta > 0 (wind from right) pushes left

    force = (lift_dir.scale(lift)
             .add(drag_dir.scale(drag))
             .add(side_dir.scale(side))
             .add(thrust_vec))
    diag = AeroDiag(airspeed=speed, alpha=alpha, beta=beta, q_dyn=q_dyn,
                    lift_coefficient=cl, lift=lift, drag=drag,
                    thrust=throttle * tables.thrust_max)
    return force, diag


def control_stability_moment(alpha: float, beta: float, servos, tables: AeroTables,
                             q_dyn: float) -> Vec3:
    """Body-frame aerodynamic moment WITHOUT damping (damping is implicit).

    Signs per the module docstring: +X moment rolls left, +Y moment pitches
    nose down, +Z moment yaws right. Friendly surface signs translate here.
    """
    aileron, elevator, rudder = (float(servos.positions[0]),
                                 float(servos.positions[1]),
                                 float(servos.positions[2]))
    # Beyond stall the moment curve flattens: clip alpha's influence so the
    # restoring moment cannot pump energy during separated flow.
    moment_alpha = max(-2.0 * tables.alpha_stall,
                       min(2.0 * tables.alpha_stall, alpha))
    moment_beta = max(-tables.alpha_stall, min(tables.alpha_stall, beta))
    ref_moment = q_dyn * tables.wing_area * tables.chord
    ref_roll = q_dyn * tables.wing_area * tables.wingspan
    cx = ref_roll * (-tables.cl_da * aileron)
    cy = ref_moment * (tables.cm0 + tables.cm_alpha * moment_alpha
                       - tables.cm_de * elevator)
    cz = ref_moment * (tables.cn_dr * rudder + tables.cn_beta * moment_beta)
    return Vec3(cx, cy, cz)


def damping_slope_per_axis(q_dyn: float, airspeed: float, tables: AeroTables,
                           rho: float = RHO_SEA_LEVEL) -> tuple:
    """Per-axis linear damping slopes (1/s) for implicit integration.

    Damping moment ~ q_dyn * S * ref * C_damp * (rate * ref / (2V)), so the
    slope on rate is q_dyn * S * ref^2 * C_damp / (2V) / I. A small base
    viscosity keeps rotation well-behaved even with zero airspeed.
    """
    ref_moment = tables.chord
    ref_roll = tables.wingspan
    if airspeed < 0.5:
        return (0.05, 0.05, 0.05)
    denom = 2.0 * airspeed
    kx = q_dyn * tables.wing_area * ref_roll * ref_roll * tables.cl_p / denom / tables.inertia_x
    ky = q_dyn * tables.wing_area * ref_moment * ref_moment * tables.cm_q / denom / tables.inertia_y
    kz = q_dyn * tables.wing_area * ref_moment * ref_moment * tables.cn_r / denom / tables.inertia_z
    return (kx + 0.05, ky + 0.05, kz + 0.05)
