"""6-DOF rigid body flight at a fixed tick.

State: position, velocity (world frame, +Z up), attitude (quaternion,
body X forward / Y right / Z up), and body angular rates (p, q, r).

Integration at each fixed dt:

1. Aero + propulsion force from velocity, attitude, surfaces, tables.
2. Velocity += (force / mass + gravity) * dt; position += velocity * dt.
3. Control/stability moment; angular rates get damping integrated
   IMPLICITLY (unconditionally stable at any dt), control explicitly.
4. Attitude advances by body rates and renormalizes.

Determinism: same state + same inputs + same dt => identical trajectory.
No randomness lives here; chaos enters the architecture at the tensor
pipeline, not in the physics.
"""
from __future__ import annotations

from skynerd.physics import aero
from skynerd.physics.aero import AeroTables, AeroDiag, GRAVITY, RHO_SEA_LEVEL
from skynerd.physics.surfaces import SurfacesState
from skynerd.vec import Quaternion, Vec3


class RigidBody:
    """One airframe's flight state and its fixed-tick dynamics."""

    def __init__(self, position: Vec3, velocity: Vec3,
                 yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                 tables: AeroTables | None = None) -> None:
        self.position = position
        self.velocity = velocity
        self.attitude = Quaternion.from_euler(yaw, pitch, roll)
        self.rate_p = 0.0   # roll rate, rad/s, about body +X
        self.rate_q = 0.0   # pitch rate, rad/s, about body +Y
        self.rate_r = 0.0   # yaw rate, rad/s, about body +Z
        self.tables = tables if tables is not None else AeroTables()

    @property
    def body_axes(self) -> tuple:
        return self.attitude.body_axes()

    def step(self, servos: SurfacesState, dt: float, *,
             gravity: Vec3 = GRAVITY, rho: float = RHO_SEA_LEVEL) -> AeroDiag:
        """Advance one fixed physics tick. Returns aero diagnostics."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        tables = self.tables
        forward, right, up = self.body_axes

        force, diag = aero.aero_force(self.velocity, forward, right, up,
                                      servos, tables, rho)
        self.velocity = self.velocity.add(
            force.scale(1.0 / tables.mass).add(gravity).scale(dt))
        self.position = self.position.add(self.velocity.scale(dt))

        alpha, beta, q_dyn = diag.alpha, diag.beta, diag.q_dyn
        moment = aero.control_stability_moment(alpha, beta, servos, tables, q_dyn)
        kx, ky, kz = aero.damping_slope_per_axis(q_dyn, diag.airspeed, tables, rho)
        # Implicit damping: rate_new = (rate + explicit_rate_delta) / (1 + k*dt).
        ix, iy, iz = tables.inertia_x, tables.inertia_y, tables.inertia_z
        self.rate_p = (self.rate_p + moment.x / ix * dt) / (1.0 + kx * dt)
        self.rate_q = (self.rate_q + moment.y / iy * dt) / (1.0 + ky * dt)
        self.rate_r = (self.rate_r + moment.z / iz * dt) / (1.0 + kz * dt)

        self.attitude = self.attitude.integrate_body_rates(
            self.rate_p, self.rate_q, self.rate_r, dt)
        return diag
