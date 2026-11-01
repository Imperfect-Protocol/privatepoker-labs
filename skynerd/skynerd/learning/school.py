"""The flight school: an instructor demonstrates, the drone only remembers.

Before a newborn ever launches, it sits in the classroom with the physics
running. Each lesson names a standard maneuver; an instructor controller
(honest feedback laws: bank, climb, speed targets -> deflections) flies it
from the lesson's start state, and the drone does exactly one thing:
observe the instrument state at each decision tick and remember what the
instructor's hands were doing, translated into its own servo vocabulary
(push / ease / hold per surface, hold-all when hands are still). No reward,
no exploration — pure watch-and-remember imprinting on the servo tensor.

The instructor is allowed to be a controller; the student never sees the
controller. It sees instrument state -> servo nudges, exactly the mapping
it will later reinforce (or anti-reinforce) with its life. Lives are short
and expensive, so whatever is cheap to demonstrate should never have to be
learned by crashing. The will's emergent desires still grow on top of this
foundation — school teaches the alphabet, not the sentences.
"""
from __future__ import annotations

import math

import numpy as np

from skynerd.learning.reward import FlightInstruments
from skynerd.learning.flight import TRIM_SPEED
from skynerd.learning.servo_tensor import (EASE_OFF, HOLD, PUSH,
                                          SURFACE_AXIS, HOLD_ALL_SURFACE,
                                          ServoPolicy)
from skynerd.physics.rigid_body import RigidBody
from skynerd.physics.surfaces import (AILERON, ELEVATOR, RUDDER, THROTTLE,
                                     SURFACE_LIMITS, SurfacesState)
from skynerd.physics.world import World
from skynerd.vec import Vec3
from skynerd.learning.flight import TRIM_COMMAND, spring_toward_neutral
from skynerd.learning.will import goal_reward

# Desire slots the school names in the will. After a lesson, waking that
# desire means wanting that maneuver — school gives the unlabeled will its
# first vocabulary, and life reshapes it afterwards.
DESIRE_LEFT = 0
DESIRE_RIGHT = 1
DESIRE_CLIMB = 2
DESIRE_DESCEND = 3
DESIRE_LEVEL = 4
DESIRE_RECOVER = 5
DESIRE_BEARING = 6

DECISION_EVERY = 10        # same 5 Hz decision clock as the flight loop
AILERON_LIMIT = SURFACE_LIMITS[AILERON][1]

# The graduation exam is scored in quality points, higher = better: the raw
# tracking error (0.0 = the student's panel sits exactly on the instructor's
# panel at every tick) is flipped into points out of QUALITY_CAP. A lesson
# graduates above GRADUATION_BAR — mean panel drift under 0.9. Weak lessons
# are drilled (re-stamped, harder each round) until they pass or the round
# cap hits.
QUALITY_CAP = 6.0        # the exam is graded 0-6: 0 = failed badly ... 6 = exceptional
QUALITY_SLOPE = 2.0      # quality = 6 - 2 * mean tracking error
DEFAULT_PASS_SCORE = 4.0 # graduation bar: at least "good" in every lesson
GRADUATION_ROUNDS = 100  # drill until every lesson passes (or this cap)

GRADE_NAMES = (
    (5.5, "exceptional"),
    (4.5, "very good"),
    (3.5, "good"),
    (2.5, "pass"),
    (1.5, "below expectations"),
    (0.5, "must repeat"),
)

def grade_name(score: float) -> str:
    """The report-card word for a 0-6 score: 6 exceptional ... 0 failed badly."""
    for threshold, name in GRADE_NAMES:
        if score >= threshold:
            return name
    return "failed badly"
DEMO_BAND = 0.15         # inside this panel band around the demo, the student
                         # keeps the controls; outside it, the instructor takes over
STRETCH_TICKS = 15       # one takeover demonstration lasts at most 3 s

# Speeds named like the C172 POH, scaled to this airframe: cruise 80 mph maps
# to the trim speed, and 60 mph (rotate AND stall, like the real 172) maps to
# 0.75 cruise. The student never sees these numbers — only the binned errors.
CRUISE_MPH = 80.0
CRUISE_SPEED = TRIM_SPEED
STALL_MPH = 60.0
STALL_SPEED = 0.75 * CRUISE_SPEED
TAKEOFF_MPH = 60.0
ROTATE_SPEED = 0.75 * CRUISE_SPEED


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _between(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bank_rad_target_of(body: RigidBody) -> float:
    """Current bank angle in radians (positive = right)."""
    _, right, _ = body.body_axes
    return -math.asin(max(-1.0, min(1.0, right.z)))


class Lesson:
    """One standard maneuver, flown by feedback to instrument targets.

    The lesson carries its INTENT: a desire slot in the will and the
    instrument targets (vs_err, bank_err, spd_err) that define the maneuver —
    "turn left 30 degrees" is desire ``turn_left`` with bank target -0.67.
    targets: the instructor holds these; rudder_law adds a yaw job
    ("oppose_spin", "into_turn"). start overrides the classroom start state
    for stalls, spins, spiral dives, takeoff.
    """

    def __init__(self, name: str, duration: float, *,
                 desire: int, targets: tuple,
                 bank_rad: float = 0.0, vs_err: float = 0.0,
                 rudder_law: str = "neutral",
                 elevator_law: str = "vs", throttle_law: str = "auto",
                 aileron_law: str = "bank", heading_offset: float = 0.0,
                 imprint_after: float = 0.0,
                 stamp_bank: tuple | None = None,
                 start: dict | None = None) -> None:
        self.name = name
        self.duration = duration
        self.desire = desire
        self.targets = targets
        self.bank_rad = bank_rad
        self.vs_err = vs_err
        self.rudder_law = rudder_law
        self.elevator_law = elevator_law
        self.throttle_law = throttle_law
        # "bank" holds a fixed bank; "heading" flies proportional navigation
        # to a bearing: bank shrinks as the HSI delta closes, so the demo
        # itself eases off approaching the bearing — no oscillation taught.
        self.aileron_law = aileron_law
        self.heading_offset = heading_offset
        # Entry transients are goal-driven (the instructor's hidden intent),
        # not state reflexes; only stamp once the maneuver is established.
        self.imprint_after = imprint_after
        # If set (lo, hi) radians: stamp only while the bank is inside the
        # intended band — a spiraling demo must not teach stabilization.
        self.stamp_bank = stamp_bank
        self.start = start or {}

    def _heading_err(self, body: RigidBody) -> float:
        """HSI delta to this lesson's bearing (zero), normalized [-1, 1]."""
        forward, _, _ = body.body_axes
        return self._wrap(-math.atan2(forward.y, forward.x)) / (math.pi / 2.0)

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _spawn(self) -> RigidBody:
        body = RigidBody(Vec3(0, 0, 80.0), Vec3(CRUISE_SPEED, 0.0, 0.0))
        if self.aileron_law == "heading":
            body.attitude = body.attitude.from_euler(
                self.heading_offset, 0.0, 0.0)   # start off bearing, wings level
            return body
        return self._spawn_kind()

    def _spawn_kind(self) -> RigidBody:
        if self.start.get("kind") == "ground":
            # Rolling start just below rotate speed: the runway roll is short.
            return RigidBody(Vec3(0, 0, 2.5),
                             Vec3(ROTATE_SPEED * 0.95, 0.0, 0.0))
        if self.start.get("kind") == "stalled":
            return RigidBody(Vec3(0, 0, 80.0), Vec3(STALL_SPEED * 0.65, 0.0, -1.0))
        if self.start.get("kind") == "spinning":
            body = RigidBody(Vec3(0, 0, 80.0), Vec3(STALL_SPEED * 0.8, 0.0, 0.0))
            body.rate_r = 2.5
            return body
        if self.start.get("kind") == "spiral":
            # Overbanked descending speed dive — the classic death spiral.
            body = RigidBody(Vec3(0, 0, 80.0), Vec3(14.0, 0.0, -4.0))
            body.attitude = body.attitude.from_euler(0.0, 0.0,
                                                     self.bank_rad)
            return body
        return RigidBody(Vec3(0, 0, 80.0), Vec3(CRUISE_SPEED, 0.0, 0.0))

    def instructor_targets(self, body: RigidBody,
                           errors) -> np.ndarray:
        """The instructor's hands: deflection targets from feedback laws."""
        forward, right, _ = body.body_axes
        bank_rad = -math.asin(max(-1.0, min(1.0, right.z)))
        # Bank error plus roll-rate damping (rate_p < 0 is rolling right).
        if self.aileron_law == "heading":
            # Proportional navigation: bank toward the bearing, shrinking
            # with the delta — the HSI needle leads the rollout.
            delta = self._heading_err(body)
            limit = 60.0 * math.pi / 180.0
            self.bank_rad = max(-limit, min(limit, -2.0 * delta))
        if self.aileron_law == "level":
            # Unusual-attitude recovery: the wings come level first, from
            # whatever bank the student finds itself in. (bank_rad here is
            # only the demo's spawn attitude, never a bank to hold.)
            aileron = _clamp(-2.0 * bank_rad + 2.5 * body.rate_p,
                             AILERON_LIMIT)
        else:
            aileron = _clamp(2.0 * (self.bank_rad - bank_rad) + 2.5 * body.rate_p,
                             AILERON_LIMIT)
        if self.elevator_law == "speed" and errors.z < -0.25:
            # Handbook recoveries: forward stick until the speed returns.
            elevator = _clamp(1.5 * errors.z, 0.25)
        elif self.elevator_law == "roll_first":
            # Unusual attitude: wings level before pulling — pulling while
            # overbanked only tightens the spiral.
            elevator = 0.0 if abs(bank_rad) > 0.5 else _clamp(
                0.4 * (self.vs_err - errors.x), 0.25)
        else:
            elevator = _clamp(0.4 * (self.vs_err - errors.x), 0.25)
        if self.rudder_law == "oppose_spin":
            rudder = _clamp(-3.0 * body.rate_r, 0.35)
        elif self.rudder_law == "into_turn":
            rudder = _clamp(0.5 * aileron, 0.35)
        else:
            rudder = 0.0
        # Trim power plus load-factor demand for the bank angle (steep turns
        # need pull power like the POH says).
        if self.throttle_law == "full":
            throttle = 1.0
        elif self.throttle_law == "idle":
            throttle = 0.1
        else:
            load = 0.4 * abs(math.sin(bank_rad))
            throttle = _between(0.15 - 0.5 * errors.z + load, 0.0, 1.0)
        if (self.start.get("kind") == "ground"
                and body.velocity.length() < ROTATE_SPEED):
            elevator, throttle = 0.05, 1.0   # ground roll: 60 mph before rotating
        return np.array([aileron, elevator, rudder, throttle])


class FlightSchool:
    """Runs the curriculum against a student's servo tensor."""

    def __init__(self, instruments: FlightInstruments, *,
                 world: World | None = None,
                 imprint_amount: float = 1.0) -> None:
        self.instruments = instruments
        self.world = world if world is not None else World(half_extent=800.0)
        self.imprint_amount = float(imprint_amount)
        self.dt = 1.0 / 50.0
        self.graduate = None   # the reference drone the whole flock copies
        self.lessons: dict = {}
        self._define_curriculum()

    def _define_curriculum(self) -> None:
        """The standard syllabus, each lesson carrying its intent: a desire
        slot in the will and the instrument targets that define the maneuver.
        Turning left is desire LEFT with bank target -0.67 (= 30 deg); the
        recovery family shares one desire — speed is what matters there.
        """
        turn = 30.0 * math.pi / 180.0
        steep = 60.0 * math.pi / 180.0
        near = 0.26   # stamp only within ~15 degrees of the intended bank
        bank30 = round(turn / (math.pi / 4.0), 3)          # 0.667
        level = (0.0, 0.0, 0.0)
        names = (
            ("takeoff", 6.0, dict(desire=DESIRE_CLIMB, targets=(0.1, 0.0, 1.0),
                                  bank_rad=0.0, throttle_law="full",
                                  start={"kind": "ground"})),
            ("level", 3.0, dict(desire=DESIRE_LEVEL, targets=level)),
            ("turn_left_30", 5.0, dict(desire=DESIRE_LEFT,
                                       targets=(0.0, -bank30, 0.0),
                                       bank_rad=-turn,
                                       imprint_after=1.5,
                                       stamp_bank=(-turn - near, -turn + near))),
            ("turn_right_30", 5.0, dict(desire=DESIRE_RIGHT,
                                        targets=(0.0, bank30, 0.0),
                                        bank_rad=+turn,
                                        imprint_after=1.5,
                                        stamp_bank=(turn - near, turn + near))),
            ("climb_turn_left_30", 5.0, dict(desire=DESIRE_LEFT,
                                             targets=(0.5, -bank30, 0.0),
                                             bank_rad=-turn, vs_err=0.5,
                                             imprint_after=1.5,
                                             stamp_bank=(-turn - near, -turn + near))),
            ("climb_turn_right_30", 5.0, dict(desire=DESIRE_RIGHT,
                                              targets=(0.5, bank30, 0.0),
                                              bank_rad=+turn, vs_err=0.5,
                                              imprint_after=1.5,
                                              stamp_bank=(turn - near, turn + near))),
            ("descend_turn_left_30", 5.0, dict(desire=DESIRE_DESCEND,
                                               targets=(-0.5, -bank30, 0.0),
                                               bank_rad=-turn, vs_err=-0.5,
                                               imprint_after=1.5,
                                               stamp_bank=(-turn - near, -turn + near))),
            ("descend_turn_right_30", 5.0, dict(desire=DESIRE_DESCEND,
                                                targets=(-0.5, bank30, 0.0),
                                                bank_rad=+turn, vs_err=-0.5,
                                                imprint_after=1.5,
                                                stamp_bank=(turn - near, turn + near))),
            ("steep_turn_left_60", 5.0,
             dict(desire=DESIRE_LEFT, targets=(0.0, -1.0, 0.0),
                  bank_rad=-steep, rudder_law="into_turn",
                  imprint_after=1.5, stamp_bank=(-steep - near, -steep + near))),
            ("steep_turn_right_60", 5.0,
             dict(desire=DESIRE_RIGHT, targets=(0.0, 1.0, 0.0),
                  bank_rad=+steep, rudder_law="into_turn",
                  imprint_after=1.5, stamp_bank=(steep - near, steep + near))),
            ("stall_recovery", 4.0, dict(desire=DESIRE_RECOVER,
                                         targets=(0.0, 0.0, 1.0),
                                         elevator_law="speed",
                                         throttle_law="full",
                                         start={"kind": "stalled"})),
            ("spin_recovery", 7.0, dict(desire=DESIRE_RECOVER,
                                        targets=(0.0, 0.0, 1.0),
                                        elevator_law="speed",
                                        throttle_law="idle",
                                        rudder_law="oppose_spin",
                                        start={"kind": "spinning"})),
            ("unusual_attitude_left", 4.0,
             dict(desire=DESIRE_RECOVER, targets=(0.2, 0.0, 0.5),
                  bank_rad=-75.0 * math.pi / 180.0, elevator_law="roll_first",
                  aileron_law="level", start={"kind": "spiral"})),
            ("unusual_attitude_right", 4.0,
             dict(desire=DESIRE_RECOVER, targets=(0.2, 0.0, 0.5),
                  bank_rad=+75.0 * math.pi / 180.0, elevator_law="roll_first",
                  aileron_law="level", start={"kind": "spiral"})),
            # Navigational lessons are parked: the proportional-nav demo
            # still overshoots the bearing, and a spiraling demo must not
            # teach stabilization. Return with rate-lead compensation.
            # ("turn_to_bearing_left/right": aileron_law="heading",
            #  heading_offset=±pi/6, desire=DESIRE_BEARING)
        )
        self.lessons = {name: Lesson(name, duration, **kwargs)
                        for name, duration, kwargs in names}

    def teach(self, policy, *, lessons: tuple | None = None,
              will=None, rehearse: bool = False,
              bar: float = DEFAULT_PASS_SCORE,
              max_rounds: int = GRADUATION_ROUNDS) -> dict:
        """Imprint the curriculum, then drill until the student graduates.

        First pass: at each decision tick the instructor's deflection targets
        are translated into the student's vocabulary — nudge any surface more
        than a deadband away from its target, hold-all otherwise ("bank
        left, hold elevator") — and stamped into the servo tensor at the
        current panel reading (VSI, bank, ASI, RPM). If a will is present,
        the lesson's intent is imprinted too: the maneuver's desire wakes
        at this context and its aim becomes the maneuver's targets.

        Then the exam, repeated: every lesson graded below the pass mark
        is drilled — the student flies the maneuver and the instructor takes
        the controls whenever it drifts out of the demo's band, showing
        the correction from the student's actual state (the readings the
        student will really see when it ends up here again) — and the exam
        is retaken, until all lessons pass or the round cap hits.
        Returns per-lesson total imprint counts for the report.
        """
        names = tuple(lessons) if lessons is not None else tuple(self.lessons)
        for name in names:
            if name not in self.lessons:
                raise KeyError(f"unknown lesson: {name}")
        counts = {name: 0 for name in names}

        for name, stamped in self._stamp_pass(policy, names, will).items():
            counts[name] += stamped
            self._print_line(policy, name, stamped)

        # The graduate the flock copies is the base curriculum: the clean
        # demonstration reflexes, life-tested in the cull. The drilling
        # below polishes the classroom exam, but its corrections make a
        # newborn act in calm air — pure risk out there — so they stay in
        # the classroom. One weights set flies the exam, another inherits.
        base = ServoPolicy()
        base.tensor.values = {k: v.copy()
                              for k, v in policy.tensor.values.items()}
        base.tensor.epochs = {k: e.copy()
                              for k, e in policy.tensor.epochs.items()}
        self.graduate = base

        # The graduation exam: repeat training until every lesson clears
        # the bar. Each drill round is a corrective show-and-repeat pass:
        # the student tries the maneuver, the instructor takes the controls
        # on every drift out of the demo's band and stamps the correction
        # at the student's actual state, so memory grows along the
        # trajectory the student really flies — not just the demo's.
        failing = [n for n in names
                   if self.student_quality(policy, self.lessons[n]) < bar]
        rounds = 0
        while failing and rounds < max_rounds:
            rounds += 1
            print(f"  {len(failing)} lesson(s) below the {bar:.1f} pass mark — "
                  f"instructor drills them again (round {rounds}/{max_rounds})")
            still_failing = []
            for name in failing:
                lesson = self.lessons[name]
                stamped, takeovers = self._corrective_pass(
                    policy, lesson, will,
                    drill_epoch=1 + (rounds - 1) % 7)
                counts[name] += stamped
                self._print_line(policy, name, stamped, takeovers)
                if self.student_quality(policy, lesson) < bar:
                    still_failing.append(name)
            failing = still_failing
        if failing:
            print(f"  still below the {bar:.1f} pass mark after "
                  f"{max_rounds} rounds: {', '.join(failing)}")
        elif rounds:
            print("  every lesson now at or above the pass mark — "
                  "the student graduates")
        print("  (the exam is flown by the drilled class; the flock inherits "
              "the base curriculum — drilled corrections are classroom polish)")
        if rehearse:
            self.rehearse(policy)
        return counts

    def _demo_tick(self, policy, lesson, body, servos, will, seconds,
                   takeover: bool = False,
                   student_throttle: float | None = None,
                   time_epoch: int = 0,
                   amount: float | None = None,
                   demo_near_trim: bool = True) -> tuple:
        """One instructor demonstration tick at the body's actual state.

        The instructor's feedback laws produce deflection targets (always
        returned — the instructor flies whatever the gates say about
        stamping). When the lesson's anti-poisoning gates allow — past
        imprint_after, inside the stamp bank — the state -> action reading
        is stamped into the student's servo tensor and, with a will
        attached, the lesson's intent is imprinted too.

        A takeover tick (the instructor correcting a drifting student)
        drops the demo's imprint_after gate — a correction at the student's
        actual state is exactly the honest reading we want stamped — but
        keeps the near-trim hands-off check (acting when nothing is needed
        is the over-control habit that kills) and a one-sided stamp bank:
        outside the demo's band a correction may only reduce bank, never
        deepen it. Takeover stamps land at double strength: a firm
        correction must outscore the hands-off memory it is correcting,
        or hold-all wins every tie and the student never changes.

        A takeover stamp is addressed to the student's state: the state key
        carries the throttle lever the student was holding, not the
        instructor's throttle target, so the exam-time student (own lever,
        own state bins) actually reads the stamp back.
        Returns (targets, stamped).
        """
        errors = self.instruments.errors(body)
        targets = lesson.instructor_targets(body, errors)
        if lesson.stamp_bank is not None:
            lo, hi = lesson.stamp_bank
            bank = bank_rad_target_of(body)
            if lo <= bank <= hi:
                pass
            elif not takeover:
                return targets, False
            else:
                # A takeover correction must fly bank TOWARD the lesson's
                # band: a shallow student is banked into the maneuver, an
                # over-banked one is rolled back. Only deepening BEYOND the
                # band stays forbidden — that is the spiral trap: "bank
                # more" stamped where wild spiral dives visit. And even the
                # toward-band deepening is barred in a dive signature
                # (strong descent or high speed): an early spiral dive
                # reads like a shallow turn in these bins, and banking
                # deeper there is the poison that kills.
                in_dive = (errors.x < -0.3 or errors.z > 0.3)
                if in_dive:
                    return targets, False
                if bank > hi and targets[AILERON] >= 0:
                    return targets, False
                if bank < lo and targets[AILERON] <= 0:
                    return targets, False
        if not takeover and seconds < lesson.imprint_after:
            return targets, False
        throttle = targets[3] if student_throttle is None else student_throttle
        state_key = self.instruments.state_key(body, throttle)
        panel = self.instruments.errors(body)
        # Feel the plane: near trim, the lesson is hands off — acting when
        # nothing is needed is the over-control habit that kills. Reflexes
        # fire only off-trim. At a takeover the same rule reads the DEMO:
        # if the demonstration is off maneuvering, a near-trim student is
        # behind the maneuver — hands-off is the failure being corrected —
        # so the instructor's real hands get stamped.
        near_trim = (abs(panel.x) < 0.15 and abs(panel.y) < 0.15
                     and abs(panel.z) < 0.2)
        if near_trim and (demo_near_trim or not takeover):
            actions = np.full(4, HOLD, dtype=np.int64)
        else:
            actions = self._translate(targets, servos)
        policy.imprint(state_key, actions,
                       amount=self.imprint_amount if amount is None else amount,
                       time_epoch=time_epoch)
        if will is not None:
            will.imprint_lesson(state_key, lesson.desire,
                                lesson.targets)
        return targets, True

    def _stamp_pass(self, policy, names, will) -> dict:
        """Demonstrate each named lesson once, stamped into the servo tensor."""
        stamped_by = {}
        for name in names:
            lesson = self.lessons[name]
            stamped = 0
            body = lesson._spawn()
            servos = SurfacesState()
            for tick in range(int(lesson.duration / self.dt)):
                seconds = tick * self.dt
                if tick % DECISION_EVERY == 0:
                    targets, hit = self._demo_tick(
                        policy, lesson, body, servos, will, seconds)
                    stamped += hit
                else:
                    targets = lesson.instructor_targets(
                        body, self.instruments.errors(body))
                servos.command(targets, self.dt)
                body.step(servos, self.dt)
                if body.position.z <= self.world.terrain_height(
                        body.position.x, body.position.y) + 1.0:
                    break
            stamped_by[name] = stamped
        return stamped_by

    def _corrective_pass(self, policy, lesson, will, drill_epoch: int) -> tuple:
        """Show-and-repeat with corrective takeover, closed on the exam.

        The student first repeats the exam exactly: memory-only, no
        exploration, the stick spring active — a deterministic replay of
        the graduation flight. Whenever its panel drifts out of the demo's
        band, the instructor takes the controls AT THAT STATE — a state
        the next exam is guaranteed to visit — stamps the firm correction
        there (the readings the student will really see when it ends up
        here again), flies the recovery until the panel is back on the
        demo's line, then hands back and lets the student continue.

        Corrections are stamped into their own time slot per round (slot 0
        stays the original demonstration), so repetition accumulates
        instead of rewriting the same saturated slot. Returns
        (stamps, takeovers).
        """
        demo = self._demo_panel(lesson)
        body = lesson._spawn()
        servos = SurfacesState()
        commands = TRIM_COMMAND.copy()
        rng = np.random.default_rng(0)
        saved, policy.epsilon = policy.epsilon, 0.0   # replay the exam exactly
        stamps, takeovers, stretch = 0, 0, 0
        student_throttle = None
        for tick in range(int(lesson.duration / self.dt)):
            seconds = tick * self.dt
            if tick % DECISION_EVERY == 0:
                panel = self.instruments.errors(body).as_array()
                goal = demo[min(tick // DECISION_EVERY, len(demo) - 1)]
                drift = max(abs(e - g) for e, g in zip(panel, goal))
                demo_near_trim = (abs(goal[0]) < 0.15 and abs(goal[1]) < 0.15
                                  and abs(goal[2]) < 0.2)
                if stretch == 0 and drift > DEMO_BAND:
                    takeovers += 1        # the student has it wrong
                    stretch = STRETCH_TICKS
                    # Address the coming corrections to the student's own
                    # lever position, so its exam-time lookups find them.
                    student_throttle = commands[3]
                    # Un-teach the hands-off reflex at this state: the drift
                    # proves that doing nothing here was wrong. Only hold-all
                    # is weakened — never a surface memory, which may be the
                    # right reading for a sibling context sharing these bins.
                    # The anti-stamp lands in slot 0, where the accumulated
                    # hands-off weight lives, and stops at a reserve: the
                    # goal is that the demonstrated correction outscores
                    # hands-off here, not that hands-off is destroyed —
                    # in the wild it is still the safe fallback.
                    takeover_key = self.instruments.state_key(
                        body, student_throttle)
                    holdall_key = takeover_key + (HOLD_ALL_SURFACE, HOLD)
                    if policy.tensor.score(holdall_key, time_epoch=0,
                                           decay=1.0) > 2.0:
                        policy.tensor.reward(holdall_key, time_epoch=0,
                                             amount=-1.5)
                if stretch > 0:
                    if drift <= DEMO_BAND:
                        stretch = 0       # back on the demo line: hands off
                    else:
                        targets, hit = self._demo_tick(
                            policy, lesson, body, servos, will, seconds,
                            takeover=True, student_throttle=student_throttle,
                            time_epoch=drill_epoch,
                            amount=2.0 * self.imprint_amount,
                            demo_near_trim=demo_near_trim)
                        stamps += hit
                        commands = targets   # the instructor's hands
                        stretch -= 1
                if stretch == 0:
                    state_key = self.instruments.state_key(body, commands[3])
                    actions = policy.choose(state_key, rng)
                    commands = policy.command_from(commands, actions)
            servos.command(commands, self.dt)
            body.step(servos, self.dt)
            commands = spring_toward_neutral(commands, self.dt)
            if body.position.z <= self.world.terrain_height(
                    body.position.x, body.position.y) + 1.0:
                break
        policy.epsilon = saved
        return stamps, takeovers

    def _print_line(self, policy, name, stamps, takeovers=None) -> None:
        score = self.student_quality(policy, self.lessons[name])
        line = (f"lesson {name:24s} stamps {stamps:3d}  "
                f"score {score:.2f} ({grade_name(score)})")
        if takeovers is not None:
            line += f"  takeovers {takeovers}"
        print(line)

    def student_quality(self, policy, lesson) -> float:
        """How well the student repeats the maneuver, graded 0-6.

        The student flies the lesson itself (memory only, no exploration);
        at each tick we measure how far its panel (VSI, bank, ASI) sits from
        the instructor's panel for that phase. The tracking error (0 =
        perfect repeat) is flipped into a grade: 6 - SLOPE * error, clamped
        to [0, 6]. 6 exceptional, 5 very good, 4 good, 3 pass,
        2 below expectations, 1 must repeat, 0 failed badly. This is the
        graduation exam.
        """
        return max(0.0, min(QUALITY_CAP,
                            QUALITY_CAP - QUALITY_SLOPE * self._tracking_error(policy, lesson)))

    def exam_scores(self, policy, names) -> tuple:
        """The whole class's exam scores, for spotting a plateau."""
        return tuple(round(self.student_quality(policy, self.lessons[n]), 2)
                     for n in names)

    def _tracking_error(self, policy, lesson) -> float:
        """Mean panel drift from the instructor's demo (0 = perfect repeat)."""
        demo = self._demo_panel(lesson)
        body = lesson._spawn()
        servos = SurfacesState()
        commands = TRIM_COMMAND.copy()
        rng = np.random.default_rng(0)
        saved, policy.epsilon = policy.epsilon, 0.0
        total, ticks = 0.0, 0
        for tick in range(int(lesson.duration / self.dt)):
            if tick % DECISION_EVERY == 0:
                state_key = self.instruments.state_key(body, commands[3])
                actions = policy.choose(state_key, rng)
                commands = policy.command_from(commands, actions)
            servos.command(commands, self.dt)
            body.step(servos, self.dt)
            commands = spring_toward_neutral(commands, self.dt)
            goal = demo[min(tick // DECISION_EVERY, len(demo) - 1)]
            errors = self.instruments.errors(body).as_array()
            total += sum(abs(e - g) for e, g in zip(errors, goal))
            ticks += 1
            if body.position.z <= self.world.terrain_height(
                    body.position.x, body.position.y) + 1.0:
                break
        policy.epsilon = saved
        return total / max(ticks, 1)

    def rehearse(self, policy, *, epsilon: float = 0.05) -> dict:
        """Instructor shows, the student repeats.

        For each lesson the instructor's flown panel (VSI, bank, ASI at each
        decision tick) becomes a moving goal. The student then flies the
        maneuver itself — its own servo tensor choosing, the stick spring
        active, small exploration — and earns dense reward for matching the
        demo's panel. Watch-and-remember trained the associations; this
        trains the production, oscillation included: over-controlling away
        from the demo's state pays negative, matching it pays positive.
        """
        rng = np.random.default_rng(0)   # the classroom's own dice
        saved = policy.epsilon
        policy.epsilon = epsilon
        counts = {}
        for name, lesson in self.lessons.items():
            # Only maneuvers that begin from trim are repeatable: the
            # student must start where the demo started. Recovery demos
            # begin in a stall/spin/spiral — matching those would teach
            # entering them. Their reflexes are already stamped.
            if lesson.start.get("kind"):
                continue
            demo = self._demo_panel(lesson)
            demo = demo[int(lesson.imprint_after
                            / (DECISION_EVERY * self.dt)):] or demo
            body = lesson._spawn()
            servos = SurfacesState()
            commands = TRIM_COMMAND.copy()
            recent: list = []
            pending = 0.0
            previous = None
            step = 0
            decisions = 0
            for tick in range(int(lesson.duration / self.dt)):
                if tick % DECISION_EVERY == 0:
                    if recent:
                        policy.reinforce(recent, pending, time_epoch=tick)
                    pending = 0.0
                    previous = None
                    state_key = self.instruments.state_key(body, commands[3])
                    actions = policy.choose(state_key, rng)
                    commands = policy.command_from(commands, actions)
                    recent.append((state_key, actions))
                    if len(recent) > 10:
                        recent.pop(0)
                    decisions += 1
                servos.command(commands, self.dt)
                body.step(servos, self.dt)
                commands = spring_toward_neutral(commands, self.dt)
                goal = demo[min(tick // DECISION_EVERY, len(demo) - 1)]
                reward, previous = goal_reward(
                    self.instruments.errors(body).as_array(), goal,
                    previous, scale=5.0)
                pending += reward
                if body.position.z <= self.world.terrain_height(
                        body.position.x, body.position.y) + 1.0:
                    policy.reinforce(recent, -2.0, time_epoch=tick)
                    break
            else:
                policy.reinforce(recent, pending + 0.5, time_epoch=10**6)
            counts[name] = decisions
        policy.epsilon = saved
        return counts

    def _demo_panel(self, lesson) -> list:
        """The instructor's panel readings at each decision tick."""
        body = lesson._spawn()
        servos = SurfacesState()
        panel = []
        for tick in range(int(lesson.duration / self.dt)):
            if tick % DECISION_EVERY == 0:
                errors = self.instruments.errors(body)
                panel.append((errors.x, errors.y, errors.z))
            targets = lesson.instructor_targets(body, self.instruments.errors(body))
            servos.command(targets, self.dt)
            body.step(servos, self.dt)
            if body.position.z <= self.world.terrain_height(
                    body.position.x, body.position.y) + 1.0:
                break
        return panel or [(0.0, 0.0, 0.0)]

    @staticmethod
    def _translate(targets: np.ndarray, servos: SurfacesState) -> np.ndarray:
        """Instructor deflections -> the student's push/ease/hold decision."""
        actions = np.full(4, HOLD, dtype=np.int64)
        changed = False
        for surface in range(4):
            delta = targets[surface] - servos.positions[surface]
            if delta > 0.04:
                actions[surface] = PUSH
                changed = True
            elif delta < -0.04:
                actions[surface] = EASE_OFF
                changed = True
        return actions
