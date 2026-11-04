/**
 * The Flight School show: the instructor demonstrates a maneuver and
 * the STUDENT LAUNCHES 10 SECONDS LATER — both fly the SAME maneuver
 * at once, the student chasing the instructor through the sky. When
 * both flights are done the result shows for a moment, then the next
 * maneuver begins (the curriculum wraps — the classroom never closes).
 *
 * While the instructor flies, the student's memory still updates (the
 * demonstration stamps into its tensors — the boot-trained brain keeps
 * learning); the student's own flight is graded against the maneuver's
 * known goals.
 *
 * EVERYTHING RUNS IN REAL TIME: one 50 Hz tick per frame tick, every
 * flight rendered as it flies. There is NO timeout: a flight lives
 * until the maneuver's element chain is complete or the wheels touch
 * the ground.
 *
 * THE SCENE holds the classroom's two aircraft — the instructor's and
 * the student's, the same ONE Aircraft class both fly. The panel shows
 * whichever is selected; the follow camera rides the student once it
 * is in the air (the one learning).
 */
import {
  FlightSchool, DemoFlight, ExamFlight, gradeName,
  type Lesson, type GoalReportRow, RNG0_STATE, RNG0_INC,
} from "../learning/school.js";
import { FlightInstruments } from "../learning/instruments.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import { TRIM_SPEED } from "../learning/flight.js";
import { NumpyPcg64 } from "../core/rng.js";
import { Aircraft } from "../physics/aircraft.js";
import { Scene, type SceneAircraft } from "./scene.js";

const TRAIL_CAP = 1200;

/** The narration phrase for each lesson ("Instructor demonstrates …"). */
export const LESSON_PHRASES: Record<string, string> = {
  takeoff: "takeoff",
  level_cruise: "level cruise",
  climb_to_5000: "climb to 5000 ft",
  descend_to_1000: "descend to 1000 ft",
  turn_left_180: "turn left",
  turn_right_180: "turn right",
  climbing_turn_left_180: "climbing turn left",
  climbing_turn_right_180: "climbing turn right",
  descending_turn_left_180: "descending turn left",
  descending_turn_right_180: "descending turn right",
  steep_turn_left_180: "steep turn left",
  steep_turn_right_180: "steep turn right",
  steep_climbing_turn_left_180: "steep climbing turn left",
  steep_climbing_turn_right_180: "steep climbing turn right",
  steep_descending_turn_left_180: "steep descending turn left",
  steep_descending_turn_right_180: "steep descending turn right",
  stall_recovery: "stall recovery",
  spin_recovery: "spin recovery",
  spin_recovery_right: "spin recovery right",
  comprehensive: "the comprehensive flight",
};

/**
 * The HIGHER-LEVEL GOAL each maneuver realizes — what the student knows
 * they are trying to achieve while they choose the element.
 */
export const GOAL_PHRASES: Record<string, string> = {
  takeoff: "take off from the runway",
  level_cruise: "hold level cruise",
  climb_to_5000: "climb to 5000 ft",
  descend_to_1000: "descend to 1000 ft",
  turn_left_180: "turn left 180\u00b0",
  turn_right_180: "turn right 180\u00b0",
  climbing_turn_left_180: "turn left 180\u00b0 while climbing",
  climbing_turn_right_180: "turn right 180\u00b0 while climbing",
  descending_turn_left_180: "turn left 180\u00b0 while descending",
  descending_turn_right_180: "turn right 180\u00b0 while descending",
  steep_turn_left_180: "steep turn left 180\u00b0",
  steep_turn_right_180: "steep turn right 180\u00b0",
  steep_climbing_turn_left_180: "steep turn left 180\u00b0 while climbing",
  steep_climbing_turn_right_180: "steep turn right 180\u00b0 while climbing",
  steep_descending_turn_left_180: "steep turn left 180\u00b0 while descending",
  steep_descending_turn_right_180: "steep turn right 180\u00b0 while descending",
  stall_recovery: "recover from the stall",
  spin_recovery: "recover from the spin",
  spin_recovery_right: "recover from the right spin",
  comprehensive: "fly the whole flight: takeoff, climb, turns, steeps, stall, and both spins",
};

/** Who is in the air: the instructor alone, or both pilots. */
export type ShowStage = "demo" | "repeat";

/** The student launches this many seconds after the instructor — both
 *  fly the SAME maneuver at once, the student chasing the instructor. */
export const STUDENT_DELAY_S = 10.0;
const DT = 1.0 / 50.0;
/** Once both flights are done, the result shows this long before the
 *  next maneuver begins. */
const HOLD_S = 3.0;

export interface SchoolMoment {
  stage: ShowStage;
  narration: string;
  lessonName: string;
  lessonIndex: number;      // 1-based, within the curriculum
  lessonCount: number;      // the curriculum size
  attempt: number;          // which student flight of this maneuver (1 = first)
  seconds: number;          // seconds into the student's flight (the demo's before)
  /** The higher-level goal being realized ("take off from the runway"). */
  goalName: string;
  /** The element being flown now — the student's chosen one once it is
   *  in the air, the instructor's before. */
  elementName: string;
  /** Both flights are over (the result shows until the next maneuver). */
  finished: boolean;
  /** The student's goal score, once its flight finished (0-6). */
  score: number | null;
  /** Both pilots are in the air right now. */
  bothFlying: boolean;
  /** Seconds until the student launches (0 once it is in the air). */
  countdown: number;
}

/**
 * The flight school show: the instructor's demonstration and — 10 s
 * behind it — the student's own flight of the same maneuver, both in
 * the scene the whole time.
 */
export class FlightSchoolShow {
  readonly lessonNames: readonly string[];
  /** The classroom's world: the instructor's and the student's aircraft. */
  readonly scene = new Scene();
  /** The report card's live bookkeeping: the student's own results. */
  readonly scores = new Map<string, number>();
  readonly goalReports = new Map<string, GoalReportRow[]>();
  readonly attempts = new Map<string, number>();
  private demo: DemoFlight | null = null;
  private studentFlight: ExamFlight | null = null;
  /** Ticks until the student launches. */
  private countdown = 0;
  /** Ticks both flights have been done (the hold before the next). */
  private hold = 0;
  private lesson: Lesson;
  private instructor: SceneAircraft;
  private student: SceneAircraft;
  private lastScore: number | null = null;
  /** The school's monotonic demo clock — each demonstration lands in the
   *  next epoch, keeping the student's memory time-ordered. */
  private demoEpoch = 0;
  /** The classroom's dice, shared by the student's flights. */
  private readonly rng = new NumpyPcg64(RNG0_STATE, RNG0_INC);

  constructor(readonly school: FlightSchool, readonly policy: ServoPolicy) {
    this.lessonNames = [...school.lessons.keys()];
    this.lesson = school.lessons.values().next().value!;
    this.instructor = this.scene.add({
      label: "Instructor's Aircraft",
      aircraft: new Aircraft(this.lesson.spawn()),
      hand: { choices: null, fresh: false },
      trail: [], flying: false, dead: false,
    });
    this.student = this.scene.add({
      label: "Student's Aircraft",
      aircraft: new Aircraft(this.lesson.spawn()),
      hand: { choices: null, fresh: false },
      trail: [], flying: false, dead: false,
    });
    this.flyLesson(this.lesson.name);
  }

  static withFreshPolicy(): FlightSchoolShow {
    const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
    return new FlightSchoolShow(school, new ServoPolicy());
  }

  /**
   * Fly the chosen maneuver from its beginning: the instructor
   * demonstrates, and 10 seconds later the student launches after it —
   * the SAME start (the lesson's spawn: the runway for takeoff, level
   * flight at 3000 ft / 80 mph for everything else).
   */
  flyLesson(name: string): void {
    const lesson = this.school.lessons.get(name);
    if (lesson === undefined) {
      throw new Error(`unknown lesson: ${name}`);
    }
    this.studentFlight?.release();
    this.studentFlight = null;
    this.demo = null;
    this.hold = 0;
    this.lesson = lesson;
    this.lastScore = null;
    this.demo = new DemoFlight(this.school, lesson, this.policy,
                               this.demoEpoch++);
    this.instructor = this.scene.replace(this.instructor, {
      label: "Instructor's Aircraft",
      aircraft: this.demo.aircraft,
      hand: { choices: null, fresh: false },
      trail: [], flying: true, dead: false,
    });
    this.student = this.scene.replace(this.student, {
      label: "Student's Aircraft",
      aircraft: new Aircraft(lesson.spawn()),
      hand: { choices: null, fresh: false },
      trail: [], flying: false, dead: false,
    });
    this.countdown = Math.round(STUDENT_DELAY_S / DT);
  }

  /** The next-maneuver button: fly the next maneuver from its beginning
   *  (the curriculum wraps at the end). */
  skipToNextManeuver(): void {
    const index = this.lessonNames.indexOf(this.lesson.name);
    this.flyLesson(this.lessonNames[(index + 1) % this.lessonNames.length]);
  }

  /** The scenario dropdown: the chosen maneuver, flown from the
   *  beginning. */
  switchToManeuver(name: string): void {
    this.flyLesson(name);
  }

  /** The instructor's scene entry (its aircraft + demonstration trail). */
  get instructorEntry(): SceneAircraft {
    return this.instructor;
  }

  /** The student's scene entry (its aircraft + flight trail). */
  get studentEntry(): SceneAircraft {
    return this.student;
  }

  /** The entry to follow: the student once it is in the air (the one
   *  learning), the instructor before it launches. */
  get flyingEntry(): SceneAircraft {
    return this.studentFlight !== null ? this.student : this.instructor;
  }

  get moment(): SchoolMoment {
    const phrase = LESSON_PHRASES[this.lesson.name] ?? this.lesson.name;
    const demoFlying = this.demo !== null;
    const studentFlying = this.studentFlight !== null;
    const finished = this.hold > 0;
    const element = this.studentFlight !== null
      ? this.studentFlight.chosenElement.name
      : this.demo !== null ? this.demo.cursor.current.name : "";
    const elementText = element === "" ? "" : ` \u2014 element: ${element}`;
    let narration: string;
    if (finished) {
      const scoreText = this.lastScore !== null
        ? ` \u2014 score ${this.lastScore.toFixed(2)}/6` : "";
      narration = `${phrase} finished${scoreText}`;
    } else if (studentFlying && demoFlying) {
      narration = `Instructor demonstrates ${phrase} \u00b7 ` +
        `Student flies it, 10 s behind${elementText}`;
    } else if (studentFlying) {
      narration = `Student flies ${phrase}${elementText}`;
    } else if (demoFlying) {
      narration = `Instructor demonstrates ${phrase}${elementText} \u00b7 ` +
        `student launches in ${(this.countdown * DT).toFixed(0)} s`;
    } else {
      narration = `Instructor demonstrates ${phrase}`;
    }
    const lessonIndex = this.lessonNames.indexOf(this.lesson.name) + 1;
    return {
      stage: studentFlying ? "repeat" : "demo",
      narration,
      lessonName: this.lesson.name,
      lessonIndex: Math.max(1, lessonIndex),
      lessonCount: this.lessonNames.length,
      attempt: (this.attempts.get(this.lesson.name) ?? 0) + 1,
      seconds: this.studentFlight !== null
        ? this.studentFlight.seconds
        : this.demo?.seconds ?? 0.0,
      goalName: GOAL_PHRASES[this.lesson.name] ?? this.lesson.name,
      elementName: element,
      finished,
      score: this.lastScore,
      bothFlying: studentFlying && demoFlying,
      countdown: this.studentFlight !== null ? 0.0 : this.countdown * DT,
    };
  }

  /**
   * Advance by one 50 Hz tick — real time, every flight rendered as it
   * flies, no shortcuts. The instructor steps; the countdown runs; the
   * student steps once it has launched; when both are done the result
   * holds a moment and the next maneuver begins.
   */
  step(): void {
    if (this.demo !== null) {
      if (!this.demo.step()) {
        this.instructor.flying = false;
        this.instructor.dead = this.school.grounded(this.demo.aircraft.body);
        this.demo = null;
      } else {
        this.refreshEntry(this.instructor, this.demo);
      }
    }
    if (this.studentFlight === null && this.countdown > 0) {
      this.countdown -= 1;
      if (this.countdown <= 0) {
        this.studentFlight = new ExamFlight(this.school, this.lesson,
                                            this.policy, this.rng);
        this.student = this.scene.replace(this.student, {
          label: "Student's Aircraft",
          aircraft: this.studentFlight.aircraft,
          hand: { choices: null, fresh: false },
          trail: [], flying: true, dead: false,
        });
      }
    }
    if (this.studentFlight !== null) {
      const flight = this.studentFlight;
      if (!flight.step()) {
        this.student.flying = false;
        this.student.dead = this.school.grounded(flight.aircraft.body);
        const score = flight.goalScore;
        const name = this.lesson.name;
        this.scores.set(name, score);
        this.goalReports.set(name, flight.goalVerdicts());
        this.attempts.set(name, (this.attempts.get(name) ?? 0) + 1);
        this.lastScore = score;
        this.studentFlight = null;
      } else {
        this.refreshEntry(this.student, flight);
      }
    }
    if (this.demo === null && this.studentFlight === null
        && this.countdown <= 0) {
      if (this.hold === 0) {
        this.hold = Math.round(HOLD_S / DT);   // both done: show the result
      } else {
        this.hold -= 1;
        if (this.hold === 0) {
          this.skipToNextManeuver();           // then the next maneuver
        }
      }
    }
  }

  /** Mirror the flying flight into its scene entry: the hand's decisions
   *  and the path it is carving. */
  private refreshEntry(entry: SceneAircraft,
                       flight: DemoFlight | ExamFlight): void {
    entry.hand = flight.handInputs;
    const position = flight.aircraft.body.position;
    entry.trail.push({ x: position.x, y: position.y, z: position.z });
    if (entry.trail.length > TRAIL_CAP) {
      entry.trail.shift();
    }
  }

  /**
   * The report card rows: attempts, score, grade, and EACH MANEUVER'S
   * KNOWN GOALS — the goal list is printed up front (before the flight)
   * and each goal earns its verdict (✓/±/✗) once the student has flown
   * the maneuver. Rows cover the whole curriculum; when a golden
   * reference is supplied (the boot brain's own scores) each student
   * flight is compared against it.
   */
  reportRows(goldenLessons: readonly {
    lesson: string; attempts: number; score: number;
    goals: readonly { label: string; fraction: number }[];
  }[]): Array<{
    lesson: string; attempts: number; score: number | null;
    grade: string | null; goals: string;
    goldenScore: number | null; exact: boolean | null;
  }> {
    const goldens = new Map(goldenLessons.map((g) => [g.lesson, g]));
    const rows: ReturnType<FlightSchoolShow["reportRows"]> = [];
    const names = goldenLessons.length > 0
      ? goldenLessons.map((g) => g.lesson)
      : [...this.school.lessons.keys()];
    for (const name of names) {
      const lesson = this.school.lessons.get(name);
      if (lesson === undefined) {
        continue;
      }
      const golden = goldens.get(name);
      const report = this.goalReports.get(name);
      const score = this.scores.get(name) ?? null;
      const attempts = this.attempts.get(name) ?? 0;
      const goals = (report ?? this.school.goalSpec(lesson).goals.map((goal) => ({
        label: goal.label, fraction: -1.0,
      }))).map((goal) =>
        goal.fraction < 0.0
          ? `\u2013 ${goal.label}`
          : `${goal.fraction >= 1.0 ? "\u2713" : goal.fraction >= 0.5 ? "\u00b1" : "\u2717"} ${goal.label}`)
        .join(" \u00b7 ");
      const exact = score !== null && golden !== undefined
        ? Math.abs(score - golden.score) < 5e-4
        : null;
      rows.push({
        lesson: name,
        attempts,
        score,
        grade: score === null ? null : gradeName(score),
        goals,
        goldenScore: golden?.score ?? null,
        exact,
      });
    }
    return rows;
  }
}
