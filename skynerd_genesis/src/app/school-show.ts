/**
 * The Flight School show: the classroom loop run live, one tick at a time.
 *
 * This is view wiring, not protocol: a SchoolSession (the teach() state
 * machine) hands out flights and the show steps them at the sim's fixed
 * 50 Hz. Narration follows the classroom loop exactly as it runs:
 * "Instructor demonstrates turn left" → "Student repeats turn left" →
 * the goal grade — pass and the next maneuver's demonstration begins,
 * fail and the instructor demonstrates the same maneuver again.
 *
 * EVERYTHING RUNS IN REAL TIME: one 50 Hz tick per frame tick, every
 * demonstration and every repeat rendered as it flies. The classroom
 * is endless (loop: true) — the curriculum wraps lap after lap and the
 * show-and-repeat loop never stops; skipToNextManeuver() is the
 * next-maneuver button, switchToManeuver() the scenario dropdown.
 *
 * THE SCENE holds the classroom's two aircraft — the instructor's and
 * the student's, the same ONE Aircraft class both fly. The panel shows
 * whichever is selected; the flying one is selected automatically. The
 * instructor's trail stays on its entry — it is the ghost path behind
 * the student's repeat.
 */
import {
  FlightSchool, SchoolSession, SchoolFlight, gradeName, currentElementName,
} from "../learning/school.js";
import { FlightInstruments } from "../learning/instruments.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import { TRIM_SPEED } from "../learning/flight.js";
import { Aircraft } from "../physics/aircraft.js";
import { Scene, type SceneAircraft } from "./scene.js";

const TRAIL_CAP = 1200;

/** The narration phrase for each lesson ("Instructor demonstrates …"). */
export const LESSON_PHRASES: Record<string, string> = {
  takeoff: "takeoff",
  level_cruise: "level cruise",
  turn_left_180: "turn left",
  turn_right_180: "turn right",
  steep_turn_left_180: "steep turn left",
  steep_turn_right_180: "steep turn right",
  stall_recovery: "stall recovery",
  spin_recovery: "spin recovery",
};

/**
 * The HIGHER-LEVEL GOAL each maneuver realizes — what the student knows
 * they are trying to achieve while they choose the element.
 */
export const GOAL_PHRASES: Record<string, string> = {
  takeoff: "take off from the runway",
  level_cruise: "hold level cruise",
  turn_left_180: "turn left 180\u00b0",
  turn_right_180: "turn right 180\u00b0",
  steep_turn_left_180: "steep turn left 180\u00b0",
  steep_turn_right_180: "steep turn right 180\u00b0",
  stall_recovery: "recover from the stall",
  spin_recovery: "recover from the spin",
};

export type ShowStage = "demo" | "repeat";

export interface SchoolMoment {
  stage: ShowStage;
  narration: string;
  lessonName: string;
  lessonIndex: number;      // 1-based, within the curriculum
  lessonCount: number;      // the curriculum size
  attempt: number;          // which demonstration of this maneuver (1 = first)
  seconds: number;          // seconds into the current flight
  /** The higher-level goal being realized ("take off from the runway"). */
  goalName: string;
  /** The element being flown now — the instructor's, or the student's
   *  chosen one ("roll", "climb", "level off"…). */
  elementName: string;
}

/**
 * Drives one classroom loop run: demonstrations, the student's repeats,
 * and the re-demonstrations a failed repeat earns — the two aircraft
 * living in the scene the whole time.
 */
export class FlightSchoolShow {
  readonly session: SchoolSession;
  readonly lessonNames: readonly string[];
  /** The classroom's world: the instructor's and the student's aircraft. */
  readonly scene = new Scene();
  private active: SchoolFlight | null = null;
  private stage: ShowStage = "demo";
  private lastAttempt = 1;
  private instructor: SceneAircraft;
  private student: SceneAircraft;
  /** The lesson the student's scene entry currently mirrors. */
  private studentLesson = "";

  constructor(readonly school: FlightSchool, policy: ServoPolicy) {
    // Endless: the classroom never closes — the curriculum wraps lap
    // after lap and the show-and-repeat loop keeps running forever.
    this.session = new SchoolSession(school, policy, { loop: true });
    this.lessonNames = [...school.lessons.keys()];
    this.instructor = this.scene.add({
      label: "Instructor's Aircraft",
      aircraft: new Aircraft(school.lessons.values().next().value!.spawn()),
      hand: { choices: null, fresh: false },
      trail: [], flying: false, dead: false,
    });
    this.student = this.scene.add({
      label: "Student's Aircraft",
      aircraft: new Aircraft(school.lessons.values().next().value!.spawn()),
      hand: { choices: null, fresh: false },
      trail: [], flying: false, dead: false,
    });
    this.beginFlight();
  }

  static withFreshPolicy(): FlightSchoolShow {
    const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
    return new FlightSchoolShow(school, new ServoPolicy());
  }

  get moment(): SchoolMoment {
    const active = this.active;
    const lesson = active?.lesson;
    const phrase = lesson ? LESSON_PHRASES[lesson.name] ?? lesson.name : "";
    const element = active === null ? "" : currentElementName(active);
    const elementText = element === "" ? "" : ` — element: ${element}`;
    let narration: string;
    if (this.stage === "demo") {
      narration = (active?.attempt ?? 1) > 1
        ? `Instructor demonstrates ${phrase} again${elementText}`
        : `Instructor demonstrates ${phrase}${elementText}`;
    } else {
      narration = `Student repeats ${phrase}${elementText}`;
    }
    const lessonIndex = lesson === undefined
      ? this.lessonNames.length
      : this.lessonNames.indexOf(lesson.name) + 1;
    return {
      stage: this.stage,
      narration,
      lessonName: lesson?.name ?? "",
      lessonIndex: Math.max(1, lessonIndex),
      lessonCount: this.lessonNames.length,
      attempt: active?.attempt ?? this.lastAttempt,
      seconds: active === null ? 0.0 : active.flight.seconds,
      goalName: lesson === undefined ? ""
        : GOAL_PHRASES[lesson.name] ?? lesson.name,
      elementName: element,
    };
  }

  /**
   * Advance the classroom by one 50 Hz tick — real time, every flight
   * rendered as it flies, no shortcuts.
   */
  step(): void {
    if (this.active === null) {
      return;
    }
    const flight = this.active;
    const more = flight.flight.step();
    this.refreshEntry(flight);
    if (!more) {
      this.finishFlight(flight);
    }
  }

  /** The next-maneuver button: leave the current flight where it stands
   *  (whatever the demonstration already stamped stays learned) and move
   *  the classroom on to the next maneuver right now. */
  skipToNextManeuver(): void {
    if (this.active === null) {
      return;
    }
    this.session.skipLesson();
    this.beginFlight();
  }

  /** The scenario dropdown: SAME student, SAME memory — every tensor and
   *  weight carried over untouched — placed with the instructor into the
   *  chosen scenario. The current flight is dropped where it stands and
   *  the chosen maneuver's demonstration begins. */
  switchToManeuver(name: string): void {
    this.session.switchLesson(name);
    this.beginFlight();
  }

  /** The instructor's scene entry (its aircraft + demonstration trail). */
  get instructorEntry(): SceneAircraft {
    return this.instructor;
  }

  /** The student's scene entry (its aircraft + repeat trail). */
  get studentEntry(): SceneAircraft {
    return this.student;
  }

  /** The entry whose flight is in the air right now (the instructor's
   *  during a demonstration, the student's during a repeat). */
  get flyingEntry(): SceneAircraft {
    return this.stage === "demo" ? this.instructor : this.student;
  }

  /** Mirror the flying flight into its scene entry: the hand's decisions
   *  and the path it is carving. */
  private refreshEntry(flight: SchoolFlight): void {
    const entry = flight.stage === "demo" ? this.instructor : this.student;
    entry.hand = flight.flight.handInputs;
    const position = flight.flight.aircraft.body.position;
    entry.trail.push({ x: position.x, y: position.y, z: position.z });
    if (entry.trail.length > TRAIL_CAP) {
      entry.trail.shift();
    }
  }

  private beginFlight(): void {
    const next = this.session.nextFlight();
    if (next === null) {
      // Unreachable in the endless classroom (loop: true) — kept for the
      // type's honesty: the show always has a flight in the air.
      this.active = null;
      return;
    }
    this.active = next;
    this.stage = next.stage === "demo" ? "demo" : "repeat";
    this.lastAttempt = next.attempt;
    if (next.stage === "demo") {
      // The instructor's next machine; the student's entry will preview
      // the start it will inherit for this lesson.
      this.instructor = this.scene.replace(this.instructor, {
        label: "Instructor's Aircraft",
        aircraft: next.flight.aircraft,
        hand: { choices: null, fresh: false },
        trail: [], flying: true, dead: false,
      });
      if (this.studentLesson !== next.lesson.name) {
        this.studentLesson = next.lesson.name;
        this.student = this.scene.replace(this.student, {
          label: "Student's Aircraft",
          aircraft: new Aircraft(this.studentStartBody(next.lesson.name)),
          hand: { choices: null, fresh: false },
          trail: [], flying: false, dead: false,
        });
      }
    } else {
      this.student = this.scene.replace(this.student, {
        label: "Student's Aircraft",
        aircraft: next.flight.aircraft,
        hand: { choices: null, fresh: false },
        trail: [], flying: true, dead: false,
      });
    }
  }

  /** The machine the student will fly this lesson from: the instructor's
   *  handoff state (a fresh copy of the same snapshot ExamFlight takes),
   *  or the lesson's spawn. */
  private studentStartBody(lessonName: string) {
    const lesson = this.school.lessons.get(lessonName)!;
    const start = this.school.preparedStart(lesson);
    return start === null ? lesson.spawn() : start.body;
  }

  private finishFlight(flight: SchoolFlight): void {
    const entry = flight.stage === "demo" ? this.instructor : this.student;
    entry.flying = false;
    entry.dead = flight.flight.finished && this.school.grounded(flight.flight.aircraft.body);
    this.beginFlight();
  }

  /**
   * The report card rows: attempts, score, grade, and EACH MANEUVER'S
   * KNOWN GOALS — the goal list is printed up front (before the repeat)
   * and each goal earns its verdict (✓/±/✗) once the student has
   * repeated the maneuver. Rows cover the whole curriculum; when a
   * golden reference is supplied its score is compared too.
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
    // Every maneuver of the curriculum gets a row; golden first when the
    // frozen reference is loaded, then any not covered by it.
    const names = goldenLessons.length > 0
      ? goldenLessons.map((g) => g.lesson)
      : [...this.school.lessons.keys()];
    for (const name of names) {
      const lesson = this.school.lessons.get(name);
      if (lesson === undefined) {
        continue;
      }
      const golden = goldens.get(name);
      const report = this.session.goalReports.get(name);
      const score = this.session.scores.get(name) ?? null;
      const attempts = this.session.attempts.get(name) ?? 0;
      const goals = (report ?? this.school.goalSpec(lesson).goals.map((goal) => ({
        label: goal.label, fraction: -1.0,
      }))).map((goal) =>
        goal.fraction < 0.0
          ? `– ${goal.label}`
          : `${goal.fraction >= 1.0 ? "✓" : goal.fraction >= 0.5 ? "±" : "✗"} ${goal.label}`)
        .join(" · ");
      const exact = score !== null && golden !== undefined
        ? Math.abs(score - golden.score) < 5e-4 && attempts === golden.attempts
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
