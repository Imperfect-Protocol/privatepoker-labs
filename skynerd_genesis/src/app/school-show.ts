/**
 * THE FLIGHT SCHOOL SHOW — the classroom's ONE aircraft, whose levers
 * belong to whoever the CONTROLS DROPDOWN says (sticky per aircraft,
 * switchable any tick): the AI, the instructor, or a both-mode (the
 * AI suggests and learns while the instructor flies, or the AI flies
 * under the instructor's guardian). The instructor's ideal path flies
 * invisibly beside — the delta column's reference and the error
 * wave's teaching source. Attempts repeat endlessly: a flight ends,
 * the result holds a moment, and the next attempt begins.
 */
import {
  FlightSchool, SoloFlight, InstructorTwin, gradeName, DECISION_EVERY,
  type Lesson, type GoalReportRow, RNG0_STATE, RNG0_INC,
} from "../learning/school.js";
import { FlightInstruments } from "../learning/instruments.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import type { TensorStateView } from "../learning/servo-tensor.js";
import { AiPilot } from "../learning/pilot.js";
import { TRIM_SPEED, springStickTowardNeutral } from "../learning/flight.js";
import { NumpyPcg64 } from "../core/rng.js";
import { Aircraft } from "../physics/aircraft.js";
import { Scene, type SceneAircraft } from "./scene.js";
import { hudDelta, readoutOf, type HudDelta } from "./hud-delta.js";
import type { ControlMode } from "../learning/guardian.js";
import { controlModeLabel } from "../learning/guardian.js";

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
  spin_recovery_left: "spin recovery left",
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
  spin_recovery_left: "recover from the spin",
  spin_recovery_right: "recover from the right spin",
  comprehensive: "fly the whole flight: takeoff, climb, turns, steeps, stall, and both spins",
};

/** What the classroom is doing right now: flying the attempt, or
 * holding the finished result before the next one. */
export type ShowStage = "flight" | "hold";

const DT = 1.0 / 50.0;
/** Once a flight is done, the result shows this long before the next
 * attempt begins. */
const HOLD_S = 3.0;

export interface SchoolMoment {
  stage: ShowStage;
  narration: string;
  lessonName: string;
  lessonIndex: number;      // 1-based, within the curriculum
  lessonCount: number;      // the curriculum size
  attempt: number;          // which attempt of this maneuver (1 = first)
  seconds: number;          // seconds into the current attempt
  /** The higher-level goal being realized ("take off from the runway"). */
  goalName: string;
  /** The element being flown now — the AI's chosen one. */
  elementName: string;
  /** The attempt is over (the result shows until the next one). */
  finished: boolean;
  /** The attempt's goal score, once it finished (0-6). */
  score: number | null;
  /** The last attempt was the instructor's hands (shown, not graded). */
  scoreByInstructor: boolean;
  /** The machine is in the air right now. */
  flying: boolean;
  /** Whose hands the controls dropdown put on the levers. */
  controlsMode: ControlMode;
  /** The instructor's hand is on the levers right now (a takeover in
   *  the guarded mode included). */
  instructorAtControls: boolean;
}

/**
 * The flight school show: ONE aircraft, one controls dropdown choice.
 */
export class FlightSchoolShow {
  readonly lessonNames: readonly string[];
  /** The classroom's world: the ONE aircraft. */
  readonly scene = new Scene();
  /** The report card's live bookkeeping: the AI's own results (the
   *  instructor's attempts are shown but never graded). */
  readonly scores = new Map<string, number>();
  readonly goalReports = new Map<string, GoalReportRow[]>();
  readonly attempts = new Map<string, number>();
  private flight: SoloFlight | null = null;
  /** Ticks the finished result holds before the next attempt. */
  private hold = 0;
  private lesson: Lesson;
  private entry: SceneAircraft;
  private lastScore: number | null = null;
  private lastScoreByInstructor = false;
  /** The school's monotonic epoch — each attempt's teach pulses land
   * in the next epoch, keeping the memory time-ordered. */
  private demoEpoch = 0;
  /** The classroom's dice, shared by the attempts. */
  private readonly rng = new NumpyPcg64(RNG0_STATE, RNG0_INC);
  /** THE CONTROLS DROPDOWN's sticky choice — one aircraft, one mode,
   *  carried across attempts. The classroom's default is PAIR:
   *  INSTRUCTOR (both-suggest): the instructor's hands fly the
   *  machine while the AI suggests and learns from them every window;
   *  the person gives the aircraft wholly to the AI with the dropdown
   *  when they want the student's solo attempt. */
  controlsMode: ControlMode = "both-suggest";

  /** THE CLASSROOM'S STUDENT — the ONE encapsulated tensor brain, its
   *  memory transplanted from the boot graduate and carried attempt to
   *  attempt (each SoloFlight resets its runtime, never its memory). */
  readonly student: AiPilot;
  /** The transplanted brain's record (the boot graduate's snapshot),
   *  kept for re-seating a restarted classroom. */
  readonly policy: ServoPolicy;

  constructor(readonly school: FlightSchool, brain: ServoPolicy) {
    this.lessonNames = [...school.lessons.keys()];
    this.lesson = school.lessons.values().next().value!;
    this.policy = brain;
    this.student = new AiPilot({ seedRng: this.rng,
      decisionEvery: DECISION_EVERY, spring: springStickTowardNeutral });
    this.student.inherit(brain.snapshot());
    this.entry = this.scene.add({
      label: "Aircraft",
      aircraft: new Aircraft(this.lesson.spawn()),
      hand: { choices: null, fresh: false },
      trail: [], flying: false, dead: false,
    });
    this.startFlight();
  }

  static withFreshPolicy(): FlightSchoolShow {
    const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
    return new FlightSchoolShow(school, new ServoPolicy());
  }

  /** Begin the next attempt of the chosen maneuver, from its start
   * (the lesson's spawn: the runway for takeoff, level flight at
   * 3000 ft / 80 mph for everything else), in the controls mode the
   * dropdown last chose. */
  private startFlight(): void {
    const twin = new InstructorTwin(this.lesson);
    this.flight = new SoloFlight(this.school, this.lesson, this.student,
                                 this.rng, this.demoEpoch++,
                                 this.controlsMode, twin);
    this.entry = this.scene.replace(this.entry, {
      label: "Aircraft",
      aircraft: this.flight.aircraft,
      hand: { choices: null, fresh: false },
      trail: [], flying: true, dead: false,
    });
  }

  /**
   * Fly the chosen maneuver from its beginning: a fresh attempt, in
   * the controls mode the dropdown holds.
   */
  flyLesson(name: string): void {
    const lesson = this.school.lessons.get(name);
    if (lesson === undefined) {
      throw new Error(`unknown lesson: ${name}`);
    }
    this.flight?.release();
    this.flight = null;
    this.hold = 0;
    this.lesson = lesson;
    this.lastScore = null;
    this.lastScoreByInstructor = false;
    this.startFlight();
  }

  /** The next-maneuver button: fly the next maneuver from its
   *  beginning (the curriculum wraps at the end). */
  skipToNextManeuver(): void {
    const index = this.lessonNames.indexOf(this.lesson.name);
    this.flyLesson(this.lessonNames[(index + 1) % this.lessonNames.length]);
  }

  /** The scenario dropdown: the chosen maneuver, flown from the
   *  beginning. */
  switchToManeuver(name: string): void {
    this.flyLesson(name);
  }

  /** The controls dropdown: whose hands are on the ONE aircraft's
   * levers. Sticky — it survives every attempt and every switch. */
  setControlsMode(mode: ControlMode): void {
    this.controlsMode = mode;
    if (this.flight !== null) {
      this.flight.controlsMode = mode;
    }
  }

  /** The machine's scene entry (its aircraft + flight trail). */
  get aircraftEntry(): SceneAircraft {
    return this.entry;
  }

  /** THE AI STATE LOG's subject: the machine's tensor network view
   *  (null while the result holds between attempts). */
  get aircraftStateView(): TensorStateView | null {
    return this.flight === null ? null : this.flight.stateView();
  }

  /** The AI log's delta column: the machine against the instructor's
   *  ideal path beside it, in pilot's units (null between attempts). */
  get instructorDelta(): HudDelta | null {
    if (this.flight === null) {
      return null;
    }
    return hudDelta(readoutOf(this.entry.aircraft),
                    readoutOf(this.flight.twinAircraft));
  }

  /** The wiki tab's scenario description: the current lesson's title,
   *  its higher-level goal, its element chain, and its known goals. */
  get wiki(): {
    title: string; goal: string;
    elements: readonly string[]; goals: readonly string[];
  } {
    const spec = this.school.goalSpec(this.lesson);
    return {
      title: LESSON_PHRASES[this.lesson.name] ?? this.lesson.name,
      goal: GOAL_PHRASES[this.lesson.name] ?? this.lesson.name,
      elements: this.lesson.phases.map((phase) => phase.name),
      goals: spec.goals.map((goal) => goal.label),
    };
  }

  /** The entry to follow: the ONE aircraft, on the ground or in the
   *  air — it is all the classroom has. */
  get flyingEntry(): SceneAircraft {
    return this.entry;
  }

  get moment(): SchoolMoment {
    const phrase = LESSON_PHRASES[this.lesson.name] ?? this.lesson.name;
    const flying = this.flight !== null;
    const finished = this.hold > 0;
    const mode = this.flight?.controlsMode ?? this.controlsMode;
    const instructorHolds = this.flight?.instructorHolds ?? false;
    const element = this.flight !== null
      ? this.flight.chosenElement.name : "";
    const elementText = element === "" ? "" : ` \u2014 element: ${element}`;
    const whoText = instructorHolds
      ? "the instructor has the controls" : "the AI has the controls";
    let narration: string;
    if (finished) {
      const by = this.lastScoreByInstructor ? " (the instructor's hands)" : "";
      const scoreText = this.lastScore !== null
        ? ` \u2014 score ${this.lastScore.toFixed(2)}/6${by}` : "";
      narration = `${phrase} finished${scoreText}`;
    } else if (flying) {
      narration = `${phrase}${elementText} \u00b7 ${whoText}`;
    } else {
      narration = `${phrase} \u2014 ready`;
    }
    const lessonIndex = this.lessonNames.indexOf(this.lesson.name) + 1;
    return {
      stage: finished ? "hold" : "flight",
      narration,
      lessonName: this.lesson.name,
      lessonIndex: Math.max(1, lessonIndex),
      lessonCount: this.lessonNames.length,
      attempt: (this.attempts.get(this.lesson.name) ?? 0) + 1,
      seconds: this.flight?.seconds ?? 0.0,
      goalName: GOAL_PHRASES[this.lesson.name] ?? this.lesson.name,
      elementName: element,
      finished,
      score: this.lastScore,
      scoreByInstructor: this.lastScoreByInstructor,
      flying,
      controlsMode: mode,
      instructorAtControls: instructorHolds,
    };
  }

  /**
   * Advance by one 50 Hz tick — real time, every flight rendered as it
   * flies, no shortcuts. The attempt flies until the machine lands or
   * its plan completes; the result holds a moment; the next attempt
   * begins (endlessly — the classroom never stops).
   */
  step(): void {
    if (this.flight !== null) {
      const flight = this.flight;
      if (!flight.step()) {
        this.entry.flying = false;
        this.entry.dead = this.school.grounded(flight.aircraft.body);
        const byInstructor = flight.controlsMode === "instructor";
        const score = flight.goalScore;
        const name = this.lesson.name;
        this.attempts.set(name, (this.attempts.get(name) ?? 0) + 1);
        this.lastScore = score;
        this.lastScoreByInstructor = byInstructor;
        // The report card grades the AI's own hands only.
        if (!byInstructor) {
          this.scores.set(name, score);
          this.goalReports.set(name, flight.goalVerdicts());
        }
        this.flight = null;
        this.hold = Math.round(HOLD_S / DT);
      } else {
        this.refreshEntry(this.entry, flight);
      }
      return;
    }
    if (this.hold > 0) {
      this.hold -= 1;
      if (this.hold === 0) {
        this.skipToNextManeuver();   // the endless classroom's next turn
      }
    }
  }

  /** Mirror the flying attempt into its scene entry: the hand's
   *  decisions and the path it is carving. */
  private refreshEntry(entry: SceneAircraft, flight: SoloFlight): void {
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
   * and each goal earns its verdict (✓/±/✗) once the AI has flown the
   * maneuver. Rows cover the whole curriculum; when a golden
   * reference is supplied (the boot brain's own scores) each flight
   * is compared against it.
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
