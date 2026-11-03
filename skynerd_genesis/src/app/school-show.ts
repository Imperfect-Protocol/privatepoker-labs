/**
 * The Flight School show: the classroom run live, one tick at a time.
 *
 * This is view wiring, not protocol: a SchoolSession (the teach() state
 * machine, held bit-exact by test/school.test.ts) hands out flights and the
 * show steps them at the sim's fixed 50 Hz, sampling the aircraft each tick
 * for the canvas and the cockpit. Narration follows the phase exactly as
 * the classroom runs it: "Instructor demonstrates turn left" during the
 * demonstration pass, "Student repeats" during the exam and the drills,
 * with the takeover moments called out while the instructor has the
 * controls.
 *
 * Drill rounds beyond the visible cap fast-forward headlessly (same code,
 * same ticks — just not rendered one by one) so the report card that ends
 * the show is the golden-exact teach() result.
 */
import {
  FlightSchool, SchoolSession, SchoolFlight, gradeName,
} from "../learning/school.js";
import { FlightInstruments } from "../learning/instruments.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import { TRIM_SPEED } from "../learning/flight.js";
import type { DroneView } from "./cockpit.js";

/** How many drill rounds are rendered tick-by-tick before fast-forward. */
export const VISIBLE_ROUNDS = 2;
/** Ticks per frame chewed while fast-forwarding the remaining rounds. */
const BURST_TICKS = 3000;

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

export interface SchoolSample {
  x: number;
  y: number;
  z: number;
}

export type ShowStage = "demo" | "exam" | "repeat" | "drilling" | "done";

export interface SchoolMoment {
  stage: ShowStage;
  narration: string;
  lessonName: string;
  lessonIndex: number;      // 1-based, within the curriculum
  lessonCount: number;     // the curriculum size
  round: number;            // drill round (0 = demo pass / first exams)
  takeover: boolean;
  seconds: number;          // seconds into the current flight
}

/**
 * Drives one classroom run: demonstrations, exams, and the corrective
 * drills, with per-tick samples for rendering.
 */
export class FlightSchoolShow {
  readonly session: SchoolSession;
  readonly lessonNames: readonly string[];
  private active: SchoolFlight | null = null;
  private stage: ShowStage = "demo";
  private lastRound = 0;
  /** Path of the current flight, sampled every tick (for the trail). */
  trail: SchoolSample[] = [];
  /** The lesson's demonstration path, kept to ghost during the repeat. */
  private ghostByLesson = new Map<string, SchoolSample[]>();
  ghost: SchoolSample[] | null = null;
  /** The last flown sample, for the frame after a flight ends. */
  private lastView: DroneView | null = null;
  private fastForwarding = false;

  constructor(readonly school: FlightSchool, policy: ServoPolicy) {
    this.session = new SchoolSession(school, policy, {});
    this.lessonNames = [...school.lessons.keys()];
    this.beginFlight();
  }

  static withFreshPolicy(): FlightSchoolShow {
    const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
    return new FlightSchoolShow(school, new ServoPolicy());
  }

  get done(): boolean {
    return this.session.done;
  }

  get graduateKeys(): number | null {
    return this.session.graduate === null
      ? null : this.session.graduate.tensor.storedCombinations;
  }

  get moment(): SchoolMoment {
    const active = this.active;
    const lesson = active?.lesson;
    const phrase = lesson ? LESSON_PHRASES[lesson.name] ?? lesson.name : "";
    let narration: string;
    if (this.done) {
      const failed = this.session.failed;
      narration = failed.length === 0
        ? "The class graduates — every lesson at or above the pass mark"
        : `The classroom closes — ${failed.length} lesson(s) still below the pass mark after ${this.session.rounds} rounds`;
    } else if (this.fastForwarding) {
      narration = "Instructor drills the failing lessons (fast-forward)…";
    } else if (this.stage === "demo") {
      narration = `Instructor demonstrates ${phrase}`;
    } else if (this.stage === "repeat" && active !== null
               && (active.flight as { takeoverActive?: boolean }).takeoverActive) {
      narration = "Instructor takes over — demonstrating the correction";
    } else {
      narration = "Student repeats";
    }
    const lessonIndex = lesson === undefined
      ? this.lessonNames.length
      : this.lessonNames.indexOf(lesson.name) + 1;
    return {
      stage: this.fastForwarding ? "drilling" : this.stage,
      narration,
      lessonName: lesson?.name ?? "",
      lessonIndex: Math.max(1, lessonIndex),
      lessonCount: this.lessonNames.length,
      round: active?.round ?? this.lastRound,
      takeover: this.stage === "repeat" && active !== null
        && !!(active.flight as { takeoverActive?: boolean }).takeoverActive,
      seconds: active === null ? 0.0 : active.flight.seconds,
    };
  }

  /** The current aircraft as a cockpit view (null before the first tick). */
  get view(): DroneView | null {
    return this.lastView;
  }

  /**
   * Advance the classroom by one 50 Hz tick. While fast-forwarding the
   * late drill rounds, one tick chews a burst of headless ticks.
   */
  step(): void {
    if (this.done || this.active === null) {
      return;
    }
    if (this.fastForwarding) {
      this.stepBurst();
      return;
    }
    const flight = this.active;
    const more = flight.flight.step();
    this.sample(flight);
    if (!more) {
      this.finishFlight(flight);
    }
  }

  /** When the visible rounds are over, finish the remaining drills headless. */
  private stepBurst(): void {
    for (let i = 0; i < BURST_TICKS && !this.done; i++) {
      const active = this.active;
      if (active === null) {
        break;
      }
      if (!active.flight.step()) {
        this.finishFlight(active);
      }
    }
    if (this.done) {
      this.fastForwarding = false;
    }
  }

  private beginFlight(): void {
    const next = this.session.nextFlight();
    if (next === null) {
      this.active = null;
      this.stage = "done";
      return;
    }
    this.active = next;
    this.stage = next.stage;
    this.lastRound = next.round;
    this.trail = [];
    if (next.stage === "demo") {
      this.ghostByLesson.set(next.lesson.name, []);
    }
    this.ghost = next.stage === "demo"
      ? null
      : this.ghostByLesson.get(next.lesson.name) ?? null;
    // Late drill rounds are not rendered one by one: fast-forward them.
    if (next.round > VISIBLE_ROUNDS) {
      this.fastForwarding = true;
    }
  }

  private finishFlight(flight: SchoolFlight): void {
    if (flight.stage === "demo") {
      this.ghostByLesson.set(flight.lesson.name, [...this.trail]);
    }
    this.beginFlight();
  }

  private sample(flight: SchoolFlight): void {
    const body = flight.flight.body;
    const servos = flight.flight.servos;
    const commands = flight.flight.commands;
    const [forward] = body.bodyAxes;
    const alive = !(flight.flight.finished
                    && this.school.grounded(body));
    const view: DroneView = {
      x: body.position.x, y: body.position.y, z: body.position.z,
      fwdX: forward.x, fwdY: forward.y,
      pitch: body.deckPitch,
      bank: body.bank,
      speed: body.velocity.length(),
      vs: body.velocity.z,
      velX: body.velocity.x,
      velY: body.velocity.y,
      throttle: commands[3],
      ail: servos.positions[0],
      elev: servos.positions[1],
      rud: servos.positions[2],
      alive,
    };
    this.lastView = view;
    this.trail.push({ x: view.x, y: view.y, z: view.z });
    if (this.trail.length > 1200) {
      this.trail.shift();
    }
  }

  /** The report card rows so far: score, grade, and golden comparison. */
  reportRows(goldenLessons: readonly {
    lesson: string; score: number; stamps: number;
  }[]): Array<{
    lesson: string; stamps: number; score: number | null;
    grade: string | null; goldenScore: number; goldenStamps: number;
    exact: boolean | null;
  }> {
    const final = this.done;
    // The report card grades like the Python capture does: a fresh exam per
    // lesson at the FINAL tensor state — later drills keep touching shared
    // state bins, so a lesson's stored passing score is not its final one.
    const finalScores = new Map<string, number>();
    if (final) {
      for (const golden of goldenLessons) {
        const lesson = this.school.lessons.get(golden.lesson);
        if (lesson !== undefined) {
          finalScores.set(golden.lesson,
            this.school.studentQuality(this.session.policy, lesson));
        }
      }
    }
    return goldenLessons.map((golden) => {
      const score = final
        ? finalScores.get(golden.lesson) ?? null
        : this.session.scores.get(golden.lesson) ?? null;
      const stamps = this.session.counts.get(golden.lesson) ?? 0;
      const scoreMatches = score !== null
        && Math.abs(score - golden.score) < 5e-4;
      const stampsMatch = final && stamps === golden.stamps;
      return {
        lesson: golden.lesson,
        stamps,
        score,
        grade: score === null ? null : gradeName(score),
        goldenScore: golden.score,
        goldenStamps: golden.stamps,
        exact: final ? scoreMatches && stampsMatch : null,
      };
    });
  }
}
