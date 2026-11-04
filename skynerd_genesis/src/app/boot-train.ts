/**
 * THE BOOT PRE-LEARNING — THE EVOLUTIONARY CLASSROOM: the flight
 * school runs OFFLINE, before the main scene ever appears — and the
 * class is a FLOCK. One instructor, many students: every repetition
 * of a lesson is flown by the WHOLE flock as a SHADOW FLIGHT (the
 * instructor's twin aircraft beside each student's, the error wave
 * teaching upstream at every decision tick), the students' grades
 * rank them, the ones that progressed STAY and are CLONED (with
 * cross-over) over the ones that deteriorated. Each student's priors
 * are jittered its own way from birth, so the flock differs from the
 * start and selection has real variety to select on.
 *
 * THE SCHEDULE: the lessons come ROUND-ROBIN, ONE repetition at a time
 * — the curriculum in order, lap after lap, forever, until every
 * lesson has had its repetitions and the schedule simply deals
 * another lap. The dialog's DONE button is the only thing that stops
 * it (finishNow): the person decides when the brain is good enough.
 * The champion (the student with the best running fitness) is the
 * brain that gets transplanted — into the live classroom's student
 * AND every newborn drone.
 *
 * The trainer is chunked: runChunk() steps the flock for a wall-time
 * budget (the students fly interleaved, one tick each, round-robin —
 * "in parallel") and returns, so the browser keeps painting the
 * dialog between chunks. The classroom itself is the honest one —
 * the same flights, the same deterministic dice, the same live
 * physics.
 */
import {
  FlightSchool, ExamFlight, ShadowFlight, InstructorTwin, type Lesson,
} from "../learning/school.js";
import { FlightInstruments } from "../learning/instruments.js";
import { ServoPolicy } from "../learning/servo-tensor.js";
import { NumpyPcg64 } from "../core/rng.js";
import { RNG0_STATE, RNG0_INC } from "../learning/school.js";
import { TRIM_SPEED } from "../learning/flight.js";

/** The class's size: TEN students fly each repetition. The genetic
 *  selection keeps the five best and weaves five new children from
 *  them, so every round ends with ten brains again. */
export const FLOCK_SIZE = 10;

/** How many repetitions each maneuver gets in total (the first pass
 *  counts as the first). */
export const BOOT_REPS = 20;

/** Each repetition's blend weight into the lesson's EMA score. 0.25:
 *  the last four repetitions dominate, but the whole history counts. */
const SCORE_EMA_ALPHA = 0.25;

/** The fitness blend: how much the FRESH rep's score weighs against
 *  the student's running level (the EMA across everything it flew). */
const FITNESS_ALPHA = 0.5;

/** THE FORGETTING SPAN: a tensor cell nobody has re-fired within this
 *  many demonstrations dies (ServoPolicy.prune). The random schedule
 *  revisits every lesson regularly, so anything the curriculum still
 *  uses survives; the endless classroom's memory stays bounded. The
 *  priors are re-imposed every cycle, so the no-brainers never die. */
const EVICTION_EPOCHS = 250;

/** The trainer's knobs (the headless probe shrinks the flock so the
 *  structural gates stay fast; the browser flies the real one). */
export interface BootOptions {
  flockSize?: number;
  reps?: number;
}

/** One golden row the boot run produces — the live report card's
 *  reference: what the boot-trained brain's EMA level is per lesson
 *  (the running level, not the best it ever flew). */
export interface BootGolden {
  lesson: string;
  attempts: number;
  score: number;
  goals: { label: string; fraction: number }[];
}

/** The boot trainer's public face: what the dialog paints. */
export interface BootProgress {
  /** The lesson table — one row per lesson: repetitions and the
   *  flock's best student's EMA score, updated live. */
  rows: BootLessonRow[];
  /** 0..1 — the bar, left to right: the averaged match across all
   *  lessons (5+/6 is fully trained, 3 or below barely flies). */
  fraction: number;
  /** The classroom completed: the brain is ready for transplant. */
  done: boolean;
}

/** One row of the boot terminal's lesson table. */
export interface BootLessonRow {
  lesson: string;
  attempts: number;
  score: number | null;
  status: "training" | "done";
}

/** One round in the schedule: the lesson, and the whole flock's
 *  shadow flights for it (stepped interleaved — the students fly in
 *  parallel). */
interface BootRound {
  lesson: Lesson;
  flights: ShadowFlight[];
  /** Which students are still in the air. */
  live: boolean[];
}

export class BootTrainer {
  readonly school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
  /** The lesson names, in curriculum order (the table's rows). */
  readonly names: readonly string[];
  /** THE FLOCK: every student is a whole brain (its own tensors, its
   *  own error-neuron signs, its own jittered priors). */
  private readonly flock: ServoPolicy[];
  /** Each student's running fitness (EMA of its grades) — the
   *  selection's long memory. */
  private readonly fitness: number[];
  /** The exams' dice: one stream shared by every flight, so every
   *  student flies its own dice but the whole classroom is
   *  deterministic. */
  private readonly rng = new NumpyPcg64(RNG0_STATE, RNG0_INC);
  /** The genetics' dice: the cross-over's weaving. */
  private readonly genes = new NumpyPcg64(770017n, 987654321n);
  /** The priors' dice: each student's jitter stream. */
  private readonly priorRng = new NumpyPcg64(20260101n, 770017n);
  /** The classroom's TIME: one fresh epoch per repetition round. */
  private demoEpoch = 0;
  /** The round currently in the air (null = the schedule hands out
   *  the next one). */
  private current: BootRound | null = null;
  /** The round-robin's position: which lesson of the curriculum the
   *  next repetition flies. */
  private roundIndex = 0;
  /** Repetitions completed per lesson. */
  private readonly attempts = new Map<string, number>();
  /** The flock's best student's EMA goal score per lesson, 0-6. */
  private readonly scores = new Map<string, number>();
  /** The latest round's per-goal fractions per lesson (the report card). */
  private readonly goalReports =
    new Map<string, { label: string; fraction: number }[]>();
  private finished = false;
  private readonly flockSize: number;
  private readonly reps: number;

  constructor(options: BootOptions = {}) {
    this.flockSize = options.flockSize ?? FLOCK_SIZE;
    this.reps = options.reps ?? BOOT_REPS;
    this.names = [...this.school.lessons.keys()];
    this.flock = [];
    for (let s = 0; s < this.flockSize; s++) {
      const student = new ServoPolicy();
      student.installPriors(0, this.priorRng);
      this.flock.push(student);
    }
    this.fitness = new Array(this.flockSize).fill(0.0);
    for (const name of this.names) {
      this.attempts.set(name, 0);
      this.scores.set(name, 0.0);
    }
  }

  /** THE CHAMPION — the student with the best running fitness. Its
   *  live tensors are what the boot dialog's cubes show. */
  get policy(): ServoPolicy {
    let best = 0;
    for (let s = 1; s < this.flock.length; s++) {
      if (this.fitness[s] > this.fitness[best]) {
        best = s;
      }
    }
    return this.flock[best];
  }

  /** The trained brain — a fresh copy of the champion's tensors,
   *  ready to transplant into the live classroom and every newborn
   *  drone. */
  get graduate(): ServoPolicy {
    const brain = new ServoPolicy();
    brain.restore(this.policy.snapshot());
    return brain;
  }

  /** What the boot brain itself scored per lesson — the golden rows the
   *  live report card compares the student's repeats against. */
  golden(): BootGolden[] {
    const rows: BootGolden[] = [];
    for (const name of this.names) {
      rows.push({
        lesson: name,
        attempts: this.attempts.get(name) ?? 0,
        score: this.scores.get(name) ?? 0.0,
        goals: this.goalReports.get(name) ?? [],
      });
    }
    return rows;
  }

  /** Step the offline classroom for up to `budgetMs` of wall time. */
  runChunk(budgetMs: number): BootProgress {
    const deadline = performance.now() + budgetMs;
    while (performance.now() < deadline) {
      if (this.current === null) {
        const next = this.nextRound();
        if (next === null) {
          this.finished = true;
          break;
        }
        this.current = next;
      }
      // The flock flies INTERLEAVED — one tick per student, round and
      // round, the students in parallel.
      let anyLive = false;
      const round = this.current;
      for (let s = 0; s < round.flights.length; s++) {
        if (!round.live[s]) {
          continue;
        }
        if (!round.flights[s].step()) {
          round.live[s] = false;
        } else {
          anyLive = true;
        }
      }
      if (!anyLive) {
        this.absorbRound(round);
        this.current = null;
      }
    }
    const done = this.finished && this.current === null;
    return { rows: this.lessonRows(), fraction: this.matchFraction(),
             done };
  }

  /**
   * THE PROGRESS BAR — the averaged match across ALL lessons: the
   * flock's best student's EMA score per lesson, averaged over the
   * whole curriculum. Averaged 5+/6 is a fully trained brain (100 %),
   * 3 or below barely flies (10 %), and the stretch between climbs
   * linearly.
   */
  private matchFraction(): number {
    let sum = 0.0;
    for (const name of this.names) {
      sum += this.scores.get(name) ?? 0.0;
    }
    const average = sum / this.names.length;
    const clamped = Math.min(1.0, Math.max(0.0, (average - 3.0) / 2.0));
    return 0.1 + 0.9 * clamped;
  }

  /**
   * THE DONE BUTTON's action: the classroom is ENDLESS — when every
   * lesson has had its repetitions the schedule simply deals another
   * cycle (random order, forever), so the tensors keep sharpening
   * until the PERSON says stop. This is the stop: the current round
   * is dropped, whatever the students learned stays, and the champion
   * is what the flock's best student has become.
   */
  finishNow(): void {
    this.finished = true;
    this.current = null;
    for (const student of this.flock) {
      student.prune(this.demoEpoch - EVICTION_EPOCHS);
    }
  }

  /** The next round on the schedule: the lessons come ROUND-ROBIN, one
   *  repetition at a time — the curriculum in order, lap after lap,
   *  the whole flock flying each one. The classroom never runs out
   *  of work. */
  private nextRound(): BootRound | null {
    if (this.finished) {
      return null;
    }
    const lesson = this.school.lessons.get(
      this.names[this.roundIndex % this.names.length])!;
    this.roundIndex += 1;
    if (this.roundIndex % this.names.length === 0) {
      this.dealCycle();   // a lap completed: the maintenance pass
    }
    const flights: ShadowFlight[] = [];
    const live: boolean[] = [];
    const twin = new InstructorTwin(lesson);
    for (let s = 0; s < this.flock.length; s++) {
      flights.push(new ShadowFlight(this.school, lesson, this.flock[s],
        this.childRng(), this.demoEpoch, twin, this.examCapFor(lesson)));
      live.push(true);
    }
    this.demoEpoch += 1;
    return { lesson, flights, live };
  }

  /** A lap of the round-robin completed: the dead cells die and the
   *  priors re-imposed (the endless classroom's memory bound, and the
   *  no-brainers' immortality). */
  private dealCycle(): void {
    for (const student of this.flock) {
      student.prune(this.demoEpoch - EVICTION_EPOCHS);
      student.installPriors(this.demoEpoch, this.priorRng);
    }
  }

  /**
   * GRADE AND SELECT: the round's grades rank the students. THE FIVE
   * BEST STAY (they progressed); the other five are REPLACED by
   * clones and cross-overs of the winners — the deteriorated
   * students' brains die, the successful ones' spread. The lesson's
   * reported score is the flock's best this round, EMA-blended.
   */
  private absorbRound(round: BootRound): void {
    const grades = round.flights.map((flight) => flight.goalScore);
    // Rank by the fresh grade, blended with the running fitness.
    const composite = grades.map((grade, s) =>
      grade * FITNESS_ALPHA + this.fitness[s] * (1.0 - FITNESS_ALPHA));
    const rank = [...this.flock.keys()].sort(
      (a, b) => composite[b] - composite[a]);
    // The running fitness follows each student's grades (the EMA the
    // champion and the selection read).
    for (let s = 0; s < this.flock.length; s++) {
      this.fitness[s] = grades[s] * FITNESS_ALPHA
                       + this.fitness[s] * (1.0 - FITNESS_ALPHA);
    }
    // SELECTION + CROSS-OVER: the winners stay; the losers are
    // rebuilt from the winners' brains — clones of the elite, and
    // cross-overs pairing them.
    const survivors = Math.max(2, Math.ceil(this.flock.length / 2));
    const elite = rank.slice(0, survivors);
    for (let loser = survivors; loser < this.flock.length; loser++) {
      const parentA = this.flock[elite[loser % survivors]];
      const parentB = this.flock[elite[(loser + 1) % survivors]];
      this.crossover(parentA, parentB, this.flock[rank[loser]]);
      // The child inherits its parents' running level, not the
      // deteriorated brain's grade history.
      this.fitness[rank[loser]] =
        (this.fitness[elite[loser % survivors]]
         + this.fitness[elite[(loser + 1) % survivors]]) / 2.0;
    }
    // The lesson's report: the flock's best flight.
    let best = 0;
    for (let s = 1; s < grades.length; s++) {
      if (grades[s] > grades[best]) {
        best = s;
      }
    }
    const name = round.lesson.name;
    const score = grades[best];
    this.attempts.set(name, this.attempts.get(name)! + 1);
    const previous = this.scores.get(name) ?? 0.0;
    this.scores.set(name, score * SCORE_EMA_ALPHA
                         + previous * (1.0 - SCORE_EMA_ALPHA));
    this.goalReports.set(name,
      round.flights[best].goalVerdicts().map((verdict) => ({
        label: verdict.label, fraction: verdict.fraction,
      })));
    // (B) THE POST-REPETITION CONSOLIDATION: every student runs the
    // signal processing over what the repetition taught — noise ->
    // Gaussian -> Haar -> threshold -> de-Haar, spreading the fresh
    // weights over their neighbor areas, once per repetition.
    for (const student of this.flock) {
      student.consolidate(this.demoEpoch, this.priorRng);
    }
  }

  /**
   * CROSS-OVER: the child's brain is woven from two parents' — each
   * tensor's memory merged key by key (each key from either parent,
   * the dice deciding), the error-neuron signs inherited the same
   * way. A clone when both parents are one: the winner's brain
   * copied whole.
   */
  private crossover(parentA: ServoPolicy, parentB: ServoPolicy,
                    child: ServoPolicy): void {
    const snapA = parentA.snapshot();
    const snapB = parentB.snapshot();
    const tensors = ["location", "energy", "flight", "controls", "skill",
                     "goal", "servo"] as const;
    const merged: Record<string, { values: Map<number, Float32Array>;
                                   epochs: Map<number, Int32Array> }> = {};
    for (const tensor of tensors) {
      const a = snapA[tensor], b = snapB[tensor];
      const values = new Map<number, Float32Array>();
      const epochs = new Map<number, Int32Array>();
      for (const [key, layers] of a.values) {
        const fromB = this.genes.uniform() < 0.5
                      && b.values.has(key);
        values.set(key, Float32Array.from(fromB ? b.values.get(key)! : layers));
        epochs.set(key, Int32Array.from(
          fromB ? b.epochs.get(key)! : a.epochs.get(key)!));
      }
      for (const [key, layers] of b.values) {
        if (!values.has(key) && this.genes.uniform() < 0.5) {
          values.set(key, Float32Array.from(layers));
          epochs.set(key, Int32Array.from(b.epochs.get(key)!));
        }
      }
      merged[tensor] = { values, epochs };
    }
    child.restore({
      location: merged.location, energy: merged.energy,
      flight: merged.flight, controls: merged.controls,
      skill: merged.skill, goal: merged.goal, servo: merged.servo,
      errorSigns: snapA.errorSigns.map((sign, i) =>
        this.genes.uniform() < 0.5 ? sign : snapB.errorSigns[i]),
    });
  }

  /** Deterministic fresh dice for one student's flight. */
  private childRng(): NumpyPcg64 {
    return new NumpyPcg64(this.rng.draw64(), this.rng.draw64());
  }

  /** The lesson table the dialog paints. */
  private lessonRows(): BootLessonRow[] {
    return this.names.map((name) => {
      const attempted = this.attempts.get(name)!;
      const trained = attempted >= this.reps;
      const currentLesson = this.current !== null
                            && this.current.lesson.name === name;
      return {
        lesson: name,
        attempts: attempted,
        score: attempted > 0 ? this.scores.get(name)! : null,
        status: currentLesson ? "training" : trained ? "done" : "training",
      };
    });
  }

  /** A fair cap for a headless shadow flight: a multiple of the
   *  demo's own length (the live classroom has NO timeout at all). */
  private examCapFor(lesson: Lesson): number {
    return this.school.demoSeconds(lesson) * 2.2 + 5.0;
  }
}

// ExamFlight is re-exported for the live classroom's solo repeats
// (the swarm's drone episodes fly their own LiveEpisode instead).
export { ExamFlight };
