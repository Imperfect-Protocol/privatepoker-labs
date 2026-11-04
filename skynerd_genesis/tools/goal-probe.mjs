/**
 * Goal-law probe for the intention-based classroom: (1) print every
 * maneuver's higher-level goal and its named element goals — the
 * curriculum's contract, (2) grade each demonstration against its own
 * goals (the bands must fit what the instructor flies), (3) run the full
 * classroom loop headlessly and print the per-maneuver outcome with the
 * per-goal fractions of the student's final repeat.
 */
import {
  FlightSchool, DemoFlight, ExamFlight, GoalTrace, DECISION_EVERY,
} from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const RNG0_STATE = 35399562948360463058890781895381311971n;
const RNG0_INC = 87136372517582989555478159403783844777n;

console.log("== every maneuver: its goal, its elements, and what each earns");
for (const [name, lesson] of school.lessons) {
  console.log(`\n${name}  (goal id ${lesson.goalId})`);
  console.log(`  elements: ${lesson.phases.map((e) => e.name).join(" -> ")}`
    + (lesson.repeatFrom > 0
      ? `   (student repeats from: ${lesson.phases[lesson.repeatFrom].name})`
      : ""));
  const spec = school.goalSpec(lesson);
  for (const goal of spec.goals) {
    console.log(`  goal: ${goal.label}`);
  }
}

console.log("\n== the instructor's own demonstrations, graded against the goals");
for (const [name, lesson] of school.lessons) {
  const policy = new ServoPolicy();
  const demo = new DemoFlight(school, lesson, policy, 0);
  const trace = new GoalTrace();
  while (demo.step()) {
    if (demo.tick % DECISION_EVERY === 0) {
      const i = Math.min(demo.cursor.index, lesson.phases.length - 1);
      trace.record(i, demo.cursor.readings(demo.body));
    }
  }
  const spec = school.goalSpec(lesson);
  let earned = 0, nominal = 0;
  const parts = [];
  for (let g = 0; g < spec.goals.length; g++) {
    const f = spec.goals[g].judge(trace, spec.phaseIndices[g]);
    earned += f * spec.nominal[g];
    nominal += spec.nominal[g];
    parts.push(`${f >= 0.999 ? "✓" : f >= 0.5 ? "±" : "✗"} ${spec.goals[g].label} (${f.toFixed(2)})`);
  }
  const score = 6 * earned / nominal;
  console.log(`${name.padEnd(21)} demo goal score ${score.toFixed(2)}  ${demo.stamped} stamps`);
  for (const p of parts) { console.log(`    ${p}`); }
}

console.log("\n== the student's first repeat per maneuver (element choices)");
for (const [name, lesson] of school.lessons) {
  const policy = new ServoPolicy();
  const demo = new DemoFlight(school, lesson, policy, 0);
  while (demo.step()) { /* teach this one maneuver */ }
  const exam = new ExamFlight(school, lesson, policy,
                              new NumpyPcg64(RNG0_STATE, RNG0_INC));
  const elements = [];
  while (exam.step()) {
    if (exam.tick % DECISION_EVERY === 0) {
      const current = exam.chosenElement.name;
      if (elements[elements.length - 1] !== current) {
        elements.push(current);
      }
    }
  }
  console.log(`${name.padEnd(21)} score ${exam.goalScore.toFixed(2)}  elements: ${elements.join(" -> ")}`);
}

console.log("\n== the classroom loop, headlessly (teach)");
const t0 = Date.now();
const student = new ServoPolicy();
const report = school.teach(student);
const dt = Date.now() - t0;
console.log(`rounds (re-demos): ${report.rounds}, failed: [${report.failed.join(", ")}], ` +
            `graduate keys ${report.graduate.storedCombinations}, ${dt} ms`);
// per-lesson detail from a fresh session (same deterministic result)
const school2 = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const policy2 = new ServoPolicy();
const session = new (await import("../build/src/learning/school.js")).SchoolSession(school2, policy2, {});
session.runToCompletion();
for (const name of school2.lessons.keys()) {
  const score = session.scores.get(name);
  const attempts = session.attempts.get(name);
  const rows = session.goalReports.get(name) ?? [];
  const ok = session.passed.includes(name) ? "PASS" : "FAIL";
  console.log(`${name.padEnd(21)} ${ok}  attempts ${String(attempts).padStart(2)}  score ${score?.toFixed(2)}`);
  for (const r of rows) {
    console.log(`    ${r.fraction >= 0.999 ? "✓" : r.fraction >= 0.5 ? "±" : "✗"} ${r.label} (${r.fraction.toFixed(2)})`);
  }
}
