/**
 * One demo + one repeat of one maneuver; prints whether the servo
 * node FIRES during the exam, the exam's forward keys, and how they
 * compare to the demo's stamped servo keys.
 */
import {
  FlightSchool, DemoFlight, ExamFlight, DECISION_EVERY,
} from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { RNG0_STATE, RNG0_INC } from "../build/src/learning/school.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const name = process.argv[2] ?? "turn_left_180";
const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const lesson = school.lessons.get(name);
if (lesson === undefined) {
  throw new Error(`unknown lesson ${name}`);
}

// --- the demo: capture the servo stamp keys -------------------------------
const policy = new ServoPolicy();
const stampKeys = [];
const rawServoFire = policy.servoTensor.fire.bind(policy.servoTensor);
policy.servoTensor.fire = (key, amount, epoch) => {
  if (stampKeys.length < 200) {
    stampKeys.push([...key.slice(0, 5)]);
  }
  return rawServoFire(key, amount, epoch);
};

const demoRng = new NumpyPcg64(RNG0_STATE, RNG0_INC);
const demo = new DemoFlight(school, lesson, policy, 0);
while (demo.step()) { /* teach */ }
console.log(`demo: ${demo.stamped} stamps, ${stampKeys.length} servo stamp keys`);

// --- the repeat: capture the servo's fired keys + acted -------------------
const examRng = new NumpyPcg64(RNG0_STATE, RNG0_INC);
const exam = new ExamFlight(school, lesson, policy, examRng,
  school.demoSeconds(lesson) * 2.2 + 5.0);
const examKeys = [];
const rawDecide = exam.network.decide.bind(exam.network);
exam.network.decide = (goalId, sense, candidates, rng) => {
  const decision = rawDecide(goalId, sense, candidates, rng);
  if (examKeys.length < 60) {
    examKeys.push({ key: [...decision.key], acted: decision.acted });
  }
  return decision;
};
while (exam.step()) {
  // the decide wrapper above records each decision
}
const actedCount = examKeys.filter((e) => e.acted).length;
console.log(`exam: ${examKeys.length} decisions, servo fired ${actedCount}`);

// --- the comparison --------------------------------------------------------
const stampSet = new Set(stampKeys.map((k) => k.join(",")));
let exact = 0;
let prefix6 = 0;
for (const e of examKeys.slice(0, 30)) {
  if (stampSet.has(e.key.join(","))) { exact += 1; }
}
for (const e of examKeys.slice(0, 30)) {
  if (stampSet.has(e.key.slice(0, 6).join(","))) { prefix6 += 1; }
}
console.log(`exam keys matching a demo stamp exactly: ${exact}/30`);
console.log(`exam keys matching on goal/element/classes (first 6): ${prefix6}/30`);
console.log("first 3 demo stamp keys:");
for (const k of stampKeys.slice(0, 3)) { console.log("  " + k.join(" ")); }
console.log("first 3 exam forward keys:");
for (const e of examKeys.slice(0, 3)) {
  console.log(`  ${e.key.join(" ")} acted=${e.acted}`);
}
