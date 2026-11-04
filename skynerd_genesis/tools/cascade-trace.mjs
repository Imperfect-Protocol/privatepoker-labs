/**
 * Trace one exam decision's dispatcher cascade: every message
 * delivered (sender -> recipient, kind) in order, plus whether the
 * servo ever fires.
 */
import {
  FlightSchool, DemoFlight, ExamFlight, RNG0_STATE, RNG0_INC,
} from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const name = process.argv[2] ?? "turn_left_180";
const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const lesson = school.lessons.get(name);
const policy = new ServoPolicy();
const demo = new DemoFlight(school, lesson, policy, 0);
while (demo.step()) { /* teach */ }
console.log(`demo: ${demo.stamped} stamps, keys ${policy.storedCombinations}`);

const exam = new ExamFlight(school, lesson, policy,
  new NumpyPcg64(RNG0_STATE, RNG0_INC),
  school.demoSeconds(lesson) * 2.2 + 5.0);
const network = exam.network;
const rawDeliver = network.deliver.bind(network);
const NAMES = ["loc", "energy", "flight", "controls", "goal", "skill", "servo"];
network.deliver = (message) => {
  console.log(`  deliver ${NAMES[message.senderId]} -> ${NAMES[message.recipientId]}` +
    ` ${message.kind} [${message.payload.join(",")}]`);
  rawDeliver(message);
};
// step the first decide manually:
// just run one exam step loop and log; the ExamFlight decides on its own cadence.
let tick = 0;
let decided = 0;
while (exam.step() && decided < 3) {
  if (tick % 10 === 0) {   // DECISION_EVERY = 10
    decided += 1;
    console.log(`--- decision ${decided} (t=${tick * 0.02}s) ---`);
  }
  tick += 1;
}
