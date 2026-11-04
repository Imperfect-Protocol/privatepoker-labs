/**
 * Boot-cost measurement: how long does ONE bounded classroom run take —
 * the same run the browser's boot pre-learning will do (fresh student,
 * maxAttempts 2, exam cap 1.5x the demo's own length). Prints wall time
 * and the per-lesson results, and the graduate's tensor size (what gets
 * transplanted).
 */
import { FlightSchool, SchoolSession } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const t0 = Date.now();
const session = new SchoolSession(school, new ServoPolicy(),
                                  { maxAttempts: 20, examCap: "auto" });
session.runToCompletion();
const dt = (Date.now() - t0) / 1000;
console.log(`boot classroom: ${dt.toFixed(1)} s wall, rounds ${session.rounds}, ` +
            `failed [${session.failed.join(", ")}], ` +
            `graduate keys ${session.graduate?.storedCombinations}`);
for (const [name, score] of session.scores) {
  console.log(`  ${name.padEnd(30)} attempts ${session.attempts.get(name)}  score ${score?.toFixed(2)}`);
}
