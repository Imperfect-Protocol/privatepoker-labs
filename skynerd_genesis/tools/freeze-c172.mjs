/**
 * Freeze the C172 flight-school reference: run the full classroom TWICE
 * with fresh student tensors and assert the two runs are bit-exact, then
 * write reference/c172-frozen.json. The Python golden is invalid for the
 * redesigned airframe/curriculum — this TS-frozen reference supersedes it
 * (migration rule: TS-frozen references supersede where the redesign
 * invalidates Python goldens).
 */
import { FlightSchool, SchoolSession } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";
import { writeFileSync } from "node:fs";

function runClassroom() {
  const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
  const session = new SchoolSession(school, new ServoPolicy());
  session.runToCompletion();
  // The report card grades like the browser does: a fresh exam per lesson
  // at the FINAL tensor state (stored passing scores go stale as later
  // drills touch shared bins).
  const lessons = [];
  for (const name of school.lessons.keys()) {
    lessons.push({
      lesson: name,
      stamps: session.counts.get(name) ?? 0,
      score: school.studentQuality(session.policy, school.lessons.get(name)),
    });
  }
  return {
    rounds: session.rounds,
    failed: [...session.failed],
    graduate_keys: session.graduate === null ? 0 : session.graduate.storedCombinations,
    lessons,
  };
}

console.log("run 1…");
const a = runClassroom();
console.log("run 2…");
const b = runClassroom();
const da = JSON.stringify(a);
const db = JSON.stringify(b);
if (da !== db) {
  console.error("NOT DETERMINISTIC — the two classroom runs differ:");
  console.error(da.slice(0, 400));
  console.error(db.slice(0, 400));
  process.exit(1);
}
console.log("bit-exact across two full classroom runs ✓");
const doc = {
  schema: "c172-frozen/1",
  note: "TS-frozen flight-school reference: 1111 kg battle-C172 airframe, " +
        "8-lesson phase curriculum. Double-run bit-exact (tools/freeze-c172.mjs).",
  rounds: a.rounds,
  failed: a.failed,
  graduate_keys: a.graduate_keys,
  lessons: a.lessons,
};
writeFileSync(new URL("../reference/c172-frozen.json", import.meta.url),
              JSON.stringify(doc, null, 2) + "\n");
console.log(`wrote reference/c172-frozen.json — rounds ${a.rounds}, ` +
            `failed [${a.failed.join(", ")}], graduate keys ${a.graduate_keys}`);
for (const l of a.lessons) {
  console.log(`  ${l.lesson.padEnd(20)} stamps ${String(l.stamps).padStart(6)}  score ${l.score.toFixed(3)}`);
}
