/**
 * Live-classroom probe: proves the browser classroom NEVER closes.
 *
 *  - the endless session (loop: true) hands out demo → repeat → demo →
 *    repeat forever, never `done`, one attempt past the old cap, and the
 *    SAME policy object (same tensors, same weights) in every flight;
 *  - skipLesson() (the next-maneuver button) moves forward and wraps;
 *  - switchLesson() (the maneuver dropdown) jumps anywhere, same student;
 *  - a passable curriculum wraps on its own (lap 2, laps counter, the
 *    graduate refreshed from the same living memory);
 *  - the bounded default (loop: false) is untouched: still ends.
 */
import { FlightSchool, SchoolSession } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const fails = [];
const check = (label, ok, info = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${info ? ` — ${info}` : ""}`);
  if (!ok) { fails.push(label); }
};

// --- 1. the endless classroom on one failing maneuver: never evicts ---------------
{
  const policy = new ServoPolicy();
  const session = new SchoolSession(school, policy,
    { lessons: ["takeoff"], maxAttempts: 3, loop: true });
  const seenPolicy = new Set();
  const stages = [];
  let flights = 0;
  let next;
  while ((next = session.nextFlight()) !== null && flights < 30) {
    while (next.flight.step()) { /* fly */ }
    seenPolicy.add(next.flight.policy);
    stages.push(next.stage);
    flights += 1;
  }
  check("takeoff classroom runs past the old cap (30 flights, no close)",
        flights === 30 && !session.done,
        `flights ${flights}, done ${session.done}`);
  check("stages alternate demo/repeat the whole way",
        stages.every((s, i) => s === (i % 2 === 0 ? "demo" : "repeat")),
        stages.slice(0, 8).join(",") + "…");
  check("attempts grow unbounded (takeoff reached "
        + `${session.attempts.get("takeoff")})`,
        session.attempts.get("takeoff") === 16);
  check("SAME student (one policy object) in every flight",
        seenPolicy.size === 1);
  check("no maneuver was evicted to failed", session.failed.length === 0);
}

// --- 2. skipLesson: forward and wrap ---------------------------------------------
{
  const policy = new ServoPolicy();
  const names = [...school.lessons.keys()];
  const session = new SchoolSession(school, policy,
    { lessons: names.slice(0, 3), loop: true });
  check("skip starts on lesson 0", session.currentLessonName === names[0]);
  session.skipLesson();
  check("skip moves to lesson 1", session.currentLessonName === names[1],
        session.currentLessonName);
  session.skipLesson();
  session.skipLesson();
  check("skip past the end wraps to lesson 0 (lap "
        + `${session.laps})`, session.currentLessonName === names[0]
        && session.laps === 1);
  const flights = [];
  let next;
  while ((next = session.nextFlight()) !== null && flights.length < 4) {
    flights.push(next);
    while (next.flight.step()) { /* fly */ }
  }
  check("the wrapped classroom keeps handing out flights",
        flights.length === 4 && !session.done);
}

// --- 3. switchLesson: the dropdown, same student ----------------------------------
{
  const policy = new ServoPolicy();
  const names = [...school.lessons.keys()];
  const session = new SchoolSession(school, policy, {});
  const spin = names[names.length - 1];
  session.switchLesson(spin);
  check("dropdown jumps straight to the chosen maneuver",
        session.currentLessonName === spin, session.currentLessonName);
  const next = session.nextFlight();
  check("the chosen maneuver's demonstration begins",
        next !== null && next.stage === "demo" && next.lesson.name === spin);
  check("the SAME student flies it (policy identity)",
        next.flight.policy === policy);
}

// --- 4. a passable curriculum wraps on its own ------------------------------------
{
  const policy = new ServoPolicy();
  const session = new SchoolSession(school, policy,
    { lessons: ["takeoff", "level_cruise"], bar: 0.0, loop: true });
  let flights = 0;
  let next;
  while ((next = session.nextFlight()) !== null && flights < 10) {
    while (next.flight.step()) { /* fly */ }
    flights += 1;
    if (session.laps >= 1 && flights >= 5) { break; }
  }
  check("a passing curriculum wraps by itself (laps "
        + `${session.laps})`, session.laps >= 1 && !session.done);
  check("the graduate is the same living memory",
        session.graduate !== null && session.graduate.storedCombinations
        === policy.storedCombinations,
        `${session.graduate.storedCombinations} keys`);
}

// --- 5. the bounded default is untouched ------------------------------------------
{
  const policy = new ServoPolicy();
  const session = new SchoolSession(school, policy,
    { lessons: ["takeoff"], maxAttempts: 2 });
  let next;
  while ((next = session.nextFlight()) !== null) {
    while (next.flight.step()) { /* fly */ }
  }
  check("bounded mode still ends (loop defaults false)",
        session.done && session.failed.length === 1,
        `failed [${session.failed.join(", ")}]`);
}

console.log(fails.length === 0
  ? "ALL CHECKS PASS — the classroom never closes"
  : `${fails.length} check(s) FAILED`);
process.exit(fails.length === 0 ? 0 : 1);
