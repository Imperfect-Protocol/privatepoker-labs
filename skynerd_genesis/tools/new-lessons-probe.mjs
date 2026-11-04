/**
 * The new lessons' instructor demonstrations: graded against their own
 * goals, with the raw flight data (VS / speed / bank in the climbing and
 * descending turn phases) so the goal bands can be checked against what
 * the machine actually does — at 2 g the climb is whatever the power
 * gives, and the bands must be honest.
 */
import { FlightSchool, DemoFlight, GoalTrace, DECISION_EVERY }
  from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const NEW = ["climb_to_5000", "descend_to_1000",
  "climbing_turn_left_180", "descending_turn_left_180",
  "steep_climbing_turn_left_180", "steep_descending_turn_left_180",
  "stall_recovery"];

for (const name of NEW) {
  const lesson = school.lessons.get(name);
  const spec = school.goalSpec(lesson);
  const demo = new DemoFlight(school, lesson, new ServoPolicy(), 0);
  const trace = new GoalTrace();
  const vsStats = new Map();    // phase index -> [minVs, maxVs, minSpd, maxSpd]
  while (demo.step()) {
    if (demo.tick % DECISION_EVERY === 0) {
      const i = Math.min(demo.cursor.index, lesson.phases.length - 1);
      const r = demo.cursor.readings(demo.body);
      trace.record(i, r);
      const s = vsStats.get(i) ?? [Infinity, -Infinity, Infinity, -Infinity];
      s[0] = Math.min(s[0], r.vsMps); s[1] = Math.max(s[1], r.vsMps);
      s[2] = Math.min(s[2], r.airspeed); s[3] = Math.max(s[3], r.airspeed);
      vsStats.set(i, s);
    }
  }
  let earned = 0, nominal = 0;
  console.log(`\n${name}: ${(demo.tick / 50).toFixed(1)} s, ${demo.stamped} stamps`);
  for (let g = 0; g < spec.goals.length; g++) {
    const f = spec.goals[g].judge(trace, spec.phaseIndices[g]);
    earned += f * spec.nominal[g];
    nominal += spec.nominal[g];
    console.log(`    ${f >= 0.999 ? "\u2713" : f >= 0.5 ? "\u00b1" : "\u2717"} ${spec.goals[g].label} (${f.toFixed(2)})`);
  }
  console.log(`    score ${(6 * earned / nominal).toFixed(2)}`);
  for (const [i, s] of vsStats) {
    console.log(`    phase ${i} (${lesson.phases[i].name}): ` +
      `VS ${s[0].toFixed(1)}..${s[1].toFixed(1)} m/s, ` +
      `${(s[2] * 2.23694).toFixed(0)}..${(s[3] * 2.23694).toFixed(0)} mph`);
  }
}
