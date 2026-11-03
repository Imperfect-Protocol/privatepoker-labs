/**
 * Takeoff-repeat probe: why doesn't the student rotate? Prints every
 * decision tick of the first takeoff demonstration and the student's
 * first repeat: the instrument bins, the argmax action, and the commands.
 */
import {
  FlightSchool, DemoFlight, ExamFlight, DECISION_EVERY,
} from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const RNG0_STATE = 35399562948360463058890781895381311971n;
const RNG0_INC = 87136372517582989555478159403783844777n;
const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const lesson = school.lessons.get("takeoff");
const policy = new ServoPolicy();
const rng = new NumpyPcg64(RNG0_STATE, RNG0_INC);

function binsOf(body, controls) {
  const sense = school.instruments.senseReading(body, controls);
  return [...sense.flight, ...sense.engine, ...sense.controls].join(",");
}

const demo = new DemoFlight(school, lesson, policy, 0);
console.log("== demo decision ticks (phase | bins | actions stamped)");
while (demo.step()) {
  if (demo.tick % DECISION_EVERY === 0) {
    const raw = demo.cursor.phaseTargets(demo.body);
    // replicate demoTick's stamp law to see what got stamped
    const phase = demo.cursor.current;
    if (phase.stamp) {
      let actions = [0, 0, 0, 0];
      for (let s2 = 0; s2 < 4; s2++) {
        const delta = raw[s2] - demo.servos.positions[s2];
        actions[s2] = delta > 0.04 ? 2 : delta < -0.04 ? 0 : 1;
      }
      if (!actions.some((a) => a !== 1)) {
        const pressures = [];
        for (let s2 = 0; s2 < 4; s2++) {
          const parked = demo.servos.positions[s2];
          const trim = [0.0, 0.02485, 0.0, 0.8][s2];
          if (Math.abs(parked - trim) > 0.05) {
            pressures.push(`${s2}:${parked > trim ? "PUSH" : "EASE"}`);
          }
        }
        actions = pressures.length ? `pressure[${pressures.join(" ")}]` : "hold-all";
      }
      console.log(`${phase.name.padEnd(9)} t${(demo.tick / 50).toFixed(1).padStart(5)} ` +
        `[${binsOf(demo.body, demo.servos.positions)}] thr_cmd ${demo.hand.commands[3].toFixed(2)} ` +
        `elev_srv ${demo.servos.positions[1].toFixed(3)} -> ${JSON.stringify(actions)}`);
    }
  }
}
console.log(`demo done: ${demo.seconds.toFixed(1)}s, stamps ${demo.stamped}`);

console.log("\n== student repeat decision ticks (bins | chosen | commands)");
const exam = new ExamFlight(school, lesson, policy, rng);
let printed = 0;
while (exam.step() && printed < 40) {
  if (exam.tick % DECISION_EVERY === 0) {
    const body = exam.body;
    const key = school.instruments.senseReading(body, exam.servos.positions);
    const flat = [...key.flight, ...key.engine, ...key.controls];
    console.log(`t${(exam.tick / 50).toFixed(1).padStart(5)} elem:${exam.chosenElement.name.padEnd(10)} ` +
      `[${flat.join(",")}] alt ${body.position.z.toFixed(1)} spd ${(body.velocity.length() * 2.237).toFixed(1)}mph ` +
      `pitch ${(body.deckPitch * 57.3).toFixed(1)}° thr ${exam.commands[3].toFixed(2)} ` +
      `elev ${exam.commands[1].toFixed(3)}`);
    printed++;
  }
}
console.log(`repeat done: ${exam.seconds.toFixed(1)}s, goal score ${exam.goalScore.toFixed(2)}`);
