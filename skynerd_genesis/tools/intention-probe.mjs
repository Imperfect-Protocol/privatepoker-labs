/**
 * Intention probe: the takeoff repeat decision by decision — the sense
 * bins, each candidate element's memory support, the chosen element and
 * the action, plus the flight state.
 */
import {
  FlightSchool, DemoFlight, ExamFlight, DECISION_EVERY,
} from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy, ELEMENT_FIRE_BAR } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const RNG0_STATE = 35399562948360463058890781895381311971n;
const RNG0_INC = 87136372517582989555478159403783844777n;
const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const lesson = school.lessons.get(process.argv[2] ?? "takeoff");
const policy = new ServoPolicy();
const rng = new NumpyPcg64(RNG0_STATE, RNG0_INC);

const demo = new DemoFlight(school, lesson, policy, 0);
while (demo.step()) { /* teach */ }

const exam = new ExamFlight(school, lesson, policy, rng);
let n = 0;
while (exam.step() && n < 400) {
  if (exam.tick % DECISION_EVERY === 0) {
    const sense = school.instruments.senseReading(exam.body, exam.servos.positions,
                                                 lesson.targetBearing);
    const projection = policy.elementProjection(lesson.goalId, sense, rng);
    const raw = lesson.phases.map((e) => {
      const fl = policy.flightNeuron.recall([...sense.flight, e.elementId]);
      const en = policy.engineNeuron.recall([...sense.engine, e.elementId]);
      const co = policy.controlsNeuron.recall([...sense.controls, e.elementId]);
      const go = policy.goalNeuron.recall([lesson.goalId, e.elementId]);
      return `${e.name}: fired=${projection[e.elementId].toFixed(1)} fl=${fl.toFixed(1)} en=${en.toFixed(1)} co=${co.toFixed(1)} goal=${go.toFixed(1)}`;
    }).join(" | ");
    const acts = [];
    for (let s2 = 0; s2 < 4; s2++) {
      for (const lv of [0, 2]) {
        acts.push(`${s2}:${lv === 2 ? "P" : "E"}=${policy.servoNeuron.recall(
          [exam.chosenElement.elementId, ...sense.flight, ...sense.controls, s2, lv]).toFixed(1)}`);
      }
    }
    acts.push(`hold=${policy.servoNeuron.recall(
      [exam.chosenElement.elementId, ...sense.flight, ...sense.controls, 4, 1]).toFixed(1)}`);
    const body = exam.body;
    console.log(`t${(exam.tick / 50).toFixed(1).padStart(5)} ` +
      `elem:${exam.chosenElement.name.padEnd(10)} ` +
      `alt ${(body.position.z * 3.281).toFixed(0).padStart(4)}ft ` +
      `spd ${(body.velocity.length() * 2.237).toFixed(0).padStart(3)}mph ` +
      `pitch ${(body.deckPitch * 57.3).toFixed(0).padStart(3)}° ` +
      `vs ${(body.velocity.z * 197).toFixed(0).padStart(4)}fpm ` +
      `elev ${exam.commands[1].toFixed(3)} ` +
      `${raw}\n         acts[${acts.join()}]`);
    n++;
  }
}
console.log(`done: ${exam.seconds.toFixed(1)}s score ${exam.goalScore.toFixed(2)}`);
