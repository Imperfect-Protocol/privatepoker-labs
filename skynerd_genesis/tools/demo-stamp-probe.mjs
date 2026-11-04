/**
 * Demo stamp probe: the takeoff demonstration decision by decision —
 * what the instructor flew, what the panel read, and which servo cells
 * the imprints landed in. The ground truth the student has to replay.
 */
import {
  FlightSchool, DemoFlight, DECISION_EVERY,
} from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const lesson = school.lessons.get("takeoff");
const policy = new ServoPolicy();
const demo = new DemoFlight(school, lesson, policy, 0);
let n = 0;
while (demo.step() && n < 200) {
  if (demo.tick % DECISION_EVERY === 0) {
    const sense = school.instruments.senseReading(demo.body,
                                                 demo.servos.positions, null);
    const element = demo.cursor.current;
    const body = demo.body;
    const recalls = [];
    for (let surface = 0; surface < 4; surface++) {
      for (const level of [0, 2]) {
        const recall = policy.servoNeuron.recall(
          [element.elementId, ...sense.flight, ...sense.controls,
           surface, level]);
        if (recall > 0.15) {
          recalls.push(`${surface}:${level === 2 ? "P" : "E"}=${recall.toFixed(1)}`);
        }
      }
    }
    console.log(`t${(demo.tick / 50).toFixed(1).padStart(5)} ${element.name.padEnd(10)} `
      + `pitch ${(body.deckPitch * 57.3).toFixed(0).padStart(3)}° `
      + `vs ${(body.velocity.z * 197).toFixed(0).padStart(4)}fpm `
      + `spd ${(body.velocity.length() * 2.237).toFixed(0).padStart(3)}mph `
      + `alt ${(body.position.z * 3.281).toFixed(0).padStart(4)}ft `
      + `elev ${demo.commands[1].toFixed(2)} `
      + `| fl=[${sense.flight}] co=[${sense.controls}] `
      + `| recall: ${recalls.join(" ")}`);
    n++;
  }
}
