/** Phase-transition probe: turned accumulator + heading at each phase change. */
import { FlightSchool, DemoFlight } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const deg = (r) => r * 57.29578;
for (const name of ["turn_left_180", "steep_turn_left_180"]) {
  const lesson = school.lessons.get(name);
  const flight = new DemoFlight(school, lesson, new ServoPolicy());
  let phase = flight.cursor.current.name;
  const hdg = () => {
    const [f] = flight.body.bodyAxes;
    let h = Math.atan2(f.y, f.x);
    while (h < 0) h += 2 * Math.PI;
    return deg(h);
  };
  console.log(`\n${name}: start phase '${phase}' heading ${hdg().toFixed(1)}°`);
  while (flight.step()) {
    const p = flight.cursor.current.name;
    if (p !== phase) {
      console.log(`  t ${flight.seconds.toFixed(2)}  -> '${p}'  turned ${deg(flight.cursor.turned).toFixed(1)}°  heading ${hdg().toFixed(1)}°  bank ${deg(flight.body.bank).toFixed(1)}°`);
      phase = p;
    }
  }
  console.log(`  END t ${flight.seconds.toFixed(2)}  turned ${deg(flight.cursor.turned).toFixed(1)}°  heading ${hdg().toFixed(1)}°  dir ${flight.cursor.turnDirection}`);
}
