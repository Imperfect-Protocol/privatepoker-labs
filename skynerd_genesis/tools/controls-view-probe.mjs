/** Drive the FlightSchoolShow headlessly and print what the PANEL sees:
 *  the scene's two aircraft (the instructor's and the student's — the one
 *  Aircraft class both fly), the selected one's instruments and the
 *  student's decision inputs during its repeat. */
import { FlightSchoolShow } from "../build/src/app/school-show.js";
import { droneViewOf } from "../build/src/app/cockpit.js";

const show = FlightSchoolShow.withFreshPolicy();
const fmt = (v) => (v >= 0 ? "+" : "") + v.toFixed(3);
const choiceChar = (c) => c === 2 ? "\u25b2" : c === 0 ? "\u25bc" : "\u00b7";

const fails = [];
const check = (label, ok, info = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${info ? ` — ${info}` : ""}`);
  if (!ok) { fails.push(label); }
};

let flight = 0;
let lastStage = "";
let ticks = 0;
const sawStudentDecisions = new Set();
let instructorCommandsSeen = false;
let studentAircraftIsOneClass = false;
while (flight < 3 && ticks < 40000) {
  show.step();
  ticks++;
  const moment = show.moment;
  if (moment.stage !== lastStage) {
    lastStage = moment.stage;
    console.log(`\n== flight ${flight}: ${moment.stage} ${moment.lessonName} (attempt ${moment.attempt})`);
    flight++;
  }
  const flying = show.flyingEntry;
  const view = droneViewOf(flying);
  if (moment.stage === "demo" && (Math.abs(view.handCommands[1]) > 0.05
      || view.throttle > 0.5)) {
    instructorCommandsSeen = true;
  }
  if (moment.stage === "repeat" && view.handChoices !== null) {
    for (let i = 0; i < 4; i++) {
      sawStudentDecisions.add(view.handChoices[i]);
    }
    studentAircraftIsOneClass = flying.aircraft.constructor.name === "Aircraft";
  }
  if (ticks % 25 === 0) {
    const dec = view.handChoices === null
      ? "instructor easing" : view.handChoices.map(choiceChar).join(" ");
    console.log(`t ${moment.seconds.toFixed(1).padStart(5)}s elem ${moment.elementName.padEnd(14)}` +
      ` [${flying.label}] ASI ${view.speed.toFixed(1)} ALT ${view.z.toFixed(0)}m` +
      ` hand A ${fmt(view.handCommands[0])} E ${fmt(view.handCommands[1])}` +
      ` T ${view.handCommands[3].toFixed(2)} dec [${dec}]` +
      `${view.handFresh ? " *beat*" : ""}` +
      (view.alive ? "" : "  CRASHED"));
  }
}

check("the scene holds exactly the classroom's two aircraft",
      show.scene.aircraft.length === 2,
      show.scene.aircraft.map((e) => e.label).join(" | "));
check("the instructor's inputs show on its machine", instructorCommandsSeen);
check("the student's decisions show on its machine (push/hold vocabulary seen)",
      sawStudentDecisions.has(1) && sawStudentDecisions.has(2),
      `seen: [${[...sawStudentDecisions].sort().join(",")}]`);
check("both pilots fly the ONE Aircraft class",
      show.instructorEntry.aircraft.constructor.name === "Aircraft"
      && studentAircraftIsOneClass);

console.log(fails.length === 0
  ? "ALL CHECKS PASS — the panel shows the selected aircraft's inputs"
  : `${fails.length} check(s) FAILED`);
process.exit(fails.length === 0 ? 0 : 1);
