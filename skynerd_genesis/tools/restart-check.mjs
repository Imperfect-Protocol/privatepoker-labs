/**
 * Headless harness for the interactive app: loads the compiled web build
 * (build-web/src/app/main.js) under a minimal DOM stub, boots it, and drives
 * the frame clock through the LIVE flight school — narration milestones,
 * the golden report card, restart, and the mode switch to carpet playback.
 * No browser needed.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// --- fetch stub: serve reference JSONs from disk ----------------------------
globalThis.fetch = async (url) => {
  const rel = String(url).replace(/^\//, "");
  const file = path.join(here, "..", rel);
  if (fs.existsSync(file)) {
    return { ok: true, json: async () => JSON.parse(fs.readFileSync(file, "utf8")) };
  }
  return { ok: false, status: 404 };
};

// --- canvas 2d context stub (records nothing, breaks nothing) ----------------
function makeCtx() {
  const noop = () => {};
  return new Proxy({}, {
    get(target, prop) {
      if (prop === "canvas") return {};
      if (prop === "createImageData") {
        return (w, h) => ({ width: w, height: h,
                            data: new Uint8ClampedArray(w * h * 4) });
      }
      return noop;
    },
    set() { return true; },
  });
}

function makeCanvas() {
  const c = {
    clientWidth: 1240,
    clientHeight: 440,
    width: 0,
    height: 0,
    style: {},
    getContext: () => makeCtx(),
  };
  c.getBoundingClientRect = () => ({ left: 0, top: 0, width: c.clientWidth, height: c.clientHeight });
  return c;
}

// --- element stubs -------------------------------------------------------------
const listeners = new Map();   // element -> { event -> [handlers] }
function makeElement(id) {
  const el = {
    id,
    textContent: "",
    innerHTML: "",
    value: "",
    hidden: false,
    disabled: false,
    className: "",
    clientWidth: 100,
    clientHeight: 100,
    addEventListener(event, handler) {
      if (!listeners.has(el)) listeners.set(el, {});
      const map = listeners.get(el);
      if (!map[event]) map[event] = [];
      map[event].push(handler);
    },
    click() {
      const map = listeners.get(el);
      for (const handler of map?.click ?? []) handler();
    },
    change() {
      const map = listeners.get(el);
      for (const handler of map?.change ?? []) handler();
    },
  };
  return el;
}

const elements = {
  sky: Object.assign(makeElement("sky"), makeCanvas()),
  chart: Object.assign(makeElement("chart"), makeCanvas()),
  status: makeElement("status"),
  narration: makeElement("narration"),
  play: makeElement("play"),
  speed: Object.assign(makeElement("speed"), { value: "1" }),
  mode: Object.assign(makeElement("mode"), { value: "school" }),
  seed: Object.assign(makeElement("seed"), { value: "42" }),
  restart: makeElement("restart"),
  badge: makeElement("badge"),
  tbody: makeElement("tbody"),
  schoolTbody: makeElement("schoolTbody"),
};
// register by HTML id so getElementById finds them
elements["cockpit"] = makeElement("cockpit");
elements["cockpit"].hidden = true;
elements["zoomIn"] = makeElement("zoomIn");
elements["zoomOut"] = makeElement("zoomOut");
elements["zoomReset"] = makeElement("zoomReset");
elements["cockpit-title"] = makeElement("cockpit-title");
elements["cockpit-foot"] = makeElement("cockpit-foot");
elements["cockpit-close"] = makeElement("cockpit-close");
elements["cockpitCanvas"] = Object.assign(makeElement("cockpitCanvas"), makeCanvas(), {
  clientWidth: 322, clientHeight: 238,
});
elements["controlsCanvas"] = Object.assign(makeElement("controlsCanvas"), makeCanvas(), {
  clientWidth: 322, clientHeight: 238,
});

globalThis.document = {
  getElementById: (id) => elements[id] ?? null,
  querySelector: (selector) =>
    String(selector).includes("#school") ? elements.schoolTbody : elements.tbody,
  createElement: () => makeCanvas(),
  body: { dataset: {} },
};

// --- rAF clock ----------------------------------------------------------------
let rafHandler = null;
globalThis.requestAnimationFrame = (fn) => { rafHandler = fn; return 1; };
globalThis.window = {
  devicePixelRatio: 1,
  addEventListener: () => {},
};

// --- checks -------------------------------------------------------------------
let failures = 0;
function check(name, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok || detail === "" ? "" : ` — ${detail}`}`);
  if (!ok) failures += 1;
}

// --- drive ---------------------------------------------------------------------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ts = 0;
function stepFrames(count, dtMs = 60) {
  for (let i = 0; i < count; i++) {
    ts += dtMs;
    const handler = rafHandler;
    rafHandler = null;
    if (handler) handler(ts);
  }
}

const app = await import("../build-web/src/app/main.js");
await sleep(50);   // let boot() resolve
stepFrames(3, 60);

console.log("== boot (flight school, live)");
check("boots into the flight school",
      elements.narration.textContent === "Instructor demonstrates takeoff",
      elements.narration.textContent);
check("cockpit is open", elements["cockpit"].hidden === false);
check("cockpit title names the school",
      elements["cockpit-title"].textContent.includes("FLIGHT SCHOOL"),
      elements["cockpit-title"].textContent);
check("status shows lesson 1/8",
      elements.status.textContent.includes("lesson 1/8"),
      elements.status.textContent);
check("status says instructor at the controls",
      elements.status.textContent.includes("instructor at the controls"),
      elements.status.textContent);
check("seed select disabled in school mode", elements.seed.disabled === true);

// run until the turn-left demo (lesson 3): 6 s takeoff + 3 s level at 1x
console.log("== live stepping to the turn-left demonstration");
let sawTurnLeft = false;
for (let i = 0; i < 4000 && !sawTurnLeft; i++) {
  stepFrames(1, 60);
  if (elements.narration.textContent === "Instructor demonstrates turn left") {
    sawTurnLeft = true;
  }
}
check("narration reads \"Instructor demonstrates turn left\"",
      sawTurnLeft, elements.narration.textContent);

// run until the first exam: the student repeats
console.log("== the student repeats (first exams)");
let sawStudentRepeats = false;
for (let i = 0; i < 6000 && !sawStudentRepeats; i++) {
  stepFrames(1, 60);
  if (elements.narration.textContent === "Student repeats") {
    sawStudentRepeats = true;
  }
}
check("narration reads \"Student repeats\"", sawStudentRepeats,
      elements.narration.textContent);

// run the classroom to completion (drills fast-forward headlessly)
console.log("== running the classroom to the report card");
// The C172 lessons are long; the app clamps wall to 0.05 s/frame, so chew
// the remaining classroom at 10x (the accumulator still steps 1/50 s).
elements.speed.value = "10";
elements.speed.change();
let sawTakeover = false;
for (let i = 0; i < 30000; i++) {
  stepFrames(1, 60);
  if (elements.narration.textContent.includes("Instructor takes over")) {
    sawTakeover = true;
  }
  if (elements.narration.className.includes("stage-done")) {
    break;
  }
}
check("classroom finishes", elements.narration.className.includes("stage-done"),
      elements.narration.textContent);
check("the instructor took over at least once during the drills", sawTakeover);
check("badge reports the golden-exact classroom",
      elements.badge.textContent === "golden: ✓ bit-exact classroom",
      elements.badge.textContent);
const reportRows = elements.schoolTbody.innerHTML;
check("report card has all 8 lessons",
      (reportRows.match(/<tr>/g) ?? []).length === 8,
      String((reportRows.match(/<tr>/g) ?? []).length));
check("report card rows all match golden exactly",
      !reportRows.includes("mismatch") && !reportRows.includes("…"),
      reportRows.slice(0, 160));
check("clock stops itself at the end", elements.play.textContent.includes("play"),
      elements.play.textContent);
check("status tells the classroom's outcome",
      elements.status.textContent.includes("round") && elements.status.textContent.includes("keys"),
      elements.status.textContent);
// back to 1x for the restart + speed checks
elements.speed.value = "1";
elements.speed.change();

// restart: the classroom replays from the takeoff demonstration
console.log("== restart");
elements.restart.click();
stepFrames(3, 60);
check("restart replays the takeoff demonstration",
      elements.narration.textContent === "Instructor demonstrates takeoff",
      elements.narration.textContent);
check("badge cleared on restart", elements.badge.textContent === "");

// speed control: the select offers 1x default with slower speeds (HTML-side
// contract; here we verify the clock honours the value)
console.log("== speed control");
elements.speed.value = "0.5";
stepFrames(10, 60);   // 0.5 s wall (clamped to 0.05/frame) -> 0.25 s sim -> t 0.4s
check("0.5x steps slower than real time",
      elements.status.textContent.includes("t 0.4s"),
      elements.status.textContent);
elements.speed.value = "1";
stepFrames(10, 60);
check("1x steps in real time",
      elements.status.textContent.includes("t 0.9s"),
      elements.status.textContent);

// 10x: the accumulator just steps ten ticks per wall tick
elements.speed.value = "10";
stepFrames(10, 60);   // 0.5 s wall (clamped) -> 5.0 s sim -> t 5.9s (takeoff is 6 s)
check("10x steps ten times real time",
      elements.status.textContent.includes("t 5.9s"),
      elements.status.textContent);
elements.speed.value = "1";

// mode switch: carpet waves (playback of the bit-exact recording)
console.log("== carpet mode (recording playback)");
elements.mode.value = "carpet";
elements.mode.change();
stepFrames(3, 60);
check("carpet mode boots the waves",
      elements.status.textContent.includes("wave 1/5"),
      elements.status.textContent);
check("carpet mode hides the cockpit until a drone is clicked",
      elements["cockpit"].hidden === true);
function clickSky(x, y) {
  const map = listeners.get(elements.sky) ?? {};
  for (const handler of map.click ?? []) handler({ clientX: x, clientY: y });
}
clickSky(507, 220);   // drone 1 spawns at world (0, -160): screen (507, 220) from-sky
stepFrames(3, 60);
check("clicking a drone opens its cockpit",
      elements["cockpit"].hidden === false
      && elements["cockpit-title"].textContent.includes("DRONE 1"),
      elements["cockpit-title"].textContent);

// and back to the live school
console.log("== back to the flight school");
elements.mode.value = "school";
elements.mode.change();
stepFrames(3, 60);
check("returning to school replays the classroom live",
      elements.narration.textContent === "Instructor demonstrates takeoff",
      elements.narration.textContent);
check("cockpit reopens for the school aircraft",
      elements["cockpit"].hidden === false);

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
