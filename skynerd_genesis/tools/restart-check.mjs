/**
 * Headless harness for the interactive app: loads the compiled web build
 * (build-web/src/app/main.js) under a minimal DOM stub with ?boot=skip
 * (no offline pre-learning — a fast, untrained world), then drives the
 * frame clock through the LIVE flight school — narration milestones, the
 * report card, restart, the speed control, the next-maneuver button, and
 * the scenario switch to the live drone attack waves. No browser needed.
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
      if (prop === "createRadialGradient" || prop === "createLinearGradient") {
        return () => ({ addColorStop: () => {} });
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
  const classes = new Set();
  const el = {
    id,
    textContent: "",
    innerHTML: "",
    value: "",
    hidden: false,
    disabled: false,
    className: "",
    style: {},
    scrollTop: 0,
    clientWidth: 100,
    clientHeight: 100,
    classList: {
      add: (...c) => c.forEach((k) => classes.add(k)),
      remove: (...c) => c.forEach((k) => classes.delete(k)),
      toggle: (k, f) => {
        const on = f === undefined ? !classes.has(k) : f;
        if (on) { classes.add(k); } else { classes.delete(k); }
        return on;
      },
      contains: (k) => classes.has(k),
    },
    closest: () => null,
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
    children: [],
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
  scenario: Object.assign(makeElement("scenario"), { value: "takeoff" }),
  aircraft: Object.assign(makeElement("aircraft"), { dataset: {} }),
  next: makeElement("next"),
  results: makeElement("results"),
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
elements["boot"] = makeElement("boot");
elements["bootFill"] = makeElement("bootFill");
elements["bootLog"] = makeElement("bootLog");
elements["bootExpand"] = makeElement("bootExpand");
elements["bootBrain"] = Object.assign(makeElement("bootBrain"), makeCanvas());

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
globalThis.HTMLCanvasElement = class HTMLCanvasElement {};   // instanceof checks
globalThis.getComputedStyle = () => ({ paddingTop: "0", paddingBottom: "0", rowGap: "8" });

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

// ?boot=skip: the developer's fast path — no offline pre-learning
globalThis.location = { search: "?boot=skip" };
const app = await import("../build-web/src/app/main.js");
await sleep(50);   // let boot() resolve
stepFrames(3, 60);
check("?boot=skip lands straight in the world (no training dialog)",
      elements["boot"].classList.contains("hidden"));

console.log("== boot (flight school, live)");
// the curriculum's lesson count: every scenario option except the waves
const lessonCount =
  (elements.scenario.innerHTML.match(/<option/g) ?? []).length - 1;
// the report card renders the TRAINED CURRICULUM's lessons — the
// boot trainer's own golden, one row per lesson
const goldenCount = lessonCount;
check("boots into the flight school",
      elements.narration.textContent.startsWith("Instructor demonstrates takeoff"),
      elements.narration.textContent);
check("the cockpit panel follows the school's aircraft",
      elements["cockpit-title"].textContent.includes("INSTRUCTOR'S AIRCRAFT"),
      elements["cockpit-title"].textContent);
check("status shows maneuver 1 of the curriculum",
      elements.status.textContent.includes(`maneuver 1/${lessonCount}`),
      elements.status.textContent);
check("status counts down to the student's launch",
      elements.status.textContent.includes("student launches in 10 s"),
      elements.status.textContent);

// the chase: the instructor demonstrates, then the student launches
// 10 s behind it and flies the same maneuver
console.log("== the student launches behind the instructor");
let sawStudentFly = false;
for (let i = 0; i < 4000 && !sawStudentFly; i++) {
  stepFrames(1, 60);
  if (elements.narration.textContent.startsWith("Student flies takeoff")) {
    sawStudentFly = true;
  }
}
check("the student launches and flies the maneuver", sawStudentFly,
      elements.narration.textContent);

// the report card and the badge (the maneuvers' golden exactness is gated
// by the frozen-reference tests and the classroom probes, not here)
console.log("== the report card and the golden badge");
const reportRows = elements.schoolTbody.innerHTML;
check("report card lists every graded lesson",
      (reportRows.match(/<tr>/g) ?? []).length === goldenCount,
      `${(reportRows.match(/<tr>/g) ?? []).length} rows, ${goldenCount} graded`);
check("no golden verdict without training (skip mode)",
      elements.badge.textContent === "",
      elements.badge.textContent);
check("the clock keeps running (the classroom is endless)",
      elements.play.textContent.includes("pause"),
      elements.play.textContent);

// restart: the one sanctioned fresh start replays the takeoff demo
console.log("== restart");
elements.restart.click();
stepFrames(3, 60);
check("restart replays the takeoff demonstration",
      elements.narration.textContent.startsWith("Instructor demonstrates takeoff"),
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

// scenario switch: the live drone attack waves (a fresh world, the
// conveyor lifting wave after wave)
console.log("== drone attack waves (live)");
elements.scenario.value = "drone-attack-waves";
elements.scenario.change();
stepFrames(3, 60);
check("the waves scenario boots wave 1",
      elements.status.textContent.includes("\u00b7 wave 1 \u00b7"),
      elements.status.textContent);
function clickSky(x, y) {
  const map = listeners.get(elements.sky) ?? {};
  for (const handler of map.click ?? []) handler({ clientX: x, clientY: y });
}
clickSky(507, 220);   // drone 1 spawns at world (0, -160): screen (507, 220) from-sky
stepFrames(3, 60);
check("clicking a drone pins its cockpit",
      elements["cockpit-title"].textContent.includes("DRONE 1"),
      elements["cockpit-title"].textContent);

// and back to the live school (the same student, mid-curriculum)
console.log("== back to the flight school");
elements.scenario.value = "takeoff";
elements.scenario.change();
stepFrames(3, 60);
check("returning to school resumes the classroom",
      elements.narration.textContent.startsWith("Instructor demonstrates takeoff"),
      elements.narration.textContent);
check("the panel follows the school's aircraft again",
      elements["cockpit-title"].textContent.includes("INSTRUCTOR'S AIRCRAFT"),
      elements["cockpit-title"].textContent);

// the next-maneuver button: skip forward through the endless curriculum
console.log("== the next-maneuver button");
elements.next.click();
stepFrames(3, 60);
check("the next button advances the curriculum",
      elements.status.textContent.includes(`maneuver 2/${lessonCount}`),
      elements.status.textContent);

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
