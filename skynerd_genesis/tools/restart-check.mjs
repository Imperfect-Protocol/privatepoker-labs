/**
 * Headless harness for the interactive app: loads the compiled web build
 * (build-web/src/app/main.js) under a minimal DOM stub with ?boot=skip
 * (no offline pre-learning — a fast, untrained world), then drives the
 * frame clock through the LIVE flight school — narration milestones, the
 * scenario wiki, restart, the speed control, the next-maneuver button, and
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
    dataset: {},
    onclick: null,
    scrollTop: 0,
    offsetWidth: 100,
    offsetHeight: 30,
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
    removeEventListener(event, handler) {
      const list = listeners.get(el)?.[event];
      if (list) {
        const i = list.indexOf(handler);
        if (i >= 0) list.splice(i, 1);
      }
    },
    setPointerCapture() {},
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
  controls: Object.assign(makeElement("controls"), { value: "ai" }),
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
// small canvases headlessly: the world buffer is per-pixel work (and
// the terrain march is per-pixel too) — the visual gates live in
// fpm-debug at full size; here the wiring is what is under test
elements["cockpitCanvas"] = Object.assign(makeElement("cockpitCanvas"), makeCanvas(), {
  clientWidth: 160, clientHeight: 90,
});
elements["controlsCanvas"] = Object.assign(makeElement("controlsCanvas"), makeCanvas(), {
  clientWidth: 160, clientHeight: 40,
});
elements["boot"] = makeElement("boot");
elements["bootFill"] = makeElement("bootFill");
elements["bootLog"] = makeElement("bootLog");
elements["bootExpand"] = makeElement("bootExpand");
elements["bootBrain"] = Object.assign(makeElement("bootBrain"), makeCanvas());
elements["bootDone"] = makeElement("bootDone");
elements["tabWiki"] = makeElement("tabWiki");
elements["tabAi"] = makeElement("tabAi");
elements["app"] = Object.assign(makeElement("app"), { clientHeight: 900 });
elements["bar"] = makeElement("bar");
elements["sideGrip"] = makeElement("sideGrip");
elements["bottomGrip"] = makeElement("bottomGrip");

globalThis.document = {
  getElementById: (id) => elements[id] ?? null,
  querySelector: (selector) =>
    String(selector).includes("#school") ? elements.schoolTbody : elements.tbody,
  querySelectorAll: () => [],
  createElement: () => makeCanvas(),
  body: { dataset: {} },
};

// --- rAF clock ----------------------------------------------------------------
let rafHandler = null;
globalThis.requestAnimationFrame = (fn) => { rafHandler = fn; return 1; };
globalThis.window = {
  devicePixelRatio: 1,
  innerWidth: 1440,
  innerHeight: 900,
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
const goldenCount = lessonCount;   // the status checks read the curriculum count
check("boots into the flight school",
      elements.narration.textContent.startsWith("takeoff") &&
      elements.narration.textContent.includes("the instructor has the controls"),
      elements.narration.textContent);
check("the cockpit panel follows the school's aircraft",
      elements["cockpit-title"].textContent.includes("AIRCRAFT") &&
      elements["cockpit-title"].textContent.includes("AIRBORNE"),
      elements["cockpit-title"].textContent);
check("status shows maneuver 1 of the curriculum",
      elements.status.textContent.includes(`maneuver 1/${lessonCount}`),
      elements.status.textContent);
check("status names the classroom's default hands (instructor-first)",
      elements.status.textContent.includes("INSTRUCTOR AT THE CONTROLS"),
      elements.status.textContent);

// the controls dropdown: whose hands are on the ONE aircraft's levers
elements.controls.value = "ai";
elements.controls.change();
stepFrames(3, 60);
check("the controls dropdown hands the levers to the AI",
      elements.status.textContent.includes("AI at the controls"),
      elements.status.textContent);
elements.controls.value = "instructor";
elements.controls.change();
stepFrames(3, 60);

// the endless classroom: the attempt flies until the machine lands or
// its plan completes, the result holds a moment, the next one begins
console.log("== the endless classroom (one aircraft, attempt after attempt)");
let sawFinish = false;
for (let i = 0; i < 4000 && !sawFinish; i++) {
  stepFrames(1, 60);
  if (elements.narration.textContent.includes("finished")) sawFinish = true;
}
check("the attempt finishes and the result holds", sawFinish,
      elements.narration.textContent);

// the results panel: the scenario wiki + the golden badge (the
// maneuvers' golden exactness is gated by the classroom probes, not here)
console.log("== the scenario wiki and the golden badge");
check("the wiki tab describes the running maneuver",
      elements["tabWiki"].innerHTML.includes("<h3>"),
      elements["tabWiki"].innerHTML.slice(0, 90));
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
      elements.narration.textContent.startsWith("takeoff") &&
      elements.narration.textContent.includes("the instructor has the controls"),
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
      elements.narration.textContent.startsWith("takeoff") &&
      elements.narration.textContent.includes("the instructor has the controls"),
      elements.narration.textContent);
check("the panel follows the school's aircraft again",
      elements["cockpit-title"].textContent.includes("AIRCRAFT"),
      elements["cockpit-title"].textContent);

// the next-maneuver button: skip forward through the endless curriculum
console.log("== the next-maneuver button");
elements.next.click();
stepFrames(3, 60);
check("the next button advances the curriculum",
      elements.status.textContent.includes(`maneuver 2/${lessonCount}`),
      elements.status.textContent);

// the resize grips: pointer drags resize the bottom panel (law: under
// 10% of the stage it hides; a 10%+ pull up from hidden reopens it) and
// the sidebar (under 30% of the window it hides; a 10%+ pull left from
// hidden reopens it at its last width)
console.log("== the resize grips");
function dragGrip(grip, growX, growY) {
  const map = listeners.get(grip) ?? {};
  const pt = { pointerId: 1, clientX: 500, clientY: 500,
               preventDefault() {} };
  for (const h of map.pointerdown ?? []) h(pt);
  for (const h of map.pointermove ?? []) h({ ...pt,
    clientX: pt.clientX - growX, clientY: pt.clientY - growY });
  for (const h of map.pointerup ?? []) h(pt);
}
const heightOf = (el) => parseInt(el.style.height, 10) || 0;
// stage = 900 - 30 (bar) - 7 (grip) = 863; initial height = 30% = 259
check("the bottom panel opens at 30% of the stage",
      Math.abs(heightOf(elements.results) - 259) <= 1,
      `${heightOf(elements.results)}px`);
dragGrip(elements.bottomGrip, 0, 200);         // pull up 200px
check("dragging the bottom grip up grows the panel",
      Math.abs(heightOf(elements.results) - 459) <= 1,
      `${heightOf(elements.results)}px`);
dragGrip(elements.bottomGrip, 0, -1000);       // slam it down
check("below 10% of the stage the panel hides",
      heightOf(elements.results) === 0, `${heightOf(elements.results)}px`);
dragGrip(elements.bottomGrip, 0, 100);         // from hidden, pull up 100px
check("a 10%+ pull up from hidden reopens the panel",
      Math.abs(heightOf(elements.results) - 100) <= 1,
      `${heightOf(elements.results)}px`);
// the sidebar: 30% of 1440 = 432 hides it; 10% = 144 reopens it
dragGrip(elements.sideGrip, 600, 0);           // pull left 600px
check("dragging the side grip left sets the sidebar's width",
      elements.cockpit.style.width === "700px",
      elements.cockpit.style.width);
dragGrip(elements.sideGrip, -600, 0);           // slam it right: hidden
check("below 30% of the window the sidebar hides",
      elements.cockpit.style.width === "0px"
      && elements.cockpit.classList.contains("collapsed"),
      elements.cockpit.style.width);
dragGrip(elements.sideGrip, 200, 0);            // from hidden, pull left
check("a 10%+ pull left reopens the sidebar (past its last width)",
      parseInt(elements.cockpit.style.width, 10) >= 700,
      elements.cockpit.style.width);

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
