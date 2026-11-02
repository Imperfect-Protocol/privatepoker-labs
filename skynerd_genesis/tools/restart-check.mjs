/**
 * Reproduction harness for the restart-button report: loads the compiled web
 * app (build-web/src/app/main.js) under a minimal DOM stub, boots it, runs
 * the frame clock to the end of playback, clicks restart, and inspects the
 * resulting state. No browser needed.
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
    value: "42",               // seed select default; options injected below
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
  play: makeElement("play"),
  speed: Object.assign(makeElement("speed"), { value: "4" }),
  seed: makeElement("seed"),
  restart: makeElement("restart"),
  badge: makeElement("badge"),
  tbody: makeElement("tbody"),
};
// register by HTML id so getElementById finds them
elements["cockpit"] = makeElement("cockpit");
elements["cockpit"].hidden = true;
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
  querySelector: () => elements.tbody,
};

// --- rAF clock ----------------------------------------------------------------
let rafHandler = null;
globalThis.requestAnimationFrame = (fn) => { rafHandler = fn; return 1; };
globalThis.window = {
  devicePixelRatio: 1,
  addEventListener: () => {},
};

// --- drive ---------------------------------------------------------------------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ts = 0;
function stepFrames(count, dtMs = 100) {
  for (let i = 0; i < count; i++) {
    ts += dtMs;
    const handler = rafHandler;
    rafHandler = null;
    if (handler) handler(ts);
  }
}

const app = await import("../build-web/src/app/main.js");

// main.js only exposes state through the DOM; observe via the status line and
// the waves table, and drive clicks through the element stubs.
await sleep(50);   // let boot() resolve
stepFrames(2, 60);
console.log("after boot:");
console.log("  status:", elements.status.textContent);
console.log("  play label:", elements.play.textContent);

// run playback to completion: 5 waves * (20 s + 1.5 s) at 4x ≈ 27 s sim time
stepFrames(2000, 60);
console.log("after playback:");
console.log("  status:", elements.status.textContent);
console.log("  play label:", elements.play.textContent);

// click restart
console.log("clicking restart…");
elements.restart.click();
stepFrames(4, 60);
console.log("after restart:");
console.log("  status:", elements.status.textContent);
console.log("  play label:", elements.play.textContent);
stepFrames(200, 60);
console.log("after 200 more frames:");
console.log("  status:", elements.status.textContent);

// --- cockpit selection flow ---------------------------------------------------
// fresh recording (restart), then click on drone 1 (slot 0): it spawns at
// world (0, -160) heading +x; with scale ~0.7045 that is screen (620, 333).
elements.restart.click();
stepFrames(2, 60);
function clickSky(x, y) {
  const map = listeners.get(elements.sky) ?? {};
  for (const handler of map.click ?? []) handler({ clientX: x, clientY: y });
}
clickSky(626, 333);
stepFrames(2, 60);
console.log("cockpit after drone click:", elements["cockpit"].hidden ? "hidden (BUG)" : "shown");
console.log("  title:", elements["cockpit-title"].textContent);
console.log("  foot:", elements["cockpit-foot"].textContent);
clickSky(40, 40);
stepFrames(2, 60);
console.log("cockpit after empty-space click:", elements["cockpit"].hidden ? "hidden" : "shown (BUG)");

// --- auto-select on crash ----------------------------------------------------
// select drone 2 (slot 1, dies at 12.08 s in wave 1); the cockpit must move
// to the next drone still flying (slot 2 survives 20 s) after the crash.
elements.restart.click();
stepFrames(2, 60);
clickSky(620, 305);   // drone 2 spawns at world (0, -120): screen (620, ~305)
stepFrames(2, 60);
console.log("selected:", elements["cockpit-title"].textContent);
const before = elements["cockpit-title"].textContent;
stepFrames(60, 60);   // t ~ 0.9 + 60*0.24 = ~15.3 s: past drone 2's 12.08 s
console.log("after its crash (12.08s), cockpit shows:",
            elements["cockpit-title"].textContent,
            before !== elements["cockpit-title"].textContent ? "(auto-advanced OK)" : "(BUG: no advance)");
console.log("  foot:", elements["cockpit-foot"].textContent);
