# Engineer.md — the logic layer's record

Working notes of the **Engineer** role (correctness of logic), paired with
the **Graphic Designer** role (look and feel). Everything below lives under
`privatepoker-labs/skynerd_genesis/`.

## The layer contract

- **Engineer owns:** `src/learning/**` (tensors, pilots, school, episodes),
  `src/physics/**`, `src/core/**`, the data behind the UI (`stateView`,
  deltas, reports), all tests and probes.
- **Graphic Designer owns:** `src/app/cockpit.ts`, `src/app/ground.ts`,
  `src/app/land.ts`, `web/style.css`, `web/index.html` presentation, all
  canvas aesthetics.
- **The seam between us:** the Designer changed the ground datum
  (`AIRFIELD` plateau at 95 ft in `src/core/world.ts`, and
  `SYLLABUS.runwayM` 10 ft → 100 ft in `src/learning/school.ts` so the
  wheels sit on terrain + 5 ft of gear). My logic reads the ground through
  `World.terrainHeight` only — never assumes flat ground.

## Principles

1. **Aircraft is sealed.** No pilot touches its members. The only write
   path is `Aircraft.command()`; everyone else reads through getters.
   No exceptions.
2. **Polymorphism over branching.** "Who flies" is a Pilot object, never
   an `if (mode === ...)` in a flight class. If a new mode is needed,
   write a new Pilot class — don't extend a switch statement.
3. **The cursor belongs to physics.** `PhaseCursor` advances by the
   machine's own state, never by a speculative "intent" of the AI. The
   early-yoke bug (see Bugs) came from breaking this.
4. **No direct member access on other classes either.** Pilots receive a
   `PilotSighting` (read-only: body, ground, cursor, sensed panels,
   teaching observation, epoch, goal, candidates) and return control
   inputs. That is their entire world.
5. **Modes are sticky per aircraft.** The controls dropdown writes one
   aircraft's mode; switching the aircraft dropdown never touches
   another machine's mode.
6. **A classroom attempt must always end.** Every SoloFlight carries a
   cap (2.2× the demo length) so a stuck AI cannot hang the endless
   classroom.
7. **The report card grades only the AI's hands.** Instructor-flown
   attempts count as attempts but never enter the EMA scores.
8. **Determinism where it matters.** The default `ai` mode is
   byte-identical with the golden episode stream (episode.test proves
   it). Any new behavior must sit behind a mode that defaults off.
9. **Verify in the browser, not just headless.** Playwright + the dev
   server; screenshots read back before claiming a UI change works.

## The Pilot classes (src/learning/pilot.ts)

All control-mode logic lives here — outside `Aircraft`, outside the
flight classes:

- **`InstructorPilot`** — the demonstration hand. Classroom: the TRUE
  element's `phaseTargets`, eased at the student's hand speed (0.05
  rad / decision window — a demo a student can physically track).
  Free flight (the swarm): the **stabilizer** — altitude hold (captured
  at takeover), wings level, trim speed, home banking past 400 m from
  origin, clearance floor 30 m AGL, a FIRM hand (4× demo rates — taking
  over a dive at demo speed arrives late with good manners).
  `reseat(commands, body)` re-seats the hand from the machine's current
  levers — takeover is seamless at any tick.
- **`AIPilot`** — the tensor brain. Decides once per window (never per
  tick — a per-tick apply is a 10× lever-speed bug), applies through
  `policy.commandFrom`, springs the stick in `relax`. In guided modes
  the same object is the suggestion engine (`decideAt`) and the error
  wave's student (`learnFromInstructor`).
- **`AIGuidedInstructorPilot`** (both a) — AI suggests + learns every
  window, instructor flies. The error wave IS the teaching: the twin's
  element/levers against the AI's decision (the proven shadow-teaching
  mechanism, made live).
- **`AIPilotWithInstructor`** (both b) — AI flies, guardian watches.
  `isDangerous` (stall family, overbank > 50°, nose < −15°, sink
  > −2400 fpm, overspeed) takes the levers (becoming mode-a during the
  recovery), 2 s of sustained safety hands them back.

`PilotSighting` is the pilots' whole world (principle 4). Flights build
it, feed `pilot.step(sighting, commands)`, pass the result to
`Aircraft.command`, then `pilot.relax` after the physics tick.

## Today's work

### The controls dropdown (the big feature)
Two dropdowns: **Aircraft** (which machine) and **Controls** (whose
hands), the latter sticky per machine, effective in every scenario.
Modes: `ai | instructor | both-suggest | both-guarded` (guardian.ts
defines the vocabulary + danger predicate + stabilizer law).

The classroom became ONE aircraft (`SoloFlight` + reworked
`FlightSchoolShow`): no demo stage, no 10 s countdown; attempts repeat
endlessly; the instructor's twin flies invisibly beside as the Δ
reference and the teaching source; default mode `instructor` (the
demo-first pedagogy — a fresh scenario demonstrates, the person hands
the machine to the AI with the dropdown).

The swarm: every drone's mode is its own; the stabilizer is the free-
flight instructor (with terrain-aware clearance — see the seam above).

### The stall family's power law (behavior fix)
The code always claimed "the stall family always adds full throttle" —
but the power half never existed. `ServoPolicy.applyStallPower(key,
actions)`: energy class 8/9 (incipient/developed stall — which is also
what a parked plane reads as) always sets throttle to full, on top of
the reflex OR the learned argmax, unconditionally. This is what lets
any plane roll on the runway at all.

### Bugs found and fixed today
- **`throttle NaN%`** — the hold-all sentinel `[HOLD_ALL_CHANNEL=4, 0]`
  reached `optionOf` directly instead of `optionOfActions`; option 37
  decoded as throttle level 7, past the 7-stop table → `undefined` →
  NaN. Fixed in both `stateView` and the consolidation tail. Verified
  with `tools/nan-hunt.mjs` (0 NaN events across boot, exams, shadow,
  waves).
- **Instructor's early yoke pull (takeoff)** — my first SoloFlight had
  the machine's cursor "adopt the AI's intent"
  (`adoptIntent(chosenIndex)`); the AI claims "level off" while parked,
  so the instructor pulled at 10 ft. The fix is principle 3: the
  cursor is physics-driven; the AI's claim drives grading only. Probed:
  elevator 0.000 below 57 mph, pull at 66 mph, climb away.
- **Stabilizer crashes** — three stacked causes, each verified: (1) a
  level drone flew straight out of the 800 m world → home banking; (2)
  the raw altitude→elevator gain couldn't hold height in a bank → a
  proper cascade (alt + sink rate → target pitch, turn back-pressure,
  bank-error → aileron); (3) after the Designer's plateau, the captured
  altitude left ~10 m AGL over the 29 m ground → the 30 m clearance
  floor in the sighting (`groundM`), and the FIRM hand for dives.
- **Per-tick lever apply** — my AIPilot applied the decision every 50 Hz
  tick instead of once per window: 10× lever speed. Window-gated now.
- **AI-log identical rows** — rows merge on qualitative identity
  (labels only; shares and the Δ never split a row; the Δ always
  overwrites with the current value). The tick cell shows
  `126 (81)` — current tick, the state's first tick in braces.

### UI work for the AI state log
`stateView()` (the log's data): fired classification + 5 dominant with
% per tensor; the servo cell shows the command and its dominant
options. The Δ column became the HUD readout (`src/app/hud-delta.ts`):
current line `55mph, 3000ft, 1200fpm, 225°SW, 10°↑, -20°←`, correction
in parens below; pitch/bank arrows are direction-relative (10°↑ held vs
5°↑ expected = `-5°↓`; 10°← vs 5°← = `-5°→`); the heading delta is
labelled with the wind the correction points at. Class labels with
parentheticals split: main line, detail in grey below.

### The Syndicate theme (paired with the Designer)
Fonts Michroma/Rajdhani, cyan-on-slate, chamfered corners; the HUD
canvas retinted to holographic cyan; pitch ladder extended to the full
−90°…+90° range (every 10°, zero = the haze horizon). Lesson rename:
`spin_recovery` → `spin_recovery_left` (display: "spin recovery left"),
matching `spin_recovery_right`.

### Speed architecture (earlier, standing)
Signal processing (noise → Gaussian [1 2 1]/4 → Haar → threshold →
de-Haar) only at three moments: (a) priors, (b) post-repetition
consolidation — incremental via the dirty set, (c) long-stun relax.
Per-decision laws are pure recall. The sweep dispatcher (relaxAt field
per node, no heap/cancel), ISRLU stun timeouts, noise only in
SkillTensor. Flock boot 10×3: ~10 m 38 s → ~5 s, champion 334 k → ~6.5 k
keys, mean EMA held (~1.34).

## The test surface

`node build/test/run.js` — 8 suites (vec, events, config, rng, physics,
episode, units, school). Note: **vec.test's terrain_height check fails
right now — that is the Designer's in-flight ground work** (the
reference JSON predates the plateau); my suites all pass, and the
episode golden stream is intact.

Probes (tools/): `solo-probe.mjs` (the four modes, takeovers,
stabilizer), `nan-hunt.mjs` (NaN sweep), `flock-quality.mjs` (EMA table),
`goal-probe.mjs` (each maneuver's goals — see below).

## Each maneuver's goals (standing report)

takeoff — roll wings-level to 60 mph · rotate at 60 mph · hold 15°
through ground effect to 100 ft · climb at 15°, wings level · level off
at 1000 ft

level_cruise — hold altitude (VSI ~0) · wings level · hold 85 mph

climb_to_5000 — climb at 15°, wings level · level off at 5000 ft

descend_to_1000 — descend nose-down, wings level · hold 85 mph ·
level off at 1000 ft

turn_left_180 / turn_right_180 — bank 20–30° · hold altitude · hold
85 mph · turn a full 180°

climbing turns — bank 20–30° · hold the climb · hold 80 mph · 180°

descending turns — bank 20–30° · hold the descent · hold 85 mph · 180°

steep turns (+ climbing/descending variants) — same four, the steep
bank band

stall_recovery / spin_recovery_left / spin_recovery_right — the
recovery sequences: entry, break, push, dive to 60 mph, climb at 15°,
level off at 3000 ft

comprehensive — the whole chain: takeoff, climb, turns, steeps, stall,
both spins

## Open items

- The takeoff EMA is still low after few reps (~0.28 at 3) — the servo
  has no rotate-pull memory yet; the power law unblocked the training
  path, and the endless browser classroom accumulates the reps. Watch
  it climb.
- The stabilizer (swarm instructor mode) still crashes in long runs on
  the new plateau — the clearance floor and firm hand were verified
  individually but the combination needs one more tuning pass over
  the Designer's final terrain. This is the next task.
- vec.test's terrain golden needs a regeneration once the Designer's
  ground work settles.
