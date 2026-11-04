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

### The second-generation attack mission: the continuous pair conveyor + THE WILL

**The spec.** No waves: TWO drones lift off at once, the next pair
rolls FIVE SECONDS behind them, pair after pair, forever. When they
split, the left seats take the left flank (HSI-45), the right seats
the right flank (HSI+45); level flight at 3000 ft — the moment in
the flank; then the left flank's steep descending turn RIGHT
(HSI+90, down to 500 ft) and the right flank's steep descending turn
LEFT (HSI-90, down to 500 ft) — the columns converge on the tank in
the area below; then the left flank's climbing turn LEFT to 1500 ft
back onto HSI-45 (the right flank mirrors, HSI+45); pitch up, spin
OWN side (left flank left, right flank right) and RECOVER; fly
level — and THE WILL DECIDES: the drone's own brain takes the
controls in free flight. No loop: the conveyor's pair-after-pair
launches ARE the repetition.

**The mode renames + the default.** The four controls modes are
named for WHOSE HANDS FLY: `Solo: Instructor` (instructor), `Solo:
AI` (ai), `Pair: Instructor` (both-suggest), `Pair: AI`
(both-guarded). ONLY THE LABELS changed (controlModeLabel in
guardian.ts + the index.html options). The default everywhere is
`Pair: Instructor` — the classroom AND every new drone
(WavesShow.defaultControlsMode), verified in the browser.

**The 28-phase chain (src/learning/mission.ts).** [takeoff] roll ->
rotate -> the ORBIT CLIMB (25-bank climbing turn held to 3000 ft,
the whole climb inside a ~300 m circle above the field; a straight
climb is 7 km and left the area) -> level off -> settle; [split]
roll in -> turn-by +-45 -> the heading-hold roll-out (turnAmount
enabler in school.ts) -> settle; [flank] the level hold, 8 s at
3000 ft; [attack] roll in at the 60-bank, idle, top rudder, -6 m/s
-> the 90-descending-turn -> **wings level, no heading capture** ->
the held-sink glide to 500 ft; [climb] roll in -> the 135-climbing
turn -> **the heading CAPTURE onto the flank heading** -> climb to
1500 ft -> level off; [spin] the school's own chain: stall entry ->
wait -> drop wing(own side) -> spin(own side) -> opposite rudder ->
dive -> climb -> level off at 1500 ft; [will] the level hold, 8 s.
MISSION_ENTRIES: takeoff 0 / split 5 / flank 9 / attack 10 / climb
14 / spin 19 / will 27.

**THE ATTACK LEG'S MUSH (the trap, and the fix).** The first plan
finished the 90-descending-turn with a precise heading-hold
roll-out — and both seats STUCK: the 60-bank idle turn bleeds the
speed to ~63 mph (near stall at 2g), the roll-out overshoots 90 by
~50 deg, and the heading-hold chases through an endless phugoid
(bank +-14 deg, pitch 0->24 deg, altitude bleeding 2987->1133 ft
over 200 s while drifting out of the area). The school's own steep
descending demo yoyos the same way after its roll-out but recovers
inside a 180; my tighter capture + deeper -8 m/s glide trapped it.
THE FIX: the roll-out is `LevelWingsElement` — wings level, NO
heading capture, the 90 lands with the natural arc overshoot
(accepted: measured +95.7/-99.4, within 10 of the spec's HSI
+-90); the glide is `DescendToElement(500 ft, -6 m/s, cruise,
descendThrottle)` — the school's own proven steep-descent regime.
The -8 m/s variant is energy starvation; ATTACK_VS_MPS = -6.0.

**THE RALLY'S HEADING CAPTURE.** The rally must end back on the
flank's own HSI +-45 — but the attack's arc overshoot VARIES, so a
fixed-amount turn lands wherever the arc left the column. The
flank heading IS the target (the split put it there, the level hold
kept it): `HeadingMark` (mission-local) is kept fresh every tick of
the flank leg by MissionFlight itself (mode- and conform-proof),
and the rally's roll-out is `RollOutToHeadingElement` — the school
gained the `headingTo?: () => number` law (bank = clamp(1.4 *
wrapPi(target - heading)), the runtime-heading twin of turnAmount)
and `PhaseReadings.heading` (the HSI, already computed in
readings(), now exposed). The 135 climbing turn flies the bulk, the
capture lands exactly: rally -46.0/+42.2 against the flank's
-46.5/+43.7.

**THE WILL PHASE.** The chain does NOT loop: when the cursor
finishes, `enterWill()` seats the AI at the levers in FREE FLIGHT
(the swarm's LiveEpisode shape: null cursor, GOAL_FREE_FLIGHT, the
free instructor's stabilizerTeaching as the error-wave source) and
the drone's own tensor brain — trained on the whole sortie's error
waves — decides every next step until it crashes or leaves the
9000 m area (MISSION_AREA_M: the attack glide spans kilometres, the
pattern points anywhere). A will-phase drone does NOT vote in the
singularity's plurality and is BEYOND conform's pull (its will has
it); it still blends memory (it remains a member of the heard
swarm).

**The conveyor (src/app/waves-show.ts).** `launchPair()` every
5.0 s: two seats abreast (PAIR_GAP 40 m), the pairs in trail
(PAIR_TRAIL 90 m), DRONE_CAP 40 (a pair launches only if
aliveCount + 2 <= 40), labels "Pair N - left/right"; the singularity
round every 50 ticks conforms everyone to the super-goal;
pruneOldCrashSites keeps the last 6 pairs' sites. main.ts's
narration: "drone attack - pair N lifted off - X flying - the will
decides after the mission".

### The third generation: the instructor's own departure (no more circles)

**The reports.** The person watched the browser: "they all turn
left all the time" (the first-generation orbit climb circled the
whole sky left), then after the mirror fix "they go in circle"
(the mirrored orbit circled both ways but STILL circled — tight
~300 m rings above the field for the whole 340 s climb), and
finally: "in the comprehensive lesson the instructor follows the
steps correctly, so the drones can too — follow the scenario."

**The fix: the orbit climb is GONE.** The mission now flies the
instructor's own departure, exactly like the comprehensive lesson:
roll -> rotate -> THE STRAIGHT CLIMB out of ground effect to 800 ft
(SPLIT_CLIMB_ALT_M) -> THE SPLIT ON THE CLIMB: the turning
departure onto the flank heading (roll in -> climbing turn-by
+-45 -> the heading-hold roll-out, the climb never stopping) ->
THE CLIMB-OUT IN THE FLANK: straight down the flank heading to
3000 ft -> level off -> settle -> the level moment. The legs:
MISSION_ENTRIES split 3 / flank 6 (was 5 / 9); the plan is still 28
phases (the split's roll-in/turn/roll-out absorbed the takeoff's
level-off slot). A turning departure off the runway is flown from
800 ft up, wings loaded and speed built — never in ground effect.

**Why the pattern is now honest geometry:** each pair's track is
the wide circuit the drawn spec shows — a straight lane out of the
field, a 45 peel onto the flank, a long straight climb-out, the
90 converging attack arc at the far end, the 135 rally home. The
probe's new checks pin it: the takeoff leg turns 0.0 deg (straight,
no circles), the split peels -45.8/+45.6, and the flank climb-out
holds the flank heading within 1.5 deg. Max radius 6712 m — inside
the 9000 m area. The sortie runs takeoff 63 s -> flank 75-80 s ->
attack 222 s -> climb 401 s -> spin 469 s -> will 521 s (half the
old time: no more 340 s of circling).

**The runway seating (the same session's earlier fix, kept):** both
drones of a pair start ON the runway — land.ts's RUNWAY is the one
source of truth, the lane's width split into two halves, each seat
at the CENTRE of its half (±7.5 m on the 30 m lane), every pair
from the same threshold (LAUNCH_X = RUNWAY.near + 50; the 5 s launch
interval is the trail, real runway ops). PAIR_GAP/PAIR_TRAIL are
gone.

### Verification (all green)
- `tools/mission-probe.mjs`: MISSION PROBE OK — both seats the full
  chain: rotate 65 mph (no early yoke), split -46.5/+43.7, the level
  moment 2987/2986 ft, attack +95.7/-99.4 at 536 ft, rally
  -46.0/+42.2 at 1490 ft, own-side spins (peak |rateR| 1.10 both
  seats), will entered at 662 s, max radius 5787/6171 m inside the
  9000 m area; the conveyor (first pair together, 4.9 s one pair,
  5.1 s the second, 10 s the third, labels, all Pair: Instructor);
  the singularity (plurality, rounds, the isolated drone untouched,
  will-phase isolation); conform (forward at a boundary, never
  backward, runway immunity).
- All 9 test suites pass (1549 checks: vec 81, events 104, config
  153, rng 174, physics 180, episode 202, units 209, school 210,
  pipeline 236) — the school's own lessons untouched.
- `tools/nan-hunt.mjs`: 0 NaN layers, 0 exam-flight NaN events,
  waves stepping 40 alive at the cap.
- Browser (Playwright, :8765 ?boot=skip): the scenario dropdown
  offers the drone attack; the controls default SELECTS "Pair:
  Instructor"; pairs roll continuously (pair 10 by ~90 s sim,
  20-40 flying, cap held); the super-goal walks the legs live
  (takeoff at t=373 -> attack at t=458 -> spin at t=654, 0 crashes
  heard through the 500 ft attack and the spins); narration and
  wiki name the will phase; no page errors.

## Earlier work

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

- The attack's natural arc lands HSI +-90 at +95.7/-99.4 (the
  accepted overshoot of the wings-level roll-out through the
  low-energy mush). The probe tolerates 10 deg; if an exact 90 is
  ever wanted, the honest path is a headingTo capture flown AFTER
  the glide (power on, speed recovered), not a tighter roll-out —
  the roll-out capture is the trap, it cost a day.
- The HeadingMark fallback: a drone conformed PAST the flank (split
  boundary -> attack/climb) never marks a flank heading, so its
  rally captures heading 0 (the runway heading). Rare, benign, and
  forward-only — note it if conform pulls harder.
- The takeoff EMA is still low after few reps (~0.28 at 3) — the servo
  has no rotate-pull memory yet; the power law unblocked the training
  path, and the endless browser classroom accumulates the reps. Watch
  it climb.
- The waves' free-flight stabilizer crash on the Designer's plateau is
  moot for the mission chain but the WILL PHASE flies the
  stabilizer's teaching as its error-wave source — watch the will
  drones' survivability over long runs on the current terrain.
- vec.test's terrain golden has settled (passing); regenerate only if
  the Designer's terrain changes again.
