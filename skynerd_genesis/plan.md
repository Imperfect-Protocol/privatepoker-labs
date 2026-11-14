# Plan — TENSOR-1: Tensor topology and dual-engine learning

Source ticket: `tensor_task.md` (untracked on `tensors_revised`). The plan is
written against commit `e3046b1` plus the uncommitted V2 additions in
`src/learning/instruments.ts` and `src/index.ts` observed in the working tree on
2026-11-14 (`git diff --stat`: 401 added lines in `instruments.ts`, four changed
lines — two added, two removed — in `index.ts`). `tensor_task.md` carries no ticket number; **TENSOR-1** is this
plan's stable id, not a claim about a ticket number.

## Scope

`tensor_task.md` has no "Done when" section. The five numbered items of its
"Summary & Implementation Roadmap" (`tensor_task.md:213-229`) are the only
acceptance statements in the ticket; they are copied verbatim as DW1–DW5.

**DW1**
1. **Eliminate Sensory Duplication**: Merge the 4 overlapping sensors into 3 orthogonal tensors:
   * `FlightAttitudeTensor` (Attitude & Airframe dynamics)
   * `EnergyPropulsionTensor` (Power & Energy state)
   * `ProprioceptionTensor` (Control surface positions)

**DW2**
2. **Decouple "Current State" from "Desired State"**:
   * Let `CurrentPhaseTensor` evaluate reality ($P_{\text{actual}}$).
   * Let `GoalPlanTensor` dictate intent ($P_{\text{desired}}$).

**DW3**
3. **Unify the Left/Right Skills into a Bidirectional Transition Engine**:
   * Compute transition vector $\Delta P = P_{\text{desired}} - P_{\text{actual}}$.
   * Fire backward interrupt override pulses if $P_{\text{actual}}$ indicates an aerodynamic emergency.

**DW4**
4. **Train Perception in Isolation**:
   * Use Monte Carlo continuous fuzzing.
   * Smooth with Gaussian kernels for adjacent-bin generalization.
   * Denoise and compress using Haar wavelet soft-thresholding.

**DW5**
5. **Combine Hebbian LTP with Cerebellar LTD**:
   * Hebbian forward association over a $0.2\text{ s} - 0.4\text{ s}$ window.
   * Signed error-neuron depression to stamp out dangerous control commands.

Each DW is a direction, not a test. The measurable acceptance for each DW is
the **Proof** of the unit that delivers it (D1).

## Out of scope

- **O1 — Mission/formation changes.** The super-goal and the guardian takeover
  stay as they are (`mission.ts:646-673`, `guardian.ts:61-75`). This ticket
  changes the brain's tensors, not who commands the drone.
- **O2 — The report's literal 4-layer / 7-tensor drawing as a schema.** The
  named tensors are delivered; the literal axis counts, the 64-vector $\Delta P$
  and a new 32-name phase vocabulary are not (D3, D4, D5).
- **O3 — Re-freezing the Python reference goldens.** The frozen values are
  already retired; only replay determinism is gated (`test/episode.test.ts:1-15`).
- **O4 — Cockpit visual redesign.** Only the AI state-log row that must name
  the three sensors and the two phases is in scope (`src/app/main.ts:812-838`).
- **O5 — GPU tensor service / performance.** No performance target is in the
  ticket; `WEBGPU-PLAN.md` §8 reserves that work.
- **O6 — Persistent snapshot format / migration.** Snapshots are in-memory
  (`pilot.ts:475-480`); no file format exists to migrate.

## Findings

**F1 — The ticket has no "Done when".** `tensor_task.md:211-229` is the only
acceptance text. The plan treats the five roadmap items as DW1–DW5 and defines
the proof for each; it does not invent new requirements.

**F2 — The current brain is the 8-tensor split the report critiques.**
`servo-tensor.ts:1-34` and `429-523` define Location, Energy, Flight, Controls,
two skill panels, Goal and Servo; `TENSOR_NODE` (`servo-tensor.ts:1268-1276`)
has eight nodes. The report's 7-tensor target removes the fourth sensor, adds a
phase tensor, and merges the two panels. So the report's diagnosis of the
current split is accurate; its target is a real topology change, not a rename.

**F3 — The report's own shapes contradict "no duplication" and the code's stop
tables.** ASI and VSI appear in *both* `FlightAttitudeTensor` and
`EnergyPropulsionTensor` (`tensor_task.md:80-81`); the uncommitted V2 code says
the overlap is intentional (`instruments.ts:113-117`). The report's
proprioception shape is `15_elev × 11_ail × 11_rud`
(`tensor_task.md:82`) and its prose says `[elev, ail, rud, trim]`
(`tensor_task.md:42`), but the code's stop tables are 11 elevator × 9 aileron ×
9 rudder (`instruments.ts:152-159`, `885-896`). D2 and D3 resolve both.

**F4 — The V2 perception vocabulary exists but is not wired.** The working tree
already contains `FLIGHT_ATTITUDE_SHAPE`, `ENERGY_PROPULSION_SHAPE`,
`PROPRIOCEPTION_SHAPE` (`instruments.ts:105,124,136`), the 32/16/64 class
vocabularies and teachers (`instruments.ts:431-697`), `SenseReadingV2`
(`instruments.ts:746-759`) and `senseReadingV2` (`instruments.ts:867-898`), plus
the `index.ts` exports. The runtime still reads the old `SenseReading`:
`servo-tensor.ts:101-109,413-421`, `school.ts:2309`, `pilot.ts:435`,
`mission.ts:700-723`, `test/episode.test.ts:78,195`. A
`tsc -p tsconfig.json --noEmit` run on 2026-11-14 printed no diagnostics on this
tree, so U1 is an adoption-and-wiring job, not a rewrite of the vocabulary.

**F5 — "Current" and "desired" are not fully conflated, but the code is closer
than the report says.** The goal tensor is a fuzzy state machine keyed on
goal × element × situation bus, output goal × element
(`servo-tensor.ts:495-503,779-780,829-833`). The bus is the sum of all four
sensor codes (`servo-tensor.ts:734-739`), so actual state does condition the
transition. The real gaps are (a) no named actual phase and (b) the goal law is
forward-only (`chosen >= current`, `servo-tensor.ts:1736-1740`). Still, the
report's direction is right: a stall cannot move the plan backwards.

**F6 — There is no runtime backward interrupt.** The only upward path is the
teaching error wave servo → skills → goal → sensors
(`servo-tensor.ts:1874-1959,2062-2111`), which needs an instructor. The guardian
can take the levers from outside the brain (`guardian.ts:1-18,61-75`;
`mission.ts:685-687`). Nothing lets the skill tensor force a goal change in
solo flight.

**F7 — Gaussian smoothing and Haar thresholding already exist; Monte Carlo
fuzzing and isolated training do not.** `denoiseProjection` runs Gaussian
[1 2 1]/4 → Haar soft-threshold → threshold (`neurons.ts:350-403`), the
consolidation runs the same law over a ±2 window (`servo-tensor.ts:1023-1108`),
and fuzzy neighbour recall exists on configured axes (`neurons.ts:55-70,215-251`;
`servo-tensor.ts:451,464`). But demonstrations stamp exact teacher labels during
lessons (`servo-tensor.ts:751-788`, `school.ts:2300-2329`) and boot training is
lesson-driven, one repetition per round-robin lesson
(`boot-train.ts:1-16,300-362`). There is no continuous sample sweep and
no perception-only training entry point. DW4 is therefore two jobs, not three.

**F8 — The Hebbian/LTD pair exists; the coincidence window does not match.**
Demo stamping (`imprint`) and reward eligibility (`reinforce`) are the LTP side
(`servo-tensor.ts:751-788,983-1005`); `ErrorNeuron` is the signed LTD pump with
the sign-flip law (`error-neuron.ts:25-60`). But `ELIGIBILITY_DECISIONS = 10`
(`servo-tensor.ts:119`) at a 5 Hz decision clock (`DECISION_EVERY = 10` at 50 Hz,
`school.ts:180`) is **2 s**, not the "1 s" the comment claims and not the
report's 0.2–0.4 s (`tensor_task.md:195`). DW5 is a tuning change plus a test.

**F9 — The tests barely gate the topology.** `test/episode.test.ts` gates replay
determinism and memory presence, and hard-codes old `SenseReading` in an imprint
(`episode.test.ts:195-197`) and an empty goal tensor in free flight
(`episode.test.ts:189-192`). `test/school.test.ts` is a placeholder. The sensor
swap and the phase split need new, specific checks, not just the existing suite.

**F10 — Every consumer hard-codes four sensors.** `stateView`
(`servo-tensor.ts:2234-2267`), the UI rows (`main.ts:812-838`), the boot
crossover list (`boot-train.ts:372-407`), `BrainSnapshot`
(`servo-tensor.ts:282-295`), `prune` (`1012-1021`), the error-wave widths
(`1927-1934`), and the spread/consolidation branches (`2132-2137`). A 3-sensor
swap touches all of them. The compiler catches all but the UI, because
`src/app/main.ts` is excluded from the Node tsconfig (`tsconfig.json:20-21`).

**F11 — The report's Servo key includes the controls class, but the code
deliberately excludes it.** `tensor_task.md:86` keys the Servo on
$\Delta P$ × control × flight; `servo-tensor.ts:504-516` explains that the
controls class is the hand's own state and keying on it would fragment the
skill memory across every lever nudge. D12 keeps the exclusion.

**F12 — `npm test` depends on a compiler the repo does not declare.**
`package.json:14` runs bare `tsc`; `package.json:17-20` lists only `vite` and
`vite-plugin-standalone` as dev dependencies and `package-lock.json` has no
TypeScript entry. The gate below
names the compiler explicitly and notes the global 5.8.3 that this machine
resolves.

**F13 — The ticket's worked stall example names a class the V2 teacher cannot
produce at the ticket's own bins.** The example says `pitch ≥ 16` and
`asi ≤ 2` should fire `11: nose high` (`tensor_task.md:93`). With wings level
and coordinated, `classifyFlightAttitude(16, 12, 3, 5, 2)` returns
**21 "wings level, nose high, slow"**; class 11 needs `asi ≥ 4`
(`instruments.ts:540-542`). The energy half is right: `asi ≤ 2`, `vsi ≤ 5`
returns **9 "developed stall"** (`instruments.ts:598-604`). ASI bin 2 is
50–60 mph, not the ticket's 55 (`instruments.ts:70`). U1's Proof pins the
teacher's real outputs, so the plan does not inherit the ticket's label.

**F14 — The three V2 panels drop heading-to-target (HSI).** The old
LocationTensor was the only carrier of heading error: `senseReading` puts
`circleBin(heading)` in `location` (`instruments.ts:835-838`), and the location
class feeds the situation bus (`servo-tensor.ts:734-740`). `SenseReadingV2`
has no heading field (`instruments.ts:746-759,867-898`), and FlightAttitude
carries bank and turn *rate*, not heading error. If U2 lands as written, two
airframes differing only by 90° of heading error are identical to perception,
and a rollout has no heading signal. D15 supplies one.

## Decisions

**D1 — DW1–DW5 are the acceptance statements, with the plan's proofs attached.**
*Directed* (the user asked for stories from `tensor_task.md`) and *settled* (the
file has no other "Done when"). If a stricter acceptance is wanted, it replaces
the matching unit's Proof, not the DW wording.

**D2 — "Orthogonal" means one tensor per role and class vocabulary, not disjoint
gauges.** ASI and VSI stay in both attitude and energy-propulsion panels, because
the attitude teacher needs speed context (classes 21/22 and stall detection) and
the energy teacher needs them for energy rate. *Recommended*, supported by the
report's own table (`tensor_task.md:80-81`) and the existing V2 comment
(`instruments.ts:113-117`). If the owner means literal disjointness, U2's
attitude vocabulary loses the speed-context classes and the V2 shapes/teachers
change; that is a different story.

**D3 — Proprioception uses the code's 11 × 9 × 9 stop counts, not the report's
15 × 11 × 11.** *Settled* by `instruments.ts:152-159`. The class axis remains
64. The report's `trim` axis (`tensor_task.md:42`) is not a panel gauge and is
dropped.

**D4 — $P_{actual}$ and $P_{desired}$ use the existing `ELEMENT_AXIS` (32 slots,
21 named elements) rather than a new 32-name phase vocabulary.** *Recommended*:
the axis, the names (`school.ts:396-418`) and the demo stamping already exist,
so the phase tensor can be supervised by the instructor's element. If the owner
wants a distinct phase vocabulary, U4 must add names and an element→phase
mapping, and every key in U4/U5 changes.

**D5 — $\Delta P$ is the existing servo option (hold-all plus the four levers'
36 stops = 37 options), not a new 64-vector vocabulary.** *Settled* by the
existing option layout (`servo-tensor.ts:850-877`): there is no 64-name
transition vocabulary in the repo to key on, and the panel+servo already
associate the phase/class company with a lever option. The report's 64 is read
as capacity. If the owner requires literal transition vectors, the 64 names and
their lever mapping are a new authoring task; U3–U5 cannot proceed unchanged.

**D6 — `GoalPlanTensor` keys on the actual phase as well as goal, desired phase
and progress.** *Recommended*: the report's input list
($goal × phase\_req × progress$, `tensor_task.md:84`) cannot trigger a normal
transition such as climb→level-off when altitude is reached; actual state must
be in the key, as the current goal tensor's bus is (`servo-tensor.ts:1723-1742`).
If the owner requires the literal shape, a separate completion/progress signal
must be supplied by the instructor, and solo transitions lose their trigger.

**D7 — Progress is an 8-bin "decisions held in the current desired phase",
owned by the network; no new sensor.** *Recommended*: it is objectively
available in both demo and solo, and it matches the report's 8-progress axis
without inventing a clock input. If it makes the stamped and recalled keys
diverge, the plan's fallback is to drop the progress axis (D6 already carries
actual state) rather than keep an unmatched key.

**D8 — An unknown actual phase is distinct from free flight.** *Recommended*:
fall back to a reserved unknown phase, not a guessed element; the state log and
the servo key must be able to tell "I do not know where I am" from "I am in free
flight". If the implementer cannot reserve a phase cleanly, the fallback is
`ELEMENT_FREE_FLIGHT` and the state log must mark it as a fallback.

**D9 — The interrupt changes the desired phase only, never the mission goal or
the formation token.** *Recommended*: the mission's super-goal is forward-only
and belongs to the formation (`mission.ts:646-673`), and the guardian already
owns takeover (`guardian.ts:61-75`). The brain may choose a recovery phase; it
may not rewrite the sortie.

**D10 — Isolated perception training uses a neutral recurrence
(prev = teacher class, element = free flight, skill bin = 0) and never replaces
runtime lesson stamping.** *Assumed*: the sensor keys are recurrent — each is
`[panel, own previous class, goal element, skill bin] -> class`
(`servo-tensor.ts:434-478,680-686`) — so a sample stamped under one context may
not recall under another. What settles it: a held-out-panel recall rate with
and without the pre-training pass. If the rate does not improve, D10 is wrong
and the pre-training pass is dropped rather than expanded. **The alternative —
dropping `prev`, `element` and `skill bin` from the three sensor keys — is a
key migration** through the goal bus, both panels, the snapshots and the
school's stamping; it is out of scope unless the neutral pass fails *and* the
owner chooses the migration. The plan does not claim DW4's "train in
isolation" while the keys stay recurrent.

**D11 — The eligibility window becomes 2 decisions (0.4 s).** *Directed* by the
report's 0.2–0.4 s range (`tensor_task.md:195`); the exact 2 is
*recommended* because `PULSE_EVERY = 2` (`servo-tensor.ts:204`) and the decision
clock is 0.2 s. This changes learning dynamics; it is proved by a unit test, not
by the retired goldens.

**D12 — The servo key keeps excluding the controls/proprioception class.**
*Recommended*, for the reason already recorded in `servo-tensor.ts:504-516`:
the hand's own state would fragment the command memory across every lever nudge.
The report's key (`tensor_task.md:86`) is not followed literally.

**D13 — Each unit replaces the old path in place; no feature flag.** *Recommended*:
the repo has no brain-level flag convention and a flag would double every key and
snapshot. Each unit's proof is the guard for the next. If parallel old/new
operation is required, that is a separate versioning story.

**D14 — The uncommitted V2 additions are treated as this ticket's starting
point.** *Assumed*: they are in the working tree on `tensors_revised`, dated
2026-11-14 00:04, and they type-check. If they are someone's scratch, U1 starts
by re-creating them; no other unit changes.

**D15 — Heading error is preserved as an input to the phase tensors, not as a
fourth sensor and not as an extra attitude axis.** *Recommended*: DW1 caps
perception at three panels, and heading error is what "where am I relative to
the leg" needs, so `SenseReadingV2` gains a 24-bin `headingError` computed from
the same `targetBearing` argument as `senseReading` (`instruments.ts:835-838`),
and U4's `currentPhaseTensor` key includes it. This keeps the three shapes
exactly as the V2 constants define them. *Alternative*: add heading error as a
sixth FlightAttitude axis — that changes `FLIGHT_ATTITUDE_SHAPE`, U1's shape
proof, every stored key and the teachers' context. If the owner wants heading
inside perception rather than in the phase, that is the swap, and U1/U2 change.

## Units

Each unit is a development story: it can be built and verified in the order
given. U2 is the only wide change; it is wide because the compiler forces every
consumer at once, not because it does more than DW1.

### U1 — Adopt and pin the V2 perception vocabulary

**Story.** As the student, I want the three perception panels and their named
classes defined once, so every later tensor keys on the same vocabulary.

**Changes.**
- Keep the existing V2 shapes, class vocabularies, teachers, `SenseReadingV2`
  and `senseReadingV2` (`instruments.ts:81-136,414-697,746-759,857-898`;
  `index.ts` exports).
- Correct the proprioception doc/comment to the code's 11 × 9 × 9 and record
  D2's ASI/VSI overlap as intentional.
- Add a test file and import it from `test/run.ts` (`test/run.ts:1-10`).

**Proof.** An automated check asserts: the three shape products are
24·24·7·16·8, 8·8·7·8·16 and 11·9·9; the class axes are 32, 16, 64; the
ticket's stall example classifies as `developed stall` (energy 9) and
`wings level, nose high, slow` (attitude 21) — **not** the ticket's
`11: nose high`, which needs `asi ≥ 4` (F13); every class index has a
non-empty label; `senseReadingV2` returns the same bin as `senseReading` for
every shared gauge on a fixed `RigidBody`; every teacher output is inside its
axis. The old 4-sensor runtime is untouched, so the existing
replay-determinism section still determines the same two runs.

**Invariant.** `SenseReading`, the old shapes, every existing tensor key and
every in-memory snapshot keep their current meaning; `test/episode.test.ts`
needs no edit. U2 may add D15's `headingError` field to `SenseReadingV2`; the
three panel fields keep their meaning.

**Touches.** `src/learning/instruments.ts`, `src/index.ts`, `test/run.ts`, new
`test/instruments.test.ts`.

### U2 — Swap the four sensors for the three V2 tensors

**Story.** As the brain, I want to classify the aircraft from three orthogonal
panels, so the same gauge is never read through two unrelated vocabularies.

**Changes.**
- Replace Location/Energy/Flight/Controls nodes with FlightAttitude (32),
  EnergyPropulsion (16), Proprioception (64). Use `SenseReadingV2` at every
  call site: `servo-tensor.ts`, `school.ts`, `pilot.ts`, `mission.ts`,
  `flight.ts`, tests.
- Do not drop heading: `SenseReadingV2` gains a `headingError` bin computed
  from the same `targetBearing` argument `senseReading` already takes
  (`instruments.ts:835-838`), as D15 requires. This is the only addition to the
  V2 interface; the three panel shapes are unchanged.
- Re-partition the two panels without changing their machinery: left =
  flight-attitude + energy-propulsion, right = proprioception, plus the element
  and servo signals and the existing bridge. A one-sensor panel still has wires
  through the element/servo pairs.
- Update `TENSOR_NODE`, `CLASS_AXES`, `BrainSnapshot`, `sensorTensorOf`,
  `teacherClass`, `stateView`, `prune`, the error-wave widths, the
  spread/consolidation branches, the boot crossover list, and the UI row
  (`main.ts:812-838`).
- Keep the sender id outside the node-id set.

**Proof.** A fixed body + fixed seed produces a deterministic decision; a
demonstration stamp lands on all three sensors; a free-flight episode still
flies; the snapshot round-trips bit-exactly; the state log names exactly three
sensors; no path constructs a four-panel `SenseReading`; two bodies differing
only by 90° of heading error get different `headingError` bins; a rollout leg
still completes. `npm run build:web` type-checks the UI row, which the Node
tsconfig excludes.

**Invariant.** The servo's option vocabulary, command semantics and teacher
fallback are unchanged; the two panels still evaluate every pair of their
signals; the 0.2 s decision clock and `PULSE_EVERY` are unchanged.

**Touches.** `src/learning/servo-tensor.ts`, `instruments.ts`, `school.ts`,
`pilot.ts`, `mission.ts`, `flight.ts`, `app/boot-train.ts`, `app/main.ts`,
`src/index.ts`, `test/episode.test.ts`, `test/run.ts`.

### U3 — Merge the two skill panels into one transition panel

**Story.** As the brain, I want one company of perception, intent and command,
so a spin recovery does not need two half-banks whose only extra coupling is a
coherence wire (the servo already mixes both classes; the bridge is extra).

**Changes.**
- Remove `LEFT_SIDE`/`RIGHT_SIDE`, the two panel nodes and the bridge
  (`servo-tensor.ts:246-261,1290-1307,1760-1832`).
- One panel over the signals `[flightAttitude, energyPropulsion,
  proprioception, element, servo]`: every unordered pair is a wire (10 wires).
  Its verdict is the mean of its wires; it forwards the three classes plus one
  coherence bin to the servo, and the servo's key has one bin axis instead of
  two.
- Keep $\Delta P$ as the servo option (D5); the panel associates the company
  with that option.

**Proof.** A check enumerates the single wire bank and asserts that every pair
the old two banks could learn is present exactly once; a demonstration stamps
all ten; a decision receives exactly one forward; no bridge message exists; the
state log shows one coherence; the snapshot round-trips.

**Invariant.** No signal pair loses its association; the servo still gets one
payload per decision and its command semantics are unchanged; the teaching error
wave still reaches the goal and the sensors.

**Touches.** `servo-tensor.ts`, `app/boot-train.ts`, `test/episode.test.ts`.

### U4 — CurrentPhaseTensor and GoalPlanTensor

**Story.** As the student, I want to know where I actually am separately from
where the plan wants me next, so a mismatch can be acted on instead of hidden
inside one transition map.

**Changes.**
- Add `currentPhaseTensor`: key
  `[attitudeClass, energyClass, proprioClass, headingError]` →
  actual phase (`ELEMENT_AXIS`), supervised at demonstration time by the
  instructor's element (the vocabulary already names `stall entry`, `break`,
  `spin`, `opposite rudder`, `dive`: `school.ts:396-418`); fallback is D8's
  unknown/free-flight marker. `headingError` is D15's bin.
- Rename `goalTensor` to `goalPlanTensor`: key
  `[goal, desiredPhase, actualPhase, progress]` → desired phase; keep the
  existing transition stamping; `progress` is D7's 0–7 held-decision count.
  The report's `phase_req` (`tensor_task.md:84`) is this `desiredPhase` slot
  and is D4's `ELEMENT_AXIS`; no separate 32-name requested-phase vocabulary
  is defined, and actual state is in the key so completion (e.g. altitude
  reached) can trigger the next transition (D6).
- Make the two phases patch signals on U3's panel (actual and desired alongside
  the three classes and the servo option), and put the phase pair into the
  servo observation key. Keep the servo's exclusion of the controls class
  (D12).
- Update the snapshot, crossover, prune, state view and the episode test's
  goal assertion (`episode.test.ts:189-192`).

**Proof.** On a stamped takeoff demonstration the current-phase tensor recalls
roll/rotate/climb/level-off at the matching panels; a stamped stall
demonstration recalls `stall entry` and the recovery recalls `break`/`push`;
two states differing only by 90° of heading error recall different actual
phases; the goal plan advances through the element chain; during a maneuver the
desired phase differs from the actual phase and the servo key carries both;
free flight reports its own phase and no plan; a stamp under the same element
history and the same progress produces the same key at recall; the snapshot
round-trips.

**Invariant.** Normal transitions remain forward-only; the mission goal is
unchanged; the servo's option count and command semantics are unchanged; the
teaching error wave still stamps the same transitions.

**Touches.** `servo-tensor.ts`, `school.ts`, `pilot.ts`, `mission.ts`,
`app/boot-train.ts`, `app/main.ts`, `test/episode.test.ts`, `src/index.ts`.

### U5 — Backward emergency interrupt

**Story.** As the student, when the aircraft is in a stall or spin but the plan
still says cruise, I want the plan to be overridden to a recovery phase in that
same decision, so the recovery does not wait for the instructor.

**Changes.**
- Define the emergency set from the V2 classes (the report's stall/spin/spiral
  family) as a named constant, and the matching recovery phases from the
  existing element vocabulary (break/push/opposite-rudder/dive).
- When actual phase is an emergency and desired phase is not the matching
  recovery, force the desired phase to the recovery element and count an
  override. Whether this is a dispatcher message or a reflex inside the goal
  plan is the implementer's choice, provided the stun timing keeps the decision
  coherent.
- The state log distinguishes a normal transition from an override.

**Proof.** A brain demonstrated to answer a stall panel with cruise, then flown
on that panel, returns the recovery desired phase and a positive override count
in the decision where the emergency first appears; a normal flight never
overrides; two networks in one process do not share override state; a repeated
decision at the same tick does not double-count; the servo still commands in
that decision.

**Invariant.** Normal forward transitions and the teaching error wave are
unchanged; the mission goal, formation token and guardian modes are unchanged
(D9); no override fires on a non-emergency class.

**Touches.** `servo-tensor.ts`, `test` (new learning test), `app/main.ts` state
row.

### U6 — Isolated perception training with Monte Carlo fuzzing

**Story.** As the instructor, I want the perception panels trained over the
continuous instrument space, so a reading one bin off a stamp still classifies.

**Changes.**
- Add a pre-training pass that samples physical gauge values uniformly with
  Gaussian jitter, runs the three deterministic teachers, and stamps the three
  perception tensors under D10's neutral recurrence. Call it before the lesson
  loop in boot training and expose it as a tool under `tools/` for inspection.
- Do not reimplement Gaussian/Haar: the firing and consolidation laws already
  exist (F7). Choose the fuzzy axes per V2 tensor from the observed
  one-bin-off behaviour, not from the old four-sensor layout.

**Proof.** On a held-out sample set the trained panels classify to the teacher's
class at a stated rate, and the rate is higher than the untrained brain's; a
panel one bin from a stamped class recalls that class or its neighbour; two
runs with the same seed store identical keys; the pass touches only the three
perception tensors; skipping the pass leaves free-flight decisions unchanged.
The measured rate is reported, not assumed.

**Invariant.** Runtime lesson stamping still works exactly as before; no cell is
invented at runtime; the brain's snapshot shape is unchanged by the pass.

**Touches.** `app/boot-train.ts`, a new `tools/perception-pretrain.mjs` (or a
test-owned harness), `test` (new learning test), `servo-tensor.ts` only if a
pre-training entry point is needed.

### U7 — Coincidence window and the LTD gate

**Story.** As the student, I want credit to attach only to the decisions that
caused the outcome, so a good landing does not reinforce a lever move from two
seconds ago.

**Changes.**
- Set the eligibility window to 2 decisions (D11) and correct the misleading
  "1 s" comment.
- Add tests for the window and for the existing error-neuron sign law: credit
  applies to the last two decisions and not the third; a worsening pump flips
  the sign; a quiet window does not; the snapshot carries `errorSigns`.

**Proof.** The new tests pass; the existing replay-determinism section still
determines identical two runs; a crash/stall epilogue still credits only the
recent window; the servo's positive/negative reward meanings are unchanged.

**Invariant.** Reward signs and the epilogue semantics are unchanged; the
teaching error wave's pumps are unchanged; the stored snapshot includes the
learned signs.

**Touches.** `servo-tensor.ts`, `test` (new learning test), `test/run.ts`.

## Final gate

Run from `skynerd_genesis/`:

1. `tsc -p tsconfig.json --noEmit` — pass = no diagnostics. (The repo does not
   declare TypeScript; this machine resolves 5.8.3 from `PATH`. If the gate is
   to be portable, the compiler must be declared first; that is a provisioning
   fix, not part of this plan.)
2. `npm test` — runs `tsc -p tsconfig.json` then `node build/test/run.js`
   (`package.json:14`). Pass = exit 0 and every test file prints its `finish`
   line, including the new tests imported from `test/run.ts`.
3. `npm run build:web` — pass = no diagnostics. This is the gate for
   `src/app/main.ts`, which the Node tsconfig excludes
   (`tsconfig.json:20-21`).
4. The deterministic replay check in `test/episode.test.ts:134-160` — pass =
   the same snapshot + the same rng produce identical outcome, survival, choices
   and trajectory.

A "pass" is a command exiting 0 with the named checks true. This plan does not
claim any of these have been run; see Unverified.

## Unverified

- **V1 — I did not run the test suite, a flight, the boot trainer or the web
  build.** I ran `tsc -p tsconfig.json --noEmit` on 2026-11-14 and it printed no
  diagnostics on `e3046b1` + the uncommitted V2 changes. That is a type-check,
  not a behavioural result.
- **V2 — Ownership of the uncommitted V2 additions is unknown.** The files are
  dated 2026-11-14 00:04 on `tensors_revised`; D14 assumes they are this
  ticket's starting point. If they are scratch, U1 re-creates them; nothing else
  changes.
- **V3 — The learning impact of U2–U7 is unmeasured.** Memory sizes, curriculum
  scores and convergence can move. What would settle it: a fixed-seed
  boot-training comparison of the report card before and after each unit. The
  plan does not promise a score floor.
- **V4 — The phase and transition vocabularies are choices, not measurements.**
  D4–D7 have not been implemented. What would settle D4/D6: a probe that stamps
  a takeoff demonstration and recalls the chain in solo; if the chain does not
  recall, the key or the progress definition is wrong.
- **V5 — The report's 64-vector $\Delta P$ and 32 named phases have no
  vocabulary in the code.** D4/D5 substitute the existing element and option
  vocabularies. If the owner wants the literal vocabularies, that is a separate
  authoring task and U3–U5 change.
- **V6 — The ticket branch was not named in the brief.** The plan is committed
  on the checked-out `tensors_revised`, which carries `tensor_task.md` and the
  V2 work. If that is the wrong branch, the plan commit must be moved before
  anyone builds; it must not be cherry-picked into a build.
- **V7 — The emergency set and the recovery-phase mapping (U5) are a first
  cut.** The report names the stall/spin example but not the full set; the
  implementer must record the chosen classes and prove no non-emergency
  override fires.
