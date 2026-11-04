# Note for the Engineer — from the Graphic Designer

The user's new altitude laws landed, and they cross into your lane (world
model, school datum, a new gear type). Everything is verified green at the
bottom of this note. Please review; the tuning constants are yours to own.

## The law (user-specified)

- Water flat at 0 ft; all land above 10 ft (a bank at every waterline).
- Runway: a flat plateau at **95 ft**.
- Aircraft datum on the runway: **100 ft = 95 ft terrain + 5 ft gear**.
  Standing on the runway, the instruments read 100 ft (they read body z,
  so this needs no instrument change).

## What changed in your files (flagged)

1. `src/core/world.ts` — new `AIRFIELD` export (elev 95 ft, the ellipse
   cx 600, rx 1050, ry 420). `World.terrainHeight` returns the plateau
   inside the ellipse, the Gaussian hills outside. This is now the
   physics ground truth; `src/app/land.ts` carves the identical shape so
   the visual ground and the physics ground agree exactly.
2. `src/learning/school.ts` — `SYLLABUS.runwayM` is now 100 ft (the
   datum), pin comment updated; `stepMachine` calls the gear after each
   `aircraft.step` (see below). One `GearContact` instance, shared by
   every classroom aircraft INCLUDING the instructor twin.
3. `test/vec.test.ts` — the terrain golden check: the frozen Python
   baseline predates the field carve, and all five sample points sit
   inside the ellipse. The check now expects the plateau inside the
   field and the Python values outside. When you next re-freeze
   `vector-ref.json`, model the field on the Python side (or move the
   points out-field) and this can go back to a straight comparison.
4. `tools/restart-check.mjs` — re-anchored to your instructor-first
   default (`controlsMode = "instructor"`), which I picked up mid-task.

## New: `src/physics/gear.ts` (your domain — please review/tune)

The user asked for physical gear instead of the pin's magic clamp
("falls into runway as if gear disappeared"). One stateless type,
applied after each physics tick wherever an aircraft is near terrain.
Laws (all gated on deck within ±15° of the ground plane):

- **SUPPORT** — wheels at/below rest (±0.5 ft band), arrival ≤ 3 m/s:
  +1g cancels gravity, the oleo absorbs the sink, the strut holds rest
  height. Rising is free — the strut never pulls the aircraft down, so
  the wing lifts it off unimpeded (verified: held rotation at 30 m/s is
  airborne in 0.7 s). Arrival > 3 m/s collapses the gear; the crash
  proceeds to your `grounded()` line.
- **LEVELING** — deck −15..0° at contact: lever torque toward level,
  proportional to nose-down depth, eased (no overshoot; −8° rights to
  level in ~2 s).
- **GLUE** — deck 0..15° within 50 ft: downforce, 0.25 g at the
  runway, relaxing completely by 50 ft.

Verified behaviors: taxi holds rest with zero leak; a 2 m/s arrival at
55 mph absorbs and settles in 1.1 s; a 20 m/s dive collapses and
crashes; the instructor takeoff demo rotates clean (no balloon) and
the AI's fumbled roll stays on the wheels (min wheel −0.00 ft).

## Open items for you

- **The pin and the gear now overlap.** The gear reproduces the pin's
  hold at rest (they agree at the datum), so you may want to retire the
  pin in favor of the gear — cleaner state machine. I left the pin
  intact to not disturb your cursor/pin logic.
- **Goldens.** physics.test has no World (unaffected, verified) and
  episode.test is replay-parity only (flight.ts untouched). But demo
  flights now fly WITH gear forces, so a future re-freeze of
  c172-frozen.json will shift slightly (aloft behavior is identical —
  the gear only acts in the contact band).
- **The waves drones step their own bodies** — no gear wired there.
  They contact terrain at your `terrainHeight + 1` line. Fine if they
  stay aloft; flagging in case they ever land.
- The 20-lesson curriculum re-freeze is still open on your side.

## Verification state (all green, just now)

- `npm test` — 8 files (vec 81 with the field-aware golden).
- `tools/restart-check.mjs` — ALL PASS (instructor-first classroom).
- `tools/fpm-debug.mjs` — terrain gates: plateau 95 ft exact, water 0,
  open land > 10 ft, hills to 1001 ft, relief intact.
- Classroom takeoffs, both control modes: no fall, clean rotation.
