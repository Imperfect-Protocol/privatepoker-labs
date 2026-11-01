# Genesis — TypeScript/WebGPU port of Skynerd

The browser-native, GPU-accelerated next stage of the Skynerd drone swarm.
The design brief lives beside this folder: `SKYNERD_Genesis_Design_Brief.docx`.

## Folder contract

- **`../skynerd/`** — the Python reference simulator. It stays the source of
  truth; behavioural changes happen there first. Its `golden/` harness
  captures the reference behaviour.
- **`reference/golden-baseline.json`** — frozen Python behaviour, captured by
  `skynerd/golden/capture.py` (schema `skynerd-golden/1`). The TypeScript
  implementation validates itself against this document. Do not edit by hand:
  re-capture with Python when the reference changes.
- **TypeScript application code** — developed here.

## Golden baseline (Milestone 0)

Captured with Python 3.14.6 / NumPy 2.5.4. Contains four scenarios with
fixed seeds — `learn_to_fly` (30 blank-policy episodes), `waves` (5 carpet
waves with distributed inheritance), `school` (14 graded lessons + graduate
tensor size), `will_flight` (one willed life) — plus the agreed clocks:

| Clock | Value | Note |
|---|---|---|
| Physics | 50 Hz (dt 0.02 s) | hardcoded in `flight.py`, fixed dt |
| Decision window | 0.2 s | every 10 physics ticks |
| Motion | 10 Hz | neighbour recomputation |
| Vision | 2 Hz | radar raycast + SwarmEye push |
| Episode horizon | 20.0 s | hardcoded in `flight.py` |

Determinism proof: two captures of the same commit match exactly
(`skynerd/golden/compare.py`, zero tolerance). Port comparisons run with
`--tol` and agreed tolerances per milestone.

## Port milestones (from the brief)

1. TypeScript core — vectors, quaternions, config, event scheduling
2. WebGPU foundation — WGSL pipelines, CPU reference path, fallback
3. One drone in browser — rigid body, aero, servos, one policy
4. Hebbian swarm — sparse tensors, reward, eligibility, inheritance, school, Will
5. SwarmEye LVM — 512×256×8 layered depth maps
6. 3D viewport — third-person swarm camera, inspection overlays
7. Swarm Commander — Nebius LLM intents, validator, command panel
8. Genesis Town — digital twins: humans, cinema, shops, buses, buildings
