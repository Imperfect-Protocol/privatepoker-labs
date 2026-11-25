# Astra kickoff prompt

You are the coding agent implementing the Hebbian Tensor Flight-Control Research Prototype.

Read `HEBBIAN_FLIGHT_CONTROL_AGENT_PLAN.md` completely before changing any source code. It is the implementation contract. Preserve the repository's existing language, architecture, simulator integration, formatting and tests. Do not assume the repository is empty or replace its stack.

## Your first task: M0 only

1. Inspect the repository and record the current Git/workspace state. Do not overwrite or discard existing user changes.
2. Identify the language, compiler/runtime, build/test commands, simulator and aircraft model, sensor fields, units, coordinate frames, update rates, ground-truth fields, control interfaces, actuator feedback, force-feedback availability, logging and rendering tools.
3. Map the plan's proposed modules to existing code. Identify what can be reused and what is missing.
4. Distinguish actual simulator data from synthetic or mocked data. Do not invent flight limits, stall thresholds, spin labels, aircraft procedures or recovery profiles.
5. Create `docs/implementation/discovery-report.md` with the findings and proposed module mapping. Record design changes that would require an Architecture Decision Record (ADR).
6. Run only safe, existing discovery/build/test commands. Do not connect to hardware, aircraft, autopilot interfaces or physical actuators.
7. Stop after M0. Do not start M1 or implement tensors yet. Present the discovery report, commands run, test results, blockers, assumptions and proposed next step for review.

## Non-negotiable constraints

- This is a simulator-only research prototype. Do not add hardware drivers, real-aircraft I/O, autopilot connectivity or flight-controller firmware output.
- Learning is disabled by default. Shadow learning must never change control outputs.
- Action arbitration, safety bounds, fault fallback and plasticity gating must remain deterministic and outside trainable weights.
- IntuitionTensor receives perception and independently measured observations, not ServoTensor's requested command.
- Mission and Intuition emit typed proposals. A deterministic ActionArbiter resolves conflicts; Hebbian weights do not decide safety authority.
- Preserve tensor notation with input axes before `->` and output axes after it. Do not interpret the arrow as an inverse or transpose.
- Do not claim simulator tests establish real-aircraft safety or certification.

After M0 is reviewed, implement one milestone at a time in the order specified by the plan. For every milestone, run its required tests, report exact commands and results, record limitations and decisions, and stop if an exit gate fails. Never silently skip a gate.
