# Astra: read these implementation appendices

**Appendix bundle version:** 1.2  
**Implementation environment:** TypeScript / JavaScript + WebGPU (WGSL)

These documents supplement `HEBBIAN_FLIGHT_CONTROL_AGENT_PLAN.md`. They do not replace or rewrite it. The original plan remains authoritative for project scope, repository integration, milestone gates and safety boundaries.

## Read in this order

1. `HEBBIAN_FLIGHT_CONTROL_AGENT_PLAN.md` - primary implementation plan and milestone gates.
2. `APPENDIX_A_TEMPORAL_DYNAMICS.md` - simulation time, current tensor scheduler, per-neuron state, per-synapse timing, EMA/readout and TypeScript/WebGPU execution.
3. `APPENDIX_B_STDP_AND_ADAPTATION.md` - the observation-to-classification-to-action boundary, multidimensional P/Q connections, STDP, differential Hebbian alternative, eligibility/reward modulation, Mission/Intuition/Servo ownership and experiments.
4. `HEBBIAN_FLIGHT_CONTROL_APPENDICES.md` - combined canonical version of Appendices A and B; use this for a single-file review or the formatted DOCX/PDF copies.

## Architecture interpretation

The intended flow is **measured instruments and control forces -> perception/classification tensors -> Mission and Intuition action proposals -> deterministic ActionArbiter -> Servo -> aircraft dynamics -> new observations**. The network is not a formal-logic engine. `P -> Q` is engineer shorthand for a soft, directed, weighted temporal association between multidimensional activity patterns. Weights connect individual source and target feature units inside tensors.

`ControlsTensor` represents measured force/feel at the pilot's controls, not yoke position. Actual actuator/surface response, if measured, is a separate feedback channel. Intuition observes aircraft consequences through the perception tensors; direct Servo-output feedback into Intuition is not required by the current design.

## Required first action

Execute **M0 only** from the primary implementation plan. Inspect the existing repository, TypeScript/JavaScript code, WebGPU/WGSL shaders, current activation model, scheduler, epoch boundary and log-sigmoid delay calculation. Produce the discovery report and stop for review. Do not start M1 or alter the current scheduler before approval.

## Non-negotiable technical distinctions

- A tensor is a population/cluster, not a single neuron.
- Instruments and force sensors provide measured observations; perception tensors classify the situation; Mission and Intuition propose actions.
- Semantic tensor axes are not automatically a time axis. Neuron state, synapse timing and optional sequence history are separate representations.
- The current tensor-wide log-sigmoid scheduler remains the baseline until controlled comparisons justify a change.
- STDP uses strict event order and pre-event arrival time at the synapse. `deltaT == 0` is coincidence, not pre-before-post.
- Canonical pair STDP: pre-before-post in the LTP window potentiates; post-before-pre in the LTD window depresses; outside both windows ordinarily has no pair update.
- A missed expected response penalty is a separate experimental mechanism, disabled by default.
- Pair-based STDP requires discrete event encoding. Do not turn every non-zero analogue activation or every sample frame into a spike. Compare explicit threshold/hysteresis event encoding with a continuous differential-Hebbian baseline if appropriate.
- Reward/outcome modulation, anti-Hebbian competition and learnable transmission delays are separate mechanisms and independent feature flags.
- Pre/post timing learns a directed temporal association; it does not by itself prove physical causation. Use simulator interventions to test candidate causal relationships.
- Positive delay prevents same-tick recursion only with deterministic tick barriers; it does not prove closed-loop stability.
- TypeScript owns deterministic simulation time and orchestration. WebGPU performs batched numerical work. GPU wall-clock execution time must never become neural simulation time.
- Keep learning disabled by default. Use synthetic tests, CPU reference, WebGPU parity, shadow mode, then guarded simulator experiments.
- Do not connect this prototype to physical flight hardware or claim flight readiness.

## M0 report additions

The discovery report must identify which values are measured observations, which are simulator-only labels, which are classified features, which are proposals, and which are actuator commands. It must also document whether the current network is event/spike-based or continuous-valued, how an activation becomes an event (if it does), and which clock the scheduler uses.
