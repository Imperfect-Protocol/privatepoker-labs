**Version:** 0.1  
**Date:** 25 November 2026  
**Status:** Executable engineering plan; simulator-only research prototype  
**Companion document:** *Hebbian Tensor Architecture for Adaptive Flight Control - Research Synthesis and Implementation Proposal*


# 0. Instructions to the coding agent

This document is the implementation contract for the first software prototype. Read it completely before changing code.

## 0.1 Execution rules

1. **Inspect before implementing.** Begin with repository reconnaissance. Identify the language, build system, simulator, existing aircraft model, state representation, control interface, test framework, logging and rendering tools. Do not assume a clean repository or introduce a replacement stack.
2. **Preserve the existing project.** Reuse existing types, simulator adapters, math libraries and conventions where appropriate. Avoid broad rewrites. Record proposed architectural changes before making them.
3. **Work milestone by milestone.** Complete one milestone, run its tests, document the result and commit or otherwise checkpoint it before beginning the next. Do not create the entire architecture in one unreviewable change.
4. **Make assumptions explicit.** If the repository does not supply an aircraft model, instrument calibration, force-sensor model or simulator interface, record the missing information. Implement a mock or synthetic test fixture where useful, but never present it as a validated aircraft model.
5. **Keep this prototype simulator-only.** Do not add hardware drivers, physical actuator interfaces, real-aircraft I/O, autopilot connectivity, flight-controller firmware output, or instructions for connecting this code to an aircraft. All control outputs must terminate at a simulator adapter or a test double.
6. **Keep adaptive learning disabled by default.** The default mode is `off`. `shadow` mode may calculate and log proposed changes but must not alter control outputs. An explicitly selected `sim_experiment` mode may alter simulated behaviour only after the relevant test gates pass.
7. **Keep safety logic outside plastic weights.** The ActionArbiter, envelope checks, actuator bounds, fault fallback and learning gate must be deterministic, inspectable and not trainable by Hebbian updates.
8. **Do not invent flight procedures or aircraft limits.** Stall/spin labels, thresholds, permitted envelopes and recovery profiles must come from the selected simulator aircraft's documented model/configuration or clearly labelled synthetic fixtures. If unavailable, represent the state as `unknown` and keep the recovery action as a test stub.
9. **Do not claim success without evidence.** Every milestone report must include commands run, tests passed/failed, known gaps, relevant logs and the exact commit/hash or workspace state.
10. **Do not silently change this design.** If implementation constraints require a change, create a short Architecture Decision Record (ADR), state the consequence and continue with the safest compatible option.

## 0.2 Required first action

Perform **Milestone M0 only**. Produce `docs/implementation/discovery-report.md` containing:

- repository tree at the relevant source depth;
- language, compiler/runtime, build commands and test commands;
- simulator name/version and the aircraft/model currently used;
- available raw sensor fields, units, coordinate frames and update rates;
- available ground-truth state fields;
- available control command and actuator-response fields;
- whether force feedback is simulated or merely planned;
- current architecture and reusable modules;
- exact missing inputs needed for a real aircraft-specific classifier (without blocking synthetic tests);
- a proposed mapping from this plan's modules to the existing repository;
- a list of any deviations requiring an ADR.

Do not start a large implementation before M0 has been reviewed. If no simulator exists, recommend the smallest mock/synthetic test harness that fits the repository; do not create a new six-degree-of-freedom aircraft model and call it validated.


# 1. Product objective and boundaries

## 1.1 Objective

Build a research prototype in which specialised, explicitly multidimensional tensors classify aircraft state and intent, propose actions, and learn associations through reward-modulated Hebbian plasticity. Validate the architecture in a repeatable flight simulator or a deterministic synthetic plant.

The prototype should answer these questions experimentally:

- Can the specialised perception tensors produce useful and confidence-calibrated classifications?
- Does a separate EnergyTensor add information beyond FlightTensor?
- Does a temporal ManeuverTensor improve recognition of manoeuvres and departures?
- Can Mission and Intuition propose different objectives without an unstable learned competition?
- Can deterministic arbitration resolve those proposals safely and explainably?
- Can a conventional baseline controller provide stable behaviour before learning is enabled?
- Can eligibility-trace Hebbian adaptation improve selected simulation metrics without degrading safety metrics?
- Does anti-Hebbian competition improve representation quality, or does it harm useful associations?

## 1.2 Non-goals

The first prototype is **not**:

- a certified flight-control system;
- an aircraft autopilot;
- a human-rated or hardware-in-the-loop control system;
- a replacement for an aircraft's approved flight manual or recovery procedures;
- proof that Hebbian learning is safe for aircraft;
- an LLM-controlled flight system;
- a dense tensor containing every possible combination of every class.

No experimental result from the simulator may be presented as evidence of real-aircraft safety or certification.

## 1.3 Definition of done for v0.1

The v0.1 prototype is complete when:

1. the sensor and tensor contracts are implemented and validated;
2. Location, Flight, Energy, Controls and Maneuver outputs are reproducible from recorded frames;
3. Mission and Intuition emit typed proposals, not actuator commands;
4. deterministic arbitration handles agreement, disagreement, hazards, low confidence and faults;
5. Servo operates through a simulator-only bounded interface and a non-learning baseline controller;
6. actuator response and post-action outcomes can be paired to the action that caused them;
7. Hebbian eligibility traces and reward modulation work in `shadow` mode with deterministic replay;
8. experiment runs compare learning-off, Hebbian and anti-Hebbian variants against fixed baselines;
9. reports include metrics, seed lists, configuration hashes and known limitations.


# 2. Architecture to implement

## 2.1 Dataflow

Implement the following logical flow. It is a dependency graph, not a requirement that every signal pass through one serial layer.

1. **Mission path:** a typed MissionRequest enters MissionTensor and becomes a MissionProposal. An optional language parser may create a candidate request, but it cannot bypass schema validation or approval.
2. **Perception path:** raw simulator frames feed LocationTensor, FlightTensor, EnergyTensor, ControlsTensor and ManeuverTensor. Each emits typed class activations, confidence and validity.
3. **Protective proposal:** IntuitionTensor observes those perception outputs and emits an IntuitionProposal. It does not receive the requested Servo command.
4. **Arbitration:** ActionArbiter receives the MissionProposal, IntuitionProposal and safety/validity state. It emits one AcceptedReference or a fallback/abstention decision.
5. **Actuation in simulation:** ServoTensor maps the AcceptedReference and current perceived state to a bounded simulated demand. The existing/baseline simulator controller applies it to the simulator plant.
6. **Feedback:** the next sensor frame returns to the perception path. Independently measured actuator state feeds ActuatorFeedbackTensor; commanded values and measured values remain separate.
7. **Learning path:** ActuatorFeedbackTensor, state-before/state-after and reference data feed ErrorTensor/OutcomeEvaluator. PlasticityGate decides whether a signed learning signal may update eligible associations. Weight updates affect simulator maps only, and only in the explicit experiment mode.

**Important signal rule:** IntuitionTensor must not receive ServoTensor's requested output as an input feature. It receives observations from perception. If the aircraft/simulator exposes measured surface position, actuator force, actuator health or tracking error, those are observations and may be exposed through a dedicated sensor/feedback tensor. Never substitute the commanded value for the measured value.

Mission and Intuition may disagree. Do not resolve this by allowing two learned weight sets to fight directly over physical output. They produce proposals; a deterministic ActionArbiter chooses or constrains the accepted reference. Log both proposals and the decision provenance for later learning and analysis.

## 2.2 Tensor shape notation

Preserve the project's explicit notation:

```text
A x B -> C x D
```

The axes before `->` are input axes; the axes after it are output axes. The arrow does not imply an inverse, transpose, gradient or one-to-one mapping.

For several forward mappings:

```text
A: A x B -> C x D
B: E x F -> G x H

A^: C x D x G x H -> A x B x E x F
```

This is a valid conceptual shape for a shared reverse/error mapping. Do **not** materialise a single dense Cartesian tensor of this size. Implement local/factorised connections, sparse active features and eligibility traces. A reverse mapping may be learned independently and need not be the mathematical inverse of the forward mappings.

Every axis must have a stable semantic identifier, cardinality, activation semantics, units or reference frame where relevant, and an explicit `unknown`/invalid policy.

## 2.3 Runtime signal contract

Use the equivalent of the following typed contract in the repository's dominant language. Adapt syntax, not semantics.

```text
AxisSpec {
    id: AxisId
    labels: ordered list<ClassLabel>
    cardinality: integer
    kind: Categorical | MultiLabel | Cyclic | ScalarBasis
    units: optional Unit
    reference_frame: optional FrameId
    unknown_policy: ExplicitUnknown | ValidityMask
}

AxisActivation {
    axis_id: AxisId
    values: float[cardinality]       // finite, normally in [0, 1]
    valid: bool
    confidence: float                // [0, 1]
    observed_at: SimTime
    source_ids: list<SensorId>
}

TensorFrame {
    tensor_id: TensorId
    frame_id: FrameId
    sim_time: SimTime
    axes: map<AxisId, AxisActivation>
    model_version: ModelVersion
}

TensorSpec {
    tensor_id: TensorId
    input_axes: ordered list<AxisRef>
    output_axes: ordered list<AxisSpec>
    model_kind: Encoder | FactorisedMap | SparseAssociativeMap
}
```

Implementation requirements:

- preserve raw sensor values in the observation frame; classification is an additional representation, not destructive quantisation;
- preserve timestamps, validity, confidence, units, source and reference frame;
- reject NaN/Inf, out-of-range confidence, wrong axis cardinality and stale frames;
- distinguish `zero activation` from `missing/invalid`;
- do not silently turn a failed sensor into a neutral class;
- validate that every tensor's runtime inputs match its declared `input_axes`;
- validate that every produced axis matches the declared output cardinality;
- attach model/configuration version to outputs so replay is reproducible.

For categorical axes, activations normally sum to 1. For multi-label hazard axes, activations are independent scores and need not sum to 1. Declare the rule in `AxisSpec`; do not infer it from the number of classes.

## 2.4 Representation and memory constraints

A full dense mapping over all input-class combinations and all output-class combinations will grow exponentially. Do not implement that as the production representation.

Use a staged representation:

1. **Reference implementation:** tiny dense maps in unit tests only, to verify tensor indexing and update equations.
2. **v0.1 runtime:** factorised or sparse mappings. Preserve named input axes and named output axes. Store only active/connected associations and declared interaction factors.
3. **Learning:** update only eligible synaptic associations, not every possible Cartesian combination.
4. **Diagnostics:** provide a way to inspect a connection by semantic source axis/class, target axis/class, context and model version.

A suitable first runtime option is a sparse associative edge bank:

```text
Synapse {
    source_feature: FeatureId       // tensor + axis + class
    target_feature: FeatureId       // output tensor + axis + class
    context_key: optional ContextId
    weight: float
    eligibility: float
    last_updated: SimTime
}
```

Use top-K or thresholded active features to bound active connections. Keep the active-feature cap configurable and log truncation. If the repository already has a stable tensor-network implementation, reuse it instead of introducing a competing representation. Record the choice in an ADR.


# 3. Tensor specifications

The class counts below are initial budgets inherited from the architecture proposal. They are not aircraft-independent physical thresholds. All continuous boundaries and class centres must be configured for the selected simulator aircraft and its sensor model.

## 3.1 LocationTensor

**Question:** Where is the aircraft in the world, relative to its navigation frame and destination?

**Inputs:** navigation/INS position; altitude and datum; heading; ground track; active waypoint/route; bearing to target; navigation-fix quality.

**Initial output axes:**

| Axis | Classes | Initial representation |
|---|---:|---|
| MapRegion | K | Configured local-map cells or route regions; K comes from the scenario map |
| AltitudeBand | 9 | Configured altitude bands, including unknown/out-of-range |
| HeadingSector | 16 | Cyclic; circular interpolation across sector boundaries |
| GroundTrackSector | 16 | Cyclic; separate from nose heading |
| BearingToTarget | 16 | Cyclic and only valid when a target exists |
| RouteRelation | 7 | Far left / left / slightly left / on route / slightly right / right / far right |
| FixConfidence | 4 | Unavailable / low / medium / high |

Do not collapse heading and ground track. Wind can make them differ. The phrase “heading north at FL100” is a useful compound classification, but it is not a complete geographic position without map position or route context.

## 3.2 FlightTensor

**Question:** What is the aircraft's instantaneous attitude and motion?

**Inputs:** ADI/attitude estimate; pitch, roll and yaw rates; heading; ASI; VSI; turn rate; slip/skid or sideslip; load factor where available.

**Initial output axes:**

| Axis | Classes | Initial labels |
|---|---:|---|
| PitchClass | 7 | Strong nose-down / nose-down / slight down / level / slight up / nose-up / strong nose-up |
| BankClass | 7 | Steep left / moderate left / shallow left / wings level / shallow right / moderate right / steep right |
| HeadingSector | 16 | Cyclic heading sectors |
| TurnRate | 5 | Rapid left / left / near zero / right / rapid right |
| VerticalMotion | 7 | Rapid descent / descent / slight descent / level / slight climb / climb / rapid climb |
| SpeedRegime | 7 | Very low through very high, relative to the aircraft/configuration profile |
| Coordination | 5 | Strong left slip / left slip / coordinated / right slip / strong right slip |
| LoadState | 5 | Low negative / low / nominal / high / very high load factor |
| Confidence | 4 | Unavailable / low / medium / high |

ASI and VSI are not attitude sensors, but they are useful shared evidence for FlightTensor and EnergyTensor. Retain raw units and do not feed only the discretised class into downstream control.

## 3.3 EnergyTensor

**Question:** What is the aircraft's energy and aerodynamic margin, and what consequences are developing?

**Inputs:** FlightTensor state; AoA or a validated AoA estimate; sideslip; load factor; engine/power indications; aircraft configuration; air density/altitude; mass/CG if available; temporal history.

**Initial output axes:**

| Axis | Classes | Initial labels |
|---|---:|---|
| KineticReserve | 5 | Very low / low / nominal / high / very high |
| PotentialReserve | 5 | Very low / low / nominal / high / very high |
| EnergyTrend | 5 | Rapid loss / loss / steady / gain / rapid gain |
| StallMargin | 7 + unknown | Comfortable / reducing / caution / warning / critical / stalled / recovering / unknown |
| WingAsymmetry | 5 + unknown | Balanced / developing left / pronounced left / developing right / pronounced right / unknown |
| DepartureState | 5 + unknown | No departure / incipient left / incipient right / established autorotation left / established autorotation right / unknown |
| PropulsiveMargin | 5 | Power deficit / low / nominal / adequate / excess |
| Confidence | 4 | Unavailable / low / medium / high |

Do not classify a stall from ASI alone. The stall label depends on exceeding critical AoA; the relationship between airspeed and stall changes with load factor, mass, configuration and other conditions. A left-wing stall/spin classification needs evidence of asymmetry and temporal rotation, such as sideslip, yaw rate, roll rate, lateral acceleration and validated aircraft-specific cues. If the required signals are absent, output `unknown` or a lower-confidence warning rather than a false definitive label.

Keep Energy separate from Flight in v0.1. Test whether it adds predictive value; do not merge it merely because it shares input sensors.

## 3.4 ControlsTensor

**Question:** What physical forces and tactile cues are acting through the controls, as felt/measured at the inceptor?

**Inputs:** measured pitch/roll/rudder force; force rate; vibration; sensor origin. This tensor does **not** infer control position from force.

**Initial input/output axes:**

| Axis | Classes | Labels |
|---|---:|---|
| PitchForce / PitchLoad | 9 input / 7 output | Strong push through neutral to strong pull |
| RollForce / RollLoad | 9 input / 7 output | Strong left through neutral to strong right |
| RudderForce / RudderLoad | 7 input / 5 output | Strong left / left / neutral / right / strong right |
| ForceRate | 5 | Rapidly increasing / increasing / stable / decreasing / rapidly decreasing |
| VibrationCue | 5 | None / light vibration / strong vibration / resistance / abnormal discontinuity |
| SensorSource | 3 | Pilot-applied force / inceptor reaction / actuator-linkage load |
| Confidence | 4 | Unavailable / low / medium / high |

Keep sensor sources separately observable. Do not merge a pilot's applied force with actuator/linkage reaction force unless the physical sensor design actually measures the combined force. Define sign convention, axis orientation, mounting position and calibration tests before training. If there is no force sensor model in the simulator, use a clearly labelled synthetic signal generator for unit/integration tests only.

## 3.5 ManeuverTensor

**Question:** What temporal manoeuvre or departure pattern is developing?

**Inputs:** LocationState, FlightState, EnergyState and three configurable temporal windows (short, medium, long). Use history, not a single frame.

**Initial output axes:**

| Axis | Classes | Labels |
|---|---:|---|
| VerticalMode | 3 | Descend / level / climb |
| TurnDirection | 3 | Left / straight / right |
| TurnSeverity | 4 | None / shallow / moderate / steep |
| AttitudeRegime | 3 | Upright / unusual / inverted |
| ManeuverPattern | 10-12 | Straight / climb / descent / level turn / climbing turn / descending turn / steep turn / spiral / loop / roll / upset / departure |
| PatternPhase | 5 | Entry / developing / peak or inverted where relevant / exit or recovery / completed |
| Confidence | 4 | Unavailable / low / medium / high |

The temporal windows are configuration values expressed in simulation time, not hard-coded frame counts. Include state-transition and hysteresis tests to avoid rapidly flickering between manoeuvre labels.

## 3.6 MissionTensor

**Question:** What task has been requested, and what reference would satisfy it?

**Input:** a typed `MissionRequest`, plus current Location and Flight classifications. Natural language is not a runtime control signal.

**MissionRequest minimum fields:**

```text
MissionRequest {
    request_id
    intent: enum
    target: { waypoint_id | coordinate, coordinate_frame }
    target_altitude: { value, unit, datum }
    target_track: { value, unit, reference }
    target_speed: { value, unit, speed_type }
    route_constraints: list<Constraint>
    allowed_modes: list<MissionMode>
    priority: enum
    valid_from
    expires_at
    issuer/source metadata
    operator_approval state
}
```

**Initial output axes:** MissionMode(9), DesiredTrack(16), DesiredAltitudeDelta(7), DesiredSpeedRegime(7), RoutePhase(5), Priority(3), Confidence(4).

MissionTensor must output a **reference/proposal**, not low-level servo or surface commands. If an LLM adapter is added, it may only convert human language into a candidate structured request. Validate schema, ranges, coordinate frame, expiry and authority. The LLM must not be called from the real-time simulation control loop and must not bypass operator approval or deterministic validation.

## 3.7 IntuitionTensor

**Question:** Given the perceived state and learned experience, what hazard or protective response should be proposed?

**Inputs:** Location, Flight, Energy, Controls and Maneuver outputs, including confidence/validity. It may receive independently measured actuator-health/response information as an observation, but must not receive ServoTensor's requested command as a feature.

**Initial output axes:**

| Axis | Classes | Initial labels |
|---|---:|---|
| HazardClass | 10 + unknown | Nominal / low-energy trend / stall warning / stall / asymmetric stall / incipient spin / established departure / unusual attitude / actuator or sensor fault / other configured hazard / unknown |
| Severity | 5 | None / advisory / caution / warning / critical |
| RecoveryProfile | 8-12 | Aircraft-profile-specific named profiles; unconfigured profiles must not issue action |
| Urgency | 4 | Monitor / prepare / act / immediate |
| Confidence | 4 | Unavailable / low / medium / high |
| AuthorityRequest | 3 | None / constrain mission / request protective override |

RecoveryProfile names are identifiers in a catalogue, not hard-coded generic flight instructions. Each configured profile must be supplied by the chosen simulation model or clearly labelled test fixture. An unrecognised profile is rejected by the arbiter.

## 3.8 ActionArbiter (deterministic component, not a trainable tensor)

**Question:** Which proposal is permitted now?

Implement a deterministic state machine with explicit reason codes. Initial priority order:

1. Reject any output that violates hard simulator envelope, actuator bounds, validity or freshness constraints.
2. If a high-confidence critical Intuition hazard is validated and a configured protective profile exists, select it and suspend/constrain Mission.
3. If sensor confidence is low, sensors disagree or a critical input is stale, select a configured fallback/abstain behaviour and disable plasticity.
4. If an advisory/caution exists, constrain the Mission reference to the permitted envelope.
5. Otherwise accept the feasible Mission reference.
6. If no feasible reference exists, abstain or use the configured safe simulator fallback.

The exact thresholds are configuration values tied to the selected simulation aircraft. The arbitration priority itself is deterministic and cannot be modified by online Hebbian learning.

Log Mission proposal, Intuition proposal, agreement/disagreement, selected source, rejected proposal(s), constraints applied and reason code.

## 3.9 ServoTensor and baseline controller

**Question:** Given the accepted reference and current perceived state, what bounded simulated control demand should be requested?

**Inputs:** AcceptedReference, FlightState, EnergyState and measured actuator health/authority summary where available.

**Output axes:** bounded PitchDemand(7), RollDemand(7), YawCoordinationDemand(5), PowerDemand(7), or a compatible continuous setpoint representation already used by the simulator.

For v0.1, ServoTensor is a **supervisory policy**. The simulator's existing conventional controller or a separately tested baseline control layer translates its reference into simulated actuator demand. Do not make the Hebbian network the only stabilising controller. If the repository's existing architecture requires Servo to emit lower-level control signals, keep them bounded by an independent simulator-side guard and document the ADR.

## 3.10 ActuatorFeedbackTensor

**Question:** Did the simulator's actuators respond as commanded, and what condition are they in?

**Inputs:** Servo demand (for comparison only), measured simulated surface/inceptor position, position rate, actuator current/torque/load, saturation and aircraft response.

**Outputs:** CommandTracking(5), Saturation(3), FaultClass(6), ControlAuthority(5), Confidence(4).

This component exists to distinguish a requested command from an achieved physical response. Its measured output may feed Error/Outcome and the safety monitor. It may supply a measured health summary to Intuition, but must never leak the requested Servo command as if it were a sensor observation.

## 3.11 ErrorTensor and OutcomeEvaluator

ErrorTensor compares reference, state before, state after, actual actuator response and the configured safety envelope.

Initial output axes: TrackError(7), AltitudeError(7), SpeedError(7), AttitudeError(7), EnergyError(7), SafetyDelta(5), ActionEffect(5), Confidence(4).

OutcomeEvaluator aggregates short- and longer-horizon evidence into a vector, not one opaque scalar:

```text
OutcomeVector {
    safety_margin_delta
    critical_envelope_event
    mission_progress_delta
    tracking_error_delta
    stability_or_oscillation_delta
    energy_margin_delta
    actuator_saturation_or_fault
    control_effort_cost
    confidence_and_data_quality
}
```

Safety is lexicographic: a gain in mission progress or reduced control effort must never compensate for a critical safety violation. A scalar plasticity modulator may be derived only after hard safety and data-quality gates pass.

## 3.12 PlasticityGate

The PlasticityGate determines whether an outcome is valid for learning. It is deterministic and non-trainable.

Initial gate conditions: valid timestamps; sufficient sensor confidence; no unresolved sensor disagreement; valid action-to-outcome pairing; no actuator fault that invalidates attribution; configured simulator-only experiment mode; no hard safety event that requires learning to be frozen; and a complete eligibility window.

When a gate condition fails, do not update weights. Log why learning was blocked.


# 4. Analogue values, class encoding and time

## 4.1 Preserve continuous values

Every input axis must retain the raw continuous measurement and its unit. The class representation is an additional semantic channel. Never discard raw pitch, roll, speed, altitude, AoA, force or engine readings after classification.

## 4.2 Soft classes for scalar values

Use overlapping membership functions rather than hard threshold bins. A suitable initial method is piecewise-linear triangular membership across configured class centres:

- between two neighbouring centres, activate both proportionally;
- clamp or use explicit underflow/overflow classes at the ends;
- include `unknown` separately from a physical zero;
- normalise only when the axis is declared categorical;
- preserve the raw value for downstream numerical control and evaluation.

The physical class centres and limits come from the simulator aircraft profile and dataset. Do not copy generic numeric thresholds from another aircraft.

## 4.3 Cyclic values

For heading, track and bearing, use a circular representation. Interpolate between adjacent sectors across the 0/360-degree boundary. A north value near 359 degrees must be close to a north value near 1 degree, not maximally distant.

## 4.4 Multi-label conditions

Hazards may coexist. Represent hazard outputs as independent class scores or multi-label activations rather than a softmax that forces exactly one hazard to be true. `unknown` and confidence must remain explicit.

## 4.5 Hysteresis and temporal windows

- Use a configurable hysteresis/debounce mechanism for categorical state transitions.
- Use simulation time rather than render-frame count.
- Every windowed feature must declare its time span and history requirements.
- Never mix observations with incompatible timestamps without an explicit resampling/alignment step.
- Test delayed, duplicated, dropped and out-of-order frames.

## 4.6 Configuration

Create an aircraft/simulator profile (file format should match the repository) containing:

- units and coordinate conventions;
- altitude datum and local/world coordinate frame;
- sensor scale, offset, range, noise, latency and availability;
- class centres and thresholds;
- aircraft configuration labels;
- simulator envelope and fault limits;
- temporal windows;
- configured hazard/recovery profile IDs;
- actuator bounds and baseline-controller limits.

Do not put aircraft-specific physical thresholds in generic tensor code.


# 5. Pretraining and online learning

## 5.1 Stage A: independent supervised pretraining

Use supervised backpropagation (or the repository's existing supervised trainer) to initialise each tensor's encoder/classifier independently.

Required labelled targets:

- Location: navigation/route state labels;
- Flight: attitude, rate, speed regime and coordination labels;
- Energy: energy trend, AoA/stall-margin and departure labels from simulator truth/model diagnostics;
- Controls: measured force class and source;
- Maneuver: sequence label and phase;
- Intuition: hazard class, severity and a configured recovery-profile proposal;
- Mission: structured request to accepted-reference translation.

Training rules:

1. Split by complete simulation episode and scenario family, never by randomly shuffling adjacent frames from the same episode across train and test.
2. Hold out entire environmental/configuration variations where possible.
3. Report per-class precision, recall, F1, confusion matrix and confidence calibration.
4. Keep a frozen model checkpoint before online adaptation.
5. Version the dataset, labels, profile and model independently.
6. Do not train on simulator truth fields unavailable to the runtime sensor interface. Truth is a label/evaluation channel, not a live input.

At first, freeze pretrained perception and policy maps during closed-loop testing. This isolates errors in the architecture from online adaptation.

## 5.2 Stage B: reward-modulated Hebbian learning

Do not use simple co-activation as a correctness test. Co-activation creates an eligibility trace; a later signed outcome signal determines whether that association is strengthened or weakened.

For source activity `x_i`, target activity `y_j`, simulation-time step `dt` and eligibility time constant `tau_e`:

```text
lambda(dt) = exp(-dt / tau_e)
e_ij(t) = lambda(dt) * e_ij(t-dt) + x_i(t) * y_j(t)
```

For a valid signed outcome modulator `delta(t)`:

```text
weight_delta_ij = learning_rate * delta(t) * e_ij(t)
weight_ij = clamp(weight_ij + weight_delta_ij, min_weight, max_weight)
```

Use a separate trace per eligible connection. `delta` must be confidence-gated and associated with a defined action/outcome window. Weight changes must be logged with the contributing trace and outcome record.

Because the result of an action may appear later, use at least two eligibility time constants (short and long) for experiments. Select actual durations from simulator response times and measured delays, not arbitrary real-time assumptions.

## 5.3 Outcome signal

The first implementation should use a transparent, deterministic outcome evaluator rather than an unconstrained learned critic.

A practical initial policy:

1. If any hard safety event occurred, freeze plasticity for the episode/window and record the event. Do not infer that every simultaneously active synapse caused the event.
2. If data quality or action attribution is inadequate, set modulator to zero.
3. Otherwise calculate separate deltas for safety margin, mission progress, state tracking, stability, energy margin and control effort.
4. Enforce lexicographic safety: only evaluate lower-priority objectives when safety remains within the configured safe region and does not deteriorate beyond its allowed tolerance.
5. Convert the eligible objective improvement to a bounded signed modulator, with a baseline or moving reference so that ordinary state drift is not automatically credited to the last action.
6. Log the vector components and the exact rule that produced the sign and magnitude.

Do not let mission progress compensate for an unsafe state. Do not treat the absence of a detected error as proof of a good action if sensor confidence is low.

## 5.4 Anti-Hebbian learning

Keep two concepts separate:

- **Outcome-based depression:** a negative third-factor `delta` weakens eligible associations that were active before an objectively poor outcome.
- **Anti-Hebbian competition:** an explicitly designated set of competing neurons/labels is decorrelated or inhibited to prevent all classes from activating together.

Implement anti-Hebbian competition only within declared groups (for example, mutually exclusive classes on one categorical output axis or competing latent units). Do not apply global anti-Hebbian depression to every co-active connection. Add homeostatic normalisation and weight bounds so activity does not collapse or saturate.

Run an ablation with anti-Hebbian competition disabled. It must demonstrate a measurable benefit before being enabled in the default experimental configuration.

## 5.5 Learning ownership

Initially plastic weights may exist only in:

- selected context-to-Intuition associations;
- accepted-reference/context-to-Servo supervisory associations;
- explicitly selected local cross-tensor mappings.

Do **not** make these components plastic in v0.1:

- sensor calibration and unit conversion;
- ActionArbiter priority logic;
- hard envelope or actuator bounds;
- safety fallback logic;
- mission authority/approval;
- simulator physics;
- output clamping and rate limits.

Keep `learning_mode` as an explicit enum: `off`, `shadow`, `sim_experiment`. Default: `off`.


# 6. Action arbitration: Mission versus Intuition

Mission and Intuition answer different questions. Mission expresses the requested objective. Intuition identifies state-dependent hazards and proposes protective responses. They are not equal-priority actuator drivers.

Record the following features for analysis:

- Mission proposal validity/confidence;
- Intuition hazard/severity/confidence;
- proposal agreement (`agree`, `neutral`, `conflict`);
- selected proposal source (`mission`, `intuition`, `fallback`);
- reference constraints applied;
- arbitration reason.

The selected source and proposal conflict may be used as contextual features for future learning, but the learned weights do not decide priority.

### Required arbiter tests

| Case | Inputs | Required result |
|---|---|---|
| A1 | Valid Mission; no hazard; sensors valid | Accept feasible Mission reference |
| A2 | Mission and Intuition agree | Accept the feasible reference; record agreement |
| A3 | Mission conflicts with high-confidence critical hazard | Select configured protective profile; suspend/constrain Mission |
| A4 | Advisory/caution only | Constrain Mission to permitted envelope; do not invent a critical override |
| A5 | Low confidence or sensor disagreement | Abstain/fallback; plasticity disabled |
| A6 | Mission request expired or malformed | Reject Mission request with explicit reason |
| A7 | Unknown recovery profile | Reject profile; use configured fallback |
| A8 | Proposed reference outside hard envelope | Reject or clamp by deterministic safety rule; record event |
| A9 | Actuator fault/saturation | Select configured simulator fallback; freeze affected learning paths |
| A10 | Repeated contradictory proposals | Deterministic result for identical inputs; no oscillation due to hidden weight updates |


# 7. Recommended repository structure

Adapt this to the existing codebase; do not force a new directory tree if the project has established conventions.

- `src/brain/core/`: `axis_spec`, `axis_activation`, `tensor_spec`, `tensor_frame`, `sim_time`, `provenance`.
- `src/brain/tensors/`: `location`, `flight`, `energy`, `controls`, `maneuver`, `mission`, `intuition`, `servo`, `actuator_feedback`, `error`.
- `src/brain/mapping/`: `encoder_interface`, `factorised_map`, `sparse_associative_map`, `active_feature_set`.
- `src/brain/arbitration/`: `action_arbiter`, `envelope_guard`, `fallback_policy`.
- `src/brain/learning/`: `eligibility_trace`, `reward_modulated_hebbian`, `anti_hebbian_competition`, `plasticity_gate`, `weight_store`.
- `src/brain/evaluation/`: `outcome_evaluator`, `metrics`, `experiment_runner`.
- `src/brain/adapters/`: `simulator_adapter`, `sensor_adapter`, `mission_request_adapter`, `dataset_adapter`.
- `src/brain/logging/`: `event_log`, `episode_record`, `model_checkpoint`.
- `config/aircraft_profiles/`, `config/tensor_classes/`, `config/learning/`: aircraft-specific and learning configuration.
- `data/schemas/`, `data/fixtures/`, `data/generated/`: schemas, small fixtures and generated datasets (do not commit large generated datasets by default).
- `experiments/configs/`, `experiments/scenarios/`, `experiments/results/`: reproducible experiment inputs and outputs.
- `docs/implementation/`, `docs/adr/`: progress, limitations, discovery report and architecture decisions.
- `tests/unit/`, `tests/integration/`, `tests/replay/`, `tests/properties/`: automated test suites.

If the project is Rust, use typed enums/structs and module boundaries consistent with the existing crate layout. If C++, use value types and explicit ownership/lifetimes consistent with the project. Do not add a second language runtime merely to implement the tensor engine.


# 8. Required data and event formats

## 8.1 SensorFrame

Each simulator observation must be convertible to a versioned `SensorFrame`:

```text
SensorFrame {
    episode_id
    frame_id
    sim_time
    dt
    raw: {
        position, altitude, altitude_datum,
        heading, ground_track,
        pitch, bank, yaw, pitch_rate, roll_rate, yaw_rate,
        airspeed, airspeed_type, vertical_speed,
        aoa, sideslip, load_factor,
        engine_power_indications, aircraft_configuration,
        control_forces, force_sensor_sources,
        measured_actuator_positions, actuator_rates,
        actuator_load_or_torque, vibration
    }
    validity: map<SensorId, Validity>
    units: map<SensorId, Unit>
    reference_frames: map<SensorId, FrameId>
    confidence: map<SensorId, float>
    simulator_truth: optional TruthFrame  // labels/evaluation only
}
```

Fields absent from the selected simulator must be omitted/invalid explicitly. Never synthesize plausible-looking real sensor values silently.

## 8.2 Control and action event

```text
ControlDecision {
    episode_id
    decision_id
    sim_time
    mission_proposal_id
    intuition_proposal_id
    selected_source
    arbitration_reason
    accepted_reference
    servo_request
    learning_mode
    model_versions
}
```

## 8.3 Outcome event

```text
OutcomeEvent {
    episode_id
    decision_id                 // exact pairing to the initiating decision
    evaluation_window_start
    evaluation_window_end
    state_before_id
    state_after_id
    measured_actuator_feedback_id
    error_axes
    outcome_vector
    plasticity_modulator
    plasticity_gate_result
    gate_reason
}
```

Pair outcomes to the initiating decision using IDs, not nearest timestamps alone. If several actions overlap in time, the agent must not pretend that a simple one-action/one-outcome attribution is proven. Mark ambiguous attribution and suppress learning until a suitable multi-action attribution method is implemented.

## 8.4 Replay and audit logging

Write deterministic structured logs (JSONL or the project's equivalent). At minimum log:

- episode/seed/configuration/data/model hashes;
- raw sensor frame IDs and validity;
- tensor class activations and confidence;
- Mission and Intuition proposals;
- arbitration decision and reason;
- Servo request and actual measured simulator response;
- error/outcome vector;
- plasticity gate and reason;
- eligibility trace values for changed connections;
- before/after weights and learning-rule version.

Support replay of an episode with learning disabled, shadow learning and simulated learning using the same initial conditions and seed.


# 9. Milestones and agent task list

Each milestone has a hard exit gate. Do not start the next milestone when the current gate fails.

## M0 - Repository and simulator reconnaissance

**Tasks**

- inspect repository, build/test commands and existing architecture;
- identify the simulator, selected aircraft and available telemetry;
- inspect control output and measured actuator response;
- document available force feedback and whether it is physical, simulated or absent;
- map this plan to existing source modules;
- list gaps and ADRs.

**Deliverable:** `docs/implementation/discovery-report.md`.

**Exit gate:** report is complete; no unverified assumption is written as fact. No major code change in this milestone.

## M1 - Core tensor contracts and deterministic time

**Tasks**

- implement AxisSpec, AxisActivation, TensorSpec, TensorFrame and typed identifiers;
- implement shape/cardinality/finite-value/validity validation;
- implement simulation timestamp and frame ordering rules;
- implement configuration loading/versioning;
- implement serialization round-trip for sensor/tensor frames;
- add small reference dense tensor mapping tests for shape semantics only.

**Tests**

- wrong cardinality rejected;
- NaN/Inf rejected;
- unknown is distinct from zero;
- stale/out-of-order frame detection;
- categorical normalisation rules enforced;
- axis order preserved across serialization;
- `A x B -> C x D` is not silently flattened into an anonymous shape.

**Exit gate:** all contract tests pass and a frame can be recorded/reloaded without semantic loss.

## M2 - Simulator adapter, recording and replay

**Tasks**

- implement an adapter over the existing simulator;
- normalise units and reference frames at the boundary;
- retain raw observations and optional simulation truth labels separately;
- implement deterministic episode reset/seed where simulator permits;
- implement recorder and replay reader;
- add a mock adapter only for fields absent from the simulator.

**Tests**

- deterministic replay yields identical frame IDs and values;
- absent sensor is invalid, not fabricated;
- unit and frame conversion tests;
- actuator command and measured response remain separate fields.

**Exit gate:** one baseline episode can be recorded and replayed exactly within the simulator's deterministic tolerance.

## M3 - Axis encoding and initial perception tensors

**Tasks**

- implement scalar soft-class encoding;
- implement cyclic sector encoding;
- implement multi-label hazard encoding;
- implement confidence and unknown policies;
- implement LocationTensor, FlightTensor and ControlsTensor;
- load thresholds and class centres from profile configuration;
- produce per-episode confusion matrices and classification metrics.

**Tests**

- values at class centres;
- values between centres activate adjacent classes;
- wraparound at 0/360 degrees;
- force polarity/sign tests for all axes;
- sensor invalidity propagates to confidence/unknown;
- no use of simulator truth as runtime input.

**Exit gate:** deterministic golden fixtures pass; held-out dataset metrics are reported; no class thresholds are silently hard-coded in generic code.

## M4 - EnergyTensor and ManeuverTensor

**Tasks**

- implement EnergyTensor inputs and output axes;
- use AoA/configuration/load/power/temporal evidence according to available simulator fields;
- implement ManeuverTensor using multi-scale temporal windows;
- explicitly represent unavailable AoA/sideslip and unknown departure state;
- implement temporal phase and hysteresis.

**Tests**

- same IAS with different AoA/load/configuration does not force the same stall-margin output;
- unknown AoA does not produce high-confidence stall classification;
- directional/asymmetric departure requires the configured evidence;
- manoeuvre labels use sequence context;
- adjacent time windows and frame loss do not corrupt the state machine.

**Exit gate:** held-out test episodes distinguish Flight state from Energy interpretation and temporal Maneuver patterns; uncertainty is surfaced honestly.

## M5 - MissionTensor and IntuitionTensor

**Tasks**

- implement schema-validated MissionRequest;
- implement MissionTensor reference proposal;
- implement IntuitionTensor hazard/protective-profile proposal;
- enforce the interface rule that Intuition has no direct Servo-command input;
- provide fixed labelled fixtures for each configured hazard/profile;
- optional: implement an LLM-to-MissionRequest parser adapter behind a feature flag, with schema validation and approval. This is not required for v0.1.

**Tests**

- malformed, expired, infeasible and unauthorised MissionRequest rejected;
- LLM/raw text cannot directly reach Servo;
- Intuition output is a proposal, never a servo command;
- Intuition input contract contains perception/measurements only;
- unknown hazard/profile produces abstention or fallback.

**Exit gate:** both tensors are independently testable and their proposals are logged with confidence and provenance.

## M6 - Deterministic ActionArbiter and safety guard

**Tasks**

- implement priority state machine from Section 6;
- implement explicit envelope/validity/freshness checks;
- implement simulator fallback and reference constraints;
- add machine-readable decision reason codes;
- ensure the arbiter cannot be updated by plasticity.

**Tests**

- all arbiter cases A1-A10;
- property tests: unsafe references are never forwarded;
- identical inputs always yield identical decisions;
- low-confidence critical signals do not become high-confidence overrides;
- no learned weight or random seed changes arbitration priority.

**Exit gate:** 100% of deterministic arbiter/property fixtures pass. The arbiter is isolated from the learning API by type/module boundary.

## M7 - ServoTensor with non-learning baseline

**Tasks**

- implement AcceptedReference contract;
- implement ServoTensor as a supervisory policy with bounded outputs;
- integrate with existing simulator controller;
- add independent output clamp, rate limit and simulator-side safety guard;
- keep learning disabled.

**Tests**

- command units and axes are explicit;
- demand bounds and rate limits always hold;
- controller is deterministic for identical state/configuration;
- nominal tracking scenarios have recorded baseline metrics;
- invalid input yields configured fallback, not undefined output.

**Exit gate:** baseline closed-loop simulation runs with learning off and produces a complete action/response log. Do not proceed to online learning if the baseline is unstable or not measurable.

## M8 - Actuator feedback, ErrorTensor and OutcomeEvaluator

**Tasks**

- implement measured actuator feedback separate from requested command;
- pair action decisions to later measured responses;
- implement ErrorTensor axes;
- implement multi-objective OutcomeVector and transparent signed modulator;
- implement data-quality and action-attribution confidence;
- implement PlasticityGate with explicit block reasons.

**Tests**

- saturated actuator is not treated as successful execution;
- command/measurement mismatch is detected;
- outcome points to the correct decision ID;
- ambiguous overlapping actions suppress learning;
- safety regression cannot be hidden by mission reward;
- invalid data produces a zero modulator and blocked gate.

**Exit gate:** replay shows correct command -> measured response -> state change -> outcome association, with an auditable reason for every modulator.

## M9 - Hebbian eligibility traces and shadow learning

**Tasks**

- implement sparse association keys and active-feature selection;
- implement elapsed-simulation-time eligibility decay;
- implement signed third-factor weight update;
- implement bounded weights, normalisation and checkpoint/revert;
- implement separately configured anti-Hebbian competition groups;
- implement `off`, `shadow`, `sim_experiment` modes;
- add trace and weight-change logging.

**Tests**

- co-activity creates trace but does not change weight without a modulator;
- positive modulator increases eligible association within bounds;
- negative modulator decreases eligible association within bounds;
- zero modulator leaves weight unchanged;
- delayed reward updates only traces still eligible;
- trace decay follows simulation time, not wall-clock;
- anti-Hebbian changes only explicitly designated competition groups;
- `off` has no trace/weight change;
- `shadow` records proposed changes but actual forward outputs are byte-for-byte identical to learning-off replay;
- save/load preserves exact weights, traces and version.

**Exit gate:** unit/integration tests pass and shadow-learning replay is control-output-identical to baseline.

## M10 - Controlled experiments and ablations

**Tasks**

- implement experiment configuration files;
- implement paired-seed experiment runner;
- run fixed baseline and learning variants;
- add ablations for Intuition, Energy, Controls, Maneuver and anti-Hebbian competition;
- compute metrics and confidence intervals;
- generate a machine-readable summary and a human-readable report.

**Exit gate:** results are reproducible from logged configuration/seed hashes; negative results are retained; no run is described as successful solely because the target was reached.

## M11 - Robustness, fault and generalisation campaign

**Tasks**

- test sensor noise, delay, dropout, bias and stuck values;
- test force-sensor polarity/calibration faults;
- test actuator lag, saturation and configured faults;
- test held-out mass/CG/environment/configuration variations if the simulator supports them;
- test unknown combinations and out-of-distribution detection;
- measure forgetting and weight drift after extended adaptation.

**Exit gate:** performance and failure cases are characterised; the report specifies what the prototype cannot generalise to.


# 10. Experiment design

## 10.1 Dataset partitioning

Split by entire episodes and scenario families. Adjacent frames from one episode must never be randomly distributed across train and test. Use:

- training episodes for supervised pretraining;
- validation episodes for calibration and hyperparameter selection;
- frozen test episodes that are not used for tuning;
- held-out configuration/environment families for generalisation tests.

Persist episode IDs, scenario config hashes, seeds, aircraft profile version, label version and model version.

## 10.2 Required controller/learning variants

Run at least the following variants on the same scenario definitions and paired seeds:

| ID | Variant | Purpose |
|---|---|---|
| B0 | Existing simulator/baseline controller, no tensor learning | Reference performance |
| B1 | Perception + Mission/Intuition + deterministic arbiter + baseline Servo, plasticity off | Value of the proposed architecture without adaptation |
| H1 | B1 + eligibility-trace reward-modulated Hebbian update, no anti-Hebbian competition | Test core learning rule |
| H2 | H1 + designated anti-Hebbian competition and homeostasis | Test whether competition adds value |
| H3 | H2 + selected cross-tensor context associations | Test higher-order context effects |
| A1 | B1 with Intuition disabled (where safe in simulation) | Intuition ablation |
| A2 | B1 with Energy removed or replaced by Flight-only features | EnergyTensor ablation |
| A3 | B1 with Controls force feedback removed | ControlsTensor ablation |
| A4 | B1 with Maneuver temporal context removed | Temporal context ablation |

Do not compare H1/H2/H3 only against each other. Compare every candidate to B0 and B1. Do not let ablation variants issue unsafe commands outside the bounded simulator test harness.

## 10.3 Scenario families

Use the simulator's existing scenario vocabulary and valid aircraft configuration. At minimum, cover:

1. nominal route/altitude/speed tracking;
2. climb and descent transitions;
3. coordinated and uncoordinated turns;
4. abrupt but model-valid power/configuration changes;
5. reducing energy/stall margin and configured recovery scenarios;
6. asymmetric departure/spin-labelled scenarios only if the simulator model provides validated truth labels;
7. turbulence/wind/environment disturbance;
8. low-quality navigation or sensor dropout;
9. AoA/sideslip sensor bias, frozen value and disagreement;
10. control-force sensor offset, polarity inversion and dropout;
11. actuator lag, saturation and configured failure;
12. Mission/Intuition conflict, infeasible request and expired request;
13. out-of-distribution combinations of otherwise familiar conditions.

Scenario details and recovery references must come from the selected simulator model or synthetic test fixture. Do not invent real-aircraft recovery actions.

## 10.4 Repetition and statistics

- For stochastic scenarios, use at least 20 paired seeds per scenario/variant as an initial prototype floor. Increase repetitions when confidence intervals are wide.
- For deterministic scenarios, vary initial conditions and configured perturbations; record the full matrix.
- Use paired comparisons: identical initial condition, simulator version, scenario and seed across variants.
- Report mean, median, spread and bootstrap confidence intervals for continuous metrics; report counts and rates for rare events.
- Keep test data frozen. Do not tune thresholds on the final test set.
- Report all failed runs and exclusions with explicit reasons.

## 10.5 Metrics

**Perception:** per-class precision/recall/F1; confusion matrices; calibration error; false hazard rate; critical hazard misses; unknown-state detection; latency; timestamp mismatch.

**Mission and Intuition:** proposal accuracy; confidence calibration; disagreement frequency; hazard-to-proposal latency; proposal rejection rate; explainability/provenance completeness.

**Control:** reference tracking error; overshoot; settling time; oscillation; control effort; saturation duration; actuator tracking error; recovery time within the simulator scenario definition.

**Safety:** hard-envelope excursions; minimum configured margins; time in warning/critical states; false negatives for configured critical labels; false overrides; fallback rate; performance under sensor/actuator faults.

**Learning:** time to improve; reward/depression balance; number and magnitude of updates; eligibility trace duration; weight saturation/drift; forgetting; sensitivity to seed; benefit over B1.

**Generalisation:** performance on held-out environment/configuration/initial conditions; performance under compound faults; unknown-state behaviour.

**Auditability:** percentage of commands with complete provenance; percentage of weight updates traceable to action/outcome IDs; replay determinism.

## 10.6 Initial acceptance gates (prototype gates, not certification)

- 100% of tensor contract, unit, timestamp and deterministic arbiter tests pass.
- 100% of generated control demands satisfy configured simulator bounds and rate limits.
- All action outputs have a provenance trail to Mission, Intuition or fallback.
- Learning `off` has no weight changes; `shadow` has no change to control outputs.
- Every plasticity update has a valid eligibility trace, outcome ID, modulator, gate approval and model/configuration version.
- No critical safety metric is worse than B1 on the frozen paired test set. Any regression blocks the candidate, even if mission progress improves.
- A candidate must show a repeatable, statistically supported improvement on at least one declared target metric before it is described as beneficial.
- A zero-event finite simulator test is not a guarantee of safety. State the test scope and statistical uncertainty.

No acceptance gate authorises physical aircraft use.


# 11. Required automated tests

Organise the tests in the repository's existing style. These cases are minimum required coverage.

## 11.1 Unit tests

- axis cardinality and label order;
- soft scalar membership functions;
- cyclic angular interpolation at north wraparound;
- multi-label output semantics;
- unknown/invalid propagation;
- unit/reference-frame conversion;
- timestamp ordering and stale-frame rejection;
- Maneuver temporal windows and phase transitions;
- MissionRequest schema, expiry and authority;
- Intuition input contract excludes Servo request;
- arbiter priority and reason codes;
- command bounds and rate limits;
- measured actuator response vs requested demand;
- ErrorTensor before/after sign conventions;
- eligibility decay as a function of sim-time delta;
- positive, negative and zero modulator weight changes;
- anti-Hebbian group scope;
- checkpoint and replay determinism.

## 11.2 Property tests

- no non-finite tensor output;
- no output cardinality mismatch;
- cyclic encoding is continuous across 0/360;
- changing a stale or invalid sensor cannot increase confidence;
- no unsafe reference reaches Servo output;
- learning mode `off` cannot mutate any weight;
- learning mode `shadow` cannot modify control outputs;
- weights remain inside configured bounds;
- hard safety logic is unchanged by all possible weight updates;
- identical input frames and configuration yield identical arbitration results.

## 11.3 Integration and replay tests

- full episode with all learning disabled;
- same episode replayed in shadow mode, with identical Servo output;
- simulated-learning episode with update trace and checkpoint;
- mission conflict with protective proposal;
- sensor fault with learning blocked;
- actuator saturation with error/outcome correctly attributed;
- saved episode can be re-run with the same model/configuration hashes.


# 12. Operational modes and configuration defaults

Use explicit, validated configuration. The following are logical defaults; tune values only through documented experiments.

```yaml
brain:
  mode: simulator_research
  learning_mode: off        # off | shadow | sim_experiment
  deterministic_replay: true
  allow_hardware_io: false  # must remain false in this project

learning:
  enabled: false
  eligibility:
    short_tau_sim_seconds: configured_from_experiments
    long_tau_sim_seconds: configured_from_experiments
  hebbian:
    learning_rate: configured_from_experiments
    min_weight: configured_from_experiments
    max_weight: configured_from_experiments
  anti_hebbian:
    enabled: false
    groups: []
  plasticity_gate:
    require_valid_outcome: true
    require_sensor_confidence: true
    freeze_on_critical_event: true
    freeze_on_actuator_fault: true

safety:
  arbiter_trainable: false
  envelope_guard_enabled: true
  actuator_bounds_enabled: true
  unknown_state_policy: abstain_or_fallback

logging:
  record_raw_frames: true
  record_tensor_activations: true
  record_decisions: true
  record_outcomes: true
  record_weight_changes: true
```

The placeholder values must be replaced by explicit profile values before a learning experiment can run. The configuration loader must fail closed on missing values required by the selected mode.


# 13. Agent deliverables and reporting format

At the end of each milestone, create or update:

1. `docs/implementation/progress.md` - milestone status, completed tasks, test commands/results, blockers and next step;
2. `docs/implementation/decisions.md` - links to ADRs and important choices;
3. `docs/implementation/known-limitations.md` - unavailable sensors, unvalidated model assumptions and unsupported scenarios;
4. experiment reports under `experiments/results/<run-id>/` containing config, seeds, metrics, plots and summary;
5. checkpoints under the project's established model-artifact location, with dataset/config/model version metadata.

Milestone report template:

```text
Milestone: Mx - Name
Repository state / commit:
Changes made:
Files added or modified:
Commands executed:
Tests passed:
Tests failed:
Experiment runs and configuration hashes:
Observed results:
Known limitations:
ADR(s) created:
Exit gate: PASS / FAIL
Next allowed milestone:
```

A test is not passed merely because a build succeeded. A report is not complete without failed tests, warnings and limitations.


# 14. Suggested initial experiment sequence

After M9 passes, run experiments in this order. Do not jump directly to full plasticity.

1. **Replay baseline:** replay a fixed episode with learning off; verify deterministic output and complete logs.
2. **Perception-only evaluation:** evaluate classifiers against ground truth without closing the control loop.
3. **Arbiter scenario suite:** run A1-A10; verify exact proposal selection and reason codes.
4. **Closed-loop B1:** run perception, arbitration and baseline Servo with plasticity off. Establish baseline metrics.
5. **Shadow H1:** calculate eligibility traces and proposed weight changes while preserving B1 control outputs exactly.
6. **Trace timing sweep:** compare short/long eligibility constants and delayed outcome windows using identical episodes.
7. **Reward sign tests:** create controlled fixtures where an association is followed by a clearly improved, worsened or neutral outcome; verify update direction and no-update conditions.
8. **Simulated H1:** only after all prior gates pass, let the learned weights affect simulator outputs under the simulator guard. Compare paired seeds with B1.
9. **H2 anti-Hebbian ablation:** add only designated competition groups; compare with H1 and B1.
10. **Cross-tensor context H3:** add one interaction family at a time; do not add a large full-product mapping.
11. **Ablations:** remove Energy, Controls, Maneuver and Intuition one at a time to estimate contribution.
12. **Fault/generalisation campaign:** only after normal scenarios are stable and reproducible.

Keep all run configs immutable. Create a new experiment ID for any parameter change.


# 15. Research basis and implementation interpretation

The implementation is informed by the following research. These works motivate components of the prototype; none demonstrates that this exact tensor architecture is safe for aircraft control.

- **Verduzco-Flores, Dorrell and De Schutter (2022), “A differential Hebbian framework for biologically-plausible motor control,” Neural Networks 150, 237-258.** Demonstrates local differential-Hebbian feedback control for selected tasks and discusses combining it with reinforcement learning where control/error relationships are non-monotonic. This motivates testing local association learning and context-dependent feedback rather than assuming one fixed sign for every control-to-error relationship. DOI: https://doi.org/10.1016/j.neunet.2022.03.002
- **Gerstner et al. (2018), “Eligibility Traces and Plasticity on Behavioral Time Scales,” Frontiers in Neural Circuits 12:53.** Reviews three-factor learning in which pre/post co-activity creates an eligibility trace and a later modulatory signal determines whether plasticity occurs. This motivates delayed, signed reward modulation. DOI: https://doi.org/10.3389/fncir.2018.00053
- **Hageman, Smith and Stachowiak (2003), NASA/TM-2003-212028, “Integration of Online Parameter Identification and Neural Network for In-Flight Adaptive Control.”** Describes a layered adaptive-control research system combining a pretrained network, online parameter identification and model-following control. It motivates a layered architecture with a conventional control baseline, not direct unconstrained actuator learning. https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20040000923.pdf
- **NASA F-15 Intelligent Flight Control System flight-test reports.** These document a research programme involving neural adaptive control, system identification and separate control/safety structures. They are historical research precedents, not an approval for this design. https://ntrs.nasa.gov/citations/20040105654
- **EASA/Daedalean (2020), Concepts of Design Assurance for Neural Networks (CoDANN).** Motivates representative datasets, generalisation analysis, robustness, verification and explicit handling of incorrect model outputs in aviation-related ML. https://www.easa.europa.eu/en/document-library/general-publications/concepts-design-assurance-neural-networks-codann
- **EASA (3 June 2026), Artificial Intelligence Concept Paper - Proposed Issue 3.** Relevant emerging guidance on reinforcement learning and advanced automation; it is explicitly a proposed concept-paper issue, not final binding certification guidance. https://www.easa.europa.eu/en/document-library/general-publications/easa-artificial-intelligence-concept-paper-proposed-issue-3
- **FAA Airplane Flying Handbook, Chapter 5, and AC 61-67C.** Provide aviation background for stall and upset/spin recognition. Use the selected aircraft's approved documentation and simulator truth for profile-specific labels and procedures. https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/airplane_handbook/06_afh_ch5.pdf and https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_61-67C.pdf


# 16. Final instruction to Astra

Implement this as a measured research prototype, not as a one-shot code-generation exercise.

**Start with M0.** Inspect the repository and report what already exists. Then implement one milestone at a time, preserving the project's language and conventions. Use the existing simulator and aircraft model where possible. Keep unknown aircraft-specific parameters explicit. Keep the ActionArbiter and safety bounds deterministic. Keep learning off by default. Prove replay, attribution and shadow-mode invariants before allowing weights to influence simulated control outputs.

When a design ambiguity remains, prefer the simplest reversible implementation, record the assumption in an ADR, and do not conceal the uncertainty. The target is an architecture whose behaviour, learning and failures can all be inspected and reproduced.
