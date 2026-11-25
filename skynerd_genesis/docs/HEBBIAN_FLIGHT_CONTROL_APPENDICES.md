# Temporal Dynamics and STDP Learning Appendices

**Companion:** `HEBBIAN_FLIGHT_CONTROL_AGENT_PLAN.md`  
**Version:** 1.2  
**Date:** 25 November 2026  
**Status:** Research-backed implementation addendum for a simulator-only prototype  
**Implementation environment:** TypeScript / JavaScript with WebGPU (WGSL compute shaders)

---

## How to use this document

This document supplements the existing implementation plan. It does **not** replace or rewrite it.

- **Appendix A** specifies temporal execution, neuron/synapse timing, output readout, and how to evaluate the current scheduler against alternatives.
- **Appendix B** specifies how multidimensional input and output patterns can learn directed soft rules through spike-timing-dependent plasticity (STDP), reward modulation, and controlled anti-Hebbian mechanisms.
- The existing plan's safety boundaries, milestone gates, and M0-first instruction remain in force.
- The project is implemented in TypeScript/JavaScript and WebGPU. Where the main plan contains generic language branches, follow the repository's actual TypeScript/WebGPU stack; do not introduce Rust or C++ as a second implementation stack.

**Astra must execute M0 first and stop for review.** During M0, inspect the current scheduler, epoch definition, log-sigmoid delay function, tensor representation, activation semantics, WebGPU dispatch model, and existing learning update. Do not replace them before recording their exact behaviour and establishing a replay baseline.

### Evidence labels

- **Research-supported:** directly motivated by published work cited in Section B.12.
- **Engineering recommendation:** a design choice proposed for this project, to be validated experimentally.
- **Open hypothesis:** a plausible extension that must remain behind a feature flag until experiments support it.

The cited studies establish useful precedents, not validation or certification of this particular architecture. The engineering recommendations below are hypotheses for a simulator research programme, not flight-ready control logic.

# Executive decision

| Design question | Recommendation | Status |
|---|---|---|
| Is a tensor one neuron? | No. A tensor is a cluster/population of neurons or feature units. Timing and plasticity must be represented below tensor level. | Required |
| Should every tensor receive a new explicit time axis? | No. Keep the tensor's semantic input/output axes. Add temporal state to each neuron and timing/eligibility state to each synapse. Use an explicit history axis only for sequence-processing tensors that need it. | Recommended |
| Should the existing tensor-wide scheduler be deleted now? | No. Preserve it as the baseline implementation. Compare it with per-neuron temporal state under identical replay scenarios before retiring or changing it. | Required experimental comparison |
| Should there be a delay per tensor? | A tensor-wide dispatch or processing latency may remain as a coarse runtime abstraction, but it is not a substitute for neuron and synapse timing. | Optional |
| Should there be delays per neuron? | Yes, represent each neuron's own activation, membrane/accumulator, refractory and event history. An intrinsic response delay may be modelled per neuron if needed. | Recommended |
| Should there be delays per synapse? | Model an explicit transmission delay per connection only where there is a reason to represent it. Start with a common or zero delay; make delays learnable only in a later experiment. | Later experiment |
| Should output be an EMA? | An EMA is useful as a rate/persistence readout, but must not replace exact event timestamps used by STDP. Keep the two channels separate. | Recommended |
| What learns `p -> q`? | Directed weights between active source-feature units in pattern P and target-feature units in pattern Q, updated from their event timing. P/Q are engineer notation for learned activity patterns, not symbols processed by a formal logic engine. | Core concept |
| Does STDP prove causality? | No. It learns a temporally ordered predictive association. Causal interpretation requires interventions and confound checks. | Required qualification |
| Does positive delay prevent unstable feedback? | No. It prevents instantaneous same-tick recursion if the runtime enforces tick barriers, but delayed recurrent loops can still oscillate or diverge. | Safety requirement |

---

# Appendix A - Temporal execution and delay architecture

## A.1 The modelling levels must remain distinct

The architecture contains several different objects. Do not use the word *tensor* to mean all of them interchangeably.

| Object | Meaning | Owns temporal state? |
|---|---|---|
| Tensor | A semantic cluster/population, such as `FlightTensor` or `EnergyTensor`, containing many feature units/neurons and declared input/output axes. | Owns references to its member state; not one shared firing time. |
| Neuron / feature unit | An individual class detector, latent unit, or output unit inside a tensor. | Yes: activation, traces, last event, refractory/adaptation state. |
| Synapse / connection | A directed weighted connection from one source unit to one target unit. | Yes: eligibility and, when modelled, transmission-delay state. |
| Tensor scheduler | Runtime mechanism that schedules tensor work or output batches. | May own dispatch timing, but is not a biological synapse model. |
| Simulation clock | Monotonic logical time of the simulator. | Authoritative time base for every event and learning update. |

A `LocationTensor` may contain many spatial class detectors. Those detectors need not fire together, and they need not share a single response time. A scalar delay assigned to the entire `LocationTensor` hides that structure.

The project's shape notation remains unchanged:

```text
LocationTensor: raw navigation inputs -> Location features
FlightTensor:   instrument inputs     -> attitude/motion features
EnergyTensor:   flight + engine state -> aerodynamic/energy features
ControlsTensor: measured forces       -> proprioceptive/control-force features
IntuitionTensor: perception patterns  -> protective action proposal
MissionTensor: mission + perception   -> mission-conditioned reference proposal
ServoTensor:   accepted reference + state -> simulated control demand
```

Temporal state is added to the individual units and connections that implement these mappings. It is not necessary to append `Time` to every tensor's semantic shape.

## A.2 Five distinct concepts that must not be conflated

1. **Simulation time:** the shared monotonically increasing logical clock (`tick`, `dt`).
2. **Neuron dynamics:** how a unit integrates, decays, adapts, crosses a threshold, fires, or becomes refractory.
3. **Synaptic transmission delay:** how long an event takes to affect a particular downstream connection.
4. **Plasticity timing window:** how much time may separate pre- and postsynaptic events for a learning update.
5. **Readout smoothing:** how recent activations/events are aggregated into a stable analogue output, for example through an EMA.

The current log-sigmoid scheduler delay is a sixth concept: **tensor-level scheduling/processing latency based on aggregate epoch input load**. It may be useful for the current model, but it is not equivalent to any of items 2-5.

## A.3 Options considered

### Option A - One scheduler delay per tensor

The current architecture updates weights on input and schedules the tensor's output for later emission. Delay grows with the accumulated number of inputs in an epoch according to the existing log-sigmoid function.

**Advantages**

- Simple scheduling abstraction.
- Captures aggregate load-dependent response latency.
- May already be well integrated with the existing WebGPU execution path.
- Provides a useful baseline for comparing new temporal representations.

**Limitations**

- Every unit in the tensor effectively shares a coarse delay, even if the tensor represents many independent neurons.
- The timing of a particular input feature and a particular output feature is obscured by aggregate epoch load.
- Two identical local pre/post event pairs can receive different effective timing because unrelated inputs elsewhere in the tensor changed the epoch total.
- It does not directly represent per-synapse transmission delay or the pre/post timing window used by STDP.

**Decision:** preserve it behind the current implementation path and instrument it. Do not treat it as the final temporal model merely because it already prevents immediate recursive calls.

### Option B - Per-neuron temporal state and event traces

Each neuron has its own state and event history. A source feature can activate at one tick, while another feature in the same tensor activates later. Target neurons receive and integrate those events independently.

**Advantages**

- Matches the fact that a tensor is a population, not a single neuron.
- Provides the event timestamps needed for local STDP.
- Supports different response dynamics within the same tensor.
- Can be implemented with compact per-neuron arrays and decaying traces rather than storing a long history for every unit.

**Limitations**

- Requires a deterministic event/tick model and more explicit state management.
- If connections have different transmission delays, traces must reflect the arrival time at each relevant synapse, not merely the source neuron's emission time.

**Decision:** recommended default temporal representation.

### Option C - Per-synapse transmission delays

Each directed connection has a delay `d_ij`. A source event emitted at `t_pre` is delivered to that connection at `t_pre + d_ij`.

**Advantages**

- Timing is local to the connection, where STDP actually evaluates pre/post order.
- Can align or separate input arrivals to detect temporal patterns.
- Recent research studies online learning of synaptic and axonal delays together with weights, although that work is not a flight-control demonstration [R7].

**Limitations**

- More state, scheduling and memory traffic.
- A different delay for every edge complicates shared per-neuron trace optimisations.
- Delay learning introduces another adaptive parameter and should not be added before weight-learning behaviour is understood.

**Decision:** implement the data model so per-connection delays can be added, but initially use zero or a small number of shared delay classes. Make delays learnable only as a separate ablation. Research has demonstrated co-learning weights, neuron adaptation and synaptic delays, as well as learning delay structure through sparse connectivity, but these demonstrations use temporal classification/speech datasets rather than aircraft control [R7, R14, R15, R16].

### Option D - A time/history axis in every tensor

A tensor could store a rolling sequence of activation states, with the oldest slice used for output and the newest slice used for accumulation.

**Advantages**

- Explicitly represents temporal patterns.
- Useful for manoeuvre/sequence recognition, temporal filters, temporal convolution, or recurrent feature extraction.

**Limitations**

- Multiplies activation storage and memory bandwidth for every tensor.
- Does not by itself define when a synapse strengthens or weakens.
- A history buffer is not the same thing as a neuron's membrane state or a synaptic delay.
- If all tensors receive this axis without evidence, it creates cost and complexity without guaranteeing better temporal learning.

**Decision:** do not add a time axis to all base tensors in v0.1. Add explicit history windows only to tensors such as `ManeuverTensor` or temporal classifiers when experiments show that an ordered sequence cannot be captured adequately with per-neuron traces.

### Option E - EMA over all layers / outputs

An exponential moving average can accumulate recent activity. Repeated output events increase the readout; an inactive unit's readout decays.

For time step `dt` and readout time constant `tau_readout`:

```text
beta = exp(-dt / tau_readout)
readout_j[t] = beta * readout_j[t-1] + (1 - beta) * activity_j[t]
```

For binary spike events, a related exponentially decaying spike-rate trace can be used. The exact normalisation depends on whether the desired output is a rate, a persistence score, or a bounded confidence-like value.

**Advantages**

- Smooth analogue readout from discrete events.
- Repetition within the readout window makes a feature stronger.
- Cheap per-neuron state; no full activation history required.

**Limitations**

- Smoothing destroys exact event order if used as the only stored signal.
- A high EMA can mean repeated firing, but it does not establish that one particular presynaptic event preceded one particular postsynaptic event.
- Different time constants can blur separate manoeuvres or merge events that should remain distinct.

**Decision:** use EMA/rate traces as a *readout channel*, alongside exact event timestamps and STDP traces. Never use an EMA-only value to calculate `deltaT` for STDP. A flight-control SNN study by Foderaro et al. used a leaky integrator to decode spike-frequency outputs into smooth continuous control values, a useful precedent for separating control readout from synaptic timing [R13].

## A.4 Recommended hybrid temporal model

Implement the following hierarchy:

1. **Global simulation clock:** integer ticks and a fixed configured `dt` for deterministic replay.
2. **Per-neuron state:** current activation/membrane or accumulator, threshold/adaptation state if applicable, refractory-until tick, last event tick, and decaying event/rate traces.
3. **Per-synapse learning state:** weight, signed eligibility, and enough timing state to evaluate the selected STDP rule.
4. **Optional connection delay:** an integer `delayTicks` on a synapse or shared projection. Events are evaluated at their arrival time.
5. **Optional temporal history buffer:** only for a tensor whose task is explicitly sequence recognition and whose ablation justifies the history axis.
6. **Tensor readout:** returns current activation plus an optional smoothed/rate representation. It does not hold a single tensor-wide `lastFiredAt` value as a substitute for unit-level timing.

This design answers the concern that `LocationTensor` cannot be treated as one neuron: every spatial feature has its own state; connections between features have their own weights; and the tensor provides a grouped, multidimensional view of those units.

## A.5 Recommended event semantics

Use the simulator's logical time, not browser wall-clock time.

```ts
export type SimTick = number; // integer tick; validate safe integer range

export interface NeuralEvent {
  readonly tick: SimTick;
  readonly tensorId: string;
  readonly neuronId: number;
  readonly kind: "activation-rise" | "spike";
  readonly amplitude: number; // finite and in the configured range
  readonly sequence: number;  // stable deterministic tie-breaker
}

export interface NeuronTemporalState {
  activation: number;
  membraneOrAccumulator: number;
  rateTrace: number;
  preTrace: number;
  postTrace: number;
  lastEventTick: SimTick | null;
  refractoryUntilTick: SimTick;
}

export interface SynapseState {
  sourceNeuronId: number;
  targetNeuronId: number;
  weight: number;
  delayTicks: number;
  eligibility: number;
  lastPreArrivalTick: SimTick | null;
  lastPostTick: SimTick | null;
}
```

Adapt the interface to the existing codebase after M0. If the current network does not emit binary spikes, do not silently convert every non-zero float to a spike. Define an event detector explicitly, for example a rising threshold crossing with hysteresis, or use a continuous-time differential Hebbian rule as a separate algorithm. Keep the original continuous activation available.

### Tick and dispatch ordering

To stop same-tick recursive feedback, use a two-phase or multi-pass tick:

1. Read the immutable state snapshot for tick `t`.
2. Integrate all scheduled arrivals and current external inputs for tick `t`.
3. Detect events from the resulting neuron states.
4. Calculate plasticity/eligibility changes from events whose arrival times are now known.
5. Write next states and future events to separate buffers.
6. Commit/swap state only at the tick boundary.

An event emitted during tick `t` must not recursively cause an unbounded chain of tensor evaluations inside the same dispatch. A zero-delay connection may be defined as arriving at the next update boundary, but the exact convention must be documented and tested.

Use a stable order for equal-tick events, such as `(tick, targetNeuronId, sourceNeuronId, sequence)`. This makes CPU replay and GPU batch execution easier to compare.

### Delay is not a stability proof

A positive delay prevents instantaneous recursion only if the runtime enforces a tick barrier. It does **not** guarantee a stable recurrent network. A delayed positive-feedback loop can still oscillate or grow. Stability also depends on weight/gain bounds, neuron thresholds, refractory/adaptation behaviour, inhibitory/competitive structure, update ordering, saturation, and the controlled plant.

Therefore:

- keep all weights bounded;
- define activation and firing-rate bounds;
- define maximum events processed per tick and a fail-closed overflow policy;
- use refractory/adaptation or explicit loop damping where justified;
- test recurrent loops under worst-case sustained input;
- keep flight-control envelope protection and fallback outside plastic weights.

## A.6 TypeScript / WebGPU implementation guidance

The implementation language is TypeScript/JavaScript. Use WebGPU for batched numeric work where profiling supports it. Do not create a JavaScript object for every neuron and synapse inside the hot loop.

### Host/runtime responsibilities (TypeScript)

- own the deterministic simulation clock and configuration;
- load aircraft profiles, class definitions, weights and experiment seeds;
- validate sensor/tensor frames and event records;
- orchestrate compute passes and tick barriers;
- manage scenario execution, logs, checkpoints and experiment reporting;
- initially manage a small deterministic event queue on CPU if that is simpler and faster for the active event count;
- perform low-frequency readback for diagnostics rather than reading every neuron state to the CPU every tick.

### GPU data layout

Prefer structure-of-arrays storage buffers, for example:

```text
neuronActivation[]
neuronAccumulator[]
neuronRateTrace[]
neuronPreTrace[]
neuronPostTrace[]
neuronLastEventTick[]
neuronRefractoryUntil[]

synapseSource[]
synapseTarget[]
synapseWeight[]
synapseDelayTicks[]
synapseEligibility[]
synapseLastPreArrival[]
```

Use integer tick counters for event ordering. Convert tick differences to seconds only when evaluating exponential time kernels. Define explicit overflow/rollover handling; do not allow an integer wrap to look like a negative elapsed time.

### Suggested compute-pass boundaries

Adapt to the current WebGPU pipeline; these are logical stages, not a demand to create one shader per bullet.

1. `decay_traces`: decay per-neuron and per-synapse traces using `dt` and configured time constants.
2. `deliver_events`: apply events whose delivery tick is the current tick.
3. `integrate_neurons`: integrate incoming weighted activity and external inputs.
4. `detect_events`: apply threshold/hysteresis and refractory logic; emit event records.
5. `update_eligibility`: update local STDP eligibility from pre/post event timing.
6. `apply_modulator`: apply a signed outcome signal to eligible synapses when a valid outcome event is received.
7. `readout`: update EMA/rate output arrays and expose tensor outputs.

Some stages may be fused after profiling, but preserve the logical tick boundaries and test invariants.

### GPU concurrency and determinism

- Avoid multiple shader invocations performing unsynchronised writes to the same target activation or weight.
- Give each synapse one owner for its weight/eligibility update, or use a deliberate gather/reduction design.
- If several edges contribute to one target, use a two-pass gather/reduce or another race-free strategy. Do not assume floating-point atomics are available or deterministic.
- Use ping-pong buffers for current/next neuron state where needed.
- Keep event ordering deterministic. If GPU event compaction changes order, sort or use stable indices before applying order-sensitive learning.
- Record the WebGPU adapter/device information, shader version and numeric mode in experiment metadata.
- Permit tolerance-based cross-device numeric comparisons, but do not permit event order or STDP sign to change because of tiny floating-point drift. Use explicit thresholds/hysteresis.
- Do not use `Date.now()`, `performance.now()`, wall-clock scheduling or GPU completion latency as neural simulation time.

### Important optimisation rule

Begin with the smallest readable implementation that preserves the math. Profile it. Optimise only after correctness tests exist. A dense all-to-all synapse array over the full Cartesian product of every input and output class is not an acceptable default. Use sparse active connections or factorised maps as specified in the main plan.

## A.7 Required comparison experiment: scheduler versus local temporal state

Do not decide from intuition alone whether the current log-sigmoid scheduler should be removed.

Implement at least these experimental modes behind configuration:

- `legacy_tensor_scheduler`: current implementation, unchanged;
- `per_neuron_traces`: per-neuron timing/trace state, no additional connection delay;
- `per_neuron_plus_projection_delay`: per-neuron state plus a fixed delay per tensor projection;
- `per_synapse_delay`: sparse edge-specific fixed delays, only if feasible;
- `temporal_history_axis`: a finite explicit history window for a selected sequence-recognition tensor;
- `ema_readout`: readout smoothing enabled/disabled independently of the temporal model.

Run identical seeded input sequences and simulator episodes. Include both fixed latency and bounded variable sensor/actuator latency. A wind-tunnel UAV pitch-control study found that controllers trained under shorter assumed delays could become oscillatory when exposed to real test delay, while longer-delay training was more conservative but could show steady-state error [R17]. Compare:

- event ordering correctness;
- ability to learn pre-before-post versus post-before-pre relationships;
- prediction accuracy for delayed output patterns;
- missed-event and false-association rates;
- recurrent-loop event growth and stability metrics;
- memory use and WebGPU buffer traffic;
- CPU/GPU execution time per simulation tick;
- deterministic replay consistency.

The legacy scheduler may remain as a separate abstraction even if per-neuron timing becomes the learning model. Do not let aggregate input count determine STDP `deltaT` unless the experiment explicitly tests that hypothesis.

## A.8 Appendix A acceptance tests

Astra must add tests for:

1. two neurons in the same tensor emitting at different ticks retain distinct timestamps;
2. two output neurons in the same tensor can respond at different ticks;
3. equal input sequences and seeds reproduce equal event order;
4. a connection delay shifts arrival time exactly by `delayTicks`;
5. a delivered event cannot be processed before its delivery tick;
6. a tick's newly emitted events cannot recursively run the whole network within the same tick;
7. trace decay matches the analytic exponential within a declared tolerance;
8. EMA readout increases under repeated events and decays under silence;
9. EMA readout does not alter the stored exact spike/event timestamps;
10. high aggregate tensor input load does not alter local STDP timing in `per_neuron_traces` mode;
11. delayed feedback loops remain bounded under configured limits, or trigger the designed protective stop;
12. event-queue overflow, stale events and invalid ticks fail visibly and deterministically.

**Exit gate:** Appendix A is not complete until the legacy scheduler and recommended local-state model can both be run from the same recorded episode and compared in one report.

---

# Appendix B - Learning soft temporal rules with STDP

## B.0 The information boundary: measured facts, classified situations, selected actions

The architecture is not a formal-logic engine. The symbols `P -> Q` used in this document are a compact notation for engineers describing a learned, directed association between two multidimensional activity patterns. They are not Boolean implications evaluated by the brain, and they do not mean that a target must fire every time a source pattern appears.

The intended information flow is:

```text
PHYSICAL WORLD / SIMULATOR
        |
        v
INSTRUMENTS AND CONTROL-FEEL SENSORS
  measured values, units, timestamps, validity, quality
        |
        v
PERCEPTION / CLASSIFICATION TENSORS
  Location | Flight | Energy | Controls | (Maneuver)
  raw observations -> classified state/activity patterns
        |
        v
ACTION-SELECTION TENSORS
  Mission | Intuition
  mission-conditioned and protective action proposals
        |
        v
DETERMINISTIC ACTION ARBITER
  fixed authority, envelope and validity constraints
        |
        v
SERVO TENSOR -> SIMULATED ACTUATORS -> AIRCRAFT DYNAMICS
        |
        +------ new instrument observations and control forces ------+
```

### B.0.1 Instruments report observations; tensors interpret them

The instruments and control-force sensors report measured observations: they tell the system what was measured, in physical units, at a particular time and with known validity/quality limits. They do not themselves perform the brain's interpretation. The perception tensors take those observations and classify the situation; Mission and Intuition consume those classifications to propose actions. Treat instrument and force-sensor inputs as measured observations, not as neural classifications. Each raw sample should retain its source, physical unit, sample timestamp, validity/calibration state and quality metadata. A bad or stale sensor sample must not silently become a confident class activation. Instruments are the factual measurement channel within their accuracy and failure limits; the tensors build an interpretation of those measurements.

Keep raw numerical channels available for diagnostics, supervised pretraining and replay. Do not throw them away after classification. Do not feed privileged simulator variables into inference unless an equivalent observable sensor channel exists in the intended runtime. Privileged simulator state may be used as a training label or evaluation oracle, but must be marked as such to prevent data leakage.

### B.0.2 Perception tensors classify the aircraft situation

| Tensor | Observed inputs | Intended classified representation |
|---|---|---|
| `LocationTensor` | Navigation/instrument observations such as heading, position/bearing and altitude | Spatial/navigation state and relation to the intended route, e.g. heading sector, altitude band, position/track state |
| `FlightTensor` | Flight-instrument observations such as ADI attitude, ASI, VSI and rates where available | Aircraft orientation and kinematic state, e.g. pitch, bank, vertical-speed and airspeed bands/trends |
| `EnergyTensor` | Flight and engine observations, plus AoA/load/configuration/asymmetry channels when available | Aerodynamic/energy state, e.g. climb/descent energy, low energy margin, stall warning, critical-AoA exceedance, asymmetric stall or spin state when supported by the model |
| `ControlsTensor` | Measured force/torque at the pilot's controls: push/pull/left/right and magnitude | Proprioceptive/control-feel state, e.g. pushing strongly, pulling, left/right force, changing resistance |
| `ManeuverTensor` (optional) | Time-ordered Flight/Energy/Location patterns | Temporal manoeuvre recognition, e.g. climbing turn, steep turn, loop or spin sequence |

`ControlsTensor` is specifically about the force felt at the controls, not yoke position. Force, control displacement, actuator command and actual control-surface deflection are distinct signals. If the simulator exposes actuator position/deflection, retain it as a separate measured actuator-feedback channel for diagnostics and outcome evaluation; do not silently substitute it for hand-force proprioception.

A tensor's classifier output should normally be a vector or multidimensional pattern of graded feature activations, not necessarily a single exclusive label. Preserve uncertainty, multi-label states and validity. For example, an aircraft can simultaneously be turning, descending and approaching a stall. Do not force physically compatible simultaneous facts into one mutually exclusive class.

### B.0.3 Control tensors select responses from classified situations

- `MissionTensor` receives a typed mission specification (goal, route/waypoint, target altitude/speed bands, constraints and completion criteria) and current perception classifications. An LLM may translate human language into this typed, validated specification; it must not bypass validation or issue raw actuator commands.
- `IntuitionTensor` receives the perception/classification patterns and learned context. It proposes protective or adaptive pilot-like responses, such as a recovery mode or a change of power/attitude target. It does not need a direct `ServoTensor` output input. The aircraft's actual response returns to it through new observations classified by `FlightTensor`, `EnergyTensor`, `ControlsTensor` and other perception tensors.
- `MissionTensor` and `IntuitionTensor` are proposal generators, not two uncontrolled writers of the same Servo weights. A deterministic `ActionArbiter` applies fixed authority, envelope, validity and conflict rules before `ServoTensor` receives an accepted reference/action proposal.
- `ServoTensor` turns the accepted proposal into bounded simulated actuator/control commands. The command is not evidence that the aircraft actually moved as intended; the subsequent instrument observations and, where available, separate actuator-feedback measurements provide that evidence.

### B.0.4 What `P -> Q` means in this architecture

For engineering discussion, `P -> Q` means: a multidimensional pattern of source-feature activity in one part of the network is temporally associated with, and may contribute to, a later multidimensional pattern of target-feature activity elsewhere. The pattern is represented by many directed weights between individual feature/neuron units. It is a soft, graded, context-dependent association. Other inputs can activate Q without P, and P may lead to different responses under different contexts.

The network does not need to store a symbolic rule such as `IF P THEN Q`. The notation is only a human-readable description of what the distributed weights and temporal activity appear to have learned. STDP supplies a local timing-dependent change to individual directed connections; it does not, by itself, establish that P is the physical cause of Q. Use controlled interventions in the simulator before describing an association as causal (see B.8).

## B.1 The intended unit of learning is a directed connection, not a whole tensor

The proposed semantics are:

```text
P  --directed weighted connections-->  Q
```

`P` is a multidimensional input pattern and `Q` is a multidimensional output pattern. Each pattern may activate many individual neurons/features. The tensor is the cluster; individual neurons/features are the units that produce pre- and postsynaptic events; weights are the connections between them.

For a particular input neuron `i` and output neuron `j`, `w[j, i]` represents the strength of the directed connection `i -> j`. The logical mapping retains the project's axis notation:

```text
P: A x B -> C x D : Q
```

The physical implementation should normally be a sparse/factorised set of connections between stable feature IDs, rather than one dense array containing every possible combination of all input classes and all output classes.

A repeated temporal relation can be described as a soft association (the word "rule" below is shorthand for a learned statistical/temporal regularity, not a formal instruction executed by the network):

```text
P  --within a learned timing window-->  Q
```

This is not Boolean implication or a symbolic IF/THEN statement. It does not mean that Q must always occur whenever P occurs, nor that Q cannot occur without P. It means that source activity in P can strengthen directed weighted pathways that make a later Q-pattern more likely or more strongly represented within a timing window and context. Other pathways can also activate Q, and the same P-pattern may participate in different Q-patterns in different contexts.

For clarity, use these notations in logs and experiment reports:

- `P ≺[W+, W-] Q`: P's presynaptic activity arrives before Q's postsynaptic event inside the potentiation window;
- `Q ≺[W-, W+] P`: Q occurs first, which may fall in the depression window for the P-to-Q connection;
- `P AND C -> Q`: a joint pattern with context C is associated with Q;
- `P !-> Q within deadline`: P occurred but the expected Q event did not occur before a declared deadline. This is a separate proposed negative-evidence rule, not ordinary pair-based STDP.

Do not use `^` in code or documentation for both conjunction and anti-causal timing. Use explicit `AND`, `PRE_BEFORE_POST`, or `POST_BEFORE_PRE` labels.

## B.2 Canonical pair-based STDP

For a synapse from presynaptic unit `i` to postsynaptic unit `j`, define:

```text
t_pre_arrival = t_pre_emission + delay_ij
delta_t       = t_post - t_pre_arrival
```

Use the following standard pair-based STDP form as the first test rule:

```text
if 0 < delta_t <= W_plus:
    delta_w = +A_plus * exp(-delta_t / tau_plus)

else if -W_minus <= delta_t < 0:
    delta_w = -A_minus * exp( delta_t / tau_minus)

else:
    delta_w = 0
```

Where:

- `W_plus` is the maximum pre-before-post potentiation interval;
- `W_minus` is the maximum post-before-pre depression interval;
- `A_plus` and `A_minus` are bounded update magnitudes;
- `tau_plus` and `tau_minus` shape the timing kernels;
- `delay_ij` is the transmission delay of that connection (zero in the initial experiment unless the selected projection has a configured delay). If the legacy tensor scheduler has already delayed the source tensor output, use the actual scheduled emission tick as `t_pre_emission`; do not add the same scheduler delay again as `delay_ij`. A configured connection delay is an additional, separate delay.

Strict temporal precedence is required for an individual pair-based update: `delta_t == 0` is coincidence, not "P before Q". The exact tie policy is no pair update in the initial implementation. Across repeated experiences, the synaptic change is the accumulated result of eligible pairings: consistently pre-before-post pairings tend to strengthen the directed connection, while reverse-order pairings inside the depression window counteract or weaken it. Training examples need not all have identical timing, but the net update must be computed from the actual distribution of event pairs rather than assuming every occurrence supports the same direction. The sign convention is deliberate:

- `delta_t > 0`: P arrived first and Q followed; this is the pre-before-post LTP side of canonical Hebbian STDP. It is a temporal-prediction signal, not proof of physical causation;
- `delta_t < 0`: Q occurred first; this is the post-before-pre LTD side for the same directed P-to-Q synapse;
- outside both windows: no pair-based update.

STDP windows are not universal constants. Biological examples often use millisecond-scale timing, but the useful window for an aircraft-simulator mapping depends on event definition, simulator sampling, aircraft response time, and task. Do not copy a biological millisecond value into a slower flight-control model without a timing sweep. Feldman's review also stresses that spike timing is one factor among firing rate, cooperativity and depolarisation, and that plasticity varies by synapse and activity regime [R1].

### Important correction: outside the window is not automatically weakening

Canonical pair-based STDP does **not** normally say that any pair separated by too much time causes depression. Its usual rule is:

- pre-before-post inside the LTP window: strengthen;
- post-before-pre inside the LTD window: weaken;
- pair outside both windows: no update.

If P occurs and no Q follows before a deadline, there is no postsynaptic event pair for the ordinary rule to evaluate. Automatically weakening the connection in that case is an additional model hypothesis. It may be useful for learning failed predictions, but only when the system has a well-defined expectation that Q should have occurred and can distinguish a genuine miss from an unrelated or suppressed Q. Implement it as a separately named `unpaired_timeout_ltd` experiment, disabled by default.

This distinction matters because Q may be produced by another input, may be inhibited by context, or may simply not be the correct target in the current situation. Treating every absent Q as a negative label would erase useful associations.

## B.3 Multidimensional P and Q: how weights represent a soft rule

Suppose P has input axes `A x B` and Q has output axes `C x D`. Let each active element in P and Q correspond to a stable feature/neuron ID after the tensor's encoding step.

For active source features `i` and target features `j`, create or use a directed synapse:

```text
source feature (P, i) -> target feature (Q, j)
weight w[j, i]
```

A P-pattern and Q-pattern are represented by many such connections. If multiple P features participate in causing a Q pattern, several incoming weights can strengthen from repeated correctly ordered events. The target neuron's integration function combines those inputs. The entire relation remains multidimensional even if the implementation flattens feature IDs for sparse storage.

**Do not reduce P to one scalar activation or Q to one scalar output merely to make STDP easy.** That would throw away the very tensor structure the architecture is designed to preserve.

### Conjunctions require context or interaction structure

A simple pairwise connection learns pairwise temporal associations. It does not automatically learn an exclusive logical conjunction.

For example:

```text
pitch-up AND low-energy-margin -> stall-risk
```

must not be approximated by two independent unconditional links:

```text
pitch-up -> stall-risk
low-energy-margin -> stall-risk
```

Those links could activate the target when only one feature is present. To represent the joint condition, use one of:

1. a pre-trained `EnergyTensor` feature that already classifies the joint aerodynamic condition;
2. an explicit context/conjunction feature that fires only when the required input pattern is jointly active;
3. a sparse higher-order/factorised interaction term whose context axes gate the P-to-Q synapses.

Start with option 1 if the selected simulator can provide a reliable labelled state and sensor mapping. Add option 2 or 3 only if an experiment demonstrates that the existing tensor encoders do not capture the required interaction.

## B.4 Pair timing versus rule/pattern timing

The input instruments are usually analogue, continuously sampled quantities, while canonical STDP operates on discrete pre- and postsynaptic events. The coding agent must therefore inspect the current activation representation before choosing the learning rule. Do not silently treat every non-zero analogue activation or every frame sample as a new spike.

If the tensors already produce discrete activation events, use those events. If they produce graded activations, compare at least two explicitly defined encodings:

1. **Threshold-crossing events:** emit an event when a feature activation crosses an `onThreshold`, and do not re-arm until it falls below a lower `offThreshold` (hysteresis). The threshold, refractory/cooldown and event polarity are configuration.
2. **Continuous differential Hebbian baseline:** use a documented local rule based on activity and its time derivative when the system is inherently continuous and thresholding would discard useful information. Keep this as a separate learning mode; do not call it pair-based STDP.

For the event path, one source pattern and one target pattern may each contain many simultaneously active feature units. The implementation must define which per-feature pairs are eligible. The update occurs on directed connections between units, not on a single scalar P/Q pair.

Start with **all-to-all pairing inside the declared projection** as a simple reference implementation, but cap event rates and record how many pairs are considered. Compare it against a nearest-neighbour pairing variant if repeated firing causes weight growth or event-pair explosion.

For each eligible connection, record:

- source/target feature IDs;
- source emission tick and connection arrival tick;
- target post-event tick;
- `deltaT` and selected window;
- rule/variant ID;
- resulting eligibility or weight delta;
- context key and model version.

A connection is not strengthened merely because P and Q are both active in the same epoch. Their event order and separation must be evaluated in simulation time.

## B.5 Eligibility traces and a third outcome factor

For control learning, a temporally correct sequence is not necessarily a *good* control action. The aircraft may respond as predicted while the response makes the mission state worse. Separate:

1. **temporal association:** did P precede Q in the relevant window?
2. **outcome evaluation:** did the selected action improve the defined objective without violating safety constraints?
3. **plasticity modulation:** should the eligible association be reinforced or weakened given that outcome?

Use an eligibility trace `e_ij` to retain a synapse's recent STDP evidence until the later outcome is available:

```text
e_ij(t + dt) = exp(-dt / tau_e) * e_ij(t)
               + STDP_pair_increment_ij(t)

when a valid outcome modulator M arrives:
    delta_w_ij = learning_rate * M * e_ij
    w_ij = clamp(w_ij + delta_w_ij, min_weight, max_weight)
```

`M` must be bounded, signed, tied to the relevant action/outcome window and set to zero when the outcome cannot be attributed confidently. The formula is a three-factor rule: pre/post timing creates eligibility; the later modulator chooses whether eligible changes are expressed. This addresses delayed reward/credit assignment and is reviewed in the SNN control tutorial [R4] and three-factor learning literature [R5, R6].

Keep two separately switchable modes:

- `unsupervised_stdp`: the pair rule directly updates weights from local timing;
- `reward_modulated_stdp`: local timing updates eligibility, and a later outcome signal changes weights.

For the flight-control learning path, use `reward_modulated_stdp` as the principal experiment. Use `unsupervised_stdp` to learn predictive sensor/feature associations or as a baseline. Do not assume a pre/post timing relation alone can tell ServoTensor that the action was beneficial.

### Outcome modulator design

The first version should reuse the main plan's deterministic `ErrorTensor` / `OutcomeEvaluator` and `PlasticityGate`.

- Positive `M`: a validated objective improved in the action's attribution window while safety margins remained acceptable.
- Negative `M`: a validated objective deteriorated in the attribution window, with adequate confidence and attribution.
- Zero `M`: neutral result, missing sensor, ambiguous attribution, stale frame, unknown aircraft state, or insufficient confidence.
- Freeze: critical/fault cases where plasticity could encode the wrong causal attribution. A critical event must not automatically broadcast punishment to every active synapse.

Use lexicographic safety: mission reward must not compensate for a safety-margin violation. Preserve outcome components separately in logs rather than collapsing them irreversibly into one scalar.

## B.6 Three mechanisms that must not share one name

| Mechanism | What it means | What it should do here |
|---|---|---|
| Anti-causal STDP / timing-dependent LTD | The postsynaptic event precedes the presynaptic arrival within the depression window. | Weaken that directed connection under the configured pair rule. |
| Negative outcome modulation | The synapse had an eligibility trace, then the associated action/outcome was judged harmful. | Reduce eligible weights, subject to attribution and safety gating. |
| Anti-Hebbian competition/decorrelation | A designated set of co-active or competing units is trained to avoid redundant/collapsed representation. | Optional local competition within named categorical/latent groups only; test by ablation. |

Do not globally weaken every co-active pair. Do not label a negative reward as anti-Hebbian STDP. These are different learning mechanisms and should have distinct configuration flags, code paths, metrics and tests.

For competition, use declared groups such as mutually exclusive classes on one categorical output axis. Keep weights bounded and add appropriate homeostatic normalisation. Demonstrate benefit before enabling competition by default. This follows the main plan's restriction on anti-Hebbian scope.

## B.7 How the aircraft examples should be encoded

The examples below are useful hypotheses for constructing controlled simulator experiments. They are **not universal flight rules** and must not be used as live-aircraft control procedures.

### Example 1 - Control input followed by attitude response

```text
ControlsTensor: PUSH_YOKE  --pre-before-post-->  FlightTensor: PITCH_DOWN
ControlsTensor: PULL_YOKE  --pre-before-post-->  FlightTensor: PITCH_UP
```

These are appropriate candidate associations for a conventional fixed-wing model, provided the sign convention, control axis, force measurement and aircraft model are verified. Use measured yoke force as the proprioceptive feature, but keep measured actuator/elevator response separate. A force at the pilot's hand is not identical to an elevator deflection or commanded actuator position.

### Example 2 - Attitude/energy evolution followed by airspeed trend

```text
FlightTensor: PITCH_DOWN  --context + timing-->  FlightTensor: ASI_INCREASING
FlightTensor: PITCH_UP    --context + timing-->  FlightTensor: ASI_DECREASING
```

Treat these as conditional, delayed relationships, not universal one-to-one implications. Airspeed trend depends on thrust, drag, flight-path angle, load factor, aircraft configuration, weight and the trajectory of the manoeuvre. Pitch attitude is not angle of attack and is not the same as flight-path angle. Put the relevant context into `EnergyTensor`/`ManeuverTensor` or explicit interaction features. The 2022 differential-Hebbian motor-control work is relevant here because it addresses control relationships and error responses that can be non-monotonic, and combines local learning with reinforcement learning for more general cases [R3].

### Example 3 - Stall-risk classification

Do not encode this as a universal rule:

```text
PITCH_UP AND ASI < Vstall -> STALL
```

The FAA states that a stall results from exceeding critical angle of attack; it can occur at any airspeed, attitude or power setting. Stall speed is affected by factors including load factor and configuration [R9]. Pitch attitude alone does not determine angle of attack.

A better modelling target is:

```text
measured_or_modelled_AOA relative to aircraft-specific critical_AOA
    + airspeed / load factor / configuration / asymmetry / sideslip context
    -> impending-stall or stall-state classification
```

Use a simulator's actual modelled stall state and aerodynamic ground truth for labels when available, while only using sensor channels actually available at inference time. Keep `stall_warning`, `critical_AOA_exceeded`, `left/right asymmetry`, `spin_entry`, and `spin_state` as different labels where the simulator provides sufficient ground truth. Do not infer a left spin from a generic stall label alone.

### Example 4 - Temporal sequence versus simultaneous condition

A conjunction such as `PITCH_UP AND LOW_AIRSPEED` is a *joint state pattern*. It is not necessarily a temporal sequence. STDP's pre-before-post rule applies to event order. If two features are simultaneously active as context and a third output class follows, use a context-gated interaction or a pre-trained classifier; do not invent an artificial event order between simultaneous inputs.

## B.8 What STDP learns, and what it does not prove

STDP is often described as strengthening a connection when presynaptic activity precedes postsynaptic activity and weakening it when the order reverses. It therefore gives the network a local mechanism for learning temporal predictive relations. Differential Hebbian learning extends this idea to continuous-valued signals by using changes/derivatives to encode temporal relations, and can produce causal, anti-causal and coincidence-sensitive learning kernels [R2].

However, `P before Q` does not by itself prove `P causes Q`. A common upstream signal could cause both P and Q, with P merely occurring first. In a controlled simulator, the project can investigate causal contribution by interventions:

1. replay the same initial state and context;
2. deliberately vary or suppress P while holding other inputs as constant as practical;
3. compare the distribution and timing of Q;
4. block/ablate the candidate P-to-Q connection and measure Q;
5. test the learned relation on held-out contexts and aircraft configurations.

Describe learned weights as **temporal predictive associations** or **candidate causal rules** until these interventions support a stronger causal interpretation. This is not a reason to abandon the `P -> Q` abstraction; it is a rule for interpreting what the weights mean.

## B.9 Pretraining and online adaptation are separate phases

Keep the original staged plan:

### Phase 1 - Pretrain individual tensor representations

Use backpropagation or other suitable supervised optimisation for individual perception classifiers/encoders where labelled data are available. This may include Location, Flight, Energy and Controls; Mission can be separately trained/validated as a typed goal encoder. Preserve raw instrument/force samples and metadata. Use simulator ground truth as a label/evaluation oracle only when it is not an input available at inference time. Split training/validation/test data by complete episode, manoeuvre, initial condition and configuration to avoid adjacent samples from one flight appearing in both train and test sets.

The purpose is to give each tensor a useful initial vocabulary. It does not mean the entire connected network already knows how those representations interact in every flight condition.

### Phase 2 - Freeze and validate the connected network

Connect the tensors; initially freeze learning. Verify that the tensor encodings, proposal interfaces, action arbitration and simulator baseline work before changing any weights online.

### Phase 3 - Enable local timing learning in shadow mode

Calculate STDP pair events and eligibility traces, log proposed changes, and prove deterministic replay. Do not change control outputs yet.

### Phase 4 - Enable reward-modulated changes in simulation experiments

Only after the shadow-mode gates pass, allow selected plastic connections to change in a simulator experiment. Use paired seeds and identical scenarios against the frozen baseline.

Pretrained encoders should remain frozen initially. If encoder plasticity is later tested, isolate it as a new experiment; do not allow BP and STDP to update the same weight bank without an explicit combined rule and ADR.

## B.10 Mission, Intuition and Servo: learning ownership

Keep the information direction explicit:

```text
measured observations -> perception classifications -> action proposals
                     -> arbiter -> Servo command -> aircraft response
                     -> new measured observations -> new classifications
```

Mission and Intuition should not fight by independently pushing unrestricted live outputs into the same Servo weights.

- Mission provides a goal/reference proposal conditioned on the validated mission specification and current perception classifications.
- Intuition provides a protective/adaptive proposal based on perception classifications and learned temporal associations. It does not need a direct Servo-output input by default; it observes the consequences through subsequent Flight/Energy/Controls classifications.
- The deterministic ActionArbiter selects, constrains or rejects proposals according to fixed authority, validity, envelope and fallback rules. Learned weights cannot modify those rules.
- Servo maps the accepted proposal and current state to bounded simulated actuator/control commands.
- A separate actuator-feedback measurement may record actual simulated actuator/surface response. It is not the same as ControlsTensor's hand-force input and is not automatically fed directly into Intuition.
- Error/OutcomeEvaluator compares the subsequent observed/classified response with the intended reference and checks safety constraints over a defined attribution horizon.
- The learning path associates the accepted decision and its outcome with the eligibility traces of connections that actually contributed to that accepted proposal.

Log which proposal was selected, which was rejected, why, the accepted action/decision ID, the contributing features/synapses, and the later perceived outcome. An outcome should update only eligible connections attributable to the accepted decision, not every neuron that happened to be active in the perception layer. Use eligibility traces, projection provenance and/or action-specific context tags for attribution.

Agreement between Mission and Intuition can be useful evidence that a proposal is consistent with both mission progress and a learned protective response, but agreement is not a reward by itself. Reward/penalty must come from observed outcome and fixed safety evaluation. When an input is invalid, context is ambiguous, or the outcome cannot be attributed, freeze plasticity for that sample rather than guessing.

## B.11 TypeScript/WebGPU learning implementation contract

Use the following as logical contracts; adapt exact names and storage to the current codebase after M0.

```ts
export interface TimedFeatureEvent {
  readonly tick: number;
  readonly featureId: number;
  readonly amplitude: number;
  readonly polarity: "positive" | "negative";
}

export interface StdpConfig {
  readonly potentiationWindowTicks: number;
  readonly depressionWindowTicks: number;
  readonly tauPotentiationTicks: number;
  readonly tauDepressionTicks: number;
  readonly aPotentiation: number;
  readonly aDepression: number;
  readonly maxAbsWeight: number;
  readonly pairing: "all-to-all" | "nearest-neighbour";
  readonly unpairedTimeoutLtdEnabled: boolean; // false initially
}

export interface PlasticSynapse {
  readonly sourceFeatureId: number;
  readonly targetFeatureId: number;
  weight: number;
  delayTicks: number;
  eligibility: number;
  lastPreArrivalTick: number | null;
  lastPostTick: number | null;
}

export interface OutcomeModulator {
  readonly tick: number;
  readonly value: number; // bounded signed value
  readonly confidence: number; // [0, 1]
  readonly decisionId: string;
  readonly outcomeWindowId: string;
  readonly reason: string;
}
```

Implementation requirements:

1. Use integer simulation ticks to decide event ordering. Never compare approximate wall-clock floats to determine whether pre preceded post.
2. Convert tick differences to configured simulation seconds only when evaluating the exponential STDP kernel.
3. Make pairing mode explicit: all-to-all or nearest-neighbour. Do not silently change semantics during optimisation.
4. Clamp weights to declared bounds; keep excitatory/inhibitory sign constraints where the chosen neuron model requires them.
5. Keep separate arrays for weight, eligibility, source/target indices and delay. Avoid object allocation in the hot path.
6. For shared delays per projection, per-neuron traces may be reused efficiently. For distinct per-edge delays, ensure the STDP calculation uses each edge's actual arrival time; use a per-edge event/trace structure or deterministic delivery queue as necessary.
7. Do not let a WebGPU workgroup update the same edge from two invocations. Assign one writer per synapse or use a deterministic reduction.
8. Keep `unsupervised_stdp`, `reward_modulated_stdp`, `unpaired_timeout_ltd`, `anti_hebbian_competition`, and `learnable_delay` as independent flags.
9. Record every sampled weight update with before/after weight, pre/post ticks, `deltaT`, eligibility, modulator, context and rule version.
10. Keep learning off by default; use shadow mode before any simulator output is affected.

### Suggested trace form

For a discrete fixed time step, exponential traces can be updated as:

```text
preTrace_i[t]  = exp(-dt / tau_pre)  * preTrace_i[t-1]  + preEvent_i[t]
postTrace_j[t] = exp(-dt / tau_post) * postTrace_j[t-1] + postEvent_j[t]
```

When a post event occurs, eligible incoming connections can receive positive pre-before-post increments from the source traces. When a pre event arrives, eligible outgoing connections can receive negative post-before-pre increments from target traces. The precise update ordering at equal ticks must be defined: do not arbitrarily treat simultaneous pre and post events as causal. A safe initial convention is `deltaT == 0 -> no pair update`; test it explicitly.

For delayed reward, accumulate signed timing evidence in `eligibility_ij` and express it only when a valid outcome modulator arrives. If no reward/outcome event arrives, eligibility decays and eventually vanishes.

## B.12 Research findings: what has already been tried

The following studies are the closest precedents located for the combination of flight control, spiking networks, timing-dependent plasticity and adaptive control.

### Directly relevant flight-control work

**Foderaro, Henriquez and Ferrari (2010), “Indirect Training of a Spiking Neural Network for Flight Control via Spike-Timing-Dependent Synaptic Plasticity.”** This is the closest conceptual precedent. It uses an action SNN as a flight controller, a separate critic SNN trained with reward-based Hebbian learning, and STDP in the action network. The critic sends spike trains that indirectly shape the action network's plasticity. The paper demonstrates aircraft stability augmentation in numerical simulation and analyses performance before/after training [R13]. It is not a demonstration of the exact tensor architecture here, nor a real-aircraft flight test. The aircraft model is deliberately narrow: longitudinal dynamics with angle of attack and pitch rate as state variables, and elevator deflection as the control input. The paper also uses a leaky integrator to decode output spike frequency into a smooth continuous control signal [R13]. This gives two particularly useful precedents for this project: a separate critic-like learning signal can modulate an action network, and an EMA/leaky readout can smooth control output without replacing precise spike timing for STDP.

**Chavez Arana et al. (2024), “Spiking Neural Network-based Flight Controller.”** This work applies an SNN designed using the Neural Engineering Framework to simulated quadrotor stabilisation and trajectory tracking, including take-off, altitude regulation and circular trajectory tracking [R8]. It is a useful modern flight-control SNN precedent, but the retrieved abstract does not establish that it uses the same STDP/anti-Hebbian mechanism proposed here.

**Oniz (2024), “Trajectory Control of Quadrotors via Spiking Neural Networks.”** The paper reports simulation and experimental quadrotor trajectory-control studies, with update rules derived using Lyapunov stability arguments and comparisons with PID and conventional neural controllers [R10]. It supports the broader feasibility of SNN-based control, not the exact learning rule or tensor arrangement proposed here.

**Stroobants, De Wagter and de Croon (2025), “Neuromorphic Attitude Estimation and Control.”** This work reports a modular SNN for attitude estimation/control deployed on a Crazyflie quadrotor. Estimation and control sub-networks are separately trained and merged; the system runs at 500 Hz and reports an average attitude tracking error around 3 degrees in real flight [R11]. This is a particularly relevant modular-perception/control precedent, but it uses imitation learning rather than demonstrating the proposed STDP rule.

**González-Álvarez et al. (2022), “Evolved neuromorphic radar-based altitude controller for an autonomous open-source blimp.”** The SNN altitude controller was evolved in simulation and evaluated in real-world experiments, with comparisons against ANN and linear controllers [R12]. It demonstrates neuromorphic altitude control with a real airship, but it is not evidence for the present Hebbian/STDP tensor design.

### STDP, temporal relations and delays

**Feldman (2012), “The spike-timing dependence of plasticity.”** A foundational review of STDP. It explains the usual pre-before-post potentiation and post-before-pre depression pattern, but also cautions that firing rate, cooperativity, depolarisation and synapse-specific conditions matter. The timing rule is not a universal one-size-fits-all law [R1].

**Cuppini et al. (2018), “General differential Hebbian learning: Capturing temporal relations between events in neural networks and the brain.”** This work formalises differential Hebbian learning rules that use signal derivatives and temporal event relationships. It can generate causal, anti-causal and coincidence-sensitive learning kernels and explicitly discusses leaky traces to bridge time gaps between events [R2]. This is relevant if the current network uses continuous activations rather than discrete spikes.

**Verduzco-Flores, Dorrell and De Schutter (2022), “A differential Hebbian framework for biologically-plausible motor control.”** This work demonstrates local differential-Hebbian feedback control for selected continuous-time motor tasks, and shows how reinforcement learning can be coupled to feedback control when control/error relationships are more general or non-monotonic [R3]. It is an especially useful bridge between the user's local control intuition and a formal learning rule.

**Madadi Asl (2025), “Time delays in computational models of neuronal and synaptic dynamics.”** This review explains how axonal and dendritic/synaptic delays change the timing of events at the synapse and can therefore change whether an STDP update potentiates or depresses a connection. This directly supports evaluating timing at the connection's arrival site rather than comparing only emission timestamps [R6].

**Vassallo and Taherinejad (2026), “Three factor delay learning rules for spiking neural networks.”** This recent work introduces online three-factor learning of synaptic and axonal delays as well as weights, using eligibility propagation and a top-down error signal. Its experiments are on temporal speech-recognition datasets, not flight control [R7]. It shows that learnable delays are a current research topic, but is a later-stage option for this project, not a reason to add trainable delays before basic STDP works.


**Deckers et al. (2024), “Co-learning synaptic delays, weights and adaptation in spiking neural networks.”** This study co-learns synaptic weights, synaptic propagation delays and neuron-adaptation parameters. It reports improvements on speech-recognition datasets and argues that delays correlate temporally separated input spikes while adaptation gives neurons additional history dependence [R14]. It is directly relevant to the design choice between per-neuron adaptation, per-synapse delay and weight learning, but it is not an aircraft-control result.

**Mészáros, Knight and Nowotny (2024), “Learning delays through gradients and structure: emergence of spatiotemporal patterns in spiking neural networks.”** This work studies learnable per-synapse delays and dynamic pruning/reconnection on a speech keyword-spotting task [R15]. It supports treating connection delay and connection existence as potentially learnable dimensions, but its method uses offline gradient-based optimisation and does not validate online Hebbian flight control.

**Garamov and Lobov (2026), “Learning to synchronize: A delay-based plasticity rule for temporal coding in spiking networks.”** This work proposes a delay-based plasticity rule that adjusts transmission delays to align temporal patterns [R16]. This is conceptually different from STDP weight plasticity: one mechanism changes connection strength, while the other changes arrival timing. It is a promising later experiment, not a first implementation requirement. The article is not a flight-control demonstration.

**Wada, Araujo-Estrada and Windsor (2021), “UAV Pitch Control under Delay Using Deep Reinforcement Learning with Continuous Action in Wind Tunnel Test.”** This is not an STDP study, but it directly demonstrates why controller delay must be measured and trained/tested rather than ignored. Networks trained with shorter assumed delays became more oscillatory when exposed to delay in wind-tunnel tests; longer-delay training was more conservative and avoided oscillation in the tested conditions but introduced some steady-state error [R17]. It supports robust delay-variation experiments, not a claim that any neural delay scheme is stable.

### Research conclusion

There is meaningful prior work on SNN flight control, aircraft stability augmentation, drone attitude control, reward-modulated STDP, differential Hebbian motor control, and learnable synaptic delays. The exact combination proposed here - instrument-fact inputs, specialised multidimensional aviation classifiers, local temporal pattern wiring, a Mission/Intuition proposal architecture, the existing log-sigmoid tensor scheduler, and reward-modulated STDP under TypeScript/WebGPU - was not identified in the literature reviewed. Treat this as a synthesis of established mechanisms and a new engineering hypothesis, not as a known validated architecture.

## B.13 Experiment plan and acceptance criteria

Run all experiments in the simulator with fixed seeds and reproducible configurations. Start with synthetic event fixtures before enabling any flight-simulator closed loop.

| ID | Experiment | Variants | Main question |
|---|---|---|---|
| STDP-00 | Pair-rule unit test | pre-before-post, post-before-pre, simultaneous, outside both windows | Are update signs and windows correct? |
| STDP-00A | Observation-to-classification boundary | calibrated raw samples, noise, stale/missing samples, conflicting instruments, simulator-only labels | Do tensors classify measured inputs without leaking privileged state or hiding invalid sensors? |
| STDP-00B | Event encoder | analogue activation, threshold crossing with hysteresis, differential-Hebbian baseline | Does the chosen event representation preserve meaningful onset/order without firing on every frame? |
| STDP-01 | Window sweep | vary `W+`, `W-`, `tau+`, `tau-` in simulation-time units | What window captures the model's event timing without cross-associating unrelated events? |
| STDP-02 | Multi-feature pattern | several P inputs, several Q outputs, sparse/factorised edge sets | Can a multidimensional P pattern wire to a multidimensional Q pattern without flattening it to one neuron? |
| STDP-02A | Pattern-to-pattern mapping | graded classifier activations, event-coded active features, sparse weights | Does the learned mapping preserve which source features contribute to which target features? |
| STDP-03 | Temporal reversal | original and reversed P/Q event sequences | Does reversed order produce the configured LTD rather than LTP? |
| STDP-04 | Outside-window policy | canonical zero-update versus optional timeout-LTD | Does penalising an unpaired prediction help or cause false unlearning? |
| STDP-05 | Conjunction/context | pairwise only versus explicit context gate/interaction | Can the model learn a joint condition without firing on either feature alone? |
| STDP-06 | Continuous versus event rule | thresholded STDP versus differential Hebbian | Which better handles existing activation signals and selected sample rate? |
| STDP-07 | Reward modulation | unsupervised STDP versus reward-modulated STDP | Does outcome gating improve action associations? |
| STDP-08 | Anti-Hebbian ablation | disabled versus designated competition groups | Does competition improve separability without erasing useful patterns? |
| STDP-09 | Delay ablation | zero delay, legacy tensor scheduler, fixed projection delay, fixed per-synapse delay, learnable delay | Do explicit delays add value beyond per-neuron traces and the current scheduler? |
| STDP-09A | Delay robustness | random bounded latency, variable compute batching, delayed actuator response, sensor jitter | Does performance remain stable under latency variation, or does the learned controller overfit one exact delay? |
| STDP-10 | Flight sequence replay | push/pull -> attitude -> energy/airspeed trends | Do learned relationships generalise across context and perturbation? |
| STDP-11 | Closed-loop stability | learning off, shadow, simulator learning under guard | Does adaptation improve metrics without unacceptable oscillation, overshoot or safety-margin degradation? |

### Required metrics

- pre/post pair counts by timing bucket;
- positive, negative and zero eligibility updates;
- weight distributions, saturation and sparsity over time;
- prediction precision/recall for Q occurring after P within the configured window;
- false positive rate when P occurs without the required context;
- context-conditioned performance on held-out manoeuvres/configurations;
- action outcome improvement and safety-margin metrics;
- oscillation amplitude, overshoot, settling time and event-rate growth;
- memory use, GPU buffer size/traffic and compute time per tick;
- determinism/replay mismatch count.

### Hard acceptance tests

Astra must demonstrate all of the following before any learned output can influence the simulated Servo path:

1. Pre-before-post inside the potentiation window gives a positive eligibility increment.
2. Post-before-pre inside the depression window gives a negative eligibility increment.
3. Repeated consistent pre-before-post pairings produce net strengthening; mixed/reversed pairings produce the mathematically expected net update.
4. A multidimensional P/Q test updates only the directed feature connections that are actually paired; it does not collapse a whole tensor into one pseudo-neuron.
5. An event pair outside both windows gives zero pair-based update.
6. Simultaneous events follow the documented `deltaT == 0` policy.
7. A configured connection delay changes effective `deltaT` using the arrival tick, not the emission tick.
8. Weight bounds and sign constraints always hold.
9. No update occurs from an invalid/stale event or a low-confidence/ambiguous outcome.
10. A positive outcome reinforces only eligible connections attributed to the selected action; a negative outcome depresses only those eligible connections.
11. The optional `unpaired_timeout_ltd` path is disabled by default and has separate tests.
12. The optional anti-Hebbian competition path affects only explicitly declared groups.
13. A conjunction/context test does not activate the target from one constituent feature alone unless that behaviour is explicitly trained and intended.
14. No online update changes ActionArbiter, envelope bounds, sensor calibration, fallback or simulator physics.
15. A complete run can be replayed with identical event order and matching learning updates within the declared numeric tolerance.

**Exit gate:** all tests pass in shadow mode; the learning report shows no unexplained weight saturation, no global punishment, no uncontrolled event-rate growth, and no hidden dependency on browser wall-clock timing. Only then may the existing main-plan gate for simulator-experiment mode be considered.

---

# References

**[R1]** Feldman, D. E. (2012). *The spike-timing dependence of plasticity*. Neuron, 75(4), 556-571. https://doi.org/10.1016/j.neuron.2012.08.001. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC3431193/

**[R2]** Cuppini, C., Magosso, E., Rowland, B. A., Stein, B. E., & Ursino, M. (2018). *General differential Hebbian learning: Capturing temporal relations between events in neural networks and the brain*. PLOS Computational Biology, 14(8), e1006227. https://doi.org/10.1371/journal.pcbi.1006227. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC6130884/

**[R3]** Verduzco-Flores, S., Dorrell, W., & De Schutter, E. (2022). *A differential Hebbian framework for biologically-plausible motor control*. Neural Networks, 150, 237-258. https://doi.org/10.1016/j.neunet.2022.03.002

**[R4]** *Brain-inspired learning rules for spiking neural network-based control: a tutorial* (2024). Frontiers in Neuroscience. https://pmc.ncbi.nlm.nih.gov/articles/PMC11704115/

**[R5]** *Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules* (2015). Frontiers in Neural Circuits. https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/

**[R6]** Madadi Asl, M. (2025). *Time delays in computational models of neuronal and synaptic dynamics*. Frontiers in Computational Neuroscience, 19, 1700144. https://doi.org/10.3389/fncom.2025.1700144. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12640968/

**[R7]** Vassallo, L., & Taherinejad, N. (2026). *Three factor delay learning rules for spiking neural networks*. Frontiers in Neuroscience, 20, 1814505. https://doi.org/10.3389/fnins.2026.1814505. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1814505/full

**[R8]** Chavez Arana, D., Garcia A., O. A., Garcia Carrillo, L. R., Rubio Scola, I., Espinoza, E. S., & Sornborger, A. T. (2024). *Spiking Neural Network-based Flight Controller*. 2024 Neuro Inspired Computational Elements Conference, 1-8. https://doi.org/10.1109/NICE61972.2024.10548609. Record: https://laro.lanl.gov/esploro/outputs/conferenceProceeding/Spiking-Neural-Network-based-Flight-Controller/9916500503603761

**[R9]** Federal Aviation Administration. *Airplane Flying Handbook*, Chapter 5, Maintaining Aircraft Control: Upset Prevention and Recovery Training. https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/airplane_handbook/06_afh_ch5.pdf

**[R10]** Oniz, Y. (2024). *Trajectory Control of Quadrotors via Spiking Neural Networks*. Electronics, 13(16), 3319. https://doi.org/10.3390/electronics13163319

**[R11]** Stroobants, S., De Wagter, C., & de Croon, G. (2025). *Neuromorphic Attitude Estimation and Control*. IEEE Robotics and Automation Letters, 10(5). https://arxiv.org/abs/2411.13945

**[R12]** González-Álvarez, M., Dupeyroux, J. J. G., Corradi, F., & de Croon, G. C. H. E. (2022). *Evolved neuromorphic radar-based altitude controller for an autonomous open-source blimp*. 2022 IEEE International Conference on Robotics and Automation, 85-90. https://doi.org/10.1109/ICRA46639.2022.9812149. Preprint: https://arxiv.org/abs/2110.00646


**[R13]** Foderaro, G., Henriquez, C., & Ferrari, S. (2010). *Indirect Training of a Spiking Neural Network for Flight Control via Spike-Timing-Dependent Synaptic Plasticity*. 49th IEEE Conference on Decision and Control, 911-916. https://doi.org/10.1109/CDC.2010.5717260. Author-hosted PDF: https://lisc.mae.cornell.edu/LISCpapers/CDC_Foderaro_SNNtraining2010.pdf

**[R14]** Deckers, L., Van Damme, L., Van Leekwijck, W., Tsang, I. J., & Latré, S. (2024). *Co-learning synaptic delays, weights and adaptation in spiking neural networks*. Frontiers in Neuroscience, 18, 1360300. https://doi.org/10.3389/fnins.2024.1360300. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11055628/

**[R15]** Mészáros, B., Knight, J. C., & Nowotny, T. (2024). *Learning delays through gradients and structure: emergence of spatiotemporal patterns in spiking neural networks*. Frontiers in Computational Neuroscience, 18, 1460309. https://doi.org/10.3389/fncom.2024.1460309. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11695354/

**[R16]** Garamov, K. A., & Lobov, S. A. (2026). *Learning to synchronize: A delay-based plasticity rule for temporal coding in spiking networks*. Chaos, Solitons & Fractals, 208(2), 118199. https://doi.org/10.1016/j.chaos.2026.118199

**[R17]** Wada, D., Araujo-Estrada, S. A., & Windsor, S. (2021). *Unmanned Aerial Vehicle Pitch Control under Delay Using Deep Reinforcement Learning with Continuous Action in Wind Tunnel Test*. Aerospace, 8(9), 258. https://doi.org/10.3390/aerospace8090258. Full text: https://www.mdpi.com/2226-4310/8/9/258

---

# Final instruction to Astra

Read this addendum together with `HEBBIAN_FLIGHT_CONTROL_AGENT_PLAN.md` and `ASTRA_START_HERE.md`.

1. Execute M0 only, as instructed by the main plan. Do not begin implementation before the discovery report is reviewed.
2. During M0, map the existing TypeScript/JavaScript and WebGPU architecture, raw instrument/force inputs, classifier tensors, control/action proposals, tensor representation, scheduler, epoch boundary, log-sigmoid delay calculation, current activation/event semantics, and exact timing of weight updates and output emission. Identify which values are measured observations, which are simulator-only labels, which are classified features, and which are actuator commands.
3. Add a section to the discovery report describing how the current implementation maps to Appendix A and Appendix B. Explicitly identify what is already implemented, what is missing, and what cannot be inferred from the current code.
4. Preserve the existing scheduler as the baseline. Do not delete or rewrite it during discovery.
5. After M0 approval, implement Appendix A temporal event/state instrumentation and tests before changing the learning rule.
6. Then implement the pair-based STDP reference rule and synthetic timing tests in shadow mode.
7. Add reward modulation and outcome attribution only after local timing correctness is demonstrated.
8. Compare scheduler, traces, delays and readouts with the experiment matrix. Do not choose a design because it looks more biological or more elegant; choose it based on repeatable results and complexity/cost.
9. Keep all work simulator-only. Learning remains off by default, and all deterministic safety and arbitration gates in the main plan remain non-plastic.
10. Stop at every existing milestone gate and report commands, test outcomes, plots, configuration hashes and limitations.

**The target is not a formal rule engine or a network that merely emits delayed outputs. Instruments and control-force sensors provide measured observations; perception tensors classify the aircraft situation; Mission and Intuition propose actions; the arbiter applies fixed authority constraints; Servo issues bounded commands. Within this system, a multidimensional population of units learns weighted, directed temporal associations between input patterns and output patterns, including their timing, context and outcome.**
