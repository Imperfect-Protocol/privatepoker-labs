Ran check bugs in Engineer.md
Analyzed Engineer.md:L270-350
Analyzed error-neuron.ts:L1-78
Analyzed neurons.ts:L1-100

Here is a deep architectural analysis and research report on the tensor topology, training methodology, and Hebbian/error-learning mechanics for the autonomous drone brain.

---

# 1. Tensor Topology: Ranks, Dimensions & Division of Labor

### Critical Evaluation of the Current Split (4 Perception + 2 Skills + 1 Goal + 1 Servo)

In the current codebase:
1. **Perception redundancy**: 
   * `LocationTensor` ($alt \times hsi$)
   * `EnergyTensor` ($alt \times asi \times vsi \times pitch \times rpm$)
   * `FlightTensor` ($vsi \times asi \times pitch \times bank \times turn$)
   * `ControlsTensor` ($ail \times elev \times rud \times thr$)
   * *Problem*: $alt$, $asi$, $vsi$, and $pitch$ are duplicated across multiple tensors. When the aircraft enters an unusual attitude, both `EnergyTensor` and `FlightTensor` receive conflicting projections, forcing downstream skills to resolve redundant and partially uncorrelated class codes.
2. **Artificial "Left/Right" Split in Skills**:
   * Dividing skills by perception (Left: location, energy) vs. execution (Right: flight, controls) is an interesting biological metaphor, but it introduces an unnecessary bottleneck: flight dynamics require coordinating *aerodynamic energy* ($vsi, asi, pitch$) directly with *aerodynamic flight surfaces* ($elev, ail, rud$). Splitting them into two disjoint panels connected only by a single 1-wire bridge delays critical cross-modal reflexes (e.g., spin recovery requires both rudder and idle throttle).
3. **Missing "Current Phase" Estimation (Interoception)**:
   * The current `GoalTensor` combines $(goal, element, situation\_bus) \to (new\_goal, new\_element)$. It conflates *where we want to be* with *where we actually are*.

---

### The Recommended Optimal Tensor Topology

To achieve the desired flow where the brain assesses **Current Phase**, computes the **Transition ($\Delta P$) to Desired Phase**, and **dispatches corrections with bidirectional interrupt capability**, the system should be organized into **4 Functional Layers (7 Tensors)**:

```
[ Airframe / Instruments / Controls ]
                 │
                 ▼
┌───────────────────────────────────────────────┐
│           LAYER 1: PERCEPTION LAYER           │
│  (Pure sensory discretization & classification)│
│                                               │
│  • FlightAttitudeTensor: [pitch, bank, turn, vsi, asi] → Flight State (32)
│  • EnergyPropulsionTensor: [alt, rpm, throttle, asi, vsi] → Energy State (16)
│  • ProprioceptionTensor: [elev, ail, rud, trim] → Control Surface Stops (64)
└───────────────────────┬───────────────────────┘
                        │ (Sensory Classes)
                        ▼
┌───────────────────────────────────────────────┐
│        LAYER 2: PHASE ESTIMATION & GOAL       │
│                                               │
│  • CurrentPhaseTensor: (Sensory State + Proprioception)
│                        → Actual Phase (P_actual)
│                                               │
│  • GoalPlanTensor:     (Mission/Goal + Elapsed/Progress)
│                        → Desired Phase (P_desired)
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│       LAYER 3: BIDIRECTIONAL SKILL TENSOR     │
│             (The Transition Engine)           │
│                                               │
│  • SkillTransitionTensor:                      │
│      Forward:  (P_actual × P_desired × Sense) → Transition Action ΔP
│      Backward: If P_actual contradicts P_desired (e.g. Stall during Cruise),
│                fires OVERRIDE interrupt to GoalPlanTensor.
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│         LAYER 4: SERVO ACTUATION TENSOR       │
│                                               │
│  • ServoTensor: (Transition Action ΔP × Proprioception × Flight State)
│                 → Δ Deflections [Δail, Δelev, Δrud, Δthr]
└───────────────────────────────────────────────┘
```

#### Tensor Rank & Dimension Budget

| Tensor Name | Input Shape (Ranks) | Output Dimension | Mathematical Role |
| :--- | :--- | :--- | :--- |
| **`FlightAttitudeTensor`** | $24_{pitch} \times 24_{bank} \times 7_{turn} \times 16_{vsi} \times 8_{asi}$ | 32 Flight States | Aerodynamic attitude classification |
| **`EnergyPropulsionTensor`**| $8_{alt} \times 8_{rpm} \times 7_{thr} \times 8_{asi} \times 16_{vsi}$ | 16 Energy States | Mechanical & kinetic potential |
| **`ProprioceptionTensor`** | $15_{elev} \times 11_{ail} \times 11_{rud}$ | 64 Control States | Tactile / control stick coordinates |
| **`CurrentPhaseTensor`** | $32_{flight} \times 16_{energy} \times 64_{control}$ | 32 Actual Phases | State estimation: *"Where are we right now?"* |
| **`GoalPlanTensor`** | $32_{goal} \times 32_{phase\_req} \times 8_{progress}$ | 32 Desired Phases | Intent: *"Where should we be next?"* |
| **`SkillTransitionTensor`** | $32_{P\_act} \times 32_{P\_des} \times 32_{flight} \times 16_{energy}$ | 64 Transition Vectors ($\Delta P$) | State-action mapping & bidirectional interrupt |
| **`ServoTensor`** | $64_{\Delta P} \times 64_{control} \times 32_{flight}$ | $4 \times \text{Deflection Stops}$ | Motor commands: $[\Delta ail, \Delta elev, \Delta rud, \Delta thr]$ |

---

### Handling the "Stall During Level Flight" Example

1. **Perception**:
   * `EnergyPropulsionTensor` observes $asi \le 2$ (airspeed falling below 55 mph) and $vsi \le 5$ (sink rate increasing) $\to$ fires class `9: developed stall`.
   * `FlightAttitudeTensor` observes $pitch \ge 16$ (nose high) $\to$ fires class `11: nose high`.
2. **Current Phase Assessment**:
   * `CurrentPhaseTensor` combines `developed stall` + `nose high` + `trim elevator` $\to$ computes **$P_{\text{actual}} = \text{Stall}$**.
3. **Intent vs. Reality Mismatch**:
   * `GoalPlanTensor` was dictating **$P_{\text{desired}} = \text{Level Cruise}$**.
4. **Bidirectional Skill Reaction**:
   * **Interrupt Pulse**: `SkillTransitionTensor` detects an invalid delta ($\text{Cruise} - \text{Stall}$). The error energy exceeds the threshold. It fires a backward inhibit signal into `GoalPlanTensor`, which forces an emergency goal change: `Goal = Spin/Stall Recovery`, `P_desired = Nose Down Unload`.
   * **Forward Motor Command**: Concurrently, `SkillTransitionTensor` outputs the learned emergency transition vector: $\Delta P = \text{"Unload Wing + Maximum Power"}$.
   * `ServoTensor` receives $\Delta P$ and current surface positions, immediately executing:
     $$\Delta elev = \text{Full Forward (Push)}, \quad \Delta thr = \text{Full Throttle}$$

---

# 2. Perception & Goal Training: Isolation, Fuzzing & Wavelets

### Should Perception Tensors Be Trained in Isolation?
**Yes, absolutely.** 

* **The Principle of Invariant Feature Representation**: In neural biological systems (e.g., visual cortex V1/V2), edge and orientation detectors do not depend on whether an animal wants to run or sleep. 
* If perception tensors are trained end-to-end with flight lessons, sensory classifications become contaminated by motor bias (e.g., an airspeed of 55 mph might be classified differently during takeoff vs. landing).
* Training perception tensors in isolation guarantees that a given flight instrument state **always deterministically maps to the exact same situation code**.

### Training Methodologies: Fuzzing, Haar Wavelets vs. Gaussian Filters

A common issue in lookup tensors is the "cliff problem": if training only stamps exact integer indices, an input that is 1 knot or 0.5 degrees off lands in an empty bin and defaults to noise.

```
       Exact Binning                 Gaussian / Fuzzy Membership               Haar Multi-Scale
     ┌───┐                          ┌───┐                                      ┌───────┐
     │   │                          │ █ │                                      │   █   │
 ────┴───┴────                  ────┴─█─┴────                              ────┴───█───┴────
 (1-off = Zero Recall)     (Spread across neighbors: exp(-d²/2σ²))      (Coarse wave + fine detail)
```

#### The Optimal 3-Stage Training Pipeline:

```mermaid
flowchart LR
    A["Grid Sampling / Monte Carlo Fuzzing"] --> B["Gaussian Kernel Smoothing"]
    B --> C["Haar Wavelet Soft-Thresholding"]
    C --> D["Sparse PulseNeuron Weights"]
```

1. **Continuous Monte Carlo Fuzzing**:
   * Rather than brute-forcing all intermediate values or just discrete class centers, sample the continuous physical space uniformly with added aerodynamic jitter:
     $$x_{\text{sampled}} = x_{\text{nominal}} + \mathcal{N}(0, \sigma_{\text{sensor}}^2)$$
   * Run the deterministic aviation equations (the "Teacher" functions) to generate ground-truth labels.

2. **Gaussian Smoothing Across Adjacent Bins**:
   * Instead of assigning 100% weight to bin $i$, spread the activation across neighboring bins using a Gaussian kernel:
     $$W(j) = \exp\left(-\frac{(j - i)^2}{2\sigma^2}\right), \quad j \in [i-1, i+1]$$
   * This gives the drone **smooth generalization**: if bank angle drifts into $16^\circ$ when the boundary is $15^\circ$, the activation smoothly shares credit between "Level" and "Shallow Turn".

3. **Haar Wavelet Decomposition & Soft-Thresholding**:
   * Decompose the multi-dimensional weight matrix into Haar wavelet bases:
     $$\mathbf{W} \xrightarrow{\text{Haar DWT}} [\mathbf{A}_{\text{coarse}}, \mathbf{D}_{\text{fine}}]$$
   * Apply soft-thresholding to the detail coefficients $\mathbf{D}$:
     $$\text{shrink}(d, \lambda) = \text{sign}(d) \cdot \max(0, |d| - \lambda)$$
   * **Why this matters for flight**:
     * Eliminates high-frequency noise from turbulent flutter.
     * Compresses the sparse memory (zeroing out $\approx 70\%$ of tiny non-informative associations).
     * Preserves sharp cliffs where physics demands them (e.g., stall angle of attack at $16^\circ$ critical AoA).

### Training the `GoalPlanTensor`
* Treat the mission curriculum as a **Hierarchical Directed Acyclic Graph (DAG)** or Semi-Markov State Machine:
  * Manually imprint canonical lesson sequences:
    $$\text{Takeoff} \to [\text{Roll} \to \text{Rotate} \to \text{Ground Effect} \to \text{Climb} \to \text{Level Off}]$$
  * Train transitions based on *completion criteria* (e.g., $alt \ge 1000\text{ ft} \land vsi \approx 0 \implies \text{Transition}(\text{Climb} \to \text{Level Off})$).

---

# 3. Dual-Engine Learning: Hebbian Association vs. Predictive Error Coding

A purely Hebbian network ("fire together, wire together") is unstable for flight control: it forms runaway positive feedback loops, reinforcing whatever the plane happens to do (including stalling or spinning).

To build a robust learning system, the **Skill Tensor** must unite two complementary biological learning laws:
1. **Hebbian Synaptic Plasticity (LTP)**: Forward pattern association.
2. **Cerebellar Predictive Error Correction (LTD / Climbing Fibers)**: Directional error trimming.

```
                      ┌───────────────────────────────────────┐
                      │        FORWARD HEBBIAN ARM            │
                      │  "What worked and fired together"     │
                      │   ΔW_assoc = η_h · (Pre ⊗ Post)       │
                      └──────────────────┬────────────────────┘
                                         │
                                         ▼
[ Sensory / Phase Inputs ] ────────► ( ⊕ ) ────────► [ Servo Actions ]
                                         ▲
                                         │
                      ┌──────────────────┴────────────────────┐
                      │       CEREBELLAR ERROR ARM            │
                      │   "Climbing fiber signed correction"  │
                      │   ΔW_error = -η_e · (Actual - Target) │
                      └───────────────────────────────────────┘
```

### 1. The Hebbian Associative Law (Forward Coincidence)
When the student observes the instructor or achieves a stable phase transition, pre-synaptic state $\mathbf{x}$ and post-synaptic action/phase $\mathbf{y}$ fire within a tight coincidence window $\Delta t$:
$$\Delta W_{\text{Hebbian}}(i, j) = \eta_{\text{Hebbian}} \cdot x_i(t) \cdot y_j(t) \cdot e^{-\frac{\Delta t}{\tau_{\text{coincidence}}}}$$
* **Togetherness Interval ($\tau_{\text{coincidence}}$)**:
  * In Skynerd's 50 Hz physics ($dt = 0.02\text{ s}$) with a decision window of $0.2\text{ s}$ (10 ticks), the coincidence window $\tau$ should span **$0.2\text{ s} \le \tau \le 0.4\text{ s}$** (1 to 2 decision cycles).
  * Anything outside this window represents a disconnected consequence (e.g., pulling elevator now does not cause a pitch-up 3 seconds later).

### 2. The Skill Tensor as an Error Neuron (Predictive Coding)
The error mechanism should follow the Purkinje cell / climbing fiber model in the cerebellum:
* The Skill Tensor maintains an **efference copy** of what outcome it predicted:
  $$\hat{s}(t+1) = \text{SkillTensor}_{\text{predict}}(s(t), a(t))$$
* When the actual outcome arrives $s(t+1)$:
  $$\mathbf{e}(t+1) = s(t+1) - \hat{s}(t+1)$$
* **Anti-Hebbian Plasticity (Long-Term Depression - LTD)**:
  If the commanded action resulted in a dangerous divergence (e.g., increased bank angle during an overbank condition):
  $$\Delta W_{\text{error}}(s, a) = -\eta_{\text{error}} \cdot \text{SignLaw}(\mathbf{e}) \cdot \|\mathbf{e}\|$$
* This suppresses wrong stick inputs immediately, while the Hebbian arm slowly consolidates the successful recoveries.

---

# Summary & Implementation Roadmap

1. **Eliminate Sensory Duplication**: Merge the 4 overlapping sensors into 3 orthogonal tensors:
   * `FlightAttitudeTensor` (Attitude & Airframe dynamics)
   * `EnergyPropulsionTensor` (Power & Energy state)
   * `ProprioceptionTensor` (Control surface positions)
2. **Decouple "Current State" from "Desired State"**:
   * Let `CurrentPhaseTensor` evaluate reality ($P_{\text{actual}}$).
   * Let `GoalPlanTensor` dictate intent ($P_{\text{desired}}$).
3. **Unify the Left/Right Skills into a Bidirectional Transition Engine**:
   * Compute transition vector $\Delta P = P_{\text{desired}} - P_{\text{actual}}$.
   * Fire backward interrupt override pulses if $P_{\text{actual}}$ indicates an aerodynamic emergency.
4. **Train Perception in Isolation**:
   * Use Monte Carlo continuous fuzzing.
   * Smooth with Gaussian kernels for adjacent-bin generalization.
   * Denoise and compress using Haar wavelet soft-thresholding.
5. **Combine Hebbian LTP with Cerebellar LTD**:
   * Hebbian forward association over a $0.2\text{ s} - 0.4\text{ s}$ window.
   * Signed error-neuron depression to stamp out dangerous control commands.
