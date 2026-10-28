# Neural Hebbian P2P Simulator

**Author:** [Sonia Code](https://github.com/sadhbh-c0d3)

*Copyright (c) 2026 Sonia Kolasinska / Imperfect Protocol. All Rights Reserved.*

----

An experimental, event-driven Python simulator for a peer-to-peer network that forms dynamically, propagates beacon and announce signals, coalesces traffic during short stun windows, and learns which next hops have been useful.

The aim is not perfect neural routing. It is to improve ordinary ripple gossip under contention by combining:

- **Stun-based coalescing** so a node can combine pending work into one outbound bundle per selected neighbour.
- **TTL and duplicate suppression** to bound propagation and prevent repeated copies from circulating indefinitely.
- **Sparse Hebbian learning** to bias forwarding toward neighbours associated with successful paths.
- **Compact fuzzy embeddings** with angle × hop dimensions.
- **Haar wavelet filtering** with background noise, Gaussian smoothing in coefficient space, soft-thresholding, and inverse reconstruction.
- **Lossless protocol metadata** for exact address identities, signal IDs, destinations, and TTL values.
- **A cancellable mesh fallback** when two active nodes have not discovered each other before the timeout.

This is a research prototype. The output measures simulated behavior; it does not establish that learned routing will outperform a baseline in every topology or workload. Use controlled comparisons with the same seed and configuration when evaluating changes.

## Requirements

- Python 3.10 or later recommended
- NumPy
- SciPy
- Optional: PyTorch, for batched embedding translation plus Haar/Gaussian/inverse-Haar processing on CUDA or Apple MPS (Metal)

The event queue, graph simulation, sparse Hebbian state, zlib compression, and serialization remain CPU-based. Installing PyTorch does **not** move the entire simulation onto a GPU. When `--device cuda` or `--device mps` is selected and available, embedding batches of at least 8 use the selected accelerator. The message/node wavelet pipeline also runs Haar decomposition, Gaussian filtering, coefficient thresholding, and inverse Haar on that accelerator. Wavelet coefficients cross back to CPU only at the compression/serialization boundary, or after inverse reconstruction when sparse node state must be updated. `--device auto` prefers CUDA, then Apple MPS, then CPU.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

PyTorch is optional. If you want accelerator support, install a PyTorch build appropriate for your operating system and accelerator using the official selector: <https://pytorch.org/get-started/locally/>. On Apple Silicon, use a build with MPS support. The event queue, sparse Hebbian tables, and zlib/JSON serialization remain on the CPU; eligible embedding batches and the Haar/Gaussian/inverse-Haar tensor pipeline use CUDA or MPS.

## Run it

The examples below assume the script is named `simulator.py` and is in the current directory. If your copy has a different filename, substitute that filename in the commands.

Run the built-in tests first:

```bash
python simulator.py --self-test
```

Run the default 100-node simulation:

```bash
python simulator.py
```

Run a reproducible 100-node simulation explicitly:

```bash
python simulator.py --nodes 100 --seed 42
```

Try a larger graph:

```bash
python simulator.py --nodes 1000 --max-events 1500000
```

The larger run may be computationally expensive, particularly when node tensor wavelet filtering is triggered. Check the processed-event count and the warning about event-budget exhaustion before comparing results.

Set the propagation TTL:

```bash
python simulator.py --nodes 100 --ttl 4
```

Force CPU embedding transforms:

```bash
python simulator.py --device cpu
```

Require CUDA for embedding transforms and wavelet tensor operations (the program exits with an argument error if CUDA is unavailable):

```bash
python simulator.py --device cuda
```

Use Apple's Metal Performance Shaders backend for embedding transforms and wavelet tensor operations on a supported Mac with an MPS-enabled PyTorch build:

```bash
python simulator.py --device mps
```

By default, `--device auto` prefers CUDA, then Apple MPS, then CPU. Verify MPS support in the active Python environment with:

```bash
python -c "import torch; print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available())"
```

If `MPS available` is `False`, `--device mps` exits with an explanatory error rather than silently pretending to use the GPU. On a CPU-only machine, use `--device cpu` or `--device auto`.

Show periodic event-loop progress:

```bash
python simulator.py --nodes 100 --progress-every 10000
```

Capture every scheduled message as one JSON Lines row, with exact addresses and
signal metadata plus the encoded tensor payload represented as a long hexadecimal
string (the on-wire tensor bytes, rendered in a blockchain-style hex field):

```bash
python simulator.py --nodes 100 --seed 42 --event-log messages.jsonl
```

Each row includes `sent_at`, `dispatch_time`, sender and recipient node IDs and
20-byte addresses, signal IDs/types/addresses/TTL/hop counts, tensor shape and
encoding, byte length, and `tensor_hex`. The log includes ordinary bundled
messages, local self-messages and internal join-intent messages. Join-intent rows
have an empty tensor because they are simulation-control events, not tensor
transmissions. The file may get large because `tensor_hex` deliberately stores
all encoded bytes as text.

Disable processing stages individually for controlled experiments:

```bash
# Neutral/unit weights, no noise, no Gaussian smoothing, with Haar + compression.
python simulator.py --nodes 100 --seed 42 --no-learning --no-noise --no-gaussian

# Keep noise but bypass Haar, Gaussian filtering, coefficient thresholding and compression.
python simulator.py --nodes 100 --no-haar

# Keep Haar thresholding and receiver-side inverse Haar, but send dense float32 coefficients.
python simulator.py --nodes 100 --no-compression

# Disable TTL enforcement, or disable embedding rotations while still shifting the 8-layer stack.
python simulator.py --nodes 100 --no-ttl
python simulator.py --nodes 100 --no-twist

# Tune background noise, coefficient blur, threshold and reward strength.
python simulator.py --noise-strength 0.00001 --gaussian-width 0.30 --threshold 0.006 --hebbian-reward 0.35
```


### No-learning baseline

Use `--no-learning` to disable Hebbian updates and node-tensor denoising while keeping message tensor processing, Haar/Gaussian filtering, compression, TTL, deduplication, beacon traffic, announcements, and fallback behavior enabled. Every logical routing weight is an implicit unit value (`1.0`), so all nodes make neutral, uniform weight-based decisions without allocating a dense tensor. Success-feedback packets remain enabled for a fair protocol comparison, but they do not change weights.

```bash
python simulator.py --nodes 100 --seed 42 --no-learning
```

Compare with the same command without `--no-learning`, keeping all other arguments identical. The difference in traffic and delivery is the incremental effect of learned weights, subject to run-to-run timing and stochastic wavelet noise.

## Command-line options

| Option | Default | Meaning |
|---|---:|---|
| `--nodes` | `100` | Number of candidate nodes to join |
| `--duration` | calculated | Simulation-time horizon. By default it follows the last planned join intent plus 8 seconds, with a minimum horizon of 10 seconds |
| `--beacon-interval` | `1.0` | Simulated seconds between beacon opportunities |
| `--join-interval` | `0.05` | Simulated spacing between initial join intents |
| `--extra-connect-requests` | `20` | Additional connection requests between existing non-neighbour nodes |
| `--announce-spacing` | `0.25` | Simulated spacing between extra announce requests |
| `--forwarding-threshold` | `0.20` | Learned-route threshold applied to the signal coefficient and learned weight; ripple fallback is available when learned routing is not selected |
| `--fallback-timeout` | `5.0` | Simulated seconds before unresolved active-active announce requests use mesh fallback |
| `--ttl` | `8` | Initial TTL per signal, decremented at each relay |
| `--seed` | `42` | Random seed for repeatable graph and simulation setup |
| `--max-events` | `750000` | Maximum number of dispatched events |
| `--progress-every` | `0` | Print progress every N events; zero disables progress output |
| `--device` | `auto` | Embedding transform device: `auto`, `cpu`, `cuda`, or `mps`. Auto prefers CUDA, then Apple MPS, then CPU |
| `--no-learning` | off | Disable Hebbian updates and node-tensor denoising, use uniform unit weights, keep message processing enabled |
| `--no-noise` | off | Disable random noise injection before tensor processing |
| `--no-haar` | off | Bypass Haar decomposition; also disables coefficient Gaussian filtering and compression, but does not disable noise |
| `--no-gaussian` | off | Disable Gaussian smoothing in Haar-coefficient space and embedding-rotation smoothing |
| `--no-compression` | off | Send dense float32 Haar coefficients instead of sparse zlib/DEFLATE. Receiver still performs inverse Haar |
| `--no-ttl` | off | Do not enforce or decrement TTL; useful for controlled propagation experiments |
| `--no-twist` | off | Do not rotate inherited embedding layers; still shift the fixed stack and add a fresh up-arrow layer |
| `--noise-strength` | `0.00001` | Standard deviation of Gaussian background noise relative to unit-valued tensor entries |
| `--gaussian-width` | `0.30` | Gaussian sigma applied in wavelet coefficient space |
| `--threshold` | `0.006` | Soft-threshold for message Haar coefficients |
| `--node-threshold` | `0.00005` | Soft-threshold for internal node-learning tensor coefficients |
| `--hebbian-reward` | `0.35` | Base successful-route reward. Default preserves the current 0.12 beacon reward and 0.35 success-feedback reward; both scale proportionally |
| `--event-log` | none | Write one JSONL row per scheduled message, with signal metadata and encoded tensor bytes in hex |
| `--self-test` | off | Run assertions and protocol tests, then exit |

## How the simulation works

### 1. Event-driven engine

A priority queue orders event tasks by simulated `dispatch_time`. The engine advances its `now` clock to the next event timestamp. The queue supports timer tasks, message deliveries, and cancellable fallback tasks. A cancelled task is lazily skipped when it reaches the queue head.

A node has one current stun timer. When a message arrives, the first message anchors the stun cluster. Later messages update the node state and replace its timer using a sigmoid duration based on the number of accepted signals in the cluster. Incoming handlers do not directly send network messages. On timer dispatch, the node chooses forwarding routes and emits at most one bundled message to each selected neighbour.

### 2. Dynamic graph formation

Nodes are initially inactive candidates. The demo starts by linking two outsiders. Further join intents connect a currently active node to a new outsider. It then schedules additional requests between active nodes that are not directly connected. These pairs send mutual announces. If discovery does not establish the link within the configured timeout, a fallback event creates the external mesh link.

The demo uses a seeded, generated join plan to make experiments repeatable. It is a simulated workload, not a live peer-discovery implementation.

### 3. Signals and exact metadata

A transport `Message` can carry multiple logical signal items, including:

- **Beacon**: identifies its origin and advertises its embedding.
- **Announce**: identifies an exact source and target address for discovery.
- **Success feedback**: travels back along recorded route state so nodes can reward paths associated with successful discovery.

The simulator uses deterministic 20-byte address-like values derived from node IDs. These are test identities, not real Ethereum addresses. Origin and target addresses remain exact 20-byte values in signal metadata. Embeddings can be smoothed or quantized, but metadata is not blurred. Signal IDs, TTL, hop count, and route state support duplicate suppression and bounded propagation.

### 4. Embeddings and learning tensor

A compact embedding is a fixed `8 × 8` array representing **angle × the eight most recent hop-history layers**. Layer 0 is always the current node's canonical unit-up arrow. The remaining layers contain inherited path directions, transformed into the current node's frame. Angles are circular, bin 0 points up (+Y), and positive rotation is clockwise. For each K→L relay, the simulator calculates the signed direction angle of the K→L displacement, rotates every occupied layer by that angle, shifts the layers down one slot, discards the oldest layer, and pushes a fresh unit-up arrow at layer 0. The occupied-layer depth is retained as exact metadata so noise and wavelet reconstruction cannot invent path history in the seven initially empty layers. Each occupied angular profile is kept as a normalized fuzzy direction distribution.

The node's logical learning tensor is:

```text
Source(8 × 8) × Target(8 × 8) × Neighbour(8 × 1) × Time(8)
```

The neighbour is represented by an angle bin with a singleton hop dimension. The learning tensor is sparse: only observed source-target-neighbour combinations allocate an eight-slot time history. Learned scores include age-based decay. Positive updates are bounded by a maximum weight.

The tensor is a fuzzy routing hint, not a guarantee of shortest-path routing. Address-derived coordinates are synthetic and do not encode true network topology.

### 5. Haar filtering and message compression

The default filtering pipeline is:

1. Add low-strength random background noise before decomposition.
2. Apply a separable, one-level orthonormal Haar transform along every tensor axis.
3. Apply Gaussian smoothing in coefficient space.
4. Soft-threshold small coefficients.
5. For node learning state, apply inverse Haar and project the result back onto already observed sparse entries.

Node tensor filtering is periodically triggered after a configured number of timer firings. Message tensors are encoded as sparse, thresholded Haar coefficients and compressed with Python's `zlib`/DEFLATE. The receiver decompresses the coefficients and applies inverse Haar, then re-applies each signal's exact occupied-layer mask and normalizes its occupied direction profiles. The message tensor is a padded stack of per-signal embeddings, with metadata rows kept in the same order so signal addresses, IDs, TTLs, targets, and embedding depth remain exact.

Pipeline switches are intended for ablation experiments. `--no-noise` bypasses noise injection. `--no-haar` sends the noisy raw float32 tensor and automatically disables Gaussian coefficient filtering, coefficient thresholding, and compression. `--no-gaussian` removes both wavelet-space Gaussian blur and the small Gaussian smoothing used during embedding rotation. `--no-compression` sends dense float32 Haar coefficients, while the receiver still applies inverse Haar. `--no-ttl` leaves TTL metadata in place but neither decrements nor enforces it. `--no-twist` shifts the eight-layer history but does not rotate or smooth inherited angle profiles. `--threshold` controls message coefficients and `--node-threshold` separately controls the learning tensor so the two very different value scales retain sensible defaults.

The current logical dense tensor has shape `8 × 8 × 8 × 8 × 8 × 8` (262,144 float32 values, about 1 MiB before overhead). Node storage is sparse, but a node wavelet-filter pass temporarily materializes a dense array. This can become a performance bottleneck as the simulation grows.

## Interpreting the output

The final report includes measurements grouped into:

- **Graph formation**: nodes joined, edge count, connectedness, announce discovery, P2P links, and fallback links.
- **Messages and routing**: outbound packet count, signal copies, packet reduction from aggregation, bundle size, wavelet payload sizes, learned-route versus ripple copies, duplicate drops, TTL drops, and no-route drops.
- **Learning and event loop**: message envelopes, timers, stale timer skips, accepted signals, success feedback, reward totals, Haar denoise passes, sparse combinations, and stun cluster size.
- **Topology summary**: minimum, median, mean, and maximum node degree.

A high packet-reduction percentage mostly reflects **bundling**. To measure the incremental value of learned routing, compare it against a ripple-routing baseline with the same graph, seed, join schedule, beacon schedule, TTL, deduplication, and bundling. Compare useful delivery and fallback rates as well as packet and signal-copy counts.

## Self-tests

The built-in self-tests cover the current implementation, including:

- Haar transform/inverse round-trip across all axes
- Eight-layer embedding rotations, accumulated hop twists, oldest-layer eviction, and preservation of zero-filled unused layers
- Noise, Gaussian coefficient blur, thresholding, and sparse node denoising
- Compressed message tensor encoding/decoding
- Exact 20-byte address metadata
- Independent pipeline switches, no-twist behavior, and JSONL event logs with hex tensor payloads
- Priority queue ordering and simulation time
- Stun anchoring, timer replacement, and stale timer cancellation
- TTL expiry and duplicate suppression
- Success feedback and route rewards
- Dynamic joins, announce discovery, and cancellable mesh fallback

Run them with:

```bash
python simulator.py --self-test
```

## Current limitations

- This is a simulator prototype, not a production P2P transport or Ethereum networking stack.
- Address-derived coordinates are synthetic. They are useful for deterministic directional hints but do not represent real graph distance.
- Learned routing is heuristic and may fall back to ripple forwarding when it is uncertain.
- The mesh server is modeled as a simulated fallback link.
- The event queue, sparse Hebbian tables, dense-to-sparse projection, zlib compression, and serialization remain CPU work. CUDA or Apple MPS accelerates eligible embedding batches and the Haar/Gaussian/inverse-Haar tensor pipeline, but not the entire event loop.
- Compression savings reported for tensor coefficients exclude signal metadata and the embedding sidecar unless the output explicitly reports otherwise.

## Suggested next experiments

1. Compare learned routing with forced ripple using identical seeds and join requests.
2. Sweep TTL values and measure delivery, traffic, duplicate drops, and fallback rate.
3. Vary the stun sigmoid and beacon interval to study burst contention.
4. Measure actual serialized bytes including metadata and sidecars.
5. Profile wavelet-filter frequency and cost before scaling to 1,000 or more nodes.

The central hypothesis is intentionally modest: merging concurrent ripples should reduce transport overhead, while TTL, deduplication, and learned forwarding may reduce unnecessary propagation further.
