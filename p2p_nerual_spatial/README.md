# Neural Hebbian P2P Simulator

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
- Optional: PyTorch, for batched embedding translation on a CUDA-capable GPU

The event queue, graph simulation, sparse Hebbian state, and Haar coefficient encoding are CPU-based. Installing PyTorch does **not** move the entire simulation onto a GPU. The embedding translation backend selects CUDA when `--device auto` detects a compatible PyTorch CUDA installation and the operation is large enough to benefit.

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

PyTorch is optional. If you want GPU-assisted embedding translation, install a PyTorch build appropriate for your operating system and CUDA setup by following the official PyTorch installation selector: <https://pytorch.org/get-started/locally/>.

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

Require CUDA for embedding transforms (the program exits with an argument error if CUDA is unavailable):

```bash
python simulator.py --device cuda
```

Show periodic event-loop progress:

```bash
python simulator.py --nodes 100 --progress-every 10000
```

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
| `--device` | `auto` | Embedding transform device: `auto`, `cpu`, or `cuda` |
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

The current filtering pipeline is:

1. Add very small random background noise to the tensor.
2. Apply a separable, one-level orthonormal Haar transform along every tensor axis.
3. Apply Gaussian smoothing in coefficient space.
4. Soft-threshold small coefficients.
5. For node learning state, inverse-transform the coefficients and project the result back onto already observed sparse entries.

Node tensor filtering is periodically triggered after a configured number of timer firings. Message tensors are encoded as sparse, thresholded Haar coefficients and compressed with Python's `zlib`/DEFLATE. The receiver decompresses the coefficients and applies the inverse Haar transform, then re-applies each signal's exact occupied-layer mask and normalizes its occupied direction profiles. The message tensor is a padded stack of per-signal embeddings, with metadata rows kept in the same order so signal addresses, IDs, TTLs, targets, and embedding depth remain exact.

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
- Haar filtering and dense tensor materialization are CPU work. CUDA acceleration, when available, applies to eligible batched embedding transformations rather than the entire event loop.
- Compression savings reported for tensor coefficients exclude signal metadata and the embedding sidecar unless the output explicitly reports otherwise.

## Suggested next experiments

1. Compare learned routing with forced ripple using identical seeds and join requests.
2. Sweep TTL values and measure delivery, traffic, duplicate drops, and fallback rate.
3. Vary the stun sigmoid and beacon interval to study burst contention.
4. Measure actual serialized bytes including metadata and sidecars.
5. Profile wavelet-filter frequency and cost before scaling to 1,000 or more nodes.

The central hypothesis is intentionally modest: merging concurrent ripples should reduce transport overhead, while TTL, deduplication, and learned forwarding may reduce unnecessary propagation further.
