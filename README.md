# Weave

**Runtime-Adaptive Distributed LLM Inference Across Heterogeneous Apple Silicon Macs**

CMPE 295A/B Master's Project — San José State University, Fall 2026 / Spring 2027

---

## Overview

Large language models are increasingly run on personal machines rather than in the cloud, and
Apple Silicon Macs suit this shift because unified memory lets the GPU address large model
weights directly. Multi-machine inference is now practical: Apple's MLX framework supports
distributed execution across physical machines over RING and JACCL, and systems such as Exo,
dnet, and prima.cpp already split one model across several consumer devices.

Splitting a model, however, does not by itself produce an efficient configuration. Existing
systems assign layers to devices from largely static information — total memory, benchmarked
compute, link bandwidth — while the best partition depends on conditions that change *within*
a session. A lengthening conversation grows the KV cache until a device that had headroom nears
its limit; concurrent requests change batching and throughput; background load and thermal
throttling make a nominally fast machine slow; and past a point, adding machines costs more in
communication than it returns.

**Weave is a runtime-adaptive placement scheduler built on top of MLX Distributed rather than
replacing it.** It profiles live cluster state, predicts each request's memory and compute
demand, observes real pipeline-stage timings, and from those signals decides *how many* Macs
should serve a workload and *how many layers* each one executes — revising that partition at
safe request boundaries.

## Research questions

| | Question |
|---|---|
| **RQ1** | Does live workload information improve model placement over equal and static hardware-based splitting? *(primary)* |
| **RQ2** | When is distributed inference actually useful — when should a workload use one, two, or three Macs? |
| **RQ3** | Can the scheduler detect changing conditions and pick a better configuration without restarting the cluster? |

## Components

| Component | Responsibility |
|---|---|
| `src/weave/profiler` | Measures per-Mac available unified memory, layer execution speed, and interconnect latency/bandwidth |
| `src/weave/estimator` | Predicts per-device memory demand from context length, output length, and request concurrency |
| `src/weave/scheduler` | Selects participating Macs and the per-device layer assignment; revises at safe request boundaries |
| `src/weave/telemetry` | Runtime monitoring — stage timings, per-node memory, network cost, OOM events |

## Evaluation

Weave is compared against progressively stronger placement baselines to isolate whether the
runtime signals actually carry value:

```
equal split  →  memory-proportional  →  static profiled  →  Weave (runtime-adaptive)
```

plus single-node MLX-LM and stock MLX distributed execution as external reference points.

**Metrics:** time to first token, time per output token, tokens/sec, aggregate throughput under
concurrency, P95 request latency, per-Mac memory usage, out-of-memory failures, scheduler
decision time, and partition-change cost.

A negative result is still a result: if adaptation pays off only within certain workload ranges,
identifying those ranges and the overhead threshold is itself a contribution.

## Tech stack

- **Hardware:** Apple Silicon Macs only (heterogeneous cluster of 2–3 machines)
- **Inference:** MLX / MLX-LM
- **Communication:** JACCL (RDMA over Thunderbolt 5) where supported; MLX RING over TCP as fallback and comparison
- **Parallelism:** pipeline parallelism (tensor parallelism as baseline/extension only)

## Repository layout

```
src/weave/       scheduler implementation (profiler, estimator, scheduler, telemetry)
benchmarks/      workload definitions and generators
experiments/     reproducible experiment configurations and run instructions
results/         experiment output (structure tracked, raw runs gitignored)
scripts/         setup and cluster utilities
docs/            project proposal and design documents
```

## Roadmap

- [ ] **M1 — Reproduce state of the art.** Configure MLX distributed, test JACCL/RING, run Exo, benchmark one/two/three Macs. *Milestone: reproducible baseline for our actual cluster.*
- [ ] **M2 — Profiling and measurement.** Free memory, per-device stage time, network performance, KV-cache estimates. *Milestone: Weave can describe cluster and workload state for any request.*
- [ ] **M3 — Static placement optimizer.** Equal, memory-proportional, and measured-performance splits under memory constraints. *Milestone: valid unequal layer assignments.*
- [ ] **M4 — Runtime/workload-aware scheduler.** Context-aware memory prediction, concurrency, live stage timing, one/two/three-Mac decision, re-evaluation at safe boundaries. *Milestone: partition changes as conditions change.*
- [ ] **M5 — Experiments and adaptation.** Long contexts, concurrency, induced background load and memory pressure, network variation.
- [ ] **M6 — Full evaluation.** Feature freeze; baselines, ablations, final figures and tables.
- [ ] **M7 — Buffer, report, and demonstration.**

## Out of scope

NVIDIA GPUs, Jetson boards, Windows/Linux clusters, iOS devices, custom distributed networking,
new quantization algorithms, cryptographic private inference, new model architectures,
distributed training, custom Metal kernels, and internet-scale P2P inference. Fault tolerance
and stall recovery are a stretch goal only.

## Team

| Member | Primary responsibility |
|---|---|
| Jainil Rana | MLX distributed execution, pipeline integration, model execution |
| Urmi Shah | Profiling, performance model, placement optimizer, adaptive scheduler |
| Mohit Barade | Benchmark framework, telemetry, workload generation, experiments |

All members contribute to integration, experiments, analysis, and the final report.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the working agreement.

**Project Advisor:** Kaikai Liu, Associate Professor, Department of Computer Engineering, SJSU

## License

[MIT](LICENSE)
