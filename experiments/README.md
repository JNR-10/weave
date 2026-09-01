# Experiments

Reproducible experiment configurations. Each experiment pairs a workload from `benchmarks/`
with a placement policy and a cluster composition.

## Comparison ladder

The core experiment walks progressively stronger placement policies over identical workloads,
so that any improvement can be attributed to a specific added signal rather than to the system
as a whole:

| Policy | Information used |
|---|---|
| Equal split | layer count only |
| Memory-proportional | + total device memory |
| Static profiled | + one-time startup benchmark |
| **Weave** | + live memory headroom, context length, concurrency, observed stage timings |

External reference points: single-node MLX-LM, stock MLX distributed execution, and Exo.

## Recording a run

An experiment record is only useful if someone else can interpret it later. Each run should
capture:

- the workload configuration and seed
- the placement policy and the partition it chose
- cluster composition — which Macs, their memory, and how they were connected
- communication backend (JACCL / RING)
- the model and quantization
- thermal and background-load state at the start of the run
- number of repetitions

Raw output goes to `results/` and is gitignored. Configurations are committed.

## Reporting

Report distributions, not single runs. macOS timing varies with thermal state and background
activity, and a difference smaller than the run-to-run spread is not a finding.
