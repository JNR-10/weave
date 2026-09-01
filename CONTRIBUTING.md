# Contributing

Working agreement for the Weave project team (CMPE 295A/B).

## Responsibilities

Each member owns one layer of the system, reviews the others, and contributes to integration,
experiments, analysis, and the final report.

| Member | Owns | Answers |
|---|---|---|
| Member 1 | MLX Distributed integration, pipeline execution, JACCL/RING backends, partition application and switching, workload generator | What does reconfiguration cost, and does the interconnect change that answer? |
| Member 2 | Cluster profiler, runtime telemetry, KV-cache and memory estimator | How accurate is the performance model — do predicted memory and stage times match reality? |
| Member 3 | Placement policies (equal, memory-proportional, static profiled, adaptive), how-many-Macs decision, ablation design | Does adaptation beat static placement, and in which workload ranges? |

Shared across the team: Month 1 reproduction of the current state of the art, integration, the
M5–M6 experiment campaign, analysis, and the final report. The failure and stall recovery
stretch goal attaches to Member 1 if it is reached.

Ownership means *responsible for*, not *sole author of*. Anyone may work anywhere; the owner
reviews changes in their area. Each role carries its own open question so that every member has
a distinct, visible contribution in the final report.

## Branches

`main` is always working. No direct pushes — everything lands through a pull request.

Branch names: `<type>/<area>-<short-description>`

```
feat/scheduler-layer-balancer
fix/profiler-memory-readout
exp/concurrency-sweep
docs/update-roadmap
```

Types: `feat`, `fix`, `exp` (experiments and benchmark runs), `docs`, `chore`.

## Pull requests

1. Branch off the latest `main`.
2. Keep the PR focused — one concern per PR.
3. Describe *what changed and why*, and note any measurements that back the change.
4. At least one teammate reviews before merge. Prefer the owner of the affected area.
5. Squash-merge unless the individual commits are independently meaningful.

## Commits

Imperative mood, present tense, explaining intent rather than mechanics.

```
Add KV-cache size estimator for grouped-query attention
Fix stage timing skew when a node is thermally throttled
```

Avoid `update`, `fix stuff`, `wip` on `main`.

## Experiments

Experiment results are evidence for the report, so they need to be reproducible.

- Commit the **configuration** that produced a run (model, context lengths, concurrency, cluster
  composition, communication backend), not the raw output — `results/` is gitignored.
- Record the hardware used. A number without the cluster it came from is not interpretable.
- Note anything that could confound the measurement: background load, thermal state, which
  machines were connected over Thunderbolt versus Ethernet.
- Repeat runs. macOS timing is noisy; single runs are not a result.

## Code

- Python, formatted with `ruff format`, linted with `ruff`.
- Type-hint public functions.
- Docstrings explain *why* a decision is made, not what the line does — the scheduler's reasoning
  is the research contribution and needs to be legible to a reader six months from now.

## Fair contribution

The course expects each member to contribute fairly, and git history is the record. Push your own
work under your own account rather than batching it through one person, and keep PR review
spread across the team.
