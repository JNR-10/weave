# Spec 0003: Predeclared evaluation plan, its checker and the result combiner

- **Status:** in review (pull request #4). Plan step 1 is done
- **Owner:** Urmi (long-term owner of ablation design). Written by Jainil's session at his request
  (schedule task T17), for Urmi's review
- **Branch:** `feat/jainilr-phase2-portable`, merged into `main` through pull request #4
- **Governs:** guide v2 section 14 gate P5 ("predeclared baseline matrix, ablations, sufficient
  repeats, tail metrics, failure/OOM accounting") and README milestones M5–M6, under interface
  2.0.0

## Problem Statement

RQ1 asks whether live workload information beats equal and static hardware-based splitting. A
comparison like that is only honest if the test is fixed before anyone sees a result: which
policies, which workloads, how many repeats, which number decides, and what counts as a
difference. Guide section 14 makes this the P5 gate, and `experiments/README.md` says a difference
smaller than the run-to-run spread is not a finding. Nothing in the repository declares any of it.

Phase 1 cannot answer the question either: its workload is three measured requests at concurrency
1, which checks reproducibility but "cannot support the later p95 claim" (guide section 14). The
team also has no code that turns many runs into the per-policy tables the report needs, and no
guard that keeps simulated and measured results apart once there are many of them.

## Solution

Commit one plan file, `experiments/plans/evaluation-plan-v1.json`, that declares the whole
comparison: the arms (the five policies of guide section 14 plus one ablation per live signal), the
workload scenarios, the repeat counts, the primary metric, the quantile method, the decision rule,
and how failures count. A checker, `python -m weave.evaluation.plan --check`, reads it strictly and
lists the cells it implies. A combiner, `python -m weave.evaluation.combine`, reads one result
record per run, refuses records that do not belong to that exact plan, and prints per-cell
summaries and each arm's verdict against the reference arm.

The plan starts as a **draft**: the two-Mac pilot (after gates P2-C to P2-E) fixes the induced
conditions and confirms the long-context and concurrency settings fit. Counted runs start only
against a **frozen** plan, and the combiner refuses measured results against a draft.

## User Stories

1. As the ablation designer, I want the comparison declared in a committed file before any run, so that no choice can follow the results.
2. As a reviewer, I want results to name the plan by the SHA-256 of its bytes, so that a plan edited after a run never matches that run's results.
3. As the policy owner, I want every policy from guide section 14 listed with an `available` flag, so that the matrix is complete even while B3 and Weave do not exist yet.
4. As the ablation designer, I want one ablation per live signal, each turning off exactly one, so that any gain can be traced to a signal (`experiments/README.md` ladder).
5. As an evaluator, I want enough requests per repeat that the p95 is not just the slowest request, and enough repeats to see the run-to-run spread.
6. As an evaluator, I want a failed or out-of-memory request counted as infinitely slow, never dropped, so that a policy cannot win by failing.
7. As a reader of the report, I want one written rule that says when one arm beats another, so that "better" means one thing.
8. As a safety reviewer, I want simulated and measured records never combined, and measured records refused against a draft plan.
9. As a teammate, I want a malformed plan or record refused with a `ValueError` that names the field, as the other readers do.
10. As the Phase 2 runner author, I want the per-request fields to match the Phase 1 `run_summary.json` request items, so that one runner can write both.
11. As a teammate on an Intel Mac or in CI, I want all of this to run without MLX.
12. As the team, I want a summary written only when asked and never over an existing file.

## Implementation Decisions

- **New package:** `weave.evaluation` (`plan.py`, `combine.py`). Field readers come from
  `weave.contracts.reading`. No change to the planner, the policies, the telemetry or any committed
  schema.
- **Version:** the plan and the result records carry `schema_version: "2.0.0"`, like every other
  artifact (freeze note section 2); anything else is refused. The plan format is a **new team
  contract**, additive under 2.0.0 like the decision record of spec 0002, and needs Urmi's and
  Mohit's approval on pull request #4 (AGENTS.md "Ask first"). There is no JSON Schema file: the
  checker is the contract, and unknown plan fields are refused so a typo cannot pass silently.
- **Arms:** the policies in declared order, then the ablations. Policies: `best_single_device`
  (B0), `equal_split` (B1), `memory_proportional` (B2) — available, spec 0002;
  `profiled_static` (B3) — not available, spec 0004; `weave_adaptive` — not available.
  `may_use_multiple_macs` is false only for B0.
- **Ablations:** one per live signal of the adaptive policy in the `experiments/README.md` ladder —
  `live_memory_headroom`, `context_length`, `concurrency`, `observed_stage_timings`. Each names
  `weave_adaptive` as its base and one disabled signal; a signal may be disabled by at most one
  ablation. An ablation cannot be available while its base policy is not.
- **Scenarios (draft):** six steady-state workloads, each with prompt-token upper bound, maximum
  new tokens, concurrency, memory pressure (`none`/`induced`), compute contention
  (`none`/`induced`) and link condition (`baseline`/`degraded`), plus `needs_two_macs` and
  `pilot_required`. `short-c1` is the Phase 1 envelope (512 / 64 / concurrency 1). `long-c1` (4096
  / 256), `short-c4` (concurrency 4; the Phase 1 runner refuses concurrency above 1), and the three
  induced-condition scenarios need the pilot first, as guide section 14 asks ("evaluate short and
  long contexts only after pilot feasibility"). A degraded link needs two Macs, so the checker
  requires `needs_two_macs` there. How each condition is induced is written down by the pilot
  before the plan is frozen.
- **Cells:** every scenario × every arm, in declared order, ID `<scenario_id>/<arm>`. A cell needs
  two Macs when its scenario does or its arm may use several Macs. The draft declares 6 × 9 = 54
  cells.
- **Repeats:** 5 repeats per cell, each with 1 warmup request (excluded, as in Phase 1) and 20
  measured requests. Why 20: with the nearest-rank method the p95 of n values is the
  ⌈0.95·n⌉-th smallest, which is the maximum for every n below 20; n = 20 is the smallest count
  whose p95 is not just the slowest request. Why 5: pooled, a cell has 100 measured requests (p95
  is the 95th), and five per-repeat values give a median and a range; macOS timing is noisy and
  single runs are not a result (CONTRIBUTING.md). For scale only: at Phase 1's measured 1.51 s
  median end to end on the M4, one short-workload cell is about 5 × 21 × 1.5 s ≈ 2.6 minutes plus
  model loads; long and concurrent cells are not measured yet.
- **Quantile method:** nearest rank, `sorted(values)[ceil(q·n) − 1]`. It always returns an observed
  value and needs no interpolation choice. Medians use the same rule with q = 0.5.
- **Failures:** a measured request with status `failed` or `oom` counts as infinite latency in every
  latency quantile, and is counted in its own column. More than 5% failed requests therefore makes
  a cell's p95 infinite. A run record must list exactly the declared number of warmup and measured
  requests, so a missing request is refused, not dropped. TPOT medians use successful requests with
  a TPOT only.
- **Primary metric and rule:** `end_to_end_ms_p95`, pooled over all measured requests of a cell.
  The reference arm is `profiled_static` (guide section 14: the main comparison is against the
  declared profiled static baseline, not a weak split). Arm A beats arm B in a scenario only when
  both cells are complete, every per-repeat p95 of A is below every per-repeat p95 of B
  (non-overlapping ranges), and A's pooled p95 is at least 5% below B's. The 5% mirrors Phase 1's
  investigation trigger for logging overhead. Otherwise the verdict is "no finding". Secondary
  metrics (median end to end, TTFT p95 and median, TPOT median, throughput, failures, OOMs) are
  reported, not decided on.
- **Throughput:** per repeat, generated tokens of successful measured requests divided by the
  record's `repeat_wall_ms`; a cell reports the median over repeats.
- **Result record (one per run):** `schema_version`, `source` (`simulated` or `measured`),
  `simulation` (must agree with `source`), `run_id` (starts with `SIMULATED-` exactly when
  simulated), `plan_id`, `plan_sha256`, `scenario_id`, `arm`, `repeat` (1 to the declared count),
  `repeat_wall_ms`, `requests`. Each request has `request_id`, `request_kind` (`warmup` or
  `measured`), `status` (`success`, `failed`, `oom`), and for a success `generation_tokens`,
  `ttft_ms`, `end_to_end_ms` and `observed_tpot_ms` (null when one token) — the Phase 1 request
  names. Extra fields are allowed, so a runner can add provenance.
- **Combining refuses:** a record for another plan or another plan hash; an undeclared scenario or
  arm; a repeat out of range; two records for one cell and repeat; a repeated `run_id`; simulated
  and measured records together; measured records against a draft plan.
- **Commands:** `python -m weave.evaluation.plan --check <plan>` prints the plan's hash, status,
  cell, run and request counts (`--cells` lists every cell). `python -m weave.evaluation.combine
  --plan <plan> <record>...` prints a Markdown table and writes nothing; `--output <file>` also
  writes the JSON summary with exclusive creation. An infinite quantile is written as JSON null; the
  failure columns say why.

## Testing Decisions

- Tests drive the public functions and commands with realistic input and assert on what a caller
  sees: the parsed plan, the cells, the summary numbers, the verdict, the written file and the
  raised error. No mocks.
- Every record in the tests is simulated (`SIMULATED-` run IDs, `simulation: true`); no measured
  numbers appear anywhere.
- The committed plan passes the checker; its cell count (54) and its draft status are locked.
- One parametrized invalid-input table per reader, asserting the field named in the error.
- Quantiles on known data: nearest rank for n = 20 is the 19th value; failures make it infinite.
- Combining: missing and incomplete cells, the plan-hash, plan-ID, draft and mixing refusals,
  duplicate repeats, the verdict rule, and the command's overwrite refusal.
- Prior art: `tests/test_policy_input.py` (invalid-input table) and `tests/test_policy_compare.py`
  (command output).

## Out of Scope

- Running anything. The campaign needs the two-Mac gates P2-C to P2-E and, for Weave, P3–P4.
- RQ3 transitions and the switching break-even: they need the runtime manager of gate P4 and a
  later plan version with time-varying scenarios.
- The pilot itself, and the exact way each condition is induced.
- The runner that writes result records, and any change to the Phase 1 runner or validator.
- External systems (exo, dnet), which guide section 14 makes conditional.

## Plan

Each step passes `make check` and is committed on its own.

1. **Spec:** add this spec.
2. **Plan and checker:** the draft plan file, `weave.evaluation.plan` and its tests.
3. **Combiner:** `weave.evaluation.combine`, the verdict rule and its tests.
4. **Review:** Urmi reviews the plan format and the numbers on pull request #4; Mohit confirms the
   result record fits the Phase 2 runner.
5. **Pilot and freeze (after P2-E):** fix the induced conditions, settle `pilot_required`
   scenarios, mark the plan `frozen` in a new version, and only then start counted runs.
