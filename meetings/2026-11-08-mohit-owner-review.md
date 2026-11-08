# Mohit owner review — telemetry, profiler, estimator (Phase 1 closeout)

- **Date:** 2026-11-08
- **Author:** Mohit Manoj Barade
- **Checkout reviewed:** `main` at `ba2b218` (after PR #1 policies and PR #2 checkpoint narrowing)
- **Scope:** Long-term owner review required by guide §3 / interface freeze §10. Not a new measured experiment.

## Decision on Urmi's placement decision-record (spec 0002)

Confirming the three choices Jainil already approved on PR #1:

1. Additive under interface **2.0.0** (no rename/removal of existing Phase 1 fields).
2. Schema lives in its own folder: `experiments/policies/schemas/v2/`.
3. Chosen plan field name: `chosen_plan_id`.

**Mohit: approve.** No interface ask-first change needed from my side.

## Telemetry (`src/weave/telemetry/`)

Reviewed `baseline.py` and `validate_evidence.py` as long-term owner (Phase 1 author: Urmi; audit: Jainil).

| Check | Result |
| --- | --- |
| Schema 2.0.0; concurrency 1 only | Pass — refused before run dir |
| Exclusive run directory; copied input hashes | Pass |
| `git_commit` / `git_dirty`; dirty rejected | Pass |
| Trace lifecycle and TTFT/e2e from `monotonic_ns` | Pass |
| Measured vs simulated labels | Pass — `source: measured`, `simulation: false` |
| Integer memory samples; empty ≠ zero | Pass |
| Parameter profile attached and `profile_id` checked | Pass |

No blocking findings. Future work I own: per-rank stage timings and reload-cost events after two-Mac bring-up (P2/P4).

## Profiler (`src/weave/profiler/model_profile.py`)

| Check | Result |
| --- | --- |
| Layer nbytes + non-layer map | Pass — every non-layer name is replicated |
| 4-bit scales/biases included | Pass — measured M4 profile sums to 613,129,216 B |
| `profile_id` includes `source` / `measurement_kind` | Pass — simulated cannot share a measured ID |
| Limitation string present | Pass — parameters only; no KV/workspace/OS |

No blocking findings. Live cluster snapshot (free memory, stage speed, link) remains Phase 2.

## Estimator (`src/weave/estimator/memory.py`) and simulated fixture

Reviewed Jainil's rebuild on `main`.

| Check | Result |
| --- | --- |
| Five explicit positive integer terms | Pass |
| Replicated weights charged every stage | Pass — matches guide §10 / Qwen pipeline |
| KV formula visible; no silent zero | Pass |
| Docstring notes bf16 / 256-token rounding as Phase 2 | Pass — do not change Phase 1 fixture |
| `memory_feasible` not overclaimed | Pass — estimate only |
| No live post-load `system_available_bytes` as full budget | Pass — avoided |

No blocking findings. Calibration of `kv_bytes_per_element`, block rounding, and workspace/safety from measured peaks is my Phase 2 work (P2-E), blocked on a second Mac.

## Out of scope for this note

- Re-running Oct 25 acceptance numbers (unchanged).
- Approving Jainil/Urmi G0 checkboxes.
- Changing committed locks, schemas, or the simulated fixture.

## Action for teammates

1. Jainil / Urmi: review Mohit's §11 positions in `meetings/phase1-interface-freeze.md` and check your boxes if you agree.
2. Optional: teammate G5 reproduce from `main` using the Phase 1 README (author self-reproduce is separate).
