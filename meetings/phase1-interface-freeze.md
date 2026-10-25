# Phase 1 Interface Freeze — Contract 2.0.0

- **Status:** Draft, pending sign-off by all three members
- **Drafted:** 2026-10-25 by Jainil
- **Governs:** Revised Phase 1 Implementation Guide v2 (document version) and interface version 2.0.0. The two identifiers are separate.
- **Sign-off:** [ ] Jainil [ ] Mohit [ ] Urmi

Guide gate G0 requires this record before Phase 1 sign-off. It states the shared names, units and semantics that the three Phase 1 components exchange.

## 1. Model

| Item | Value |
| --- | --- |
| Bring-up model | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| Revision | `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed` |
| `config.json` SHA-256 | `1661a349986919d13820d3981623776138d50783d13e247cdc5b075a22b62698` (matches upstream at that revision) |
| Geometry | 28 layers, hidden 3584, 28 attention heads, 4 KV heads, head dim 128, 4-bit, group size 64 |
| Rescue model | Llama-3.2-3B-Instruct-4bit, for diagnosing installation or resource problems only; labeled separately, never an equivalent result |
| Deferred | Qwen2.5-14B-Instruct-4bit (48 layers): not a Phase 1 dependency |

## 2. Versioning

- `schema_version` is `"2.0.0"` on every JSON artifact and every trace row. 1.0.0 inputs are rejected, never reinterpreted.
- The committed schemas in `experiments/phase1/schemas/v2` are the contract. `tests/test_schemas.py` fails if the code drifts from them; a change means a reviewed regeneration, not an edit in place.

## 3. Units and conventions

- **Bytes:** integer bytes internally. GiB = bytes / 2^30, for display only. Library-reported GB values stay labeled separately.
- **Time:** `monotonic_ns` comes from `perf_counter_ns` and is comparable only within one process. `wall_time_utc` is for approximate alignment only. Never subtract monotonic values from different machines.
- **Layer ranges:** half-open `[start_layer, end_layer_exclusive)`. The old inclusive `end_layer` field is not reused.
- **Stage vs rank:** `logical_stage` 0 runs the earliest layers; current MLX rank 0 runs the latest. Both are stored. `mlx_split_by_rank` is `logical_split` reversed, and `device_id_by_mlx_rank` is the device order reversed. Logical `[10, 18]` on `[A, B]` launches as native `[18, 10]` on `[B, A]`. A rank belongs to one launch group, never permanently to a Mac.
- **Single-device fallback:** one stage at rank 0, loaded with ordinary `mlx_lm.load`. Never a two-rank group with a zero-layer rank.

## 4. Identifiers

| Field | Meaning |
| --- | --- |
| `run_id` | The run directory name; the validator enforces this |
| `request_id` | Unique within a run (`req-0000`, …) |
| `device_id` | Stable physical identifier (e.g. `mohit-m4-air`) |
| `partition_version` | Links metrics to the placement that produced them (`single-device-v2` in Phase 1) |
| `source_snapshot_id`, `profile_id` | Link candidate plans to the planning inputs they came from |
| `SIMULATED-` prefix | Required on every synthetic identity, so it cannot join with measured evidence |

## 5. Workload envelope

`benchmarks/workloads/phase1-smoke.json` (`qwen7b-smoke-v2`): fixed prompt, `prompt_max_tokens` 512, `max_new_tokens` 64, concurrency 1, 1 warmup and 3 measured requests, temperature 0, seed 0. Warmup is excluded from measured aggregates, and model load is timed separately from requests. Three requests check reproducibility; they do not support a p95 claim.

## 6. Trace contract

- **Columns:** `schema_version, sequence, run_id, device_id, source, wall_time_utc, monotonic_ns, request_id, request_kind, event, policy, partition_version, workload_id, model_revision, token_count`, plus memory columns `process_rss_bytes, system_available_bytes, mlx_active_bytes, mlx_cache_bytes, mlx_peak_bytes`. An empty memory cell means "not sampled", never zero.
- **Successful lifecycle:** `process_start → model_load_start → model_load_end →` serial requests `→ process_end`. Each request is `request_start, first_token, token…, first_visible_text, request_end`. `error` ends a failed attempt and is never a successful lifecycle.
- **Token trace modes:** `full` (per-token rows) is the acceptance mode. `boundary` omits per-token rows and is used only for the matched logging-overhead check.

## 7. Run provenance (proposed on `fix/telemetry-phase1-evidence-checks`; needs agreement)

- `run_summary.json` records `workload_sha256` and `model_lock_sha256` for byte-exact copies of the inputs in the run directory.
- It records `git_commit` for the checkout the runner was imported from, and `git_dirty` for uncommitted tracked changes. A dirty run also saves `git-diff.patch`.
- The validator rejects runs that are dirty, have no commit, or whose copied inputs no longer match their hashes.
- These are additions within 2.0.0; no field is renamed or removed. **Decision needed:** keep 2.0.0 (proposed) or bump the interface version.

## 8. Model-lock format

`experiments/phase1/model-lock.json` holds `schema_version`, `model_id`, `model_revision` (40 hex characters), `model_dir` (repository-relative, never a home path), `config_sha256`, the embedded `config`, and `source: "downloaded_metadata"`. A lock is immutable: a new revision gets a new directory and a new manifest, never a relabeled old one.

## 9. Measured vs simulated evidence

- **Measured:** `source: "measured"`, `simulation: false`, under `results/<run_id>/`.
- **Simulated:** `source: "simulated"`, `simulation: true`, `SIMULATED-` identities, under `results/SIMULATED-<id>/`. Candidate output always has `simulation: true`, `executable: false`, `runtime_verified: false` and `selected_plan_id: null`.
- `memory_feasible` means "passes a stated estimate". It never means executable, verified, fastest or selected.
- Raw evidence is gitignored. **Archive location: to be agreed.**

## 10. Phase 1 authorship and long-term ownership

Phase 1 authorship is temporary and does not reassign the long-term roles in `README.md`. An owner is responsible for an area, not its sole author.

| Files | Phase 1 author / operator | Phase 1 reviewer | Long-term owner |
| --- | --- | --- | --- |
| `scripts/phase1/prepare_model.py`, `probe_mlx.py`, installed mapping test | Mohit | Jainil | Jainil (execution and rank mapping) |
| `weave.telemetry.baseline`, `validate_evidence` | Urmi | Mohit | Mohit (telemetry) |
| `benchmarks/workloads/phase1-smoke.json` | Urmi | Jainil | Jainil (workload generator) |
| `weave.profiler.model_profile` | Urmi (runner handoff) | Mohit | Mohit (profiler) |
| `weave.scheduler.candidates` | Jainil | Urmi | Urmi (placement policies) |
| `weave.profiler.simulated_fixture`, `weave.estimator.memory` | Jainil | Mohit | Mohit (profiler, estimator) |
| `weave.contracts`, `pyproject.toml`, this note | All | All | All |

## 11. Open items for sign-off

1. Keep the provenance additions in section 7 under 2.0.0?
2. Where is the shared evidence archive?
3. Should `.rocketride/` (a personal tool directory) stay in the shared `.gitignore`, or move to a local `.git/info/exclude`?
4. KV bytes per element: the synthetic fixture assumes 4 bytes. The installed MLX cache is bf16 (2 bytes) and allocates in 256-token steps. Calibrate this in Phase 2 rather than changing the Phase 1 fixture.
