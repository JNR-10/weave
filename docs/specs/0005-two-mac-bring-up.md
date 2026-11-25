# Spec 0005: Two-Mac bring-up tooling (T13–T15)

- **Status:** in review (pull request #4). Code written and tested on Intel only; **unverified on
  Apple Silicon** until gates P2-B to P2-D pass on two real Macs
- **Owner:** Jainil (execution, backends, partition application)
- **Branch:** `feat/jainilr-phase2-portable`
- **Governs:** guide v2 sections 7 and 14 (gates P2-B transport, P2-C equal split, P2-D unequal
  split) and README milestone M1, under interface 2.0.0

## Problem Statement

Phase 1 ran the model on one Mac. Every later question needs the same model running across two
Macs with a split we choose, and three things stand in the way:

1. **Transport.** Nothing in the repository yet proves two launched processes can exchange data.
2. **Unequal splits.** The installed `sharded_load` (mlx-lm 0.32.2) calls `pipeline(group)`
   twice without a `split`, so it can only run MLX's even split. B2, B3 and Weave all need
   unequal ones (guide section 7).
3. **Rank order.** Logical stage 0 runs the first layers, but MLX rank 0 runs the last. Reversing
   the counts without reversing the hosts still covers every layer, on the wrong Macs, and gives
   plausible numbers. Nothing would catch it.

The second Mac has not arrived, so this work cannot be run yet. It can be written and its logic
tested, so that the first two-Mac session is spent running checks, not writing code.

## Solution

- **Portable launch view** (`weave.runtime.launch`). From a logical device order and split, it
  derives the MLX split by rank and the device order by rank using the one mapping in
  `weave.scheduler.candidates.assignments`, and cross-checks that against the installed
  `PipelineMixin` formula. It orders hostfile entries by rank, picks the checkpoint files a rank
  needs the way `sharded_load` does, and checks a finished run's rank records.
- **Split-aware loader** (`weave.runtime.mlx_pipeline`, guide section 7.3's simple path): every
  Mac holds the full locked snapshot, the model loads lazily, and `pipeline(group, split=...)` is
  called exactly once on a fresh model before weights are materialized. A one-stage plan uses
  ordinary `mlx_lm.load` with no group (guide section 7.2).
- **Scripts** in `scripts/phase2/`: `distributed_smoke.py` (P2-B, from guide section 14) and
  `pipeline_generate.py` (P2-C, P2-D and the single-device reference).
- **Runbook** in `experiments/phase2/README.md`.

## Implementation Decisions

- **Identity is checked twice.** Each Mac reads its stable device ID from `~/.weave-device-id`.
  Before reading weights, a rank refuses to run if it is not the Mac the plan puts at its rank or
  the group size differs. After `pipeline`, it refuses if `start_idx`/`end_idx` differ from the
  plan. A rank belongs to one launch; the record stores rank and device per run.
- **Agreement is exact.** Decoding is greedy (the workload must have temperature 0). Every rank
  must generate the same token IDs, and the two-Mac tokens must equal the single-device
  reference from the same script on the same prompt. The installed Qwen2Model all-gathers the
  final hidden state, so every rank computes the same logits. A mismatch is reported with its
  token index; no tolerance is applied, because none has been agreed.
- **Every layer exactly once.** `check-run` requires one record per rank, the planned Mac and
  range per rank, and a union of ranges equal to `[0, L)` with no overlap.
- **Records are a draft format** (`weave-phase2-bring-up`, `draft-1`), outside the 2.0.0
  evidence contract. The Phase 1 validator does not accept them. The P2-C/P2-D evidence format
  is agreed by the team after a real run, not before.
- **Evidence rules still apply.** A run refuses a checkout with uncommitted changes or no commit,
  refuses a `SIMULATED-` run ID, checks the model config against the lock, writes each record
  with exclusive create (never overwrites), and records the git commit, package versions,
  workload and lock hashes. Times come from each process's `perf_counter_ns` and are never
  compared across Macs.
- **A hang becomes a failed run.** After `--timeout-s` (default 900 s) a rank writes an error
  record and exits 3, because a peer stuck in a collective never returns.
- **Hostfile entry format is not invented.** Entries pass through unchanged from a per-device
  map; only their order is computed. Their format comes from `mlx.launch --help` in the pinned
  environment.
- **No MLX at module level** under `src/weave/` (repo rule); the scripts import on Intel.

## Testing Decisions

- `tests/test_runtime_launch.py` (portable): the guide's `[10, 18]` example; the installed
  formula agrees with `assignments` for every 2- and 3-stage split of 28 layers; rank checks
  refuse a wrong size, host or range; shard-file selection keeps non-layer tensors, ignores
  layers past the model and refuses a missing layer; hostfile order; `check-run` passes
  agreeing ranks and reports divergences, inconsistent ranks and non-comparable references.
- `tests/test_runtime_mlx_fake.py` (portable, fake `mlx`/`mlx_lm` modules): `pipeline` called
  once with the reversed split; a wrong host refused before any load; a mis-landed pipeline
  refused; the single-device path uses the ordinary loader; a skipping token counter refused.
  The fake copies the installed `PipelineMixin` arithmetic, so these tests prove control flow
  only. They are not a substitute for P2-C or P2-D.

## Out of Scope

- Running anything on two Macs (needs the second Mac; T12 first).
- JACCL/RDMA (its own gate), a disk-saving loader that downloads only local shards, and
  switching splits without a reload (P4).
- A 2.0.0 evidence schema for two-Mac runs and per-stage timing (T16, Mohit).
