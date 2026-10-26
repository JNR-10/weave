# Phase 1 reproduction (M4 single-device baseline)

Checkpoint: hardware-limited bring-up on Mohit's M4 MacBook Air (16 GB, Mac16,13). Two-Mac inference is out of scope for this gate. Simulated candidate output is not measured evidence.

Pinned source commit at first measured run: `8ac11046aa181d9122bb362f94b4cabb85e238e5` (working tree then had Phase 1 files uncommitted). Those runs are superseded. The 2026-10-25 acceptance rerun used clean commit `0875258c1fb1fe2c1e44328ae7a4b686390aef8f`; see [Measured evidence](#measured-evidence-gitignored-under-results).

## Portable setup (Intel or Apple Silicon, no MLX)

Use any native Python 3.11+ (`python3.11`; Homebrew on Intel is under `/usr/local/bin`). The portable lock is the `[dev]` closure frozen on Intel, at the same versions as the M4 lock.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/phase1/requirements-portable-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest -q
```

`tests/test_installed_pipeline.py` skips without MLX; a skip is not a passed hardware gate. `tests/test_schemas.py` regenerates the 2.0.0 schemas into a temporary directory and compares them with the committed `experiments/phase1/schemas/v2`; never regenerate into the committed directory (the generator refuses to overwrite it).

## Simulated candidates (portable)

The input is wholly synthetic and the output is never measured evidence.

```sh
PHASE1_SIM_ID="SIMULATED-$(date -u +%Y%m%dT%H%M%SZ)-candidates-v2"
mkdir -p "results/$PHASE1_SIM_ID"
python -m weave.scheduler.candidates \
  --state experiments/phase1/cluster-state-SIMULATED.json \
  --output "results/$PHASE1_SIM_ID/candidate_plans.json"
```

Expected: 56 candidates, 15 memory-feasible, `selected_plan_id` null. The `[10, 18]` split with the pressure device first is rejected with `MEMORY_BUDGET_EXCEEDED`.

For the G4 checkpoint (guide section 12.2), one command prints the four demo items (a single-Mac plan, an accepted split, the rejected `[10, 18]` split, and the `[10, 18]` → MLX `[18, 10]` rank mapping) and writes the evidence package into a new `results/SIMULATED-<UTC time>-checkpoint/`: the fixture copy and its hash, the candidate file, schema and planner-test results, the git commit and a README. It exits non-zero if a check fails. Run it from a clean, committed checkout so the evidence names the exact source:

```sh
python -m weave.scheduler.checkpoint
```

## M4 setup

Native arm64 Python 3.11.17 at `/opt/homebrew/bin/python3.11`. Do not measure under Rosetta.

```sh
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,m4]"
python scripts/phase1/prepare_model.py \
  --repo mlx-community/Qwen2.5-7B-Instruct-4bit \
  --model-dir models/qwen7b \
  --manifest experiments/phase1/model-lock.json
mlx_lm.generate --model models/qwen7b \
  --prompt "Explain model partitioning in two short sentences." --max-tokens 64
python scripts/phase1/probe_mlx.py --output-dir results/ENV-<id>/compatibility
python -m pytest -q tests/test_installed_pipeline.py
```

`prepare_model.py` refuses an existing directory, checks the full Qwen2.5-7B geometry (layers, hidden size, attention and KV heads, 4-bit), and verifies every file of the revision against the hub (SHA-256 for the LFS weights, git blob SHA-1 for the rest) before writing the lock. Files that fail verification are fetched again once. A stray `*.safetensors` file also fails, because `mlx_lm.load` would read it, and must be removed by hand.

If a download was interrupted, finish it with the explicit revision instead of writing a lock by hand:

```sh
python scripts/phase1/prepare_model.py --resume \
  --revision c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed \
  --model-dir models/qwen7b --manifest experiments/phase1/model-lock.json
```

To check a snapshot that already has a lock (for example one written before verification existed), write a report; this downloads nothing and changes no lock, and exits nonzero on any mismatch:

```sh
python scripts/phase1/prepare_model.py --verify-report results/ENV-<id>/snapshot-verification.json
```

On mlx-lm 0.32.2 the probe's `compatibility.json` should show `qwen2_inherits_pipeline_mixin: true` and `sharded_load_pipeline_calls: 2`; it stops if `Qwen2Model` does not inherit `PipelineMixin`.

## Measured runs (M4)

Start from a clean, committed checkout. The runner records the commit of the checkout it was imported from (unknown if it runs from a non-editable install), whether tracked files had uncommitted changes or `src/` had untracked importable files (copied under `git-untracked/`), the Python and macOS versions, the machine architecture, and every installed package version. The validator rejects runs with uncommitted changes or no commit, and runs not recorded as native arm64 with MLX. It also requires `model_profile.json` to describe the locked model, with a complete layer tally and a `profile_id` that matches its content. Every run gets a new directory; never delete or reuse an attempt.

```sh
git status --short   # tracked files must be clean
PHASE1_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m4air-b0-qwen7b-v2"
python -m weave.telemetry.baseline \
  --workload benchmarks/workloads/phase1-smoke.json \
  --model-lock experiments/phase1/model-lock.json \
  --output-dir "results/$PHASE1_RUN_ID" \
  --device-id mohit-m4-air
python -m weave.telemetry.validate_evidence "results/$PHASE1_RUN_ID"
```

Token-logging overhead: four more runs in the order full, boundary, boundary, full, each validated. Report the raw results and the median end-to-end difference; above 5% is an investigation trigger.

```sh
for mode in full boundary boundary full; do
  id="$(date -u +%Y%m%dT%H%M%SZ)-m4air-overhead-$mode"
  python -m weave.telemetry.baseline \
    --workload benchmarks/workloads/phase1-smoke.json \
    --model-lock experiments/phase1/model-lock.json \
    --output-dir "results/$id" --device-id mohit-m4-air --token-trace "$mode"
  python -m weave.telemetry.validate_evidence "results/$id"
done
```

Negative checks (guide §9.3). Both must fail; keep their output with the evidence.

```sh
# 1. Invalid input: concurrency 2 is refused before any run directory is created.
mkdir -p results/INVALID-CHECK
python -c 'import json; w = json.load(open("benchmarks/workloads/phase1-smoke.json")); w["concurrency"] = 2; json.dump(w, open("results/INVALID-CHECK/concurrency2.json", "w"))'
python -m weave.telemetry.baseline --workload results/INVALID-CHECK/concurrency2.json \
  --model-lock experiments/phase1/model-lock.json \
  --output-dir results/INVALID-CHECK/run --device-id mohit-m4-air

# 2. Tampered copy of the real run: same directory name, TTFT changed by 1 ms.
mkdir -p results/TAMPER-CHECK && cp -R "results/$PHASE1_RUN_ID" results/TAMPER-CHECK/
python -c 'import json, sys; p = sys.argv[1]; s = json.load(open(p)); s["requests"][-1]["ttft_ms"] += 1; json.dump(s, open(p, "w"))' \
  "results/TAMPER-CHECK/$PHASE1_RUN_ID/run_summary.json"
python -m weave.telemetry.validate_evidence "results/TAMPER-CHECK/$PHASE1_RUN_ID"
```

## Frozen identities

- Model: `mlx-community/Qwen2.5-7B-Instruct-4bit` revision `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`
- Geometry: 28 layers, hidden 3584, 28 heads, 4 KV heads, head dim 128, 4-bit
- Installed: mlx 0.32.4, mlx-lm 0.32.2; `PipelineMixin.pipeline(self, group, split=None)`
- Unequal mapping test: native `split=[18, 10]` maps rank 0 → `[10, 28)`, rank 1 → `[0, 10)`
- `runtime_two_device_verified` is false
- Fresh-process planning cap for later calibration: 8 GiB. This is not measured capacity.

## Measured evidence (gitignored under `results/`)

Sharing: raw evidence is shared directly between teammates as a zip plus its SHA-256, outside the repository and with no team archive; see Experiments in [CONTRIBUTING.md](../../CONTRIBUTING.md). Keep the zips in your local `donotcheckin/` folder, which is gitignored. Mohit's zip of `results/` from 2026-10-25 is `weave-phase1-m4-evidence-20261025T224835Z.zip`, SHA-256 `81cfedc907de3a7c237e052bab64d31670595377097b7fa9707261ab62ec15fe`. It holds the runs below and the 2026-10-22 attempts.

### 2026-10-25 acceptance rerun

Recorded on the M4 at `0875258c1fb1fe2c1e44328ae7a4b686390aef8f` with no uncommitted changes and no untracked source (`git_dirty: false`). The checkout's only untracked files were outside `src/` (`.claude/`, `.cursor/` and a copy of the guide).

Environment: MacBook Air Mac16,13, Apple M4, 17,179,869,184 bytes (16 GiB), macOS 26.6.2 (25G83), native arm64 Python 3.11.17, mlx 0.32.4, mlx-lm 0.32.2. The installed packages match `requirements-m4-lock.txt` exactly. AC power, Low Power Mode off, foreground apps Finder, Electron, WhatsApp and Cursor. Thermal state was not recorded. Weights were cached at `models/qwen7b`. `pytest` on the M4: 86 passed, 0 skipped, so the installed `[18, 10]` mapping test ran.

| Run | Role | Result |
| --- | --- | --- |
| `ENV-20261025T224835Z-m4air-v3` | snapshot verification, MLX probe, session logs | snapshot `ok: true`; Qwen2 inherits `PipelineMixin`; `sharded_load` makes 2 pipeline calls |
| `20261025T224855Z-m4air-b0-qwen7b-v2` | acceptance, full token trace | validator PASS |
| `20261025T224908Z-m4air-overhead-full` | overhead full | validator PASS |
| `20261025T224918Z-m4air-overhead-boundary` | overhead boundary | validator PASS |
| `20261025T224928Z-m4air-overhead-boundary` | overhead boundary | validator PASS |
| `20261025T224938Z-m4air-overhead-full` | overhead full | validator PASS |
| `INVALID-CHECK` | negative check 1: concurrency 2 | refused with `ValueError`, exit 1, no run directory created |
| `TAMPER-CHECK` | negative check 2: acceptance copy with one TTFT raised by 1 ms | validator fails with `summary/trace mismatch: ttft_ms`, exit 1 |

Acceptance run, three measured requests after the excluded warmup: 39 prompt tokens and 28 generated tokens each, with the same text every time. TTFT 360.4, 353.8 and 360.5 ms (median 360.4 ms). End to end 1511.7, 1509.3 and 1521.9 ms (median 1511.7 ms). Median observed time per output token 42.0 ms. Times are the instrumented stream callback's wall time, including tokenization and excluding model load.

Token-logging overhead, in the order full, boundary, boundary, full (median of each run's measured end-to-end times): full 1507.6 and 1517.8 ms, boundary 1501.5 and 1525.5 ms. Median full 1512.7 ms vs boundary 1513.5 ms, a difference of −0.8 ms (−0.05%). This is below the 5% investigation trigger and isolates per-token CSV rows only.

Memory: request MLX peak 4,441,795,936 bytes (4.14 GiB) and model-load MLX peak 4,287,507,464 bytes (3.99 GiB), both from `mx.get_peak_memory`. The `mlx_lm.generate` log's "4.442 GB" is the same request peak in decimal GB. Parameter profile `profile-b92f137a195e799f` is identical in all five runs: 28 × 131,111,936 layer bytes plus 613,129,216 replicated bytes (embeddings, final norm, output head). It excludes KV, workspace and OS memory.

Independent check: the validator from `0875258`, run on Intel against the unzipped runs, gives the same results. The tamper copy differs from the acceptance run only in that one TTFT, and the invalid input differs from the smoke workload only in `concurrency`.

The 2026-10-22 attempts took about twice as long (end to end about 3.1 s, TTFT about 0.81 s), with the same tokens, text and memory peaks. Their power state was not recorded, so timings from the two days must not be compared.

### 2026-10-22 runs (superseded)

These were recorded at `8ac1104` with the Phase 1 files uncommitted, and before the runner copied its inputs byte-for-byte, so they fail the current validator's provenance and input-hash checks. Keep them as attempts; the 2026-10-25 rerun from a clean commit is the acceptance evidence.

| Run | Role |
| --- | --- |
| `20261022T080429Z-m4air-b0-qwen7b-v2` | acceptance full token-trace (passed the validator of that time) |
| `20261022T080503Z-m4air-overhead-boundary` | overhead boundary |
| `20261022T080521Z-m4air-overhead-boundary` | overhead boundary |
| `20261022T080539Z-m4air-overhead-full` | overhead full |
| `20261022T071329Z-m4air-env-v2` | environment inventory before and after install |
| `20261022T071535Z-m4air-env-v2` | environment freeze, model download logs, CLI smoke and MLX probe |

Token-logging overhead then: median measured end-to-end 3133.2 ms (full) vs 3093.6 ms (boundary), +1.3%. The parameter profile bytes and memory peaks were the same as on 2026-10-25.

## Reproduce from the M4 lock

Weights are already cached at `models/qwen7b`; a fresh process is not a cold filesystem cache.

```sh
python3.11 -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r experiments/phase1/requirements-m4-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -c "import weave; print(weave.__file__)"
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest -q
# Repeat the measured runs above with NEW results/ run IDs.
```
