# Phase 1 reproduction (M4 single-device baseline)

Checkpoint: hardware-limited bring-up on Mohit's M4 MacBook Air (16 GB, Mac16,13). Two-Mac inference is out of scope for this gate. Simulated candidate output is not measured evidence.

Pinned source commit at first measured run: `8ac11046aa181d9122bb362f94b4cabb85e238e5` (working tree then had Phase 1 files uncommitted).

## Portable setup (Intel or Apple Silicon, no MLX)

```sh
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_candidates.py tests/test_evidence.py
python -m weave.contracts.schemas --output-dir experiments/phase1/schemas/v2
```

On later checkouts, regenerate schemas into `results/SCHEMA-CHECK-<id>/` and compare; do not overwrite committed `experiments/phase1/schemas/v2`.

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

`prepare_model.py` refuses an existing directory. If a previous download was interrupted, resume `snapshot_download` into `models/qwen7b` and then write the lock; do not relabel a partial snapshot as a new revision.

## Frozen identities

- Model: `mlx-community/Qwen2.5-7B-Instruct-4bit` revision `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`
- Geometry: 28 layers, hidden 3584, 28 heads, 4 KV heads, head dim 128, 4-bit
- Installed: mlx 0.32.4, mlx-lm 0.32.2; `PipelineMixin.pipeline(self, group, split=None)`
- Unequal mapping test: native `split=[18, 10]` maps rank 0 → `[10, 28)`, rank 1 → `[0, 10)`
- `runtime_two_device_verified` is false
- Fresh-process planning cap for later calibration: 8 GiB. This is not measured capacity.

## Measured evidence (gitignored under `results/`)

| Run | Role |
| --- | --- |
| `20261022T080429Z-m4air-b0-qwen7b-v2` | acceptance full token-trace (validator PASS) |
| `20261022T080503Z-m4air-overhead-boundary` | overhead boundary |
| `20261022T080521Z-m4air-overhead-boundary` | overhead boundary |
| `20261022T080539Z-m4air-overhead-full` | overhead full |
| `20261022T071535Z-m4air-env-v2` | environment freeze and MLX probe |

Token-logging overhead: median measured end-to-end 3133.2 ms (full) vs 3093.6 ms (boundary), +1.3%. Below the 5% investigation trigger. Isolates per-token CSV rows only.

Parameter profile: 28 × 131,111,936 layer bytes plus 613,129,216 replicated bytes (embeddings, final norm, output head). Excludes KV, workspace, and OS memory. Request MLX peak ≈ 4.44 GiB.

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
python -m pytest -q
# Repeat baseline with a NEW results/ run ID.
```
