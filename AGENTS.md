# AGENTS.md

Weave decides how many Apple Silicon Macs serve an LLM workload and how many transformer layers
each one runs, on top of MLX. It is a CMPE 295 research project: correctness of measured evidence
matters more than speed. Python 3.11+, src layout, import name `weave` (distribution
`weave-inference`).

## Commands

- Setup, any Mac, no MLX: `make setup`
- Setup, Apple Silicon with MLX: `make setup-m4`
- Verify (lint, format check, tests): `make check`
- Auto-fix lint and formatting: `make fmt`
- One test file: `.venv/bin/python -m pytest -q tests/test_candidates.py`
- Model lock, measured runs, validator, negative checks: `experiments/phase1/README.md`

## Done means

- `make check` passes, with no new skips and no tests weakened.
- Commit only after `make check` exits 0. A pipe such as `make check | tail` hides a failure;
  `make hooks` installs a pre-commit check.
- Code that needs MLX (the measured runner, `scripts/phase1/`) cannot run on Intel or in CI. Say
  it is unverified instead of claiming it works; the M4 owner runs it.
- If a check fails, report it with its output. A failing check is a finding.

## Never

- Commit anything under `results/`, `models/` or `donotcheckin/` (evidence zips shared between
  teammates), model weights, or secrets (`.env*`, HF tokens).
- Delete, skip, or loosen a test or a validator check to get a green run.
- Create, edit, or relabel measured evidence, or mix simulated data (`SIMULATED-` IDs,
  `simulation: true`) with measured data.
- Edit committed contracts in place: `experiments/phase1/schemas/v2/`, `model-lock.json`, the
  `requirements-*-lock.txt` files. Their bytes are hashed into evidence.
- Import MLX, `mlx_lm`, `psutil` or `huggingface_hub` at module level under `src/weave/`.
  Portable code must import on Intel; import inside the function that needs it.
- Start a measured run from a checkout with uncommitted changes; the validator rejects it.

## Ask first

- Changing interface 2.0.0: field names, units, meanings, or the version.
- Adding or upgrading a dependency; both locks must be regenerated, one of them on the M4.
- Moving or renaming files or directories.

## Conventions that are easy to get wrong

- Memory is integer bytes. GiB = bytes / 2^30, for display only.
- Layer ranges are half-open: `[start_layer, end_layer_exclusive)`.
- Logical stage 0 runs the first layers; MLX rank 0 runs the last. `mlx_split_by_rank` is
  `logical_split` reversed. A rank belongs to one launch, never to a Mac.
- `monotonic_ns` compares only within one process, never across Macs.
- `memory_feasible` means "passes the stated estimate", not executable, verified, or chosen.
- Every run writes a new directory. Never overwrite, reuse, or delete an attempt.

## Git

- Branches: `<type>/<area>-<short-description>`, types `feat`, `fix`, `exp`, `docs`, `chore`.
- `main` stays green. Commit messages are imperative and say why.
- Keep moves and renames in their own commit, separate from behavior changes.

## Where to look

- Project, roles, roadmap: `README.md`. Working agreement: `CONTRIBUTING.md`.
- Phase 1 runbook: `docs/Initial_Implementation_Guide.md` (rank mapping in section 7, evidence
  rules in section 8, gates in section 12).
- Interface 2.0.0 contract (draft): `meetings/phase1-interface-freeze.md`.
- Feature specs and their step plans: `docs/specs/`. Read the relevant spec before changing that
  feature.

## Keeping this file useful

When an agent makes the same mistake twice, add one line here in the same change. Keep it short,
link to details instead of copying them, and delete lines that stop being true.
