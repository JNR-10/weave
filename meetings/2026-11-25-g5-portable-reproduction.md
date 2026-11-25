# G5 portable reproduction — 2026-11-25

- **Gate:** G5 (guide §12, commands in §12.1, Intel path)
- **Run by:** Jainil, on his Intel MacBook Air (x86_64, macOS 12.7.6, python.org Python 3.11.1),
  with Claude Code driving the shell
- **Source:** a fresh `git clone` of `JNR-10/weave` checked out at `main`
  `2d5ae14896fddb53dadc8f29bae50139fbc28492`, with no uncommitted changes
- **Environment:** a new venv outside the clone, made from the base interpreter (not the working
  `.venv`), with `PATH` reduced to system paths
- **Log:** `donotcheckin/g5-portable-reproduction-20261125T191233Z.log` on Jainil's machine,
  SHA-256 `a6667373b1df1ab36a74ce6ac6851d48fdf09ebb6f2e68bb285e92d573e96abd` (gitignored; it
  goes in the team Drive folder with the other evidence)

## Result

The portable half of G5 passes. The M4 half is still open (see the last section).

| Step (guide §12.1) | Result |
| --- | --- |
| `pip install -r experiments/phase1/requirements-portable-lock.txt`, then `pip install --no-build-isolation --no-deps -e .` | exit 0; `pip freeze --all` matches the lock exactly (14 packages), plus the editable `weave-inference` at `2d5ae14` |
| `pip check` | No broken requirements |
| `import weave` | imports from the clone's `src/weave` |
| `ruff check src scripts tests`, `ruff format --check src scripts tests` | both pass |
| `pytest -q` | 370 passed, 1 skipped (`tests/test_installed_pipeline.py`, M4 only: a skip, not a passed hardware gate) |
| `python -m weave.scheduler.candidates` on the committed simulated fixture | 56 candidates, 15 memory-feasible, `selected_plan_id` null |
| `python -m weave.scheduler.policies --compare` | matches `tests/golden/policy-comparison.md`: second Mac alone; `[14, 14]` over by 838,336,512 B on the pressure Mac; `[1, 27]` feasible |
| `python -m weave.scheduler.checkpoint` | all four G4 demo items printed; schema validation and planner tests pass; evidence written to a new `results/SIMULATED-20261125T191356Z-checkpoint` in the clone |
| Mohit's evidence zip `weave-phase1-m4-evidence-20261025T224835Z.zip` | SHA-256 `81cfedc9…62ec15fe` matches the Phase 1 README; unzipped into a new, empty directory |
| `validate_evidence` on the five 2026-10-25 runs (acceptance plus four overhead runs) | all five `PASS` |
| `validate_evidence` on `TAMPER-CHECK/20261025T224855Z-m4air-b0-qwen7b-v2` | fails with `summary/trace mismatch: ttft_ms`, exit 1, as required |
| `git check-ignore` on the four `results/IGNORE-CHECK/...` paths | all four ignored |
| `git diff --check`, `git status --short` after every command | clean: running the commands changed no tracked file |

The script also pointed the validator at `TAMPER-CHECK/validator.stdout.txt` and
`validator.stderr.txt` (a loose glob). Those are not run directories, so the two failures are
meaningless and are not counted above.

## Other G5 items

- **Public functions annotated:** every public function under `src/weave/` has parameter and
  return annotations. The only unannotated functions are `event` and `memory`, closures nested
  inside the measured runner in `weave/telemetry/baseline.py`. They are not public, and editing
  the runner would change what later measured runs record, so they are left as they are.
- **Raw evidence ignored:** checked above.
- **Owner-reviewed PR before merge:** PRs #1–#3 each had a teammate review. Phase 1 itself
  reached `main` without a PR (`e7400c1`, 2026-10-25); Mohit's owner review in
  `meetings/2026-11-08-mohit-owner-review.md` covers the estimator and profiler. Urmi reviewed
  PR #2, which changed only `tests/test_checkpoint.py`, so no owner review of the Phase 1
  candidate generator in `src/weave/scheduler/` is recorded.
- **Evidence labels:** the simulated outputs above carry `SIMULATED-` IDs and `simulation: true`;
  the measured runs pass the validator's label checks.

## Still open for G5

- **The M4 half of §12.1:** a teammate other than the operator installs the M4 lock into a fresh
  environment on the M4, reruns the baseline into a new `results/` run ID and validates it. This
  needs Mohit's M4 and cannot be done on Intel. Re-validating Mohit's zip (above) checks his files,
  not a fresh run.
- **Urmi's owner review of the Phase 1 candidate generator** (`src/weave/scheduler/candidates.py`,
  CODEOWNERS), which no PR has recorded.
