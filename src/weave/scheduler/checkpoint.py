"""Phase 1 planner checkpoint: show the guide section 12.2 demo and keep the G4 evidence.

Run from the repository root:  python -m weave.scheduler.checkpoint

It regenerates the candidates from the committed simulated fixture, prints the four items the
section 12.2 demonstration asks for, and writes the evidence package into a new, gitignored
results/SIMULATED-<UTC time>-checkpoint/ directory. Everything here is simulated planning output:
nothing is selected, executed or measured.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weave.scheduler.candidates import generate
from weave.telemetry.baseline import git_provenance, source_checkout

DEFAULT_STATE = Path("experiments/phase1/cluster-state-SIMULATED.json")
SCHEMA_DIR = Path("experiments/phase1/schemas/v2")
# The planner's own suites (spec 0001) and the guide's reference tests, which gate G4 runs. This
# command's tests are deliberately not listed, so running the suite can never start it again.
PLANNER_TESTS = (
    "tests/test_candidates.py",
    "tests/test_planner_input.py",
    "tests/test_planner_mapping.py",
    "tests/test_memory_estimate.py",
    "tests/test_planner_candidates.py",
    "tests/test_simulated_fixture.py",
    "tests/test_phase1_golden.py",
)
# Names the package writes itself, git provenance included; an input state copied in under one of
# these names would be overwritten.
PACKAGE_FILES = frozenset(
    {
        "candidate_plans.json",
        "planner-tests.txt",
        "checkpoint.json",
        "README.md",
        "git-diff.patch",
        "git-untracked",
    }
)
GIB = 2**30


@dataclass(frozen=True)
class SuiteResult:
    passed: bool
    output: str
    command: list[str]


def run_planner_suite() -> SuiteResult:
    """Run the planner tests in a fresh interpreter, from the checkout this module came from."""
    repo = source_checkout()
    if repo is None:
        return SuiteResult(False, "planner tests need a source checkout (editable install)\n", [])
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *PLANNER_TESTS]
    done = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
    return SuiteResult(done.returncode == 0, done.stdout + done.stderr, command)


def _validate_schemas(
    state_file: Path, candidate_file: Path
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Check the files as written against the committed 2.0.0 schema files.

    The committed schema bytes are the contract, so they are what this checks against, and each
    one's hash is recorded. A missing validator or schema file is a failure, never a skip.
    """
    checks = (
        ("cluster_state", "cluster_state_simulated", state_file),
        ("candidate_set", "candidate_set", candidate_file),
    )
    try:
        import jsonschema
    except ImportError:
        missing = "fail: jsonschema is not installed; run make setup"
        return {name: missing for name, _, _ in checks}, {}
    repo = source_checkout()
    results: dict[str, str] = {}
    schema_files: dict[str, dict[str, str]] = {}
    for name, schema_name, document in checks:
        relative = SCHEMA_DIR / f"{schema_name}.schema.json"
        schema_path = repo / relative if repo else None
        if schema_path is None or not schema_path.is_file():
            results[name] = f"fail: committed schema {relative} not found"
            continue
        schema_bytes = schema_path.read_bytes()
        schema_files[name] = {
            "file": str(relative),
            "sha256": hashlib.sha256(schema_bytes).hexdigest(),
        }
        try:
            jsonschema.validate(json.loads(document.read_text()), json.loads(schema_bytes))
            results[name] = "pass"
        except jsonschema.ValidationError as exc:
            results[name] = f"fail: {exc.message}"
    return results, schema_files


def _gib(n: int) -> str:
    return f"{n / GIB:.2f} GiB"


def _devices(plan: dict[str, Any]) -> list[str]:
    return [stage["device_id"] for stage in plan["assignments"]]


def _bracketed(names: list[str]) -> str:
    return "[" + ", ".join(names) + "]"


def _stage_costs(plan: dict[str, Any]) -> str:
    return "; ".join(
        f"{s['device_id']} {_gib(s['predicted_memory_bytes'])} of {_gib(s['safe_process_budget_bytes'])}"
        for s in plan["assignments"]
    )


def _demo(candidate_set: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """The four section 12.2 items: fallback, a fitting split, a rejection, and the rank mapping."""
    plans = candidate_set["candidates"]
    single = next((p for p in plans if p["mode"] == "single_device" and p["memory_feasible"]), None)
    accepted = next((p for p in plans if p["mode"] == "pipeline" and p["memory_feasible"]), None)
    # The guide's worked rejection is logical [10, 18]; any rejected split stands in otherwise.
    rejected = next(
        (p for p in plans if p["logical_split"] == [10, 18] and not p["memory_feasible"]), None
    ) or next((p for p in plans if p["mode"] == "pipeline" and not p["memory_feasible"]), None)
    mapped = rejected or accepted

    lines = []
    if single:
        stage = single["assignments"][0]
        lines.append(
            f"1. Single-Mac plan: {stage['device_id']} runs all {stage['layer_count']} layers; "
            f"predicted {_gib(stage['predicted_memory_bytes'])} of "
            f"{_gib(stage['safe_process_budget_bytes'])}, fits"
        )
    else:
        lines.append("1. Single-Mac plan: no single Mac fits this state")
    if accepted:
        lines.append(
            f"2. Accepted split: logical {accepted['logical_split']} on "
            f"{', '.join(_devices(accepted))}; {_stage_costs(accepted)}; fits"
        )
    else:
        lines.append("2. Accepted split: no split fits this state")
    if rejected:
        stages = {s["device_id"]: s for s in rejected["assignments"]}
        reasons = "; ".join(
            f"{r['device_id']} needs {_gib(stages[r['device_id']]['predicted_memory_bytes'])} of "
            f"{_gib(stages[r['device_id']]['safe_process_budget_bytes'])} and exceeds its budget "
            f"by {r['excess_bytes']:,} bytes ({r['code']})"
            for r in rejected["rejection_reasons"]
        )
        lines.append(
            f"3. Rejected split: logical {rejected['logical_split']} on "
            f"{', '.join(_devices(rejected))}; {reasons}"
        )
    else:
        lines.append("3. Rejected split: every split fits this state")
    if mapped:
        lines.append(
            f"4. Rank mapping: logical {mapped['logical_split']} on {_bracketed(_devices(mapped))} "
            f"launches as MLX split {mapped['mlx_split_by_rank']} on ranks "
            f"{_bracketed(mapped['device_id_by_mlx_rank'])}; MLX rank 0 runs the last layers"
        )
    else:
        lines.append("4. Rank mapping: no multi-Mac candidate to show")
    rank_mapping = None
    if mapped:
        rank_mapping = {
            "logical_split": mapped["logical_split"],
            "device_order": _devices(mapped),
            "mlx_split_by_rank": mapped["mlx_split_by_rank"],
            "device_id_by_mlx_rank": mapped["device_id_by_mlx_rank"],
        }
    summary = {
        "single_mac_plan_id": single["plan_id"] if single else None,
        "accepted_split_plan_id": accepted["plan_id"] if accepted else None,
        "rejected_split_plan_id": rejected["plan_id"] if rejected else None,
        "rank_mapping": rank_mapping,
    }
    return lines, summary


def _readme(
    summary: dict[str, Any], state_name: str, demo_lines: list[str], output_dir: Path
) -> str:
    commit = summary["git_commit"] or "unknown (not a git checkout)"
    kept = [f"`{n}`" for n in ("git-diff.patch", "git-untracked/") if (output_dir / n).exists()]
    dirty = {True: f"yes, kept as {' and '.join(kept)}", False: "no", None: "unknown"}
    tests = "passed" if summary["planner_tests"]["passed"] else "FAILED"
    schemas = ", ".join(f"{k} {v}" for k, v in summary["schema_validation"].items())
    return f"""# Phase 1 planner checkpoint: {summary["run_id"]}

Simulated planning evidence for gate G4 (spec 0001; guide sections 11 and 12). Nothing in this
directory was measured, selected or executed.

| File | What it is |
| --- | --- |
| `{state_name}` | Byte copy of the input state, sha256 `{summary["state_sha256"]}` |
| `candidate_plans.json` | All {summary["candidate_count"]} candidates, \
{summary["memory_feasible_count"]} memory-feasible, sha256 `{summary["candidate_file_sha256"]}` |
| `checkpoint.json` | Machine-readable summary: hashes, counts, checks, git commit, demo |
| `planner-tests.txt` | Output of the planner test suites |

- Produced by `python -m weave.scheduler.checkpoint` at {summary["generated_at_utc"]}
- Source commit: `{commit}`; uncommitted source changes: {dirty[summary["git_dirty"]]}
- Schema validation: {schemas}
- Planner tests: {tests}

## Demonstration (guide section 12.2)

{chr(10).join(demo_lines)}
"""


def run(
    state_path: Path,
    output_dir: Path,
    run_suite: Callable[[], SuiteResult] | None = None,
) -> tuple[int, str]:
    """Write the evidence package and return (exit code, report). Exit 1 if any check failed."""
    output_dir = Path(output_dir)
    if not output_dir.name.startswith("SIMULATED-"):
        raise ValueError(
            "output_dir: simulated evidence must live in a SIMULATED-<id> directory, "
            f"got {output_dir.name!r}"
        )
    state_path = Path(state_path)
    if state_path.name in PACKAGE_FILES:
        raise ValueError(
            f"state_path: {state_path.name!r} is a name the package writes itself, so its copy "
            "would be overwritten; rename the input"
        )
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    candidate_set = generate(state)
    output_dir.mkdir(parents=True, exist_ok=False)  # every evidence package is a new attempt

    (output_dir / state_path.name).write_bytes(state_bytes)
    candidate_file = output_dir / "candidate_plans.json"
    with candidate_file.open("x") as f:
        json.dump(candidate_set, f, indent=2)
    schema_validation, schema_files = _validate_schemas(
        output_dir / state_path.name, candidate_file
    )
    suite = (run_suite or run_planner_suite)()
    (output_dir / "planner-tests.txt").write_text(suite.output)
    demo_lines, demo = _demo(candidate_set)

    summary = {
        "schema_version": "2.0.0",
        "simulation": True,
        "run_id": output_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state_file": str(state_path),
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "candidate_file_sha256": hashlib.sha256(candidate_file.read_bytes()).hexdigest(),
        "candidate_count": candidate_set["candidate_count"],
        "memory_feasible_count": candidate_set["memory_feasible_count"],
        "schema_validation": schema_validation,
        "schema_files": schema_files,
        "planner_tests": {"passed": suite.passed, "command": suite.command},
        **git_provenance(source_checkout(), output_dir),
        "demo": demo,
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "README.md").write_text(_readme(summary, state_path.name, demo_lines, output_dir))

    passed = suite.passed and all(v == "pass" for v in schema_validation.values())
    report = "\n".join(
        [
            "Phase 1 planner checkpoint (spec 0001, guide section 12.2); simulated, nothing selected",
            f"State: {state_path} (sha256 {summary['state_sha256'][:16]})",
            f"Candidates: {summary['candidate_count']}, memory-feasible: "
            f"{summary['memory_feasible_count']}",
            "",
            *demo_lines,
            "",
            f"Evidence: {output_dir}",
            f"Schema validation: {', '.join(f'{k} {v}' for k, v in schema_validation.items())}",
            f"Planner tests: {'passed' if suite.passed else 'FAILED'}",
        ]
    )
    return (0 if passed else 1), report


def default_paths() -> tuple[Path, Path] | None:
    """The committed fixture and the gitignored results/ folder, or None outside a checkout.

    Both come from the checkout this module was imported from, not the working directory: run
    from elsewhere, a relative results/ would land outside the gitignored folder.
    """
    repo = source_checkout()
    return None if repo is None else (repo / DEFAULT_STATE, repo / "results")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", help="default: the committed fixture in this checkout")
    parser.add_argument("--results-dir", help="default: results/ in this checkout (gitignored)")
    args = parser.parse_args(argv)
    defaults = default_paths()
    if defaults is None:
        # The package records this checkout's planner tests and committed schemas; without them
        # it could only record failures, so refuse before writing anything.
        parser.error("run from a source checkout (editable install): the evidence needs its tests")
    state = Path(args.state) if args.state else defaults[0]
    results_dir = Path(args.results_dir) if args.results_dir else defaults[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exit_code, report = run(state, results_dir / f"SIMULATED-{stamp}-checkpoint")
    print(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
