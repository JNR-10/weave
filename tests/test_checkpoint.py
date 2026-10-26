"""Spec 0001 step 8: the checkpoint command prints the section 12.2 demo and writes the evidence.

One real run (with the real planner test suite) is shared by the tests that inspect its output;
the remaining tests replace only the test-suite subprocess, a system boundary, with a fake.
"""

import hashlib
import json
from pathlib import Path

import pytest

from weave.scheduler import checkpoint
from weave.scheduler.checkpoint import SuiteResult, run

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments/phase1/cluster-state-SIMULATED.json"
GOLDEN = ROOT / "tests/golden/phase1-candidate-plans.json"


def passing_suite() -> SuiteResult:
    return SuiteResult(passed=True, output="1 passed", command=["fake"])


def failing_suite() -> SuiteResult:
    return SuiteResult(passed=False, output="1 failed", command=["fake"])


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results") / "SIMULATED-test-checkpoint"
    exit_code, report = run(FIXTURE, out)
    return exit_code, report, out


def test_real_run_with_passing_tests_exits_zero(real_run):
    exit_code, _, _ = real_run
    assert exit_code == 0


def test_report_shows_a_single_mac_plan_that_fits(real_run):
    _, report, _ = real_run
    assert "1. Single-Mac plan: SIMULATED-second-mac runs all 28 layers" in report


def test_report_shows_an_accepted_split(real_run):
    _, report, _ = real_run
    assert (
        "2. Accepted split: logical [1, 27] on SIMULATED-m4-pressure, SIMULATED-second-mac"
        in report
    )


def test_report_shows_the_rejected_10_18_split_with_its_excess(real_run):
    _, report, _ = real_run
    assert "exceeds its budget by 292,028,416 bytes (MEMORY_BUDGET_EXCEEDED)" in report


def test_report_shows_logical_10_18_launching_as_mlx_18_10(real_run):
    _, report, _ = real_run
    assert (
        "launches as MLX split [18, 10] on ranks [SIMULATED-second-mac, SIMULATED-m4-pressure]"
        in report
    )


def test_package_holds_a_byte_copy_of_the_fixture_and_its_hash(real_run):
    _, _, out = real_run
    summary = json.loads((out / "checkpoint.json").read_text())
    copy = (out / "cluster-state-SIMULATED.json").read_bytes()
    assert (copy, summary["state_sha256"]) == (
        FIXTURE.read_bytes(),
        hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    )


def test_package_candidate_file_is_the_golden_file(real_run):
    _, _, out = real_run
    assert (out / "candidate_plans.json").read_bytes() == GOLDEN.read_bytes()


def test_package_records_schema_and_test_results(real_run):
    _, _, out = real_run
    summary = json.loads((out / "checkpoint.json").read_text())
    assert (summary["schema_validation"], summary["planner_tests"]["passed"]) == (
        {"cluster_state": "pass", "candidate_set": "pass"},
        True,
    )


def test_package_keeps_the_planner_test_output(real_run):
    _, _, out = real_run
    assert " passed" in (out / "planner-tests.txt").read_text()


def test_package_readme_names_every_file(real_run):
    _, _, out = real_run
    readme = (out / "README.md").read_text()
    files = ["cluster-state-SIMULATED.json", "candidate_plans.json", "checkpoint.json"]
    assert all(name in readme for name in [*files, "planner-tests.txt"])


def test_failing_planner_tests_exit_non_zero_and_are_recorded(tmp_path):
    out = tmp_path / "SIMULATED-failing-checkpoint"
    exit_code, _ = run(FIXTURE, out, run_suite=failing_suite)
    summary = json.loads((out / "checkpoint.json").read_text())
    assert (exit_code, summary["planner_tests"]["passed"]) == (1, False)


def test_existing_output_folder_is_refused(tmp_path):
    out = tmp_path / "SIMULATED-earlier-checkpoint"
    out.mkdir()
    with pytest.raises(FileExistsError):
        run(FIXTURE, out, run_suite=passing_suite)


def test_output_folder_must_be_labeled_simulated(tmp_path):
    with pytest.raises(ValueError, match=r"^output_dir: "):
        run(FIXTURE, tmp_path / "checkpoint", run_suite=passing_suite)


def test_command_writes_a_timestamped_simulated_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(checkpoint, "run_planner_suite", passing_suite)
    exit_code = checkpoint.main(["--state", str(FIXTURE), "--results-dir", str(tmp_path)])
    (made,) = tmp_path.iterdir()
    assert (exit_code, made.name.startswith("SIMULATED-"), made.name.endswith("-checkpoint")) == (
        0,
        True,
        True,
    )


def test_command_finds_the_committed_fixture_from_any_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # not the repository
    monkeypatch.setattr(checkpoint, "run_planner_suite", passing_suite)
    assert checkpoint.main(["--results-dir", str(tmp_path / "results")]) == 0


def test_default_results_folder_is_the_repositorys_own(tmp_path, monkeypatch):
    # A results/ folder outside the repository would not be gitignored. Checked without running
    # the command, so no test writes into, or deletes from, the real results/ folder.
    monkeypatch.chdir(tmp_path)
    assert checkpoint.default_paths() == (FIXTURE, ROOT / "results")


def test_package_was_validated_against_the_committed_schema_files(real_run):
    # The committed files are the 2.0.0 contract; regenerated schemas are not.
    _, _, out = real_run
    schema_dir = ROOT / "experiments/phase1/schemas/v2"
    recorded = json.loads((out / "checkpoint.json").read_text())["schema_files"]
    assert {name: entry["sha256"] for name, entry in recorded.items()} == {
        name: hashlib.sha256((schema_dir / f"{file}.schema.json").read_bytes()).hexdigest()
        for name, file in (
            ("cluster_state", "cluster_state_simulated"),
            ("candidate_set", "candidate_set"),
        )
    }


def test_package_records_the_hash_of_the_candidate_file_it_wrote(real_run):
    _, _, out = real_run
    summary = json.loads((out / "checkpoint.json").read_text())
    written = (out / "candidate_plans.json").read_bytes()
    assert summary["candidate_file_sha256"] == hashlib.sha256(written).hexdigest()


@pytest.mark.parametrize(
    "name",
    ["candidate_plans.json", "checkpoint.json", "planner-tests.txt", "README.md", "git-diff.patch"],
)
def test_state_named_like_a_package_file_is_refused_before_writing(tmp_path, name):
    # The copy would be overwritten, and the recorded state hash would no longer match it.
    state = tmp_path / name
    state.write_bytes(FIXTURE.read_bytes())
    out = tmp_path / "SIMULATED-clash-checkpoint"
    with pytest.raises(ValueError, match=r"^state_path: "):
        run(state, out, run_suite=passing_suite)
    assert not out.exists()
