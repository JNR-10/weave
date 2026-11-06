"""Step 7 of spec 0002: the --compare command and its golden file.

The candidate set is read from the golden candidate file rather than generated here: policies
consume the generator's output file, and tests/test_phase1_golden.py already holds the generator to
that file byte for byte. It also keeps these tests out of the planner's checkpoint evidence, which
covers every test file that uses the planner modules.

The comparison output for that file is locked in tests/golden/policy-comparison.md, generated once
with the --compare command and reviewed by hand, so a change to any policy's choice is deliberate
and named.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from weave.scheduler import policies

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "tests/golden/phase1-candidate-plans.json"
COMPARISON = ROOT / "tests/golden/policy-comparison.md"

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"

REGENERATE = (
    "The comparison changed. If the change is deliberate, regenerate "
    "tests/golden/policy-comparison.md with `.venv/bin/python -m weave.scheduler.policies "
    "--candidates tests/golden/phase1-candidate-plans.json --compare "
    "> tests/golden/policy-comparison.md` in the same commit and name the change in its message."
)

SEPARATOR = "| --- | --- | --- | --- | --- | --- |"
OUT = "<out>"


def run_command(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [policies.__name__, *argv])
    policies.main()


def table_rows(text):
    """The policy rows: every line after the separator."""
    lines = text.splitlines()
    return lines[lines.index(SEPARATOR) + 1 :]


def row_for(text, policy):
    rows = [row for row in table_rows(text) if row.startswith(f"| {policy} |")]
    assert len(rows) == 1, f"expected one row for {policy}, got {rows}"
    return rows[0]


def test_comparison_of_the_golden_file_matches_the_golden_comparison(monkeypatch, capsys):
    run_command(monkeypatch, "--candidates", str(CANDIDATES), "--compare")
    captured = capsys.readouterr()
    assert captured.out == COMPARISON.read_text(), REGENERATE


def test_comparison_names_the_input_and_each_policys_choice(golden_set):
    text = policies.compare(CANDIDATES.read_bytes())
    sha = hashlib.sha256(CANDIDATES.read_bytes()).hexdigest()
    first = text.splitlines()[0]
    assert golden_set["source_snapshot_id"] in first
    assert sha in first
    assert not text.endswith("\n")

    best = row_for(text, "best_single_device")
    assert best.startswith("| best_single_device | MEMORY_FEASIBLE | ")
    assert best.endswith("| [28] | SIMULATED-second-mac | 19798163456 |")

    equal = row_for(text, "equal_split")
    assert equal.startswith("| equal_split | MEMORY_INFEASIBLE | ")
    assert equal.endswith(
        "| [14, 14] | SIMULATED-m4-pressure, SIMULATED-second-mac | -838336512, 21710241792 |"
    )

    proportional = row_for(text, "memory_proportional")
    assert proportional.startswith("| memory_proportional | MEMORY_FEASIBLE | ")
    assert proportional.endswith(
        "| [1, 27] | SIMULATED-m4-pressure, SIMULATED-second-mac | 937164800, 19934740480 |"
    )

    rows = table_rows(text)
    assert len(rows) == 3
    assert all(row.startswith("| ") for row in rows)
    for row, name in zip(rows, policies.POLICIES, strict=True):
        assert row.startswith(f"| {name} | ")


def test_a_policy_with_no_plan_gets_a_row_of_nones(golden_set, make_unfit):
    second_alone = golden_set["candidates"][1]
    assert golden_set["candidates"][0]["assignments"][0]["device_id"] == PRESSURE
    assert second_alone["assignments"][0]["device_id"] == SECOND
    assert len(second_alone["logical_split"]) == 1
    make_unfit(second_alone)

    lines = policies.compare(json.dumps(golden_set).encode()).splitlines()

    assert "| best_single_device | NO_FEASIBLE_PLAN | none | none | none | none |" in lines


def test_the_table_shows_the_plan_ids_of_the_decision_records():
    data = CANDIDATES.read_bytes()
    text = policies.compare(data)
    for name in policies.POLICIES:
        chosen = policies.decide(data, name)["chosen_plan_id"]
        assert chosen is not None
        assert chosen in row_for(text, name)


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(["--compare", "--policy", "equal_split"], id="compare_with_policy"),
        pytest.param(["--compare", "--output", OUT], id="compare_with_output"),
        pytest.param(["--policy", "equal_split"], id="policy_without_output"),
        pytest.param(["--output", OUT], id="output_without_policy"),
        pytest.param([], id="neither_mode"),
    ],
)
def test_command_refuses_a_mixed_or_incomplete_mode(extra, tmp_path, monkeypatch):
    out = tmp_path / "decision.json"
    argv = [str(out) if arg == OUT else arg for arg in extra]
    with pytest.raises(SystemExit):
        run_command(monkeypatch, "--candidates", str(CANDIDATES), *argv)
    assert not out.exists()


def test_the_policy_mode_still_writes_one_record(tmp_path, monkeypatch):
    out = tmp_path / "d.json"
    run_command(
        monkeypatch,
        "--candidates",
        str(CANDIDATES),
        "--policy",
        "equal_split",
        "--output",
        str(out),
    )
    record = json.loads(out.read_text())
    # Literal values: the field list from the decision schema, the plan and outcome from the
    # equal_split row of tests/golden/policy-comparison.md. The full record is in
    # test_policy_decision.py.
    assert list(record) == [
        "schema_version",
        "simulation",
        "policy",
        "source_snapshot_id",
        "candidate_set_sha256",
        "chosen_plan_id",
        "outcome",
        "reason",
        "rejection_reasons",
        "executable",
        "runtime_verified",
    ]
    assert record["policy"] == "equal_split"
    assert record["chosen_plan_id"] == "candidate-f52712c34dc8ad3a"
    assert record["outcome"] == "MEMORY_INFEASIBLE"
    assert record["source_snapshot_id"] == "SIMULATED-qwen7-pressure-v2"
    assert record["candidate_set_sha256"] == hashlib.sha256(CANDIDATES.read_bytes()).hexdigest()
    # The command writes exactly what the library returns for the same input.
    assert record == policies.decide(CANDIDATES.read_bytes(), "equal_split")
