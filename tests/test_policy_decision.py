"""Step 6 of spec 0002: the decision record, its schema and the --policy command.

The candidate set is read from the golden candidate file rather than generated here: policies
consume the generator's output file, and tests/test_phase1_golden.py already holds the generator to
that file byte for byte. It also keeps these tests out of the planner's checkpoint evidence, which
covers every test file that uses the planner modules.

A decision record names the candidate file it was made from by the SHA-256 of that file's bytes, so
the tests hash the golden file the same way. In the golden file, the plan with logical_split
[14, 14] and the pressure Mac first is the equal split of 28 layers over two Macs, and it fails the
pressure Mac's budget. The record is checked against the committed schema in
experiments/policies/schemas/v2/, not against the generator's output.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import jsonschema
import pytest

from weave.scheduler import policies

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/phase1-candidate-plans.json"
SCHEMA = ROOT / "experiments/policies/schemas/v2/placement_decision.schema.json"
SHA = hashlib.sha256(GOLDEN.read_bytes()).hexdigest()

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"

EQUAL_SPLIT_REASON = (
    "equal split [14, 14] on ['SIMULATED-m4-pressure', 'SIMULATED-second-mac'] "
    "fails the memory estimate on SIMULATED-m4-pressure"
)

FIELD_ORDER = [
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


def run_command(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [policies.__name__, *argv])
    policies.main()


def golden():
    return json.loads(GOLDEN.read_text())


def schema():
    return json.loads(SCHEMA.read_text())


def even_split_plan(raw, first_device):
    """The raw [14, 14] candidate whose stage 0 runs on first_device."""
    for candidate in raw["candidates"]:
        if candidate["logical_split"] == [14, 14]:
            if candidate["assignments"][0]["device_id"] == first_device:
                return candidate
    raise AssertionError(f"no [14, 14] candidate starts on {first_device}")


def equal_split_record():
    return policies.decide(GOLDEN.read_bytes(), "equal_split")


@pytest.mark.parametrize("name", ["best_single_device", "equal_split", "memory_proportional"])
def test_every_policy_gives_a_record_that_validates(name):
    record = policies.decide(GOLDEN.read_bytes(), name)
    jsonschema.validate(record, schema())
    assert record["policy"] == name
    assert record["candidate_set_sha256"] == SHA
    assert record["simulation"] is True
    assert record["executable"] is False
    assert record["runtime_verified"] is False


def test_equal_split_record_on_the_golden_file_is_exact():
    raw = golden()
    record = equal_split_record()
    assert list(record) == FIELD_ORDER
    assert record == {
        "schema_version": "2.0.0",
        "simulation": True,
        "policy": "equal_split",
        "source_snapshot_id": raw["source_snapshot_id"],
        "candidate_set_sha256": SHA,
        "chosen_plan_id": even_split_plan(raw, PRESSURE)["plan_id"],
        "outcome": "INFEASIBLE",
        "reason": EQUAL_SPLIT_REASON,
        "rejection_reasons": [
            {"code": "MEMORY_BUDGET_EXCEEDED", "device_id": PRESSURE, "excess_bytes": 838336512},
        ],
        "executable": False,
        "runtime_verified": False,
    }


def test_no_feasible_plan_record_has_no_chosen_plan():
    raw = golden()
    second_alone = raw["candidates"][1]
    assert second_alone["assignments"][0]["device_id"] == SECOND
    assert len(second_alone["logical_split"]) == 1
    second_alone["memory_feasible"] = False
    second_alone["rejection_reasons"] = [
        {"code": "MEMORY_BUDGET_EXCEEDED", "device_id": SECOND, "excess_bytes": 1},
    ]
    second_alone["assignments"][0]["headroom_bytes"] = -1
    edited = json.dumps(raw).encode()

    record = policies.decide(edited, "best_single_device")

    assert record["chosen_plan_id"] is None
    assert record["outcome"] == "NO_FEASIBLE_PLAN"
    assert record["rejection_reasons"] == []
    assert record["candidate_set_sha256"] == hashlib.sha256(edited).hexdigest()
    assert record["candidate_set_sha256"] != SHA
    jsonschema.validate(record, schema())


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="policy: expected one of"):
        policies.decide(GOLDEN.read_bytes(), "adaptive")


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"executable": True}, id="executable"),
        pytest.param({"runtime_verified": True}, id="runtime_verified"),
        pytest.param({"simulation": False}, id="not_simulated"),
        pytest.param({"selected_plan_id": None}, id="extra_field"),
        pytest.param({"chosen_plan_id": None}, id="infeasible_without_plan"),
        pytest.param({"outcome": "NO_FEASIBLE_PLAN"}, id="no_feasible_plan_with_plan"),
        pytest.param({"candidate_set_sha256": "abc"}, id="short_hash"),
        pytest.param({"policy": "adaptive"}, id="unknown_policy"),
        pytest.param({"schema_version": "1.0.0"}, id="old_version"),
    ],
)
def test_schema_refuses_a_broken_record(changes):
    record = equal_split_record()
    jsonschema.validate(record, schema())
    broken = {**copy.deepcopy(record), **changes}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema())


def test_command_writes_the_record_without_a_trailing_newline(tmp_path, monkeypatch):
    out = tmp_path / "decision.json"
    run_command(
        monkeypatch,
        "--candidates",
        str(GOLDEN),
        "--policy",
        "memory_proportional",
        "--output",
        str(out),
    )
    record = policies.decide(GOLDEN.read_bytes(), "memory_proportional")
    assert json.loads(out.read_text()) == record
    assert out.read_text() == json.dumps(record, indent=2)


def test_command_refuses_to_overwrite(tmp_path, monkeypatch):
    out = tmp_path / "decision.json"
    out.write_text("keep")
    with pytest.raises(FileExistsError):
        run_command(
            monkeypatch,
            "--candidates",
            str(GOLDEN),
            "--policy",
            "equal_split",
            "--output",
            str(out),
        )
    assert out.read_text() == "keep"


def test_command_refuses_an_unknown_policy(tmp_path, monkeypatch):
    out = tmp_path / "decision.json"
    with pytest.raises(SystemExit):
        run_command(
            monkeypatch,
            "--candidates",
            str(GOLDEN),
            "--policy",
            "adaptive",
            "--output",
            str(out),
        )
    assert not out.exists()
