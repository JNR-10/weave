"""Spec 0003: the committed evaluation plan is read strictly and lists its cells in one order."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from weave.evaluation import plan as evaluation_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/plans/evaluation-plan-v1.json"


def committed() -> dict:
    return json.loads(PLAN.read_text())


def parse(raw: dict) -> evaluation_plan.Plan:
    return evaluation_plan.plan_from_bytes(json.dumps(raw).encode(), "test plan")


def frozen(raw: dict) -> dict:
    """The committed draft with every arm available and no pilot left: a valid frozen plan."""
    raw = copy.deepcopy(raw)
    raw["status"] = "frozen"
    for entry in raw["policies"] + raw["ablations"]:
        entry["available"] = True
    for scenario in raw["scenarios"]:
        scenario["pilot_required"] = False
    return raw


def test_committed_plan_passes_and_is_a_draft():
    plan = evaluation_plan.load_plan(PLAN)
    assert plan.plan_id == "evaluation-plan-v1"
    assert plan.status == "draft"
    assert plan.sha256 == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    assert (plan.repeats_per_cell, plan.measured_requests_per_repeat) == (5, 20)
    assert plan.decision_rule.reference_arm == "profiled_static"


def test_committed_plan_declares_every_policy_and_one_ablation_per_signal():
    plan = evaluation_plan.load_plan(PLAN)
    assert [(p.policy, p.label, p.available) for p in plan.policies] == [
        ("best_single_device", "B0", True),
        ("equal_split", "B1", True),
        ("memory_proportional", "B2", True),
        ("profiled_static", "B3", False),
        ("weave_adaptive", "Weave", False),
    ]
    assert sorted(a.disables_signal for a in plan.ablations) == sorted(evaluation_plan.SIGNALS)


def test_cells_are_every_scenario_times_every_arm_in_declared_order():
    plan = evaluation_plan.load_plan(PLAN)
    cells = evaluation_plan.cells(plan)
    assert len(cells) == 6 * 9 == 54
    assert cells[0].cell_id == "short-c1/best_single_device"
    assert cells[8].cell_id == "short-c1/weave-no-stage-timings"
    assert cells[-1].cell_id == "short-c1-degraded-link/weave-no-stage-timings"
    assert len({c.cell_id for c in cells}) == 54


def test_a_cell_needs_two_macs_when_its_scenario_or_its_arm_does():
    by_id = {c.cell_id: c for c in evaluation_plan.cells(evaluation_plan.load_plan(PLAN))}
    assert not by_id["short-c1/best_single_device"].needs_two_macs
    assert by_id["short-c1/equal_split"].needs_two_macs
    assert by_id["short-c1-degraded-link/best_single_device"].needs_two_macs
    assert by_id["short-c1/equal_split"].available
    assert not by_id["short-c1/profiled_static"].available
    assert not by_id["short-c1/weave-no-live-memory"].available


def test_the_hash_is_of_the_bytes_as_read():
    # Reformatting the same content changes the hash, so results never match an edited plan.
    raw = committed()
    compact = evaluation_plan.plan_from_bytes(json.dumps(raw).encode(), "compact")
    spaced = evaluation_plan.plan_from_bytes(json.dumps(raw, indent=4).encode(), "spaced")
    assert compact.plan_id == spaced.plan_id
    assert compact.sha256 != spaced.sha256


def test_a_valid_frozen_plan_passes():
    assert parse(frozen(committed())).status == "frozen"


def _set(path, value):
    def edit(raw):
        *parents, last = path
        target = raw
        for key in parents:
            target = target[key]
        target[last] = value

    return edit


def _drop(path):
    def edit(raw):
        *parents, last = path
        target = raw
        for key in parents:
            target = target[key]
        del target[last]

    return edit


def _add_ablation(raw):
    raw["ablations"].append({**raw["ablations"][0], "ablation_id": "weave-no-live-memory-again"})


def _duplicate_arm(raw):
    raw["ablations"][0]["ablation_id"] = "equal_split"


def _duplicate_scenario(raw):
    raw["scenarios"][1]["scenario_id"] = "short-c1"


def _available_ablation_on_unavailable_base(raw):
    raw["ablations"][0]["available"] = True


INVALID = [
    ("version", _set(["schema_version"], "1.0.0"), "schema_version"),
    ("unknown field", _set(["repeats"], 5), "repeats: unknown field"),
    ("missing field", _drop(["plan_id"]), "plan_id: required"),
    ("bad status", _set(["status"], "final"), "status"),
    ("bad revision", _set(["model_revision"], "c26a38f"), "model_revision"),
    ("zero repeats", _set(["repeats_per_cell"], 0), "repeats_per_cell"),
    ("bool repeats", _set(["repeats_per_cell"], True), "repeats_per_cell"),
    ("negative warmup", _set(["warmup_requests_per_repeat"], -1), "warmup_requests_per_repeat"),
    (
        "too few requests",
        _set(["measured_requests_per_repeat"], 19),
        "measured_requests_per_repeat",
    ),
    ("quantile method", _set(["quantile_method"], "linear"), "quantile_method"),
    ("failure rule", _set(["failed_request_latency"], "dropped"), "failed_request_latency"),
    ("primary metric", _set(["primary_metric"], "end_to_end_ms_mean"), "primary_metric"),
    ("secondary metric", _set(["secondary_metrics"], ["tokens"]), "secondary_metrics[0]"),
    ("repeated metric", _set(["secondary_metrics"], ["ttft_ms_p95"] * 2), "secondary_metrics[1]"),
    (
        "reference arm",
        _set(["decision_rule", "reference_arm"], "b3"),
        "decision_rule.reference_arm",
    ),
    (
        "margin zero",
        _set(["decision_rule", "min_relative_difference"], 0),
        "decision_rule.min_relative",
    ),
    ("margin one", _set(["decision_rule", "min_relative_difference"], 1), "decision_rule.min_rel"),
    (
        "margin bool",
        _set(["decision_rule", "min_relative_difference"], True),
        "decision_rule.min_rel",
    ),
    ("no policies", _set(["policies"], []), "policies: expected at least one"),
    ("policy id", _set(["policies", 0, "policy"], "Best Single"), "policies[0].policy"),
    ("policy spec", _set(["policies", 0, "spec"], ""), "policies[0].spec"),
    ("policy field", _set(["policies", 0, "owner"], "Urmi"), "policies[0].owner: unknown"),
    ("ablation base", _set(["ablations", 0, "base_policy"], "nope"), "ablations[0].base_policy"),
    ("ablation signal", _set(["ablations", 0, "disables_signal"], "x"), "ablations[0].disables_s"),
    ("signal twice", _add_ablation, "ablations[4].disables_signal"),
    ("arm twice", _duplicate_arm, "ablations[0].ablation_id"),
    ("ablation ahead of base", _available_ablation_on_unavailable_base, "ablations[0].available"),
    ("no scenarios", _set(["scenarios"], []), "scenarios: expected at least one"),
    ("scenario twice", _duplicate_scenario, "scenarios[1].scenario_id"),
    ("scenario slash", _set(["scenarios", 0, "scenario_id"], "a/b"), "scenarios[0].scenario_id"),
    (
        "prompt tokens",
        _set(["scenarios", 0, "prompt_tokens_upper_bound"], 0),
        "scenarios[0].prompt",
    ),
    ("concurrency", _set(["scenarios", 0, "concurrency"], "1"), "scenarios[0].concurrency"),
    ("pressure level", _set(["scenarios", 0, "memory_pressure"], "high"), "scenarios[0].memory"),
    ("contention level", _set(["scenarios", 0, "compute_contention"], "x"), "scenarios[0].compute"),
    ("link level", _set(["scenarios", 0, "link_condition"], "fast"), "scenarios[0].link"),
    (
        "one-Mac degraded link",
        _set(["scenarios", 5, "needs_two_macs"], False),
        "scenarios[5].needs",
    ),
    ("pilot flag", _set(["scenarios", 0, "pilot_required"], 0), "scenarios[0].pilot_required"),
]


@pytest.mark.parametrize("name,edit,field", INVALID, ids=[case[0] for case in INVALID])
def test_invalid_plan_is_refused_naming_the_field(name, edit, field):
    raw = committed()
    edit(raw)
    with pytest.raises(ValueError, match=r"^" + field.replace("[", r"\[").replace("]", r"\]")):
        parse(raw)


def test_frozen_plan_refuses_a_scenario_still_waiting_for_the_pilot():
    raw = frozen(committed())
    raw["scenarios"][2]["pilot_required"] = True
    with pytest.raises(ValueError, match=r"^scenarios\[2\]\.pilot_required: a frozen plan"):
        parse(raw)


def test_frozen_plan_refuses_an_arm_that_does_not_exist_yet():
    raw = frozen(committed())
    raw["policies"][3]["available"] = False
    with pytest.raises(ValueError, match=r"^policies\[3\]\.available: a frozen plan"):
        parse(raw)


def test_plan_that_is_not_json_names_the_file():
    with pytest.raises(ValueError, match="^broken.json: not valid JSON"):
        evaluation_plan.plan_from_bytes(b"{", "broken.json")


def test_plan_that_is_not_an_object_is_refused():
    with pytest.raises(ValueError, match="^plan: expected a JSON object"):
        evaluation_plan.plan_from_bytes(b"[]", "list.json")


def test_check_command_prints_the_counts(capsys):
    assert evaluation_plan.main(["--check", str(PLAN)]) == 0
    out = capsys.readouterr().out
    assert "evaluation-plan-v1 (draft)" in out
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() in out
    assert "54 cells" in out
    assert "270 runs" in out
    assert "5670 requests" in out


def test_check_command_lists_cells_on_request(capsys):
    assert evaluation_plan.main(["--check", str(PLAN), "--cells"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  ")]
    assert len(lines) == 54
    assert lines[0].split() == ["short-c1/best_single_device", "one-Mac", "available"]
    assert lines[1].split() == ["short-c1/equal_split", "two-Macs", "available"]
    assert lines[3].split() == ["short-c1/profiled_static", "two-Macs", "not-available"]


def test_check_command_reports_an_invalid_plan_and_exits_nonzero(tmp_path, capsys):
    raw = committed()
    raw["status"] = "final"
    bad = tmp_path / "plan.json"
    bad.write_text(json.dumps(raw))
    assert evaluation_plan.main(["--check", str(bad)]) == 1
    assert "invalid plan" in capsys.readouterr().err
