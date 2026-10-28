"""Step 2 of spec 0002: read_candidate_set reads the generator's output once and refuses bad input.

Every refusal is a ValueError whose message starts with the offending field's path, so the person
fixing the candidate set knows exactly where to look. The input is the committed fixture's
candidate set, edited per case.

The candidate set is read from the golden candidate file rather than generated here: policies
consume the generator's output file, and tests/test_phase1_golden.py already holds the generator to
that file byte for byte. It also keeps these tests out of the planner's checkpoint evidence, which
covers every test file that uses the planner modules.
"""

import copy
import dataclasses
import functools
import json
import re
from pathlib import Path

import pytest

from weave.scheduler.policies import read_candidate_set

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/phase1-candidate-plans.json"

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"
SECOND_BUDGET = 25769803776


@functools.cache
def _golden():
    return json.loads(GOLDEN.read_text())


def candidate_set():
    """A fresh copy of the fixture's candidate set, safe to edit."""
    return copy.deepcopy(_golden())


def test_fixture_layout_the_invalid_table_relies_on():
    # The generator emits the single-device plans first, then the pressure-first group from [1, 27].
    candidates = candidate_set()["candidates"]
    assert [c["logical_split"] for c in candidates[:4]] == [[28], [28], [1, 27], [2, 26]]
    assert [a["device_id"] for a in candidates[2]["assignments"]] == [PRESSURE, SECOND]
    assert candidates[0]["assignments"][0]["device_id"] == PRESSURE
    assert candidates[1]["assignments"][0]["device_id"] == SECOND
    assert len(candidates) == 56


def test_valid_fixture_is_read_into_typed_records():
    result = read_candidate_set(candidate_set())
    assert len(result.candidates) == 56
    assert result.total_layers == 28
    assert result.source_snapshot_id == candidate_set()["source_snapshot_id"]
    assert tuple(d.device_id for d in result.devices) == (PRESSURE, SECOND)
    pressure, second = result.devices
    assert pressure.safe_process_budget_bytes == 3221225472
    assert pressure.fixed_stage_bytes == 2147483648
    assert second.safe_process_budget_bytes == SECOND_BUDGET


def test_find_returns_the_candidate_with_its_memory_verdict():
    result = read_candidate_set(candidate_set())
    plan = result.find((PRESSURE, SECOND), (10, 18))
    assert plan.device_order == (PRESSURE, SECOND)
    assert plan.logical_split == (10, 18)
    assert not plan.memory_feasible
    assert plan.headroom_bytes[0] == -292028416
    assert plan.rejection_reasons == (
        {"code": "MEMORY_BUDGET_EXCEEDED", "device_id": PRESSURE, "excess_bytes": 292028416},
    )
    alone = result.find([SECOND], [28])
    assert alone.memory_feasible
    assert alone.headroom_bytes == (19798163456,)
    assert alone.rejection_reasons == ()


@pytest.mark.parametrize(
    "device_order, logical_split",
    [
        ((PRESSURE, SECOND), (0, 28)),
        ((PRESSURE, SECOND), (14, 13)),
        ((PRESSURE,), (10,)),
        (("SIMULATED-third",), (28,)),
    ],
)
def test_find_refuses_a_split_that_is_not_among_the_candidates(device_order, logical_split):
    result = read_candidate_set(candidate_set())
    with pytest.raises(ValueError, match="not among the candidates"):
        result.find(device_order, logical_split)


def test_reading_never_mutates_the_input_or_points_into_it():
    raw = candidate_set()
    before = copy.deepcopy(raw)
    result = read_candidate_set(raw)
    assert raw == before
    plan = result.find((PRESSURE, SECOND), (10, 18))
    plan.rejection_reasons[0]["excess_bytes"] = 0
    assert raw == before


def test_records_are_frozen():
    result = read_candidate_set(candidate_set())
    records = [
        (result, "total_layers"),
        (result.devices[0], "device_id"),
        (result.candidates[0], "plan_id"),
    ]
    for record, attribute in records:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, attribute, "changed")


# Each edit takes the fixture's candidate set and returns what to hand to read_candidate_set.


def _apply(edit):
    def mutate(raw):
        edit(raw)
        return raw

    return mutate


def _top(**changes):
    return _apply(lambda raw: raw.update(changes))


def _cand(i, **changes):
    return _apply(lambda raw: raw["candidates"][i].update(changes))


def _drop(i, key):
    return _apply(lambda raw: raw["candidates"][i].pop(key))


def _assign(i, j, **changes):
    return _apply(lambda raw: raw["candidates"][i]["assignments"][j].update(changes))


def _parts(i, j, **changes):
    def edit(raw):
        raw["candidates"][i]["assignments"][j]["memory_components"].update(changes)

    return _apply(edit)


def _repeat_plan_id(raw):
    raw["candidates"][1]["plan_id"] = raw["candidates"][0]["plan_id"]


def _change_total_to_29(raw):
    # Consistent within the candidate, so only the cross-candidate total check can refuse it.
    plan = raw["candidates"][2]
    plan["logical_split"][0] = 2
    plan["assignments"][0]["layer_count"] = 2


def _add_second_single(raw):
    extra = copy.deepcopy(raw["candidates"][0])
    extra["plan_id"] = "candidate-extra"
    raw["candidates"].append(extra)


INVALID = [
    # The set as a whole.
    ("set-not-object", lambda raw: [], "candidate_set: expected a JSON object, got list"),
    ("set-version-old", _top(schema_version="1.0.0"), "schema_version: unsupported '1.0.0'"),
    (
        "set-version-missing",
        _apply(lambda raw: raw.pop("schema_version")),
        "schema_version: required",
    ),
    ("set-not-simulated", _top(simulation=False), "simulation: expected true, got False"),
    ("set-simulation-not-bool", _top(simulation=1), "simulation: expected true, got 1"),
    (
        "set-already-selected",
        _top(selected_plan_id="candidate-x"),
        "selected_plan_id: expected null",
    ),
    (
        "set-snapshot-empty",
        _top(source_snapshot_id=""),
        "source_snapshot_id: expected a non-empty string",
    ),
    ("set-candidates-not-list", _top(candidates={}), "candidates: expected a non-empty list"),
    ("set-candidates-empty", _top(candidates=[]), "candidates: expected a non-empty list"),
    # One candidate.
    (
        "candidate-not-object",
        _apply(lambda raw: raw["candidates"].__setitem__(1, "x")),
        "candidates[1]: expected a JSON object, got str",
    ),
    (
        "candidate-version-old",
        _cand(1, schema_version="1.0.0"),
        "candidates[1].schema_version: unsupported '1.0.0'",
    ),
    (
        "candidate-not-simulated",
        _cand(1, simulation=False),
        "candidates[1].simulation: expected true, got False",
    ),
    (
        "candidate-selected",
        _cand(1, selected=True),
        "candidates[1].selected: expected false, got True",
    ),
    (
        "candidate-executable",
        _cand(1, executable=True),
        "candidates[1].executable: expected false, got True",
    ),
    (
        "candidate-runtime-verified",
        _cand(1, runtime_verified=True),
        "candidates[1].runtime_verified: expected false, got True",
    ),
    (
        "candidate-policy",
        _cand(1, policy="best_single_device"),
        "candidates[1].policy: expected 'candidate_only', got 'best_single_device'",
    ),
    (
        "candidate-plan-id-empty",
        _cand(1, plan_id=""),
        "candidates[1].plan_id: expected a non-empty string",
    ),
    (
        "candidate-plan-id-repeated",
        _apply(_repeat_plan_id),
        "candidates[1].plan_id: duplicate",
    ),
    (
        "candidate-snapshot-differs",
        _cand(1, source_snapshot_id="other"),
        "candidates[1].source_snapshot_id: 'other' differs from the set's",
    ),
    (
        "candidate-mode-unknown",
        _cand(0, mode="mesh"),
        "candidates[0].mode: expected 'single_device' or 'pipeline', got 'mesh'",
    ),
    (
        "candidate-mode-pipeline-with-one-stage",
        _cand(0, mode="pipeline"),
        "candidates[0].mode: 'pipeline' is inconsistent with 1 stage(s)",
    ),
    (
        "candidate-mode-single-with-two-stages",
        _cand(2, mode="single_device"),
        "candidates[2].mode: 'single_device' is inconsistent with 2 stage(s)",
    ),
    (
        "candidate-split-empty",
        _cand(0, logical_split=[]),
        "candidates[0].logical_split: expected a non-empty list",
    ),
    (
        "candidate-split-not-list",
        _cand(0, logical_split="28"),
        "candidates[0].logical_split: expected a non-empty list",
    ),
    (
        "candidate-split-not-positive",
        _cand(0, logical_split=[0]),
        "candidates[0].logical_split[0]: expected a positive integer, got 0",
    ),
    (
        "candidate-feasible-not-bool",
        _cand(0, memory_feasible="yes"),
        "candidates[0].memory_feasible: expected true or false",
    ),
    (
        "candidate-feasible-missing",
        _drop(0, "memory_feasible"),
        "candidates[0].memory_feasible: required",
    ),
    (
        "candidate-feasible-with-reasons",
        _cand(0, memory_feasible=True),
        "candidates[0].memory_feasible: True contradicts 1 rejection reason(s)",
    ),
    (
        "candidate-infeasible-without-reasons",
        _cand(1, memory_feasible=False),
        "candidates[1].memory_feasible: False contradicts 0 rejection reason(s)",
    ),
    (
        "candidate-reasons-not-list",
        _cand(0, rejection_reasons="none"),
        "candidates[0].rejection_reasons: expected a list, got str",
    ),
    (
        "candidate-reason-not-object",
        _cand(0, rejection_reasons=[5]),
        "candidates[0].rejection_reasons[0]: expected a JSON object, got int",
    ),
    (
        "candidate-assignments-not-list",
        _cand(2, assignments={}),
        "candidates[2].assignments: expected a list, got dict",
    ),
    (
        "candidate-assignments-too-few",
        _apply(lambda raw: raw["candidates"][2]["assignments"].pop()),
        "candidates[2].assignments: expected one entry per logical_split entry (2), got 1",
    ),
    # One assignment.
    (
        "assignment-not-object",
        _apply(lambda raw: raw["candidates"][2]["assignments"].__setitem__(1, "x")),
        "candidates[2].assignments[1]: expected a JSON object, got str",
    ),
    (
        "assignment-device-empty",
        _assign(2, 0, device_id=""),
        "candidates[2].assignments[0].device_id: expected a non-empty string",
    ),
    (
        "assignment-device-repeated",
        _assign(2, 1, device_id=PRESSURE),
        f"candidates[2].assignments[1].device_id: duplicate '{PRESSURE}'",
    ),
    (
        "assignment-stage-wrong",
        _assign(2, 1, logical_stage=0),
        "candidates[2].assignments[1].logical_stage: expected 1, got 0",
    ),
    (
        "assignment-layer-count-wrong",
        _assign(2, 0, layer_count=2),
        "candidates[2].assignments[0].layer_count: expected 1, got 2",
    ),
    (
        "assignment-headroom-bool",
        _assign(3, 0, headroom_bytes=True),
        "candidates[3].assignments[0].headroom_bytes: expected an integer, got True",
    ),
    (
        "assignment-headroom-float",
        _assign(3, 0, headroom_bytes=1.5),
        "candidates[3].assignments[0].headroom_bytes: expected an integer, got 1.5",
    ),
    (
        "assignment-budget-zero",
        _assign(2, 0, safe_process_budget_bytes=0),
        "candidates[2].assignments[0].safe_process_budget_bytes: "
        "expected a positive integer, got 0",
    ),
    (
        "assignment-components-not-object",
        _assign(2, 0, memory_components=[]),
        "candidates[2].assignments[0].memory_components: expected a JSON object, got list",
    ),
    (
        "assignment-replicated-zero",
        _parts(2, 0, replicated_weight_bytes=0),
        "candidates[2].assignments[0].memory_components.replicated_weight_bytes: "
        "expected a positive integer, got 0",
    ),
    (
        "assignment-workspace-bool",
        _parts(2, 1, workspace_reserve_bytes=True),
        "candidates[2].assignments[1].memory_components.workspace_reserve_bytes: "
        "expected a positive integer, got True",
    ),
    (
        "assignment-safety-float",
        _parts(2, 1, safety_margin_bytes=2.5),
        "candidates[2].assignments[1].memory_components.safety_margin_bytes: "
        "expected a positive integer, got 2.5",
    ),
    # Across candidates.
    (
        "layer-totals-differ",
        _apply(_change_total_to_29),
        "candidates[2].logical_split: sums to 29 layers, expected 28",
    ),
    (
        "single-device-candidate-missing",
        _apply(lambda raw: raw["candidates"].pop(0)),
        f"candidates: no single-device candidate for '{PRESSURE}'",
    ),
    (
        "single-device-candidate-twice",
        _apply(_add_second_single),
        f"candidates[56]: second single-device candidate for '{PRESSURE}'",
    ),
    (
        "budget-disagrees",
        _assign(2, 1, safe_process_budget_bytes=SECOND_BUDGET - 1),
        "candidates[2].assignments[1].safe_process_budget_bytes: "
        f"{SECOND_BUDGET - 1} differs from {SECOND_BUDGET}",
    ),
    (
        "fixed-component-disagrees",
        _parts(2, 0, replicated_weight_bytes=1),
        "candidates[2].assignments[0].memory_components.replicated_weight_bytes: 1 differs from",
    ),
]


@pytest.mark.parametrize("case", INVALID, ids=lambda case: case[0])
def test_invalid_candidate_set_is_refused_with_the_field_named(case):
    _, mutate, fragment = case
    raw = mutate(candidate_set())
    with pytest.raises(ValueError, match=rf"^{re.escape(fragment)}"):
        read_candidate_set(raw)
