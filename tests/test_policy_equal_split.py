"""Step 4 of spec 0002: the equal-split policy and its rounding rule.

The candidate set is read from the golden candidate file rather than generated here: policies
consume the generator's output file, and tests/test_phase1_golden.py already holds the generator to
that file byte for byte. It also keeps these tests out of the planner's checkpoint evidence, which
covers every test file that uses the planner modules.

In the file, the plan with logical_split [14, 14] and the pressure Mac first is the equal split of
28 layers over two Macs, and it fails the pressure Mac's budget. The reader refuses a candidate
whose memory_feasible contradicts its rejection reasons, so every edit below keeps the two
consistent.
"""

import copy
import functools
import json
from pathlib import Path

import pytest

from weave.scheduler.policies import equal_split, equal_split_counts, read_candidate_set

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/phase1-candidate-plans.json"

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"

FEASIBLE_REASON = (
    "equal split [14, 14] on ['SIMULATED-m4-pressure', 'SIMULATED-second-mac'] "
    "passes the memory estimate"
)
INFEASIBLE_REASON = (
    "equal split [14, 14] on ['SIMULATED-m4-pressure', 'SIMULATED-second-mac'] "
    "fails the memory estimate on SIMULATED-m4-pressure"
)


@functools.cache
def _golden():
    return json.loads(GOLDEN.read_text())


def candidate_set():
    """A fresh copy of the fixture's candidate set, safe to edit."""
    return copy.deepcopy(_golden())


def _even_split_plan(raw, first_device):
    """The raw [14, 14] candidate whose stage 0 runs on first_device."""
    for candidate in raw["candidates"]:
        if candidate["logical_split"] == [14, 14]:
            if candidate["assignments"][0]["device_id"] == first_device:
                return candidate
    raise AssertionError(f"no [14, 14] candidate starts on {first_device}")


def _choose(raw):
    return equal_split(read_candidate_set(raw))


def test_fixture_equal_split_is_infeasible_on_the_pressure_mac():
    choice = _choose(candidate_set())
    assert choice.policy == "equal_split"
    assert choice.outcome == "INFEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (PRESSURE, SECOND)
    assert choice.plan.logical_split == (14, 14)
    assert choice.plan.plan_id == _even_split_plan(_golden(), PRESSURE)["plan_id"]
    assert choice.plan.headroom_bytes == (-838336512, 21710241792)
    assert choice.plan.rejection_reasons == (
        {"code": "MEMORY_BUDGET_EXCEEDED", "device_id": PRESSURE, "excess_bytes": 838336512},
    )
    assert choice.reason == INFEASIBLE_REASON


def test_an_equal_split_that_fits_is_feasible():
    raw = candidate_set()
    plan = _even_split_plan(raw, PRESSURE)
    plan["memory_feasible"] = True
    plan["rejection_reasons"] = []
    plan["assignments"][0]["headroom_bytes"] = 1
    choice = _choose(raw)
    assert choice.policy == "equal_split"
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.plan_id == plan["plan_id"]
    assert choice.plan.rejection_reasons == ()
    assert choice.reason == FEASIBLE_REASON


@pytest.mark.parametrize(
    ("total", "count", "expected"),
    [
        (28, 1, (28,)),
        (28, 2, (14, 14)),
        (28, 3, (10, 9, 9)),
        (28, 5, (6, 6, 6, 5, 5)),
        (29, 2, (15, 14)),
        (7, 7, (1,) * 7),
        (1, 1, (1,)),
    ],
)
def test_counts_rule(total, count, expected):
    counts = equal_split_counts(total, count)
    assert counts == expected
    assert sum(counts) == total
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize(
    ("total", "count", "message"),
    [
        (1, 2, "cannot give each of 2 Macs at least one"),
        (0, 1, "total_layers: expected a positive integer"),
        (28, 0, "device_count: expected a positive integer"),
        (True, 1, "total_layers: expected a positive integer"),
        (28, 2.0, "device_count: expected a positive integer"),
    ],
)
def test_counts_rule_refuses_bad_arguments(total, count, message):
    with pytest.raises(ValueError, match=message):
        equal_split_counts(total, count)


def test_a_model_with_fewer_layers_than_macs_is_refused():
    raw = candidate_set()
    raw["candidates"] = raw["candidates"][:2]
    for candidate in raw["candidates"]:
        candidate["logical_split"] = [1]
        candidate["assignments"][0]["layer_count"] = 1
    candidates = read_candidate_set(raw)
    with pytest.raises(ValueError, match="cannot give each of 2 Macs at least one"):
        equal_split(candidates)


def test_a_missing_equal_split_plan_is_refused_not_replaced():
    raw = candidate_set()
    raw["candidates"].remove(_even_split_plan(raw, PRESSURE))
    candidates = read_candidate_set(raw)
    with pytest.raises(ValueError, match="not among the candidates"):
        equal_split(candidates)


def test_the_equal_split_never_searches_device_orders():
    raw = candidate_set()
    other = read_candidate_set(raw).find((SECOND, PRESSURE), (14, 14))
    choice = _choose(raw)
    assert choice.plan is not None
    assert choice.plan.device_order == (PRESSURE, SECOND)
    assert choice.plan.plan_id != other.plan_id
