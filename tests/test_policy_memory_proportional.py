"""Step 5 of spec 0002: the memory-proportional policy and its rounding rules.

The candidate set is read from the golden candidate file rather than generated here: policies
consume the generator's output file, and tests/test_phase1_golden.py already holds the generator to
that file byte for byte. It also keeps these tests out of the planner's checkpoint evidence, which
covers every test file that uses the planner modules.

In the file, the plan with logical_split [1, 27] and the pressure Mac first is the memory-
proportional split of 28 layers: the two Macs have 1073741824 and 23622320128 bytes of room for
layers (1 : 22), and it passes the memory estimate. The reader refuses a candidate whose
memory_feasible contradicts its rejection reasons, and a Mac whose budget differs between
candidates, so every edit below keeps both consistent.
"""

import copy
import functools
import json
import re
from pathlib import Path

import pytest

from weave.scheduler.policies import (
    memory_proportional,
    memory_proportional_counts,
    read_candidate_set,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/phase1-candidate-plans.json"

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"

FEASIBLE_REASON = (
    "memory-proportional split [1, 27] on ['SIMULATED-m4-pressure', 'SIMULATED-second-mac'] "
    "from room [1073741824, 23622320128] bytes passes the memory estimate"
)
INFEASIBLE_REASON = (
    "memory-proportional split [1, 27] on ['SIMULATED-m4-pressure', 'SIMULATED-second-mac'] "
    "from room [1073741824, 23622320128] bytes fails the memory estimate on "
    "SIMULATED-m4-pressure"
)


@functools.cache
def _golden():
    return json.loads(GOLDEN.read_text())


def candidate_set():
    """A fresh copy of the fixture's candidate set, safe to edit."""
    return copy.deepcopy(_golden())


def _plan(raw, split, first_device):
    """The raw candidate with this logical_split whose stage 0 runs on first_device."""
    for candidate in raw["candidates"]:
        if candidate["logical_split"] == split:
            if candidate["assignments"][0]["device_id"] == first_device:
                return candidate
    raise AssertionError(f"no {split} candidate starts on {first_device}")


def _choose(raw):
    return memory_proportional(read_candidate_set(raw))


def test_fixture_memory_proportional_split_is_feasible():
    choice = _choose(candidate_set())
    assert choice.policy == "memory_proportional"
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (PRESSURE, SECOND)
    assert choice.plan.logical_split == (1, 27)
    assert choice.plan.plan_id == _plan(_golden(), [1, 27], PRESSURE)["plan_id"]
    assert choice.plan.headroom_bytes == (937164800, 19934740480)
    assert choice.plan.rejection_reasons == ()
    assert choice.reason == FEASIBLE_REASON


def test_a_split_that_fails_the_estimate_is_reported_not_replaced():
    raw = candidate_set()
    plan = _plan(raw, [1, 27], PRESSURE)
    plan["memory_feasible"] = False
    plan["rejection_reasons"] = [
        {"code": "MEMORY_BUDGET_EXCEEDED", "device_id": PRESSURE, "excess_bytes": 1}
    ]
    plan["assignments"][0]["headroom_bytes"] = -1
    choice = _choose(raw)
    assert choice.policy == "memory_proportional"
    assert choice.outcome == "INFEASIBLE"
    assert choice.plan is not None
    assert choice.plan.plan_id == plan["plan_id"]
    assert choice.plan.headroom_bytes == (-1, 19934740480)
    assert choice.reason == INFEASIBLE_REASON


def test_the_split_follows_the_budgets():
    raw = candidate_set()
    # The pressure Mac gets the second Mac's budget in every candidate, so the reader's check that
    # a Mac states one budget still passes. The edited set keeps the golden verdicts, which the
    # policy reports as given; only the split is asserted.
    for candidate in raw["candidates"]:
        for assignment in candidate["assignments"]:
            if assignment["device_id"] == PRESSURE:
                assignment["safe_process_budget_bytes"] = 25769803776
    choice = _choose(raw)
    assert choice.policy == "memory_proportional"
    assert choice.plan is not None
    assert choice.plan.device_order == (PRESSURE, SECOND)
    assert choice.plan.logical_split == (14, 14)


@pytest.mark.parametrize(
    ("total", "rooms", "expected"),
    [
        (28, [1073741824, 23622320128], (1, 27)),
        (28, [1, 1], (14, 14)),
        (28, [0, 0], (14, 14)),
        (28, [0, 5], (1, 27)),
        (10, [1, 2, 2], (2, 4, 4)),
        (3, [1, 1], (2, 1)),
        (28, [1, 1, 1], (10, 9, 9)),
        (3, [0, 0, 5], (1, 1, 1)),
        (4, [0, 1, 1], (1, 1, 2)),
        (1, [7], (1,)),
    ],
)
def test_counts_rule(total, rooms, expected):
    counts = memory_proportional_counts(total, rooms)
    assert counts == expected
    assert sum(counts) == total
    assert min(counts) >= 1


@pytest.mark.parametrize(
    ("total", "rooms", "message"),
    [
        (0, [1], "total_layers: expected a positive integer"),
        (28, [], "rooms: expected a non-empty sequence"),
        (28, [1, -1], "rooms[1]: expected a non-negative integer"),
        (28, [True, 1], "rooms[0]: expected a non-negative integer"),
        (28, [1.5, 1], "rooms[0]: expected a non-negative integer"),
        (2, [1, 1, 1], "cannot give each of 3 Macs at least one"),
        (28, "ab", "rooms: expected a non-empty sequence"),
    ],
)
def test_counts_rule_refuses_bad_arguments(total, rooms, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        memory_proportional_counts(total, rooms)
