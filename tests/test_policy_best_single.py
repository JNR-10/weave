"""Step 3 of spec 0002: the best-single-Mac policy, its tie rule and its no-fit rule.

The candidate set is read from the golden candidate file rather than generated here: policies
consume the generator's output file, and tests/test_phase1_golden.py already holds the generator to
that file byte for byte. It also keeps these tests out of the planner's checkpoint evidence, which
covers every test file that uses the planner modules.

In the file, candidates[0] is the pressure Mac alone (does not fit) and candidates[1] is the
second Mac alone (fits). The reader refuses a candidate whose memory_feasible contradicts its
rejection reasons, so every edit below keeps the two consistent.
"""

import copy
import dataclasses
import functools
import json
from pathlib import Path

import pytest

from weave.scheduler.policies import best_single_device, read_candidate_set

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/phase1-candidate-plans.json"

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"
SECOND_HEADROOM = 19798163456


@functools.cache
def _golden():
    return json.loads(GOLDEN.read_text())


def candidate_set():
    """A fresh copy of the fixture's candidate set, safe to edit."""
    return copy.deepcopy(_golden())


def _reason(device_id, headroom):
    return (
        f"{device_id} alone has the most headroom of the single-device plans that fit: "
        f"{headroom} bytes"
    )


def _make_feasible(raw, i, headroom):
    """Make single-device candidate i pass the estimate, keeping verdict and reasons consistent."""
    candidate = raw["candidates"][i]
    candidate["memory_feasible"] = True
    candidate["rejection_reasons"] = []
    candidate["assignments"][0]["headroom_bytes"] = headroom


def _choose(raw):
    return best_single_device(read_candidate_set(raw))


def test_fixture_picks_the_second_mac_alone():
    choice = _choose(candidate_set())
    assert choice.policy == "best_single_device"
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (SECOND,)
    assert choice.plan.plan_id == _golden()["candidates"][1]["plan_id"]
    assert choice.plan.headroom_bytes == (SECOND_HEADROOM,)
    assert choice.reason == _reason(SECOND, SECOND_HEADROOM)


def test_no_mac_fits_alone_reports_no_feasible_plan():
    raw = candidate_set()
    second = raw["candidates"][1]
    second["memory_feasible"] = False
    second["rejection_reasons"] = [
        {"code": "MEMORY_BUDGET_EXCEEDED", "device_id": SECOND, "excess_bytes": 1}
    ]
    second["assignments"][0]["headroom_bytes"] = -1
    choice = _choose(raw)
    assert choice.policy == "best_single_device"
    assert choice.outcome == "NO_FEASIBLE_PLAN"
    assert choice.plan is None
    assert choice.reason == "no single-device plan fits its Mac's budget"


def test_both_fit_and_the_earlier_mac_has_more_headroom():
    raw = candidate_set()
    _make_feasible(raw, 0, SECOND_HEADROOM + 1)
    choice = _choose(raw)
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (PRESSURE,)
    assert choice.reason == _reason(PRESSURE, SECOND_HEADROOM + 1)


def test_equal_headroom_goes_to_the_earlier_mac():
    raw = candidate_set()
    _make_feasible(raw, 0, SECOND_HEADROOM)
    choice = _choose(raw)
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (PRESSURE,)
    assert choice.reason == _reason(PRESSURE, SECOND_HEADROOM)


def test_both_fit_and_the_later_mac_has_more_headroom():
    raw = candidate_set()
    _make_feasible(raw, 0, 1)
    choice = _choose(raw)
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (SECOND,)
    assert choice.reason == _reason(SECOND, SECOND_HEADROOM)


def test_an_infeasible_plan_is_never_chosen_whatever_its_headroom_number():
    raw = candidate_set()
    # Its rejection reasons stay, so the verdict is still "does not fit".
    raw["candidates"][0]["assignments"][0]["headroom_bytes"] = 10**12
    choice = _choose(raw)
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (SECOND,)


def test_the_choice_is_frozen():
    choice = _choose(candidate_set())
    with pytest.raises(dataclasses.FrozenInstanceError):
        choice.outcome = "INFEASIBLE"
