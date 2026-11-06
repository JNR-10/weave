"""Spec 0002: the three policies on a cluster of three Macs, which the golden file does not have.

The golden candidate file has two Macs, so this file generates a three-Mac set with the real
generator: the committed simulated cluster state plus a copy of the second Mac, with the Phase 1
device limit raised to 3 for the test only. Every byte value is the generator's own.

The room for layers is the budget minus 2147483648 fixed bytes: 1073741824 on the pressure Mac
and 23622320128 on each of the others. For 28 layers the floors are [0, 13, 13], so 2 layers are
left over. The two larger remainders belong to the second and third Macs, which gives
[0, 14, 14]. The pressure Mac cannot run zero layers, so it takes one from the earliest Mac with
the most, giving [1, 13, 14]. The equal split of 28 layers over three Macs is [10, 9, 9]; its 10
layers on the pressure Mac exceed that Mac's budget by 292,028,416 bytes.
"""

import json
from pathlib import Path

import pytest

from weave.scheduler import candidates
from weave.scheduler.policies import (
    best_single_device,
    compare,
    equal_split,
    memory_proportional,
    read_candidate_set,
)

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "experiments/phase1/cluster-state-SIMULATED.json"

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"
THIRD = "SIMULATED-third-mac"
ORDER = [PRESSURE, SECOND, THIRD]
SECOND_HEADROOM = 19798163456
SEPARATOR = "| --- | --- | --- | --- | --- | --- |"


@pytest.fixture
def three_mac_set(monkeypatch):
    """The generator's candidate set for the simulated state plus a copy of the second Mac."""
    state = json.loads(STATE.read_text())
    state["devices"].append({**state["devices"][1], "device_id": THIRD})
    monkeypatch.setattr(candidates, "PHASE1_MAX_DEVICES", 3)
    return candidates.generate(state)


def find_plan(raw, order, split):
    """The generator's plan for this device order and logical split."""
    matches = [
        plan
        for plan in raw["candidates"]
        if [a["device_id"] for a in plan["assignments"]] == order and plan["logical_split"] == split
    ]
    assert len(matches) == 1
    return matches[0]


def plan_id(raw, order, split):
    """The generator's plan_id for this device order and logical split."""
    return find_plan(raw, order, split)["plan_id"]


def test_the_reader_sees_three_macs_in_the_order_of_the_single_device_plans(three_mac_set):
    candidate_set = read_candidate_set(three_mac_set)
    assert candidate_set.device_ids == tuple(ORDER)
    assert candidate_set.total_layers == 28
    # 3 alone + 6 ordered pairs x 27 cuts + 6 orders of all three x C(27, 2).
    assert len(candidate_set.candidates) == 2271
    rooms = [device.room_for_layers_bytes for device in candidate_set.devices]
    assert rooms == [1073741824, 23622320128, 23622320128]


def test_equal_split_gives_the_extra_layer_to_the_first_mac_and_fails_on_it(three_mac_set):
    choice = equal_split(read_candidate_set(three_mac_set))
    assert choice.outcome == "MEMORY_INFEASIBLE"
    assert choice.plan is not None
    assert choice.plan.plan_id == plan_id(three_mac_set, ORDER, [10, 9, 9])
    assert choice.plan.device_order == tuple(ORDER)
    assert choice.plan.logical_split == (10, 9, 9)
    assert choice.plan.headroom_bytes == (-292028416, 22393126912, 22393126912)
    assert choice.reason == (
        f"equal split [10, 9, 9] on {ORDER} fails the memory estimate on {PRESSURE}"
    )
    assert [dict(r) for r in choice.rejection_reasons] == [
        {"code": "MEMORY_BUDGET_EXCEEDED", "device_id": PRESSURE, "excess_bytes": 292028416}
    ]


def test_memory_proportional_follows_the_room_and_keeps_every_stage_non_empty(three_mac_set):
    choice = memory_proportional(read_candidate_set(three_mac_set))
    assert choice.outcome == "MEMORY_FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.plan_id == plan_id(three_mac_set, ORDER, [1, 13, 14])
    assert choice.plan.device_order == tuple(ORDER)
    assert choice.plan.logical_split == (1, 13, 14)
    assert choice.plan.headroom_bytes == (937164800, 21846818816, 21710241792)
    assert choice.reason == (
        f"memory-proportional split [1, 13, 14] on {ORDER} "
        "from room [1073741824, 23622320128, 23622320128] bytes passes the memory estimate"
    )


def test_best_single_device_breaks_a_tie_toward_the_earlier_mac(three_mac_set):
    third_alone = find_plan(three_mac_set, [THIRD], [28])
    assert third_alone["assignments"][0]["headroom_bytes"] == SECOND_HEADROOM

    choice = best_single_device(read_candidate_set(three_mac_set))
    assert choice.outcome == "MEMORY_FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.plan_id == plan_id(three_mac_set, [SECOND], [28])
    assert choice.plan.device_order == (SECOND,)
    assert choice.plan.headroom_bytes == (SECOND_HEADROOM,)
    assert choice.reason == (
        f"{SECOND} alone has the most headroom of the single-device plans that pass the "
        f"memory estimate: {SECOND_HEADROOM} bytes"
    )


def test_compare_has_one_row_per_policy_on_three_macs(three_mac_set):
    lines = compare(json.dumps(three_mac_set).encode()).splitlines()
    rows = lines[lines.index(SEPARATOR) + 1 :]
    order_cell = ", ".join(ORDER)
    equal_id = plan_id(three_mac_set, ORDER, [10, 9, 9])
    proportional_id = plan_id(three_mac_set, ORDER, [1, 13, 14])
    assert len(rows) == 3
    assert rows[0].startswith("| best_single_device | MEMORY_FEASIBLE | ")
    assert rows[0].endswith(f"| [28] | {SECOND} | {SECOND_HEADROOM} |")
    assert rows[1] == (
        f"| equal_split | MEMORY_INFEASIBLE | {equal_id} | [10, 9, 9] | {order_cell} "
        "| -292028416, 22393126912, 22393126912 |"
    )
    assert rows[2] == (
        f"| memory_proportional | MEMORY_FEASIBLE | {proportional_id} | [1, 13, 14] "
        f"| {order_cell} | 937164800, 21846818816, 21710241792 |"
    )
