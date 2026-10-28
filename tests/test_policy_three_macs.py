"""Spec 0002: the three policies on a cluster of three Macs, which the golden file does not have.

The golden candidate file has two Macs. Policy tests may not import the planner, so this file
hand-builds a minimal three-Mac candidate set from golden pieces instead of generating one. The
reader needs only each Mac's single-device plan plus the plans a policy looks up, so the set holds
three single-device plans and two three-stage plans: the equal split and the
memory-proportional split.

The third Mac is a copy of the second, so both state the same budget and the reader's check that
a Mac has one budget passes. Byte figures other than the headrooms are copied, not recomputed;
the policies read only the layer counts, the headrooms, the verdicts and the budgets.

The room for layers is the budget minus 2147483648 fixed bytes: 1073741824 on the pressure Mac
and 23622320128 on each of the others, 48318382080 in all. For 28 layers the floors are 0, 13 and
13, so 2 layers are left over. The two larger remainders belong to the second and third Macs,
which gives [0, 14, 14]. The pressure Mac cannot run zero layers, so it takes one from the
earliest Mac with the most, giving [1, 13, 14]. The equal split of 28 layers over three Macs is
[10, 9, 9].
"""

import copy
import functools
import json
from pathlib import Path

from weave.scheduler.policies import (
    best_single_device,
    compare,
    equal_split,
    memory_proportional,
    read_candidate_set,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/phase1-candidate-plans.json"

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"
THIRD = "SIMULATED-third-mac"
ORDER = [PRESSURE, SECOND, THIRD]
SECOND_HEADROOM = 19798163456
SEPARATOR = "| --- | --- | --- | --- | --- | --- |"


@functools.cache
def _golden():
    return json.loads(GOLDEN.read_text())


def _pipeline(base, plan_id, split, feasible):
    """A three-stage plan on (pressure, second, third) built from the golden two-stage plan."""
    plan = copy.deepcopy(base)
    pressure_stage, second_stage = plan["assignments"]
    assert pressure_stage["device_id"] == PRESSURE
    assert second_stage["device_id"] == SECOND
    templates = [pressure_stage, second_stage, second_stage]
    assignments = []
    start = 0
    for stage, count in enumerate(split):
        assignment = copy.deepcopy(templates[stage])
        assignment.update(
            device_id=ORDER[stage],
            logical_stage=stage,
            mlx_rank=len(ORDER) - 1 - stage,
            start_layer=start,
            end_layer_exclusive=start + count,
            layer_count=count,
        )
        assignments.append(assignment)
        start += count
    reasons = []
    if not feasible:
        assignments[0]["headroom_bytes"] = -1
        reasons = [{"code": "MEMORY_BUDGET_EXCEEDED", "device_id": PRESSURE, "excess_bytes": 1}]
    plan.update(
        plan_id=plan_id,
        mode="pipeline",
        logical_split=list(split),
        mlx_split_by_rank=list(reversed(split)),
        device_id_by_mlx_rank=list(reversed(ORDER)),
        assignments=assignments,
        memory_feasible=feasible,
        rejection_reasons=reasons,
    )
    return plan


def three_mac_set():
    """A fresh three-Mac candidate set: three single-device plans and two three-stage plans."""
    raw = copy.deepcopy(_golden())
    pressure_alone, second_alone, two_stage = raw["candidates"][:3]
    assert pressure_alone["assignments"][0]["device_id"] == PRESSURE
    assert second_alone["assignments"][0]["device_id"] == SECOND
    third_alone = copy.deepcopy(second_alone)
    third_alone["plan_id"] = "candidate-third-alone"
    third_alone["assignments"][0]["device_id"] = THIRD
    third_alone["device_id_by_mlx_rank"] = [THIRD]
    raw["candidates"] = [
        pressure_alone,
        second_alone,
        third_alone,
        _pipeline(two_stage, "candidate-three-10-9-9", (10, 9, 9), feasible=False),
        _pipeline(two_stage, "candidate-three-1-13-14", (1, 13, 14), feasible=True),
    ]
    raw["candidate_count"] = len(raw["candidates"])
    raw["memory_feasible_count"] = 3
    return raw


def test_the_reader_sees_three_macs_in_the_order_of_the_single_device_plans():
    candidates = read_candidate_set(three_mac_set())
    assert candidates.device_ids == tuple(ORDER)
    assert candidates.total_layers == 28
    assert len(candidates.candidates) == 5
    rooms = [device.room_for_layers_bytes for device in candidates.devices]
    assert rooms == [1073741824, 23622320128, 23622320128]


def test_equal_split_gives_the_extra_layer_to_the_first_mac_and_fails_on_it():
    choice = equal_split(read_candidate_set(three_mac_set()))
    assert choice.outcome == "INFEASIBLE"
    assert choice.plan is not None
    assert choice.plan.plan_id == "candidate-three-10-9-9"
    assert choice.plan.device_order == tuple(ORDER)
    assert choice.plan.logical_split == (10, 9, 9)
    assert choice.reason == (
        f"equal split [10, 9, 9] on {ORDER} fails the memory estimate on {PRESSURE}"
    )


def test_memory_proportional_follows_the_room_and_keeps_every_stage_non_empty():
    choice = memory_proportional(read_candidate_set(three_mac_set()))
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.plan_id == "candidate-three-1-13-14"
    assert choice.plan.device_order == tuple(ORDER)
    assert choice.plan.logical_split == (1, 13, 14)
    assert choice.reason == (
        f"memory-proportional split [1, 13, 14] on {ORDER} "
        "from room [1073741824, 23622320128, 23622320128] bytes passes the memory estimate"
    )


def test_best_single_device_breaks_a_tie_toward_the_earlier_mac():
    choice = best_single_device(read_candidate_set(three_mac_set()))
    assert choice.outcome == "FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (SECOND,)
    assert choice.plan.headroom_bytes == (SECOND_HEADROOM,)
    assert choice.reason == (
        f"{SECOND} alone has the most headroom of the single-device plans that fit: "
        f"{SECOND_HEADROOM} bytes"
    )


def test_compare_has_one_row_per_policy_on_three_macs():
    lines = compare(json.dumps(three_mac_set()).encode()).splitlines()
    rows = lines[lines.index(SEPARATOR) + 1 :]
    order_cell = ", ".join(ORDER)
    equal_tail = f"| [10, 9, 9] | {order_cell} | -1, 19934740480, 19934740480 |"
    proportional_tail = f"| [1, 13, 14] | {order_cell} | 937164800, 19934740480, 19934740480 |"
    assert len(rows) == 3
    assert rows[0].startswith("| best_single_device | FEASIBLE | ")
    assert rows[0].endswith(f"| [28] | {SECOND} | {SECOND_HEADROOM} |")
    assert rows[1].startswith("| equal_split | INFEASIBLE | candidate-three-10-9-9 | ")
    assert rows[1].endswith(equal_tail)
    assert rows[2].startswith("| memory_proportional | FEASIBLE | candidate-three-1-13-14 | ")
    assert rows[2].endswith(proportional_tail)
