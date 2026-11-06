"""Step 12 of spec 0002: a model with fewer layers than Macs.

With fewer layers than Macs there is no split that gives every Mac a layer, so the generator emits
only the single-device plans. Equal split and memory-proportional then report NO_FEASIBLE_PLAN with
a reason instead of raising, and --compare keeps the best-single-device row.

The one-layer set is hand-edited from the golden candidate file, which tests/test_phase1_golden.py
holds to the generator byte for byte. Its byte values are SYNTHETIC, chosen by hand (headrooms 1000
and 2000 bytes); they are not generator output. The tests assert which plan each policy picks and
the outcome and reason, never bytes as if they were real.
"""

import json
from pathlib import Path

import jsonschema

from weave.scheduler import policies
from weave.scheduler.policies import (
    best_single_device,
    equal_split,
    memory_proportional,
    read_candidate_set,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "experiments/policies/schemas/v2/placement_decision.schema.json"

PRESSURE = "SIMULATED-m4-pressure"
SECOND = "SIMULATED-second-mac"

# Synthetic headrooms, by hand: the second Mac has more, so it is the best single device.
HEADROOM = {PRESSURE: 1000, SECOND: 2000}


def one_layer_set(raw):
    """The golden set (the fixture's fresh copy) cut down to the two single-device plans of a
    one-layer model.

    The reader does not check start_layer, mlx_split_by_rank or the byte components against each
    other, so only the fields below change. end_layer_exclusive is set to 1 to keep the layer
    range honest.
    """
    raw["candidates"] = raw["candidates"][:2]
    for candidate in raw["candidates"]:
        assignment = candidate["assignments"][0]
        candidate["logical_split"] = [1]
        candidate["mlx_split_by_rank"] = [1]
        assignment["layer_count"] = 1
        assignment["end_layer_exclusive"] = 1
        candidate["memory_feasible"] = True
        candidate["rejection_reasons"] = []
        assignment["headroom_bytes"] = HEADROOM[assignment["device_id"]]
    if "candidate_count" in raw:
        raw["candidate_count"] = 2
    if "memory_feasible_count" in raw:
        raw["memory_feasible_count"] = 2
    return raw


def test_the_one_layer_set_is_accepted_by_the_reader(golden_set):
    candidates = read_candidate_set(one_layer_set(golden_set))
    assert candidates.total_layers == 1
    assert candidates.device_ids == (PRESSURE, SECOND)


def test_equal_split_reports_no_feasible_plan_when_layers_are_fewer_than_macs(golden_set):
    choice = equal_split(read_candidate_set(one_layer_set(golden_set)))
    assert choice.policy == "equal_split"
    assert choice.outcome == "NO_FEASIBLE_PLAN"
    assert choice.plan is None
    assert choice.reason == "no equal split: 1 layers cannot give each of 2 Macs at least one"


def test_memory_proportional_reports_no_feasible_plan_when_layers_are_fewer_than_macs(golden_set):
    choice = memory_proportional(read_candidate_set(one_layer_set(golden_set)))
    assert choice.policy == "memory_proportional"
    assert choice.outcome == "NO_FEASIBLE_PLAN"
    assert choice.plan is None
    assert choice.reason == (
        "no memory-proportional split: 1 layers cannot give each of 2 Macs at least one"
    )


def test_best_single_device_still_picks_a_mac_when_layers_are_fewer_than_macs(golden_set):
    choice = best_single_device(read_candidate_set(one_layer_set(golden_set)))
    assert choice.outcome == "MEMORY_FEASIBLE"
    assert choice.plan is not None
    assert choice.plan.device_order == (SECOND,)


def test_compare_keeps_the_best_single_device_row_when_layers_are_fewer_than_macs(golden_set):
    text = policies.compare(json.dumps(one_layer_set(golden_set)).encode())
    rows = text.splitlines()[-3:]
    assert rows[0].startswith("| best_single_device | MEMORY_FEASIBLE | ")
    assert rows[0].endswith("| [1] | SIMULATED-second-mac | 2000 |")
    assert rows[1] == "| equal_split | NO_FEASIBLE_PLAN | none | none | none | none |"
    assert rows[2] == "| memory_proportional | NO_FEASIBLE_PLAN | none | none | none | none |"


def test_the_decision_record_for_no_split_has_no_chosen_plan_and_validates(golden_set):
    record = policies.decide(json.dumps(one_layer_set(golden_set)).encode(), "equal_split")
    assert record["chosen_plan_id"] is None
    assert record["outcome"] == "NO_FEASIBLE_PLAN"
    assert record["rejection_reasons"] == []
    jsonschema.validate(record, json.loads(SCHEMA.read_text()))
