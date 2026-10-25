"""Seam 1 of spec 0001: generate (and evaluate, and the candidate command) build the candidates.

The golden test pins the committed fixture's whole output; these tests cover what one fixture
cannot: every layer count, degenerate inputs, the budget boundary and the safety flags.
"""

import json
import sys
from pathlib import Path

import jsonschema
import pytest

from weave.profiler.simulated_fixture import make_state
from weave.scheduler import candidates
from weave.scheduler.candidates import evaluate, generate

SCHEMA = (
    Path(__file__).resolve().parents[1] / "experiments/phase1/schemas/v2/candidate_set.schema.json"
)
PRESSURE, SECOND = "SIMULATED-m4-pressure", "SIMULATED-second-mac"
# Guide section 10 worked by hand for logical [10, 18] with the pressure Mac first: 10 x 128 MiB
# + 1 GiB replicated + 23,592,960 KV bytes + 2 x 512 MiB reserves.
PRESSURE_STAGE_10_LAYERS = 3_513_253_888


def state_with_layers(total: int) -> dict:
    state = make_state()
    state["model"].update(total_layers=total, layer_weight_bytes=[2**27] * total)
    return state


def test_count_is_both_single_mac_plans_plus_every_cut_in_both_orders():
    # Guide section 11: two single-Mac plans plus (L - 1) cuts in each of two device orders.
    for total in range(1, 29):
        assert generate(state_with_layers(total))["candidate_count"] == 2 + 2 * (total - 1)


def test_every_candidate_covers_each_layer_exactly_once():
    for total in range(1, 29):
        for plan in generate(state_with_layers(total))["candidates"]:
            covered = [
                layer
                for a in plan["assignments"]
                for layer in range(a["start_layer"], a["end_layer_exclusive"])
            ]
            assert covered == list(range(total)), plan["logical_split"]


def test_plan_ids_are_unique_within_a_set():
    plans = generate(make_state())["candidates"]
    assert len({p["plan_id"] for p in plans}) == len(plans)


def test_one_device_gives_exactly_one_single_mac_candidate():
    state = make_state()
    state["devices"] = state["devices"][:1]
    plans = generate(state)["candidates"]
    assert [(p["mode"], p["device_id_by_mlx_rank"]) for p in plans] == [
        ("single_device", [PRESSURE])
    ]


def test_one_layer_model_gives_only_single_mac_candidates():
    plans = generate(state_with_layers(1))["candidates"]
    assert [p["mode"] for p in plans] == ["single_device", "single_device"]


def test_every_candidate_is_simulated_unselected_and_unverified():
    result = generate(make_state())
    flags = {
        (p["simulation"], p["selected"], p["policy"], p["executable"], p["runtime_verified"])
        for p in result["candidates"]
    }
    assert (result["selected_plan_id"], flags) == (
        None,
        {(True, False, "candidate_only", False, False)},
    )


def test_logical_10_18_launches_as_18_10_with_ranks_reversed():
    plan = evaluate([PRESSURE, SECOND], [10, 18], make_state())
    assert (plan["mlx_split_by_rank"], plan["device_id_by_mlx_rank"]) == (
        [18, 10],
        [SECOND, PRESSURE],
    )


def test_budget_equal_to_the_prediction_fits():
    state = make_state()
    state["devices"][0]["safe_process_budget_bytes"] = PRESSURE_STAGE_10_LAYERS
    assert evaluate([PRESSURE, SECOND], [10, 18], state)["memory_feasible"] is True


def test_budget_one_byte_short_is_rejected_with_an_excess_of_one():
    state = make_state()
    state["devices"][0]["safe_process_budget_bytes"] = PRESSURE_STAGE_10_LAYERS - 1
    assert evaluate([PRESSURE, SECOND], [10, 18], state)["rejection_reasons"] == [
        {"code": "MEMORY_BUDGET_EXCEEDED", "device_id": PRESSURE, "excess_bytes": 1}
    ]


def test_plan_ids_are_stable_for_the_same_input():
    first = [p["plan_id"] for p in generate(make_state())["candidates"]]
    assert [p["plan_id"] for p in generate(make_state())["candidates"]] == first


def test_plan_id_ignores_budgets_as_the_guide_defines():
    # Guide section 11 identity is snapshot, profile, order and counts only, so a changed input
    # must come with a new snapshot ID. This guards the rule against a silent "fix".
    state = make_state()
    before = evaluate([PRESSURE, SECOND], [10, 18], state)["plan_id"]
    state["devices"][0]["safe_process_budget_bytes"] *= 2
    assert evaluate([PRESSURE, SECOND], [10, 18], state)["plan_id"] == before


def test_candidate_set_matches_the_committed_schema():
    jsonschema.validate(generate(make_state()), json.loads(SCHEMA.read_text()))


def test_candidate_command_refuses_to_overwrite(tmp_path, monkeypatch):
    state_file, out = tmp_path / "state.json", tmp_path / "candidate_plans.json"
    state_file.write_text(json.dumps(make_state()))
    out.write_text("earlier output")
    monkeypatch.setattr(
        sys, "argv", ["candidates", "--state", str(state_file), "--output", str(out)]
    )
    with pytest.raises(FileExistsError):
        candidates.main()


def test_unknown_device_is_named():
    with pytest.raises(ValueError, match=r"^device_order\[1\]: 'SIMULATED-unknown' is not in"):
        evaluate([PRESSURE, "SIMULATED-unknown"], [10, 18], make_state())


def test_split_that_misses_layers_is_named():
    with pytest.raises(ValueError, match=r"^logical_counts: expected 28 layers in total, got 27"):
        evaluate([PRESSURE, SECOND], [10, 17], make_state())


def test_plan_id_ignores_model_revision_as_the_guide_defines():
    # The other half of the guide section 11 identity rule (budgets are tested above).
    state = make_state()
    before = evaluate([PRESSURE, SECOND], [10, 18], state)["plan_id"]
    state["model"]["model_revision"] = "SIMULATED-another-revision"
    assert evaluate([PRESSURE, SECOND], [10, 18], state)["plan_id"] == before


def test_raising_the_device_limit_plans_three_macs(monkeypatch):
    # Phase 2 lifts the limit in one place; the planning core already handles any number of
    # Macs. 3 alone + 6 ordered pairs x 27 cuts + 6 orders of all three x C(27, 2) = 2,271.
    monkeypatch.setattr(candidates, "PHASE1_MAX_DEVICES", 3)
    state = make_state()
    state["devices"].append({**state["devices"][1], "device_id": "SIMULATED-third-mac"})
    assert generate(state)["candidate_count"] == 2271
