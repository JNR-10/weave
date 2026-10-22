import jsonschema
import pytest

from weave.contracts.schemas import build_schemas
from weave.profiler.simulated_fixture import make_state
from weave.scheduler.candidates import assignments, evaluate, generate, validate_state
from weave.telemetry.baseline import validate_workload


@pytest.mark.parametrize("total", [1, 2, 3, 28])
def test_complete_candidates_and_exact_cover(total):
    s = make_state()
    s["model"]["total_layers"] = total
    s["model"]["layer_weight_bytes"] = [2**27] * total
    result = generate(s)
    assert result["candidate_count"] == 2 + 2 * (total - 1)
    assert len({p["plan_id"] for p in result["candidates"]}) == result["candidate_count"]
    for p in result["candidates"]:
        covered = [
            i for a in p["assignments"] for i in range(a["start_layer"], a["end_layer_exclusive"])
        ]
        assert covered == list(range(total))
        assert p["mlx_split_by_rank"] == p["logical_split"][::-1]
        assert [a["mlx_rank"] for a in p["assignments"]] == list(
            range(len(p["assignments"]) - 1, -1, -1)
        )
        assert p["selected"] is False and p["executable"] is False
    jsonschema.validate(result, build_schemas()["candidate_set"])


def test_unequal_rank_mapping_and_single_device_candidates():
    assert assignments(["A", "B"], [10, 18]) == [
        {
            "device_id": "A",
            "logical_stage": 0,
            "mlx_rank": 1,
            "start_layer": 0,
            "end_layer_exclusive": 10,
            "layer_count": 10,
        },
        {
            "device_id": "B",
            "logical_stage": 1,
            "mlx_rank": 0,
            "start_layer": 10,
            "end_layer_exclusive": 28,
            "layer_count": 18,
        },
    ]
    plans = generate(make_state())["candidates"]
    assert sum(p["mode"] == "single_device" for p in plans) == 2
    assert {tuple(p["device_id_by_mlx_rank"]) for p in plans if p["mode"] == "pipeline"} == {
        ("SIMULATED-second-mac", "SIMULATED-m4-pressure"),
        ("SIMULATED-m4-pressure", "SIMULATED-second-mac"),
    }


def test_budget_boundary_and_rejection_explanation():
    s = make_state()
    order = [d["device_id"] for d in s["devices"]]
    p = evaluate(order, [10, 18], s)
    required = p["assignments"][0]["predicted_memory_bytes"]
    s["devices"][0]["safe_process_budget_bytes"] = required
    assert evaluate(order, [10, 18], s)["memory_feasible"]
    s["devices"][0]["safe_process_budget_bytes"] -= 1
    p = evaluate(order, [10, 18], s)
    assert not p["memory_feasible"]
    assert p["rejection_reasons"][0]["excess_bytes"] == 1


def test_memory_grows_with_context_and_single_node_state():
    s = make_state()
    s["devices"] = s["devices"][:1]
    before = generate(s)["candidates"][0]["assignments"][0]["predicted_memory_bytes"]
    s["request_envelope"]["prompt_tokens_upper_bound"] *= 2
    result = generate(s)
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["assignments"][0]["predicted_memory_bytes"] > before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(schema_version="1.0.0"),
        lambda s: s.update(simulation=False),
        lambda s: s["model"].update(total_layers=True),
        lambda s: s["model"].update(layer_weight_bytes=[1]),
        lambda s: s["devices"][0].update(workspace_reserve_bytes=0),
        lambda s: s["devices"][1].update(device_id=s["devices"][0]["device_id"]),
        lambda s: s["request_envelope"].update(concurrency=2),
    ],
)
def test_invalid_state_fails_closed(mutation):
    s = make_state()
    mutation(s)
    with pytest.raises(ValueError):
        validate_state(s)


def test_invalid_split_and_workload_rejected():
    s = make_state()
    ids = [d["device_id"] for d in s["devices"]]
    for split in ([0, 28], [-1, 29], [10, 17], [True, 27]):
        with pytest.raises(ValueError):
            evaluate(ids, split, s)
    w = {
        "schema_version": "2.0.0",
        "workload_id": "test",
        "prompt": "test",
        "prompt_max_tokens": 512,
        "max_new_tokens": 64,
        "concurrency": 1,
        "warmup_requests": 1,
        "measured_requests": 3,
        "temperature": 0,
        "seed": 0,
    }
    validate_workload(w)
    jsonschema.validate(w, build_schemas()["workload"])
    w["concurrency"] = 2
    with pytest.raises(ValueError):
        validate_workload(w)
