"""Portable checks for the two-Mac launch view (spec 0005); no MLX, no real hardware."""

import json
from itertools import product

import pytest

from weave.runtime.launch import (
    RECORD,
    RECORD_VERSION,
    check_bring_up,
    check_rank,
    first_divergence,
    hostfile,
    installed_range,
    launch_plan,
    local_weight_files,
    main,
    read_record,
)


def test_guide_example_10_18_launches_reversed():
    plan = launch_plan(["A", "B"], [10, 18])
    assert plan.mlx_split_by_rank == (18, 10)
    assert plan.device_id_by_mlx_rank == ("B", "A")
    a, b = plan.stages
    assert (a.device_id, a.logical_stage, a.mlx_rank) == ("A", 0, 1)
    assert (a.start_layer, a.end_layer_exclusive) == (0, 10)
    assert (b.device_id, b.logical_stage, b.mlx_rank) == ("B", 1, 0)
    assert (b.start_layer, b.end_layer_exclusive) == (10, 28)


def test_equal_split_is_its_own_reverse_but_hosts_are_not():
    plan = launch_plan(["m4", "second"], [14, 14])
    assert plan.mlx_split_by_rank == (14, 14)
    assert plan.device_id_by_mlx_rank == ("second", "m4")
    assert plan.stage_for_rank(0).device_id == "second"


def test_single_device_is_rank_zero_with_every_layer():
    plan = launch_plan(["m4"], [28])
    (stage,) = plan.stages
    assert (stage.mlx_rank, stage.start_layer, stage.end_layer_exclusive) == (0, 0, 28)


@pytest.mark.parametrize("stages", [2, 3])
def test_installed_formula_matches_assignments_for_every_split(stages):
    devices = [f"d{i}" for i in range(stages)]
    for split in product(range(1, 28), repeat=stages):
        if sum(split) != 28:
            continue
        plan = launch_plan(devices, split)
        for stage in plan.stages:
            assert installed_range(plan.mlx_split_by_rank, stage.mlx_rank) == (
                stage.start_layer,
                stage.end_layer_exclusive,
            )


def test_installed_range_refuses_a_rank_outside_the_group():
    with pytest.raises(ValueError, match="rank: expected 0 to 1"):
        installed_range([18, 10], 2)


def test_check_rank_accepts_the_planned_rank():
    plan = launch_plan(["A", "B"], [10, 18])
    stage = check_rank(plan, rank=1, size=2, device_id="A", start_idx=0, end_idx=10)
    assert stage.logical_stage == 0


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(rank=0, size=3, device_id="B", start_idx=10, end_idx=28), "3 ranks"),
        (dict(rank=0, size=2, device_id="A", start_idx=10, end_idx=28), "hostfile order"),
        (dict(rank=0, size=2, device_id="B", start_idx=0, end_idx=18), r"\[10, 28\)"),
        (dict(rank=2, size=2, device_id="B", start_idx=0, end_idx=10), "rank: expected"),
    ],
)
def test_check_rank_refuses_a_mismatch(kwargs, message):
    plan = launch_plan(["A", "B"], [10, 18])
    with pytest.raises((RuntimeError, ValueError), match=message):
        check_rank(plan, **kwargs)


def _weight_map(layers: int) -> dict[str, str]:
    """Two-file checkpoint: layers 0-13 in file 1, 14 onwards and the head in file 2."""
    weights = {"model.embed_tokens.weight": "model-00001.safetensors"}
    for layer in range(layers):
        file = "model-00001.safetensors" if layer < 14 else "model-00002.safetensors"
        weights[f"model.layers.{layer}.self_attn.q_proj.weight"] = file
        weights[f"model.layers.{layer}.mlp.down_proj.scales"] = file
    weights["model.norm.weight"] = "model-00002.safetensors"
    weights["lm_head.weight"] = "model-00002.safetensors"
    return weights


def test_weight_files_include_shared_tensors_and_only_local_layers():
    weights = _weight_map(28)
    # Rank 1 of logical [10, 18] runs layers 0-9, all in file 1, but still needs the final norm
    # and head from file 2: every stage keeps the non-layer tensors.
    assert local_weight_files(weights, 0, 10, 28) == [
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]
    only_one = {k: v for k, v in weights.items() if "layers" in k or "embed" in k}
    assert local_weight_files(only_one, 14, 28, 28) == [
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]
    assert local_weight_files(
        {k: v for k, v in only_one.items() if "embed" not in k}, 14, 28, 28
    ) == ["model-00002.safetensors"]


def test_weight_files_ignore_layers_past_the_model():
    weights = _weight_map(28)
    weights["model.layers.28.mtp.weight"] = "model-00003.safetensors"
    assert "model-00003.safetensors" not in local_weight_files(weights, 14, 28, 28)


def test_weight_files_refuse_a_checkpoint_missing_a_layer():
    weights = {k: v for k, v in _weight_map(28).items() if ".layers.5." not in k}
    with pytest.raises(ValueError, match=r"no weights for layers \[5\]"):
        local_weight_files(weights, 0, 10, 28)


def test_weight_files_refuse_a_range_outside_the_model():
    with pytest.raises(ValueError, match="not inside"):
        local_weight_files(_weight_map(28), 10, 29, 28)


def test_hostfile_orders_entries_by_rank_and_allows_extra_devices():
    plan = launch_plan(["A", "B"], [10, 18])
    hosts = {"A": {"ssh": "host-a"}, "B": {"ssh": "host-b"}, "C": {"ssh": "host-c"}}
    assert hostfile(plan, hosts) == [{"ssh": "host-b"}, {"ssh": "host-a"}]


def test_hostfile_refuses_a_missing_device():
    with pytest.raises(ValueError, match=r"no entry for \['B'\]"):
        hostfile(launch_plan(["A", "B"], [10, 18]), {"A": {}})


@pytest.mark.parametrize(
    "left,right,expected",
    [([1, 2, 3], [1, 2, 3], None), ([1, 2, 3], [1, 9, 3], 1), ([1, 2], [1, 2, 3], 2)],
)
def test_first_divergence(left, right, expected):
    assert first_divergence(left, right) == expected


def _record(rank: int, **changes) -> dict:
    """A simulated rank record for logical [10, 18] on [A, B]; never measured evidence."""
    plan = launch_plan(["A", "B"], [10, 18])
    stage = plan.stage_for_rank(rank)
    record = {
        "record": RECORD,
        "record_version": RECORD_VERSION,
        "status": "success",
        "run_id": "SIMULATED-bring-up-test",
        "simulation": True,
        "rank": rank,
        "size": 2,
        "device_id": stage.device_id,
        "model_revision": "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
        "start_idx": stage.start_layer,
        "end_idx": stage.end_layer_exclusive,
        "plan": plan.to_json(),
        "prompt_tokens": [5, 6, 7],
        "tokens": [11, 12, 13],
    }
    record.update(changes)
    return read_record(record, f"rank-{rank}.json")


def _reference(**changes) -> dict:
    plan = launch_plan(["A"], [28])
    record = {
        "record": RECORD,
        "record_version": RECORD_VERSION,
        "status": "success",
        "run_id": "SIMULATED-single-test",
        "simulation": True,
        "rank": 0,
        "size": 1,
        "device_id": "A",
        "model_revision": "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
        "start_idx": 0,
        "end_idx": 28,
        "plan": plan.to_json(),
        "prompt_tokens": [5, 6, 7],
        "tokens": [11, 12, 13],
    }
    record.update(changes)
    return read_record(record, "reference.json")


def test_bring_up_passes_when_ranks_agree_and_match_the_reference():
    result = check_bring_up([_record(0), _record(1)], _reference())
    assert result["passed"] and result["ranks_agree"] and result["matches_reference"]
    assert result["layers_loaded_once"]
    assert result["plan"]["mlx_split_by_rank"] == [18, 10]


def test_bring_up_reports_a_token_divergence_instead_of_hiding_it():
    result = check_bring_up([_record(0), _record(1, tokens=[11, 99, 13])], _reference())
    assert not result["passed"]
    assert result["rank_divergence"] == {1: 1}
    off = check_bring_up([_record(0), _record(1)], _reference(tokens=[11, 12, 14]))
    assert not off["passed"] and off["reference_divergence"] == 2


@pytest.mark.parametrize(
    "records,message",
    [
        (lambda: [_record(0)], "one record per rank"),
        (lambda: [_record(0), _record(0)], "one record per rank"),
        (lambda: [_record(0), _record(1, device_id="B")], "hostfile order"),
        (lambda: [_record(0), _record(1, start_idx=0, end_idx=18)], r"\[0, 10\)"),
        (lambda: [_record(0), _record(1, run_id="SIMULATED-other")], "run_id"),
        (lambda: [_record(0), _record(1, prompt_tokens=[5])], "different prompt"),
    ],
)
def test_bring_up_refuses_inconsistent_ranks(records, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        check_bring_up(records())


def test_reference_must_be_a_comparable_single_device_run():
    with pytest.raises(ValueError, match="single-device"):
        check_bring_up([_record(0), _record(1)], _record(0))
    with pytest.raises(ValueError, match="model revision"):
        check_bring_up([_record(0), _record(1)], _reference(model_revision="other"))


def test_record_refuses_simulated_and_measured_labels_that_disagree():
    with pytest.raises(ValueError, match="SIMULATED- run ID disagree"):
        _record(0, simulation=False)


def test_record_refuses_a_failed_rank():
    with pytest.raises(ValueError, match="did not finish"):
        _record(0, status="error")


def test_hostfile_command_writes_rank_order_and_never_overwrites(tmp_path, capsys):
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps({"A": {"ssh": "host-a"}, "B": {"ssh": "host-b"}}))
    out = tmp_path / "hostfile.json"
    argv = ["hostfile", "--device-order", "A", "B", "--split", "10", "18"]
    argv += ["--hosts", str(hosts), "--output", str(out)]
    assert main(argv) == 0
    assert json.loads(out.read_text()) == [{"ssh": "host-b"}, {"ssh": "host-a"}]
    with pytest.raises(FileExistsError):
        main(argv)


def test_check_run_command_exit_code_follows_the_result(tmp_path, capsys):
    paths = []
    for rank, tokens in ((0, [11, 12, 13]), (1, [11, 12, 99])):
        plan = launch_plan(["A", "B"], [10, 18])
        stage = plan.stage_for_rank(rank)
        path = tmp_path / f"rank-{rank}.json"
        path.write_text(
            json.dumps(
                {
                    "record": RECORD,
                    "record_version": RECORD_VERSION,
                    "status": "success",
                    "run_id": "SIMULATED-cli",
                    "simulation": True,
                    "rank": rank,
                    "size": 2,
                    "device_id": stage.device_id,
                    "model_revision": "rev",
                    "start_idx": stage.start_layer,
                    "end_idx": stage.end_layer_exclusive,
                    "plan": plan.to_json(),
                    "prompt_tokens": [1],
                    "tokens": tokens,
                }
            )
        )
        paths.append(str(path))
    assert main(["check-run", *paths]) == 1
    assert json.loads(capsys.readouterr().out)["rank_divergence"] == {"1": 2}
