"""Control flow of the split-aware loader against fake MLX modules (spec 0005).

These tests prove the loader calls `pipeline` once with the reversed split, checks the rank's
identity before reading weights and its layers after, and keeps the single-device path ordinary.
They prove nothing about real MLX: the fake `pipeline` copies the installed PipelineMixin logic
recorded in Mohit's M4 probe (mlx-lm 0.32.2), and only a two-Mac run can verify the real one.
"""

import json
import sys
import types

import pytest

from weave.runtime.launch import launch_plan
from weave.runtime.mlx_pipeline import generate, load_stage

LAYERS = 28


class FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class FakeInner:
    """Same arithmetic as the installed PipelineMixin.pipeline (mlx-lm 0.32.2)."""

    def __init__(self, calls: list):
        self.layers = list(range(LAYERS))
        self.start_idx, self.end_idx = 0, None
        self.calls = calls

    def pipeline(self, group, split=None):
        self.calls.append(("pipeline", group.rank(), list(split)))
        rank = group.rank()
        self.start_idx = sum(split[rank + 1 :])
        self.end_idx = self.start_idx + split[rank]
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx


class FakeModel:
    def __init__(self, calls: list):
        self.model = FakeInner(calls)

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    def parameters(self):
        return {}


@pytest.fixture
def fake_mlx(monkeypatch):
    """Install fake mlx.core / mlx_lm modules for one test; returns the call log."""
    calls: list = []
    core = types.ModuleType("mlx.core")
    core.synchronize = lambda: None
    core.reset_peak_memory = lambda: None
    core.get_peak_memory = lambda: 123
    core.eval = lambda *a: calls.append(("eval",))
    core.array = lambda x: x
    core.cpu = "cpu"
    core.random = types.SimpleNamespace(seed=lambda s: calls.append(("seed", s)))
    core.distributed = types.SimpleNamespace(all_sum=lambda x, stream=None: x)
    mlx = types.ModuleType("mlx")
    mlx.core = core
    mlx_lm = types.ModuleType("mlx_lm")

    def load(path, lazy=False):
        calls.append(("load", path, lazy))
        return FakeModel(calls), FakeTokenizer()

    mlx_lm.load = load
    mlx_lm.stream_generate = None  # set per test
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda temp: ("greedy", temp)
    for name, module in (
        ("mlx", mlx),
        ("mlx.core", core),
        ("mlx_lm", mlx_lm),
        ("mlx_lm.sample_utils", sample_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return types.SimpleNamespace(calls=calls, mlx_lm=mlx_lm)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [5, 6, 7]


@pytest.fixture
def snapshot(tmp_path):
    """A model directory with a two-file index and both files present."""
    weight_map = {"model.embed_tokens.weight": "a.safetensors", "lm_head.weight": "b.safetensors"}
    for layer in range(LAYERS):
        weight_map[f"model.layers.{layer}.mlp.weight"] = (
            "a.safetensors" if layer < 14 else "b.safetensors"
        )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "a.safetensors").write_bytes(b"")
    (tmp_path / "b.safetensors").write_bytes(b"")
    return tmp_path


def test_unequal_split_pipelines_once_with_the_reversed_split(fake_mlx, snapshot):
    plan = launch_plan(["A", "B"], [10, 18])
    stage = load_stage(snapshot, plan, FakeGroup(rank=1, size=2), "A")
    assert (stage.rank, stage.start_idx, stage.end_idx) == (1, 0, 10)
    pipeline_calls = [c for c in fake_mlx.calls if c[0] == "pipeline"]
    assert pipeline_calls == [("pipeline", 1, [18, 10])]
    assert ("load", str(snapshot), True) in fake_mlx.calls
    assert stage.weight_files == ("a.safetensors", "b.safetensors")


def test_wrong_host_is_refused_before_any_weight_is_read(fake_mlx, snapshot):
    plan = launch_plan(["A", "B"], [10, 18])
    with pytest.raises(RuntimeError, match="hostfile order"):
        load_stage(snapshot, plan, FakeGroup(rank=0, size=2), "A")
    assert not [c for c in fake_mlx.calls if c[0] == "load"]


def test_group_size_must_match_the_plan(fake_mlx, snapshot):
    plan = launch_plan(["A", "B"], [10, 18])
    with pytest.raises(RuntimeError, match="3 ranks"):
        load_stage(snapshot, plan, FakeGroup(rank=0, size=3), "B")


def test_a_pipeline_that_lands_elsewhere_is_refused(fake_mlx, snapshot, monkeypatch):
    def wrong(self, group, split=None):
        self.start_idx, self.end_idx = 0, 18  # as if the split were not reversed

    monkeypatch.setattr(FakeInner, "pipeline", wrong)
    with pytest.raises(RuntimeError, match=r"plan expects \[10, 28\)"):
        load_stage(snapshot, launch_plan(["A", "B"], [10, 18]), FakeGroup(0, 2), "B")


def test_single_device_uses_the_ordinary_loader_without_a_group(fake_mlx, snapshot):
    stage = load_stage(snapshot, launch_plan(["A"], [LAYERS]), None, "A")
    assert (stage.rank, stage.size, stage.start_idx, stage.end_idx) == (0, 1, 0, LAYERS)
    assert ("load", str(snapshot), False) in fake_mlx.calls
    assert not [c for c in fake_mlx.calls if c[0] == "pipeline"]


def test_single_device_runs_only_on_its_planned_mac(fake_mlx, snapshot):
    with pytest.raises(RuntimeError, match="hostfile order"):
        load_stage(snapshot, launch_plan(["A"], [LAYERS]), None, "B")


def test_missing_checkpoint_file_is_refused(fake_mlx, snapshot):
    (snapshot / "b.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="b.safetensors"):
        load_stage(snapshot, launch_plan(["A", "B"], [10, 18]), FakeGroup(1, 2), "A")


def _responses(*tokens):
    for count, token in enumerate(tokens, start=1):
        yield types.SimpleNamespace(generation_tokens=count, token=token, text=f"<{token}>")


def test_generate_collects_one_token_per_counter_step(fake_mlx, snapshot):
    fake_mlx.mlx_lm.stream_generate = lambda model, tok, prompt, max_tokens, sampler: _responses(
        11, 12, 13
    )
    stage = load_stage(snapshot, launch_plan(["A"], [LAYERS]), None, "A")
    result = generate(stage, "hi", max_new_tokens=3, seed=0)
    assert result["prompt_tokens"] == [5, 6, 7]
    assert result["tokens"] == [11, 12, 13]
    assert result["text"] == "<11><12><13>"
    assert ("seed", 0) in fake_mlx.calls


def test_generate_refuses_a_counter_that_skips(fake_mlx, snapshot):
    def skipping(*args, **kwargs):
        yield types.SimpleNamespace(generation_tokens=2, token=1, text="x")

    fake_mlx.mlx_lm.stream_generate = skipping
    stage = load_stage(snapshot, launch_plan(["A"], [LAYERS]), None, "A")
    with pytest.raises(ValueError, match="token counter"):
        generate(stage, "hi", max_new_tokens=3, seed=0)
