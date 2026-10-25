"""Seam 4 of spec 0001: validate_state reads the cluster state once and rejects bad input.

Every rejection is a ValueError whose message starts with the offending field's path, so the
person fixing the input knows exactly where to look.
"""

import re

import pytest

from weave.profiler.simulated_fixture import make_state
from weave.scheduler.candidates import validate_state


def test_valid_state_is_returned_unchanged():
    state = make_state()
    assert validate_state(state) is state


def test_non_object_state_is_rejected():
    with pytest.raises(ValueError, match=r"^state: expected a JSON object"):
        validate_state([])


TOP_LEVEL = [
    ("schema_version", lambda s: s.update(schema_version="1.0.0")),
    ("schema_version", lambda s: s.pop("schema_version")),
    ("snapshot_id", lambda s: s.update(snapshot_id="")),
    ("snapshot_id", lambda s: s.pop("snapshot_id")),
    ("simulation", lambda s: s.update(simulation="yes")),
    ("model", lambda s: s.pop("model")),
    ("model", lambda s: s.update(model=[])),
    ("request_envelope", lambda s: s.update(request_envelope=[])),
    ("devices", lambda s: s.update(devices={})),
    ("devices", lambda s: s.update(devices=[])),
]


@pytest.mark.parametrize("path, mutate", TOP_LEVEL)
def test_bad_top_level_field_is_named(path, mutate):
    state = make_state()
    mutate(state)
    with pytest.raises(ValueError, match=rf"^{re.escape(path)}: "):
        validate_state(state)


MODEL = [
    ("model.model_id", lambda m: m.update(model_id="")),
    ("model.model_revision", lambda m: m.pop("model_revision")),
    ("model.profile_id", lambda m: m.update(profile_id=5)),
    ("model.total_layers", lambda m: m.update(total_layers=True)),
    ("model.total_layers", lambda m: m.update(total_layers=0)),
    ("model.layer_weight_bytes", lambda m: m.update(layer_weight_bytes="128MiB")),
    ("model.layer_weight_bytes", lambda m: m.update(layer_weight_bytes=[2**27] * 27)),
    ("model.layer_weight_bytes[3]", lambda m: m["layer_weight_bytes"].__setitem__(3, 0)),
    ("model.layer_weight_bytes[0]", lambda m: m["layer_weight_bytes"].__setitem__(0, True)),
    ("model.replicated_weight_bytes", lambda m: m.update(replicated_weight_bytes=0)),
    ("model.kv_heads", lambda m: m.update(kv_heads=-1)),
    ("model.head_dim", lambda m: m.update(head_dim=1.5)),
    ("model.kv_bytes_per_element", lambda m: m.pop("kv_bytes_per_element")),
    ("model.source", lambda m: m.pop("source")),
]


@pytest.mark.parametrize("path, mutate", MODEL)
def test_bad_model_field_is_named(path, mutate):
    state = make_state()
    mutate(state["model"])
    with pytest.raises(ValueError, match=rf"^{re.escape(path)}: "):
        validate_state(state)


REQUEST = [
    ("request_envelope.prompt_tokens_upper_bound", lambda r: r.update(prompt_tokens_upper_bound=0)),
    ("request_envelope.prompt_tokens_upper_bound", lambda r: r.pop("prompt_tokens_upper_bound")),
    ("request_envelope.max_new_tokens", lambda r: r.update(max_new_tokens="64")),
    ("request_envelope.concurrency", lambda r: r.update(concurrency=0)),
]


@pytest.mark.parametrize("path, mutate", REQUEST)
def test_bad_request_field_is_named(path, mutate):
    state = make_state()
    mutate(state["request_envelope"])
    with pytest.raises(ValueError, match=rf"^{re.escape(path)}: "):
        validate_state(state)


DEVICES = [
    ("devices[0]", lambda d: d.__setitem__(0, "SIMULATED-m4-pressure")),
    ("devices[0].device_id", lambda d: d[0].update(device_id="")),
    ("devices[1].device_id", lambda d: d[1].pop("device_id")),
    ("devices[1].device_id", lambda d: d[1].update(device_id=d[0]["device_id"])),
    ("devices[0].source", lambda d: d[0].pop("source")),
    ("devices[1].budget_basis", lambda d: d[1].pop("budget_basis")),
    ("devices[0].safe_process_budget_bytes", lambda d: d[0].update(safe_process_budget_bytes=0)),
    ("devices[1].workspace_reserve_bytes", lambda d: d[1].update(workspace_reserve_bytes=0)),
    ("devices[1].safety_margin_bytes", lambda d: d[1].update(safety_margin_bytes=True)),
]


@pytest.mark.parametrize("path, mutate", DEVICES)
def test_bad_device_field_is_named(path, mutate):
    state = make_state()
    mutate(state["devices"])
    with pytest.raises(ValueError, match=rf"^{re.escape(path)}: "):
        validate_state(state)
