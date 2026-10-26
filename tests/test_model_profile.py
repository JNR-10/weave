"""Parameter-byte profile logic on fake tensor sizes; the MLX flattening is a thin wrapper."""

import pytest

from weave.profiler.model_profile import profile_from_tensor_bytes, profile_id

# Qwen2 names as MLX-LM flattens them: quantized tensors carry scales and biases.
TENSORS = [
    ("model.embed_tokens.weight", 100),
    ("model.embed_tokens.scales", 10),
    ("model.layers.0.self_attn.q_proj.weight", 40),
    ("model.layers.0.mlp.down_proj.weight", 60),
    ("model.layers.1.self_attn.q_proj.weight", 40),
    ("model.layers.1.mlp.down_proj.weight", 60),
    ("model.norm.weight", 2),
    ("lm_head.weight", 100),
]


def make(tensors=TENSORS, total_layers=2):
    return profile_from_tensor_bytes(
        tensors,
        model_id="m",
        model_revision="a" * 40,
        config_sha256="e" * 64,
        total_layers=total_layers,
    )


def test_layer_and_replicated_tallies():
    p = make()
    assert p["layer_weight_bytes"] == [100, 100]
    assert p["replicated_tensor_bytes"] == {
        "lm_head.weight": 100,
        "model.embed_tokens.scales": 10,
        "model.embed_tokens.weight": 100,
        "model.norm.weight": 2,
    }
    assert p["replicated_weight_bytes"] == 212


def test_profile_id_is_content_addressed():
    assert make()["profile_id"] == make()["profile_id"] == profile_id(make())
    changed = make([*TENSORS[:-1], ("lm_head.weight", 101)])
    assert changed["profile_id"] != make()["profile_id"]


def test_simulated_profile_never_shares_a_measured_id():
    simulated = {**make(), "source": "simulated"}
    assert profile_id(simulated) != make()["profile_id"]


@pytest.mark.parametrize(
    "tensors, total_layers",
    [
        (TENSORS, 3),  # locked geometry says 3 layers, the weights cover 2
        ([*TENSORS, ("model.layers.2.mlp.down_proj.weight", 1)], 2),  # index beyond the geometry
    ],
)
def test_geometry_mismatch_rejected(tensors, total_layers):
    with pytest.raises(ValueError):
        make(tensors, total_layers)
