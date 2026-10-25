"""Geometry gate for the model lock; needs no download and no MLX."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_model", ROOT / "scripts/phase1/prepare_model.py"
)
prepare_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_model)


def locked_config():
    return json.loads((ROOT / "experiments/phase1/model-lock.json").read_text())["config"]


def test_locked_qwen7b_config_passes():
    prepare_model.check_geometry(locked_config())


@pytest.mark.parametrize(
    "changes",
    [
        # mlx-community/Qwen2.5-1.5B-Instruct-4bit: also qwen2, 28 layers, 4-bit.
        {"hidden_size": 1536, "num_attention_heads": 12, "num_key_value_heads": 2},
        # Llama-3.2-3B, the guide's rescue model, also has 28 layers.
        {"model_type": "llama", "hidden_size": 3072, "num_key_value_heads": 8},
        {"quantization": {"group_size": 64, "bits": 8}},
        {"quantization": None},
    ],
)
def test_other_geometry_or_quantization_rejected(changes):
    config = {**locked_config(), **changes}
    if config["quantization"] is None:
        del config["quantization"]
    with pytest.raises(ValueError):
        prepare_model.check_geometry(config)
