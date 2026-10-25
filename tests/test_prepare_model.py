"""Geometry and snapshot gates for the model lock; need no download, hub access, or MLX."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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


def test_git_blob_sha1_matches_git(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b'{"model_type": "qwen2"}\n')
    git = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
    assert prepare_model.git_blob_sha1(path) == git


def test_expected_files_uses_lfs_sha256_or_blob_id():
    siblings = [
        SimpleNamespace(rfilename="config.json", size=3, blob_id="b" * 40, lfs=None),
        SimpleNamespace(
            rfilename="model.safetensors",
            size=130,
            blob_id="c" * 40,
            lfs=SimpleNamespace(size=9, sha256="d" * 64),
        ),
    ]
    assert prepare_model.expected_files(siblings) == {
        "config.json": {"size": 3, "git_blob_sha1": "b" * 40},
        "model.safetensors": {"size": 9, "sha256": "d" * 64},
    }


def snapshot(root):
    (root / "config.json").write_bytes(b"{}")
    (root / "model.safetensors").write_bytes(b"weights")
    return {
        "config.json": {
            "size": 2,
            "git_blob_sha1": prepare_model.git_blob_sha1(root / "config.json"),
        },
        "model.safetensors": {"size": 7, "sha256": hashlib.sha256(b"weights").hexdigest()},
    }


def test_verified_snapshot_passes(tmp_path):
    assert prepare_model.verify_snapshot(tmp_path, snapshot(tmp_path))["ok"]


@pytest.mark.parametrize(
    "damage, failed",
    [
        (lambda root: (root / "model.safetensors").write_bytes(b"weightz"), "model.safetensors"),
        (lambda root: (root / "config.json").unlink(), "config.json"),
        # mlx_lm.load would also read this file, so it must not pass.
        (
            lambda root: (root / "model-00002.safetensors").write_bytes(b"x"),
            "model-00002.safetensors",
        ),
    ],
)
def test_damaged_snapshot_fails(tmp_path, damage, failed):
    expected = snapshot(tmp_path)
    damage(tmp_path)
    report = prepare_model.verify_snapshot(tmp_path, expected)
    assert not report["ok"]
    assert prepare_model.failures(report) == [failed]
