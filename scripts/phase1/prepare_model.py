"""Resolve one immutable HF revision, then download exactly that snapshot."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Layer count alone is not enough: Qwen2.5-1.5B is also qwen2 with 28 layers.
QWEN25_7B = {
    "model_type": "qwen2",
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
}


def check_geometry(config: dict[str, Any]) -> None:
    wrong = {k: config.get(k) for k, v in QWEN25_7B.items() if config.get(k) != v}
    if wrong:
        raise ValueError(f"this Phase 1 profile requires Qwen2.5-7B geometry; got {wrong}")
    if config.get("quantization", {}).get("bits") != 4:
        raise ValueError("expected 4-bit converted model")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    p.add_argument("--revision", default="main")
    p.add_argument("--model-dir", default="models/qwen7b")
    p.add_argument("--manifest", default="experiments/phase1/model-lock.json")
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    out, model_dir = Path(args.manifest), Path(args.model_dir)
    if out.exists() or model_dir.exists():
        raise FileExistsError("choose new paths; never overwrite an existing model lock/snapshot")
    info = HfApi().model_info(args.repo, revision=args.revision)
    revision = info.sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("expected immutable 40-character model commit")
    snapshot_download(repo_id=args.repo, revision=revision, local_dir=str(model_dir))
    config_bytes = (model_dir / "config.json").read_bytes()
    config = json.loads(config_bytes)
    check_geometry(config)
    manifest = {
        "schema_version": "2.0.0",
        "model_id": args.repo,
        "model_revision": revision,
        "model_dir": str(model_dir),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": config,
        "source": "downloaded_metadata",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(manifest, f, indent=2)
    print(
        json.dumps(
            {
                "model_id": args.repo,
                "model_revision": revision,
                "manifest": str(out),
                "model_dir": str(model_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
