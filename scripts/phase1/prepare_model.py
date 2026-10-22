"""Resolve one immutable HF revision, then download exactly that snapshot."""

import argparse
import hashlib
import json
import re
from pathlib import Path


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
    if config.get("model_type") != "qwen2" or config.get("num_hidden_layers") != 28:
        raise ValueError("this Phase 1 profile requires Qwen2.5-7B geometry")
    quant = config.get("quantization", {})
    if quant.get("bits") != 4:
        raise ValueError("expected 4-bit converted model")
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
