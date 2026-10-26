"""Resolve one immutable HF revision, download exactly that snapshot, and verify every file."""

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
    if (config.get("quantization") or {}).get("bits") != 4:
        raise ValueError("expected 4-bit converted model")


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Git's object id for a regular (non-LFS) file, which the hub reports as blob_id."""
    header = f"blob {path.stat().st_size}\0".encode()
    with path.open("rb") as f:
        return hashlib.file_digest(f, lambda: hashlib.sha1(header)).hexdigest()


def expected_files(siblings: list[Any]) -> dict[str, dict[str, Any]]:
    """LFS files are pinned by SHA-256; regular git files by their blob SHA-1."""
    files = {}
    for s in siblings:
        if s.lfs is not None:
            files[s.rfilename] = {"size": s.lfs.size, "sha256": s.lfs.sha256}
        else:
            files[s.rfilename] = {"size": s.size, "git_blob_sha1": s.blob_id}
    return files


def verify_snapshot(model_dir: Path, expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check every file of the revision; a config hash alone does not pin the weights.

    mlx_lm.load reads every *.safetensors in the directory, so a weight file that is not part of
    the revision (left over from another download) also fails verification.
    """
    files = {}
    for name, want in sorted(expected.items()):
        path = model_dir / name
        if not path.is_file():
            files[name] = {"expected": want, "actual": None, "ok": False}
            continue
        digest = ("sha256", file_sha256) if "sha256" in want else ("git_blob_sha1", git_blob_sha1)
        got = {"size": path.stat().st_size, digest[0]: digest[1](path)}
        files[name] = {"expected": want, "actual": got, "ok": got == want}
    stray = sorted(p.name for p in model_dir.glob("*.safetensors") if p.name not in expected)
    ok = all(f["ok"] for f in files.values()) and not stray
    return {"ok": ok, "files": files, "unexpected_weight_files": stray}


def failures(report: dict[str, Any]) -> list[str]:
    return [n for n, f in report["files"].items() if not f["ok"]] + report[
        "unexpected_weight_files"
    ]


def hub_files(repo: str, revision: str) -> tuple[str, dict[str, dict[str, Any]]]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise ValueError("expected immutable 40-character model commit")
    return info.sha, expected_files(info.siblings)


def verify_existing(manifest: Path, report_path: Path) -> None:
    """Verify the snapshot an existing lock names, without downloading or rewriting the lock."""
    lock = json.loads(manifest.read_text())
    model_dir = Path(lock["model_dir"])
    if file_sha256(model_dir / "config.json") != lock["config_sha256"]:
        raise ValueError("config.json differs from the model lock")
    revision, expected = hub_files(lock["model_id"], lock["model_revision"])
    report = {
        "schema_version": "2.0.0",
        "model_id": lock["model_id"],
        "model_revision": revision,
        "model_dir": str(model_dir),
        **verify_snapshot(model_dir, expected),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"report": str(report_path), "ok": report["ok"]}))
    if not report["ok"]:
        raise SystemExit(f"snapshot does not match the lock's revision: {failures(report)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    p.add_argument("--revision", default="main")
    p.add_argument("--model-dir", default="models/qwen7b")
    p.add_argument("--manifest", default="experiments/phase1/model-lock.json")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="finish an interrupted download of an explicit 40-character --revision "
        "into the existing --model-dir, verify it, then write the lock",
    )
    mode.add_argument(
        "--verify-report",
        help="verify the snapshot named by the existing --manifest and write this report; "
        "downloads nothing and changes no lock",
    )
    args = p.parse_args()
    if args.verify_report:
        verify_existing(Path(args.manifest), Path(args.verify_report))
        return
    from huggingface_hub import hf_hub_download, snapshot_download

    out, model_dir = Path(args.manifest), Path(args.model_dir)
    if out.exists():
        raise FileExistsError("choose a new manifest; never overwrite an existing model lock")
    if args.resume:
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            raise ValueError(
                "--resume needs the 40-character --revision of the interrupted download"
            )
        if not model_dir.is_dir():
            raise FileNotFoundError("--resume needs the existing --model-dir")
    elif model_dir.exists():
        raise FileExistsError("choose a new model dir, or --resume an interrupted download")
    revision, expected = hub_files(args.repo, args.revision)
    snapshot_download(repo_id=args.repo, revision=revision, local_dir=str(model_dir))
    report = verify_snapshot(model_dir, expected)
    damaged = [name for name in failures(report) if name in expected]
    if damaged:
        # snapshot_download can skip a file its local metadata marks complete even when it is
        # damaged, so fetch failing files again once. Stray weight files must be removed by hand.
        for name in damaged:
            hf_hub_download(
                args.repo, name, revision=revision, local_dir=str(model_dir), force_download=True
            )
        report = verify_snapshot(model_dir, expected)
    if not report["ok"]:
        raise ValueError(f"snapshot does not match revision {revision}: {failures(report)}")
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
