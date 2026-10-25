"""Apple Silicon only. Observed callback timings, serial requests, fresh KV per request."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weave.profiler.model_profile import build_parameter_profile

VERSION = "2.0.0"
FIELDS = [
    "schema_version",
    "sequence",
    "run_id",
    "device_id",
    "source",
    "wall_time_utc",
    "monotonic_ns",
    "request_id",
    "request_kind",
    "event",
    "policy",
    "partition_version",
    "workload_id",
    "model_revision",
    "token_count",
    "process_rss_bytes",
    "system_available_bytes",
    "mlx_active_bytes",
    "mlx_cache_bytes",
    "mlx_peak_bytes",
]


def validate_workload(w: dict[str, Any]) -> None:
    if (
        w.get("schema_version") != VERSION
        or type(w.get("concurrency")) is not int
        or w["concurrency"] != 1
    ):
        raise ValueError("requires schema 2.0.0 and concurrency=1")
    for k in ("prompt_max_tokens", "max_new_tokens", "warmup_requests", "measured_requests"):
        if type(w.get(k)) is not int or w[k] <= 0:
            raise ValueError(f"{k} must be a positive integer")
    if not isinstance(w.get("prompt"), str) or not w["prompt"].strip():
        raise ValueError("prompt must be nonempty")
    if not w.get("workload_id") or w.get("temperature") != 0 or type(w.get("seed")) is not int:
        raise ValueError("requires workload_id, temperature=0, integer seed")


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_provenance(out: Path) -> dict[str, Any]:
    """A commit hash alone hides uncommitted edits, so record whether the tree was dirty.

    Git runs against the checkout that holds this module, not the working directory, so the
    commit identifies the source that was actually imported. Untracked files are ignored; tracked
    changes are saved beside the evidence so a dirty run still shows exactly what executed.
    """
    repo = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}
    if diff:
        (out / "git-diff.patch").write_text(diff)
    return {"git_commit": commit, "git_dirty": bool(diff)}


def environment() -> dict[str, Any]:
    """Record what actually ran; a hash of the committed lock file only says what should have."""
    packages = {}
    for dist in importlib.metadata.distributions():
        if dist.metadata["Name"]:
            packages[re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()] = dist.version
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
        "packages": dict(sorted(packages.items())),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--model-lock", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device-id", required=True)
    p.add_argument("--token-trace", choices=["full", "boundary"], default="full")
    args = p.parse_args()
    # Keep the exact input bytes: the copies in the run directory must match their hashes.
    workload_bytes = Path(args.workload).read_bytes()
    w = json.loads(workload_bytes)
    validate_workload(w)
    lock_bytes = Path(args.model_lock).read_bytes()
    lock = json.loads(lock_bytes)
    if lock.get("schema_version") != VERSION or not re.fullmatch(
        r"[0-9a-f]{40}", lock.get("model_revision", "")
    ):
        raise ValueError("requires a real immutable model lock")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "workload.json").write_bytes(workload_bytes)
    (out / "model-lock.json").write_bytes(lock_bytes)
    summary = {
        "schema_version": VERSION,
        "run_id": out.name,
        "device_id": args.device_id,
        "source": "measured",
        "simulation": False,
        "status": "failed",
        "policy": "single_device",
        "partition_version": "single-device-v2",
        "model_id": lock["model_id"],
        "model_revision": lock["model_revision"],
        "workload_id": w["workload_id"],
        "requests": [],
        "timing_definition": "instrumented stream callback wall time; includes tokenization; excludes model load",
        "token_count_definition": "MLX-LM generation_tokens, may include terminal EOS",
        "token_trace": args.token_trace,
        "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "model_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        **git_provenance(out),
        "environment": environment(),
    }
    if Path("experiments/phase1/requirements-m4-lock.txt").exists():
        summary["requirements_sha256"] = sha("experiments/phase1/requirements-m4-lock.txt")
    trace = (out / "trace.csv").open("x", newline="")
    writer = csv.DictWriter(trace, fieldnames=FIELDS)
    writer.writeheader()
    sequence, request_id, request_kind, memory = 0, "", "", None

    def event(name, token_count="", sample_memory=True, timestamp=None):
        nonlocal sequence
        stamp = time.perf_counter_ns() if timestamp is None else timestamp
        row = dict.fromkeys(FIELDS, "")
        row.update(
            {
                "schema_version": VERSION,
                "sequence": sequence,
                "run_id": out.name,
                "device_id": args.device_id,
                "source": "measured",
                "wall_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "monotonic_ns": stamp,
                "request_id": request_id,
                "request_kind": request_kind,
                "event": name,
                "policy": "single_device",
                "partition_version": "single-device-v2",
                "workload_id": w["workload_id"],
                "model_revision": lock["model_revision"],
                "token_count": token_count,
            }
        )
        if sample_memory and memory is not None:
            row.update(memory())
        writer.writerow(row)
        sequence += 1
        if name != "token":
            trace.flush()
        return stamp

    exit_code = 1
    try:
        event("process_start")
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            raise RuntimeError("run measured inference on native Apple Silicon macOS")
        import mlx.core as mx
        import psutil
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        process = psutil.Process()

        def memory():
            return {
                "process_rss_bytes": process.memory_info().rss,
                "system_available_bytes": psutil.virtual_memory().available,
                "mlx_active_bytes": int(mx.get_active_memory()),
                "mlx_cache_bytes": int(mx.get_cache_memory()),
                "mlx_peak_bytes": int(mx.get_peak_memory()),
            }

        model_dir = Path(lock["model_dir"])
        config_bytes = (model_dir / "config.json").read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != lock["config_sha256"]:
            raise ValueError("model config differs from the lock")
        # Geometry comes from the hashed file, so the lock's embedded copy must agree with it.
        config = json.loads(config_bytes)
        if config != lock["config"]:
            raise ValueError("model lock's embedded config differs from config.json")
        mx.random.seed(w["seed"])
        mx.reset_peak_memory()
        event("model_load_start")
        model, tokenizer = load(str(model_dir))
        mx.eval(model.parameters())
        mx.synchronize()
        event("model_load_end")
        summary["model_load_peak_bytes"] = int(mx.get_peak_memory())
        profile = build_parameter_profile(
            model.parameters(),
            model_id=lock["model_id"],
            model_revision=lock["model_revision"],
            total_layers=config["num_hidden_layers"],
        )
        (out / "model_profile.json").write_text(json.dumps(profile, indent=2))
        for index in range(w["warmup_requests"] + w["measured_requests"]):
            request_kind = "warmup" if index < w["warmup_requests"] else "measured"
            request_id = f"req-{index:04d}"
            mx.synchronize()
            mx.reset_peak_memory()
            start = event("request_start")
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": w["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if not 0 < len(tokens) <= w["prompt_max_tokens"]:
                raise ValueError("actual tokenized prompt exceeds the frozen workload envelope")
            if len(tokens) + w["max_new_tokens"] > config["max_position_embeddings"]:
                raise ValueError("request exceeds model context limit")
            first, first_text, last_token, previous, pieces, last = None, None, None, 0, [], None
            for response in stream_generate(
                model,
                tokenizer,
                tokens,
                max_tokens=w["max_new_tokens"],
                sampler=make_sampler(temp=0),
            ):
                now = time.perf_counter_ns()
                count = int(response.generation_tokens)
                if count < previous or count > previous + 1:
                    raise ValueError("unexpected stream token counter; review installed API")
                if count > previous:
                    if first is None:
                        first = event("first_token", count, False, now)
                    if args.token_trace == "full":
                        event("token", count, False, now)
                    last_token, previous = now, count
                if response.text and first_text is None:
                    first_text = event("first_visible_text", count, False)
                pieces.append(response.text)
                last = response
            mx.synchronize()
            end = event("request_end", previous)
            text = "".join(pieces)
            if first is None or last is None or not text.strip():
                raise ValueError("empty/no-token output is not a successful baseline")
            item = {
                "request_id": request_id,
                "request_kind": request_kind,
                "status": "success",
                "prompt_tokens": len(tokens),
                "generation_tokens": previous,
                "ttft_ms": (first - start) / 1e6,
                "end_to_end_ms": (end - start) / 1e6,
                "first_visible_text_ms": None if first_text is None else (first_text - start) / 1e6,
                "observed_tpot_ms": (last_token - first) / (previous - 1) / 1e6
                if previous > 1
                else None,
                "library_prompt_tps": float(last.prompt_tps),
                "library_generation_tps": float(last.generation_tps),
                "request_peak_mlx_bytes": int(mx.get_peak_memory()),
                "generated_text": text,
            }
            summary["requests"].append(item)
        request_id, request_kind = "", ""
        event("process_end")
        summary["status"], exit_code = "success", 0
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        (out / "error.txt").write_text(traceback.format_exc())
        event("error", sample_memory=False)
    finally:
        trace.close()
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"run_id": out.name, "status": summary["status"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
