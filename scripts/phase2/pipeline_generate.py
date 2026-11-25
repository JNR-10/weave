"""T14 / T15 (gates P2-C, P2-D): load a logical split on the launched Macs and generate.

Every rank runs this same command under mlx.launch. Each rank checks that it is the Mac the plan
puts there and that it loaded the planned layers, generates the smoke workload's prompt greedily,
and writes its own `rank-<r>.json` into a new run directory on its own Mac. Collect the records on
one Mac and check them together with `python -m weave.runtime.launch check-run`.

A one-stage plan (`--split 28`) runs without mlx.launch and is the single-device reference the
two-Mac tokens are compared with.

UNVERIFIED: written on an Intel Mac; it has not yet run on Apple Silicon. Records are a draft
bring-up format outside the 2.0.0 evidence contract (weave.runtime.launch.RECORD_VERSION).
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weave.runtime.launch import RECORD, RECORD_VERSION, launch_plan
from weave.runtime.mlx_pipeline import generate, init_group, load_stage
from weave.telemetry.baseline import git_provenance, source_checkout


def _write_once(path: Path, record: dict[str, Any]) -> None:
    with path.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device-order", nargs="+", required=True, help="logical order, first layers")
    p.add_argument("--split", nargs="+", type=int, required=True, help="logical layer counts")
    p.add_argument("--model-lock", default="experiments/phase1/model-lock.json")
    p.add_argument("--workload", default="benchmarks/workloads/phase1-smoke.json")
    p.add_argument("--backend", default="ring")
    p.add_argument("--device-id-file", default=str(Path.home() / ".weave-device-id"))
    p.add_argument("--output-dir", required=True, help="new results/<run-id> on each Mac")
    p.add_argument("--timeout-s", type=float, default=900.0, help="a hang becomes a failed run")
    args = p.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("run on native Apple Silicon macOS")

    out = Path(args.output_dir)
    if out.name.startswith("SIMULATED-"):
        raise ValueError("a real run cannot use a SIMULATED- run ID")
    plan = launch_plan(args.device_order, args.split)
    device_id = Path(args.device_id_file).read_text().strip()
    lock_bytes = Path(args.model_lock).read_bytes()
    lock = json.loads(lock_bytes)
    workload_bytes = Path(args.workload).read_bytes()
    workload = json.loads(workload_bytes)
    if workload["temperature"] != 0:
        raise ValueError("token agreement is defined for greedy decoding only (temperature 0)")
    model_dir = Path(lock["model_dir"])
    config_bytes = (model_dir / "config.json").read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != lock["config_sha256"]:
        raise ValueError("model config differs from the lock")
    if json.loads(config_bytes) != lock["config"]:
        raise ValueError("model lock's embedded config differs from config.json")

    out.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance(source_checkout(), out)
    if provenance["git_commit"] is None or provenance["git_dirty"] is not False:
        raise RuntimeError(f"start from a clean, committed checkout: {provenance}")

    record: dict[str, Any] = {
        "record": RECORD,
        "record_version": RECORD_VERSION,
        "status": "error",
        "run_id": out.name,
        "source": "measured",
        "simulation": False,
        "wall_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "device_id": device_id,
        "backend": args.backend if plan.size > 1 else None,
        "rank": 0 if plan.size == 1 else None,
        "size": plan.size if plan.size == 1 else None,
        "plan": plan.to_json(),
        "model_id": lock["model_id"],
        "model_revision": lock["model_revision"],
        "model_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "workload_id": workload["workload_id"],
        "workload_sha256": hashlib.sha256(workload_bytes).hexdigest(),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        **provenance,
    }
    lock_finish = threading.Lock()
    finished: list[bool] = []

    def finish(status: str, extra: dict[str, Any]) -> None:
        """Write this rank's record once: the first outcome wins, a later one is dropped."""
        with lock_finish:
            if finished:
                return
            finished.append(True)
            record.update(extra, status=status)
            rank = record["rank"]
            name = f"rank-{rank}.json" if rank is not None else f"rank-unknown-{device_id}.json"
            _write_once(out / name, record)

    def on_timeout() -> None:
        finish(
            "error", {"error": {"type": "Timeout", "message": f"no result in {args.timeout_s}s"}}
        )
        os._exit(3)  # a peer stuck in a collective never returns; leave without joining it

    timer = threading.Timer(args.timeout_s, on_timeout)
    timer.daemon = True
    timer.start()
    try:
        group = init_group(args.backend) if plan.size > 1 else None
        if group is not None:
            record["rank"], record["size"] = group.rank(), group.size()
        stage = load_stage(model_dir, plan, group, device_id)
        result = generate(stage, workload["prompt"], workload["max_new_tokens"], workload["seed"])
        timer.cancel()
        finish(
            "success",
            {
                "start_idx": stage.start_idx,
                "end_idx": stage.end_idx,
                "layers_loaded": stage.end_idx - stage.start_idx,
                "weight_files": list(stage.weight_files),
                "load_ms": stage.load_ms,
                "load_peak_bytes": stage.load_peak_bytes,
                **result,
            },
        )
    except Exception as exc:
        timer.cancel()
        finish(
            "error",
            {
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": traceback.format_exc(),
            },
        )
        raise
    if record["rank"] == 0:
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in ("run_id", "rank", "size", "device_id", "start_idx", "end_idx")
                }
                | {"text": record["text"], "generated_tokens": len(record["tokens"])},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
