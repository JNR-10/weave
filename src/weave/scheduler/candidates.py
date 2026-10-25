"""Offline Phase 1 candidates. This module never starts MLX or selects a winner."""

import argparse
import hashlib
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weave.estimator.memory import estimate_stage_memory

VERSION = "2.0.0"


def positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name}: expected {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Fail closed with ValueError, including when a field is missing or has the wrong JSON type."""
    try:
        return _check_state(state)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed cluster state: {exc!r}") from exc


def _check_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("schema_version") != VERSION:
        raise ValueError("unsupported schema_version; migrate explicitly to 2.0.0")
    if state.get("simulation") is not True:
        raise ValueError("Phase 1 fixture runner accepts simulation=true only")
    if not isinstance(state.get("snapshot_id"), str) or not state["snapshot_id"]:
        raise ValueError("snapshot_id is required")
    model, req = state["model"], state["request_envelope"]
    for key in ("model_id", "model_revision", "profile_id"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ValueError(f"model.{key} is required")
    total = positive_int(model["total_layers"], "total_layers")
    weights = model["layer_weight_bytes"]
    if not isinstance(weights, list) or len(weights) != total:
        raise ValueError("layer_weight_bytes must contain one entry per layer")
    for n in weights:
        positive_int(n, "layer_weight_bytes entry")
    for key in ("replicated_weight_bytes", "kv_heads", "head_dim", "kv_bytes_per_element"):
        positive_int(model[key], key)
    if model["source"] != "simulated":
        raise ValueError("fixture model profile must be labeled simulated")
    for key in ("prompt_tokens_upper_bound", "max_new_tokens", "concurrency"):
        positive_int(req[key], key)
    if req["concurrency"] != 1:
        raise ValueError("Phase 1 supports concurrency=1 only")
    devices = state["devices"]
    if not 1 <= len(devices) <= 2:
        raise ValueError("Phase 1 supports one or two candidate devices")
    ids = [d["device_id"] for d in devices]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("device_id values must be nonempty and unique")
    for d in devices:
        if d["source"] != "simulated":
            raise ValueError("every fixture device must be labeled simulated")
        if d["budget_basis"] != "fresh_process_absolute":
            raise ValueError("unsupported budget basis")
        for key in ("safe_process_budget_bytes", "workspace_reserve_bytes", "safety_margin_bytes"):
            positive_int(d[key], key)
    return state


def assignments(device_order: Sequence[str], logical_counts: Sequence[int]) -> list[dict[str, Any]]:
    if len(device_order) != len(logical_counts) or not logical_counts:
        raise ValueError("device/count length mismatch")
    if len(set(device_order)) != len(device_order):
        raise ValueError("duplicate device assignment")
    start, rows = 0, []
    for stage, (device_id, count) in enumerate(zip(device_order, logical_counts)):
        positive_int(count, "layer_count")
        rows.append(
            {
                "device_id": device_id,
                "logical_stage": stage,
                "mlx_rank": len(logical_counts) - 1 - stage,
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "layer_count": count,
            }
        )
        start += count
    return rows


def evaluate(order: Sequence[str], counts: Sequence[int], state: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    model, req = state["model"], state["request_envelope"]
    if sum(counts) != model["total_layers"]:
        raise ValueError("counts must sum to total_layers")
    devices = {d["device_id"]: d for d in state["devices"]}
    if not set(order) <= devices.keys():
        raise ValueError("order names a device that is not in the state")
    rows, reasons = assignments(order, counts), []
    tokens = req["prompt_tokens_upper_bound"] + req["max_new_tokens"]
    for row in rows:
        d = devices[row["device_id"]]
        parts = estimate_stage_memory(
            layer_weight_bytes=model["layer_weight_bytes"][
                row["start_layer"] : row["end_layer_exclusive"]
            ],
            replicated_weight_bytes=model["replicated_weight_bytes"],
            total_tokens=tokens,
            kv_heads=model["kv_heads"],
            head_dim=model["head_dim"],
            kv_bytes_per_element=model["kv_bytes_per_element"],
            concurrency=req["concurrency"],
            workspace_reserve_bytes=d["workspace_reserve_bytes"],
            safety_margin_bytes=d["safety_margin_bytes"],
        )
        predicted = sum(parts.values())
        row.update(
            {
                "memory_components": parts,
                "predicted_memory_bytes": predicted,
                "safe_process_budget_bytes": d["safe_process_budget_bytes"],
                "headroom_bytes": d["safe_process_budget_bytes"] - predicted,
            }
        )
        if predicted > d["safe_process_budget_bytes"]:
            reasons.append(
                {
                    "code": "MEMORY_BUDGET_EXCEEDED",
                    "device_id": d["device_id"],
                    "excess_bytes": predicted - d["safe_process_budget_bytes"],
                }
            )
    identity = {
        "order": list(order),
        "counts": list(counts),
        "snapshot": state["snapshot_id"],
        "profile": model["profile_id"],
    }
    plan_id = (
        "candidate-"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    )
    return {
        "schema_version": VERSION,
        "plan_id": plan_id,
        "source_snapshot_id": state["snapshot_id"],
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "profile_id": model["profile_id"],
        "simulation": True,
        "selected": False,
        "policy": "candidate_only",
        "mode": "single_device" if len(counts) == 1 else "pipeline",
        "logical_split": list(counts),
        "mlx_split_by_rank": list(reversed(counts)),
        "device_id_by_mlx_rank": list(reversed(order)),
        "assignments": rows,
        "memory_feasible": not reasons,
        "rejection_reasons": reasons,
        "runtime_verified": False,
        "executable": False,
    }


def generate(state: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    total = state["model"]["total_layers"]
    ids = [d["device_id"] for d in state["devices"]]
    plans = [evaluate([d], [total], state) for d in ids]
    if len(ids) == 2:
        for order in itertools.permutations(ids):
            for left in range(1, total):
                plans.append(evaluate(order, [left, total - left], state))
    return {
        "schema_version": VERSION,
        "simulation": True,
        "source_snapshot_id": state["snapshot_id"],
        "selected_plan_id": None,
        "candidate_count": len(plans),
        "memory_feasible_count": sum(p["memory_feasible"] for p in plans),
        "candidates": plans,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = generate(json.loads(Path(args.state).read_text()))
    with Path(args.output).open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
