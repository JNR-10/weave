"""Offline Phase 1 candidates. This module never starts MLX or selects a winner."""

import argparse
import hashlib
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weave.estimator.memory import estimate_stage_memory

VERSION = "2.0.0"


def positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name}: expected {'nonnegative' if allow_zero else 'positive'} integer")
    return value


# Reading the cluster state. Every error starts with the field's path ("devices[1].device_id: ...")
# so whoever fixes the input knows exactly where to look.


def _path(parent: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    return f"{parent}.{key}" if parent else key


def _get(obj: dict[str, Any], key: str, parent: str) -> Any:
    if key not in obj:
        raise ValueError(f"{_path(parent, key)}: required")
    return obj[key]


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(value).__name__}")
    return value


def _text(obj: dict[str, Any], key: str, parent: str) -> str:
    value = _get(obj, key, parent)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{_path(parent, key)}: expected a non-empty string, got {value!r}")
    return value


def _boolean(obj: dict[str, Any], key: str, parent: str) -> bool:
    value = _get(obj, key, parent)
    if not isinstance(value, bool):
        raise ValueError(f"{_path(parent, key)}: expected true or false, got {value!r}")
    return value


def _positive_int(value: Any, path: str) -> int:
    # bool is a subclass of int in Python, so `type(...) is int` keeps true/false out.
    if type(value) is not int or value < 1:
        raise ValueError(f"{path}: expected a positive integer, got {value!r}")
    return value


def _positive_field(obj: dict[str, Any], key: str, parent: str) -> int:
    return _positive_int(_get(obj, key, parent), _path(parent, key))


@dataclass(frozen=True)
class _ModelProfile:
    """Per-layer and replicated weight bytes plus the KV-cache geometry the estimate needs."""

    model_id: str
    model_revision: str
    profile_id: str
    layer_weight_bytes: tuple[int, ...]
    replicated_weight_bytes: int
    kv_heads: int
    head_dim: int
    kv_bytes_per_element: int
    source: str

    @property
    def total_layers(self) -> int:
        return len(self.layer_weight_bytes)


def _parse_model(model: dict[str, Any]) -> _ModelProfile:
    total = _positive_field(model, "total_layers", "model")
    weights = _get(model, "layer_weight_bytes", "model")
    if not isinstance(weights, list):
        raise ValueError(f"model.layer_weight_bytes: expected a list, got {type(weights).__name__}")
    if len(weights) != total:
        raise ValueError(
            f"model.layer_weight_bytes: expected one entry per layer ({total}), got {len(weights)}"
        )
    return _ModelProfile(
        model_id=_text(model, "model_id", "model"),
        model_revision=_text(model, "model_revision", "model"),
        profile_id=_text(model, "profile_id", "model"),
        layer_weight_bytes=tuple(
            _positive_int(n, f"model.layer_weight_bytes[{i}]") for i, n in enumerate(weights)
        ),
        replicated_weight_bytes=_positive_field(model, "replicated_weight_bytes", "model"),
        kv_heads=_positive_field(model, "kv_heads", "model"),
        head_dim=_positive_field(model, "head_dim", "model"),
        kv_bytes_per_element=_positive_field(model, "kv_bytes_per_element", "model"),
        source=_text(model, "source", "model"),
    )


@dataclass(frozen=True)
class _RequestEnvelope:
    """The largest request the plan must hold: prompt and output token bounds, and concurrency."""

    prompt_tokens_upper_bound: int
    max_new_tokens: int
    concurrency: int


def _parse_request(request: dict[str, Any]) -> _RequestEnvelope:
    parent = "request_envelope"
    return _RequestEnvelope(
        prompt_tokens_upper_bound=_positive_field(request, "prompt_tokens_upper_bound", parent),
        max_new_tokens=_positive_field(request, "max_new_tokens", parent),
        concurrency=_positive_field(request, "concurrency", parent),
    )


@dataclass(frozen=True)
class _DeviceBudget:
    """One Mac's absolute budget for a fresh model process, and the reserves held out of it."""

    device_id: str
    source: str
    budget_basis: str
    safe_process_budget_bytes: int
    workspace_reserve_bytes: int
    safety_margin_bytes: int


def _parse_devices(devices: list[Any]) -> tuple[_DeviceBudget, ...]:
    parsed: list[_DeviceBudget] = []
    for i, raw in enumerate(devices):
        parent = f"devices[{i}]"
        device = _object(raw, parent)
        device_id = _text(device, "device_id", parent)
        if device_id in {d.device_id for d in parsed}:
            raise ValueError(f"{parent}.device_id: duplicate {device_id!r}")
        parsed.append(
            _DeviceBudget(
                device_id=device_id,
                source=_text(device, "source", parent),
                budget_basis=_text(device, "budget_basis", parent),
                safe_process_budget_bytes=_positive_field(
                    device, "safe_process_budget_bytes", parent
                ),
                workspace_reserve_bytes=_positive_field(device, "workspace_reserve_bytes", parent),
                safety_margin_bytes=_positive_field(device, "safety_margin_bytes", parent),
            )
        )
    return tuple(parsed)


@dataclass(frozen=True)
class _ClusterState:
    """The cluster state after its one read; planning trusts these fields without re-checking."""

    snapshot_id: str
    simulation: bool
    model: _ModelProfile
    request: _RequestEnvelope
    devices: tuple[_DeviceBudget, ...]


def _parse_state(raw: Any) -> _ClusterState:
    state = _object(raw, "state")
    version = _get(state, "schema_version", "")
    if version != VERSION:
        raise ValueError(
            f"schema_version: unsupported {version!r}; migrate explicitly to {VERSION}"
        )
    snapshot_id = _text(state, "snapshot_id", "")
    simulation = _boolean(state, "simulation", "")
    model = _parse_model(_object(_get(state, "model", ""), "model"))
    request = _parse_request(_object(_get(state, "request_envelope", ""), "request_envelope"))
    devices = _get(state, "devices", "")
    if not isinstance(devices, list):
        raise ValueError(f"devices: expected a list, got {type(devices).__name__}")
    if not devices:
        raise ValueError("devices: expected at least one device")
    return _ClusterState(snapshot_id, simulation, model, request, _parse_devices(devices))


PHASE1_MAX_DEVICES = 2


def _phase1_limit(path: str, accepted: str, got: Any) -> ValueError:
    return ValueError(f"{path}: Phase 1 accepts {accepted} only, got {got!r}")


def _check_phase1_limits(state: _ClusterState) -> None:
    """Phase 1 plans only wholly simulated, fresh-process, one-request, at-most-two-Mac input.

    Guide section 8 makes this a safety boundary: an uncalibrated estimate must never pass as a
    measured deployment. The limits live here, apart from parsing and planning, so Phase 2 lifts
    them by changing this check alone.
    """
    if state.simulation is not True:
        raise _phase1_limit("simulation", "simulated input (true)", state.simulation)
    if state.model.source != "simulated":
        raise _phase1_limit("model.source", "'simulated'", state.model.source)
    if state.request.concurrency != 1:
        raise _phase1_limit("request_envelope.concurrency", "1", state.request.concurrency)
    if len(state.devices) > PHASE1_MAX_DEVICES:
        raise _phase1_limit("devices", f"at most {PHASE1_MAX_DEVICES} devices", len(state.devices))
    for i, device in enumerate(state.devices):
        if device.source != "simulated":
            raise _phase1_limit(f"devices[{i}].source", "'simulated'", device.source)
        if device.budget_basis != "fresh_process_absolute":
            raise _phase1_limit(
                f"devices[{i}].budget_basis", "'fresh_process_absolute'", device.budget_basis
            )


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Fail closed with a ValueError naming the field; returns the state unchanged when valid."""
    _check_phase1_limits(_parse_state(state))
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
