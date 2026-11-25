"""Offline Phase 1 candidates. This module never starts MLX or selects a winner."""

import argparse
import hashlib
import itertools
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weave.contracts.reading import (
    as_object,
    bool_field,
    positive_field,
    positive_int,
    required,
    text_field,
)
from weave.estimator.memory import estimate_stage_memory

VERSION = "2.0.0"


# Reading the cluster state. Every error starts with the field's path (weave.contracts.reading).


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
    total = positive_field(model, "total_layers", "model")
    weights = required(model, "layer_weight_bytes", "model")
    if not isinstance(weights, list):
        raise ValueError(f"model.layer_weight_bytes: expected a list, got {type(weights).__name__}")
    if len(weights) != total:
        raise ValueError(
            f"model.layer_weight_bytes: expected one entry per layer ({total}), got {len(weights)}"
        )
    return _ModelProfile(
        model_id=text_field(model, "model_id", "model"),
        model_revision=text_field(model, "model_revision", "model"),
        profile_id=text_field(model, "profile_id", "model"),
        layer_weight_bytes=tuple(
            positive_int(n, f"model.layer_weight_bytes[{i}]") for i, n in enumerate(weights)
        ),
        replicated_weight_bytes=positive_field(model, "replicated_weight_bytes", "model"),
        kv_heads=positive_field(model, "kv_heads", "model"),
        head_dim=positive_field(model, "head_dim", "model"),
        kv_bytes_per_element=positive_field(model, "kv_bytes_per_element", "model"),
        source=text_field(model, "source", "model"),
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
        prompt_tokens_upper_bound=positive_field(request, "prompt_tokens_upper_bound", parent),
        max_new_tokens=positive_field(request, "max_new_tokens", parent),
        concurrency=positive_field(request, "concurrency", parent),
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
        device = as_object(raw, parent)
        device_id = text_field(device, "device_id", parent)
        if device_id in {d.device_id for d in parsed}:
            raise ValueError(f"{parent}.device_id: duplicate {device_id!r}")
        parsed.append(
            _DeviceBudget(
                device_id=device_id,
                source=text_field(device, "source", parent),
                budget_basis=text_field(device, "budget_basis", parent),
                safe_process_budget_bytes=positive_field(
                    device, "safe_process_budget_bytes", parent
                ),
                workspace_reserve_bytes=positive_field(device, "workspace_reserve_bytes", parent),
                safety_margin_bytes=positive_field(device, "safety_margin_bytes", parent),
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
    state = as_object(raw, "state")
    version = required(state, "schema_version", "")
    if version != VERSION:
        raise ValueError(
            f"schema_version: unsupported {version!r}; migrate explicitly to {VERSION}"
        )
    snapshot_id = text_field(state, "snapshot_id", "")
    simulation = bool_field(state, "simulation", "")
    model = _parse_model(as_object(required(state, "model", ""), "model"))
    request = _parse_request(as_object(required(state, "request_envelope", ""), "request_envelope"))
    devices = required(state, "devices", "")
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


def _load_state(state: Any) -> _ClusterState:
    """Read the state once and hold it to the Phase 1 limits; everything after trusts the result."""
    cluster = _parse_state(state)
    _check_phase1_limits(cluster)
    return cluster


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Fail closed with a ValueError naming the field; returns the state unchanged when valid."""
    _load_state(state)
    return state


def assignments(device_order: Sequence[str], logical_counts: Sequence[int]) -> list[dict[str, Any]]:
    """Lay the layers out stage by stage, and give each stage its MLX rank (guide section 7).

    Logical stage 0 runs the earliest layers, but MLX's PipelineMixin gives rank 0 the *last*
    layers, so ranks count down from the final stage. Ranges are half-open,
    [start_layer, end_layer_exclusive), matching Python slicing and MLX's start_idx/end_idx.
    """
    if not logical_counts:
        raise ValueError("logical_counts: expected at least one stage")
    if len(device_order) != len(logical_counts):
        raise ValueError(
            f"device_order: expected {len(logical_counts)} devices to match logical_counts, "
            f"got {len(device_order)}"
        )
    counts = [positive_int(c, f"logical_counts[{i}]") for i, c in enumerate(logical_counts)]
    seen: set[str] = set()
    for i, device_id in enumerate(device_order):
        if not isinstance(device_id, str) or not device_id:
            raise ValueError(f"device_order[{i}]: expected a non-empty string, got {device_id!r}")
        if device_id in seen:
            raise ValueError(f"device_order[{i}]: duplicate {device_id!r}")
        seen.add(device_id)
    last_rank = len(counts) - 1
    starts = itertools.accumulate(counts, initial=0)
    return [
        {
            "device_id": device_id,
            "logical_stage": stage,
            "mlx_rank": last_rank - stage,
            "start_layer": start,
            "end_layer_exclusive": start + count,
            "layer_count": count,
        }
        for stage, (device_id, count, start) in enumerate(zip(device_order, counts, starts))
    ]


def _plan_id(order: Sequence[str], counts: Sequence[int], cluster: _ClusterState) -> str:
    """Guide section 11: a plan's identity is snapshot, profile, device order and layer counts.

    It deliberately ignores the model revision and the budgets, so a changed input must come with
    a new snapshot ID. Changing what goes in here changes every plan ID and the 2.0.0 contract.
    """
    identity = {
        "order": list(order),
        "counts": list(counts),
        "snapshot": cluster.snapshot_id,
        "profile": cluster.model.profile_id,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return "candidate-" + digest[:16]


def _evaluate(
    cluster: _ClusterState, order: Sequence[str], counts: Sequence[int]
) -> dict[str, Any]:
    """One candidate: each stage's memory against its Mac's budget. Never selects or runs it."""
    model, request = cluster.model, cluster.request
    rows = assignments(order, counts)
    if sum(counts) != model.total_layers:
        raise ValueError(
            f"logical_counts: expected {model.total_layers} layers in total, got {sum(counts)}"
        )
    devices = {device.device_id: device for device in cluster.devices}
    for i, device_id in enumerate(order):
        if device_id not in devices:
            raise ValueError(f"device_order[{i}]: {device_id!r} is not in the state")
    total_tokens = request.prompt_tokens_upper_bound + request.max_new_tokens
    reasons = []
    for row in rows:
        device = devices[row["device_id"]]
        parts = estimate_stage_memory(
            layer_weight_bytes=model.layer_weight_bytes[
                row["start_layer"] : row["end_layer_exclusive"]
            ],
            replicated_weight_bytes=model.replicated_weight_bytes,
            total_tokens=total_tokens,
            kv_heads=model.kv_heads,
            head_dim=model.head_dim,
            kv_bytes_per_element=model.kv_bytes_per_element,
            concurrency=request.concurrency,
            workspace_reserve_bytes=device.workspace_reserve_bytes,
            safety_margin_bytes=device.safety_margin_bytes,
        )
        predicted = sum(parts.values())
        budget = device.safe_process_budget_bytes
        row.update(
            {
                "memory_components": parts,
                "predicted_memory_bytes": predicted,
                "safe_process_budget_bytes": budget,
                "headroom_bytes": budget - predicted,
            }
        )
        if predicted > budget:
            reasons.append(
                {
                    "code": "MEMORY_BUDGET_EXCEEDED",
                    "device_id": device.device_id,
                    "excess_bytes": predicted - budget,
                }
            )
    by_rank = sorted(rows, key=lambda row: row["mlx_rank"])
    return {
        "schema_version": VERSION,
        "plan_id": _plan_id(order, counts, cluster),
        "source_snapshot_id": cluster.snapshot_id,
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "profile_id": model.profile_id,
        # Fixed by the guide section 8 safety boundary: a candidate is never a decision.
        "simulation": True,
        "selected": False,
        "policy": "candidate_only",
        "mode": "single_device" if len(rows) == 1 else "pipeline",
        "logical_split": [row["layer_count"] for row in rows],
        "mlx_split_by_rank": [row["layer_count"] for row in by_rank],
        "device_id_by_mlx_rank": [row["device_id"] for row in by_rank],
        "assignments": rows,
        "memory_feasible": not reasons,
        "rejection_reasons": reasons,
        "runtime_verified": False,
        "executable": False,
    }


def evaluate(order: Sequence[str], counts: Sequence[int], state: dict[str, Any]) -> dict[str, Any]:
    """One candidate for an explicit device order and logical layer split."""
    return _evaluate(_load_state(state), order, counts)


def _splits(total_layers: int, stages: int) -> Iterator[list[int]]:
    """Every way to give each stage at least one layer, first stage's share rising first."""
    for cuts in itertools.combinations(range(1, total_layers), stages - 1):
        bounds = (0, *cuts, total_layers)
        yield [end - start for start, end in itertools.pairwise(bounds)]


def _layouts(cluster: _ClusterState) -> Iterator[tuple[tuple[str, ...], list[int]]]:
    """Every placement: each Mac alone, then every ordered group of Macs with every split.

    The order is part of the output and the golden file: single-Mac plans in device order, then
    groups by size, each size in permutation order. Nothing here assumes two Macs; the Phase 1
    limit check is what caps the group size today.
    """
    device_ids = [device.device_id for device in cluster.devices]
    for size in range(1, len(device_ids) + 1):
        for order in itertools.permutations(device_ids, size):
            for counts in _splits(cluster.model.total_layers, size):
                yield order, counts


def generate(state: dict[str, Any]) -> dict[str, Any]:
    """Every candidate for the state, all unselected: ranking and selection are later phases."""
    cluster = _load_state(state)
    plans = [_evaluate(cluster, order, counts) for order, counts in _layouts(cluster)]
    return {
        "schema_version": VERSION,
        "simulation": True,
        "source_snapshot_id": cluster.snapshot_id,
        "selected_plan_id": None,
        "candidate_count": len(plans),
        "memory_feasible_count": sum(plan["memory_feasible"] for plan in plans),
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
