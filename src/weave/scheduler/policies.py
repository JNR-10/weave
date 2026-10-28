"""Static placement policies over the Phase 1 candidate set (spec 0002).

This module reads and chooses. It never builds plans, because layer cover, rank mapping and the
memory estimate have one implementation in weave.scheduler.candidates, and it never starts MLX.
The candidate set is read once into immutable records, so a policy never works from raw JSON or
from a set that is malformed, partly selected, or not wholly simulated.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from weave.scheduler.candidates import VERSION

# Reading the candidate set. Every error starts with the field's path ("candidates[2].mode: ..."),
# as in weave.scheduler.candidates. The small helpers are repeated here on purpose: modules do not
# import each other's private helpers.


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


def _flag(obj: dict[str, Any], key: str, expected: bool, parent: str) -> None:
    """Refuse anything but the one safe value; `is not` also keeps 0 and 1 out."""
    value = _get(obj, key, parent)
    if value is not expected:
        word = "true" if expected else "false"
        raise ValueError(f"{_path(parent, key)}: expected {word}, got {value!r}")


def _positive_int(value: Any, path: str) -> int:
    # bool is a subclass of int in Python, so `type(...) is int` keeps true/false out.
    if type(value) is not int or value < 1:
        raise ValueError(f"{path}: expected a positive integer, got {value!r}")
    return value


def _positive_field(obj: dict[str, Any], key: str, parent: str) -> int:
    return _positive_int(_get(obj, key, parent), _path(parent, key))


def _expect_int(obj: dict[str, Any], key: str, expected: int, parent: str) -> None:
    value = _get(obj, key, parent)
    if type(value) is not int or value != expected:
        raise ValueError(f"{_path(parent, key)}: expected {expected}, got {value!r}")


def _check_version(obj: dict[str, Any], parent: str) -> None:
    version = _get(obj, "schema_version", parent)
    if version != VERSION:
        where = _path(parent, "schema_version")
        raise ValueError(f"{where}: unsupported {version!r}; migrate explicitly to {VERSION}")


@dataclass(frozen=True)
class Device:
    """One Mac as its single-device candidate states it: budget and the reserves held out of it."""

    device_id: str
    safe_process_budget_bytes: int
    replicated_weight_bytes: int
    workspace_reserve_bytes: int
    safety_margin_bytes: int

    @property
    def fixed_stage_bytes(self) -> int:
        """Bytes every stage on this Mac pays however many layers it runs.

        Replicated weights, the workspace reserve and the safety margin are charged in full to
        every stage, so only the budget left after them is room for layers.
        """
        return (
            self.replicated_weight_bytes + self.workspace_reserve_bytes + self.safety_margin_bytes
        )


@dataclass(frozen=True)
class Candidate:
    """One plan from the generator, reduced to what a policy needs to find it and report on it.

    `memory_feasible` means "passes the stated estimate", not executable, verified, or chosen.
    Headroom is per logical stage and is negative where the stage exceeds its Mac's budget.
    """

    plan_id: str
    device_order: tuple[str, ...]
    logical_split: tuple[int, ...]
    memory_feasible: bool
    headroom_bytes: tuple[int, ...]
    rejection_reasons: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CandidateSet:
    """A whole candidate set after its one read; policies trust these fields without re-checking.

    `devices` follows the order of the single-device candidates, which is the cluster state's
    device order. A multi-Mac policy places its stages in that order.
    """

    source_snapshot_id: str
    total_layers: int
    devices: tuple[Device, ...]
    candidates: tuple[Candidate, ...]

    def find(self, device_order: Sequence[str], logical_split: Sequence[int]) -> Candidate:
        """The candidate for this device order and split, or ValueError if there is none.

        A policy computes the split it wants and looks it up here instead of building a plan. A
        missing plan means the set does not match the generator.
        """
        order, split = tuple(device_order), tuple(logical_split)
        for candidate in self.candidates:
            if candidate.device_order == order and candidate.logical_split == split:
                return candidate
        raise ValueError(
            f"candidates: split {list(split)} on {list(order)} is not among the candidates; "
            "the set does not match the generator"
        )


@dataclass(frozen=True)
class _Stage:
    """One parsed assignment: the Mac it names, with its numbers as this candidate states them."""

    device: Device
    headroom_bytes: int


@dataclass(frozen=True)
class _ParsedCandidate:
    """A candidate plus the Mac records of its stages, kept for the cross-candidate checks."""

    candidate: Candidate
    stage_devices: tuple[Device, ...]


def _parse_assignment(raw: Any, parent: str, stage: int, layers: int) -> _Stage:
    obj = _object(raw, parent)
    device_id = _text(obj, "device_id", parent)
    _expect_int(obj, "logical_stage", stage, parent)
    _expect_int(obj, "layer_count", layers, parent)
    headroom = _get(obj, "headroom_bytes", parent)
    if type(headroom) is not int:
        where = _path(parent, "headroom_bytes")
        raise ValueError(f"{where}: expected an integer, got {headroom!r}")
    budget = _positive_field(obj, "safe_process_budget_bytes", parent)
    components_path = _path(parent, "memory_components")
    components = _object(_get(obj, "memory_components", parent), components_path)
    replicated = _positive_field(components, "replicated_weight_bytes", components_path)
    workspace = _positive_field(components, "workspace_reserve_bytes", components_path)
    safety = _positive_field(components, "safety_margin_bytes", components_path)
    return _Stage(Device(device_id, budget, replicated, workspace, safety), headroom)


def _parse_assignments(raw: Any, parent: str, split: tuple[int, ...]) -> tuple[_Stage, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{parent}: expected a list, got {type(raw).__name__}")
    if len(raw) != len(split):
        raise ValueError(
            f"{parent}: expected one entry per logical_split entry ({len(split)}), got {len(raw)}"
        )
    stages: list[_Stage] = []
    for j, item in enumerate(raw):
        assignment_path = _path(parent, j)
        stage = _parse_assignment(item, assignment_path, j, split[j])
        device_id = stage.device.device_id
        if device_id in {s.device.device_id for s in stages}:
            raise ValueError(f"{assignment_path}.device_id: duplicate {device_id!r}")
        stages.append(stage)
    return tuple(stages)


def _parse_split(raw: Any, path: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list, got {raw!r}")
    return tuple(_positive_int(n, _path(path, k)) for k, n in enumerate(raw))


def _parse_reasons(raw: Any, path: str) -> tuple[dict[str, Any], ...]:
    """Rejection reasons are carried through as given; a copy keeps the input untouched."""
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list, got {type(raw).__name__}")
    return tuple(copy.deepcopy(_object(item, _path(path, k))) for k, item in enumerate(raw))


def _parse_candidate(raw: Any, index: int, snapshot_id: str) -> _ParsedCandidate:
    parent = _path("candidates", index)
    obj = _object(raw, parent)
    _check_version(obj, parent)
    # The Phase 1 safety boundary: a candidate is never a decision, and never measured data.
    _flag(obj, "simulation", True, parent)
    _flag(obj, "selected", False, parent)
    _flag(obj, "executable", False, parent)
    _flag(obj, "runtime_verified", False, parent)
    policy = _get(obj, "policy", parent)
    if policy != "candidate_only":
        where = _path(parent, "policy")
        raise ValueError(f"{where}: expected 'candidate_only', got {policy!r}")
    plan_id = _text(obj, "plan_id", parent)
    own_snapshot = _text(obj, "source_snapshot_id", parent)
    if own_snapshot != snapshot_id:
        where = _path(parent, "source_snapshot_id")
        raise ValueError(f"{where}: {own_snapshot!r} differs from the set's {snapshot_id!r}")
    mode = _get(obj, "mode", parent)
    if mode not in ("single_device", "pipeline"):
        where = _path(parent, "mode")
        raise ValueError(f"{where}: expected 'single_device' or 'pipeline', got {mode!r}")
    split = _parse_split(_get(obj, "logical_split", parent), _path(parent, "logical_split"))
    if (mode == "single_device") != (len(split) == 1):
        where = _path(parent, "mode")
        raise ValueError(f"{where}: {mode!r} is inconsistent with {len(split)} stage(s)")
    feasible = _boolean(obj, "memory_feasible", parent)
    reasons = _parse_reasons(
        _get(obj, "rejection_reasons", parent), _path(parent, "rejection_reasons")
    )
    # A policy reports the verdict and carries the reasons through, so the two must agree.
    if feasible == bool(reasons):
        where = _path(parent, "memory_feasible")
        raise ValueError(f"{where}: {feasible} contradicts {len(reasons)} rejection reason(s)")
    assignments_path = _path(parent, "assignments")
    stages = _parse_assignments(_get(obj, "assignments", parent), assignments_path, split)
    candidate = Candidate(
        plan_id=plan_id,
        device_order=tuple(s.device.device_id for s in stages),
        logical_split=split,
        memory_feasible=feasible,
        headroom_bytes=tuple(s.headroom_bytes for s in stages),
        rejection_reasons=reasons,
    )
    return _ParsedCandidate(candidate, tuple(s.device for s in stages))


# Checks across candidates. They run after every candidate parsed on its own, so a malformed
# candidate is named by its own field before any comparison with the others.

# (Device attribute, its path inside an assignment)
_DEVICE_FIELDS = (
    ("safe_process_budget_bytes", "safe_process_budget_bytes"),
    ("replicated_weight_bytes", "memory_components.replicated_weight_bytes"),
    ("workspace_reserve_bytes", "memory_components.workspace_reserve_bytes"),
    ("safety_margin_bytes", "memory_components.safety_margin_bytes"),
)


def _check_layer_totals(parsed: list[_ParsedCandidate]) -> int:
    expected = sum(parsed[0].candidate.logical_split)
    for i, entry in enumerate(parsed):
        total = sum(entry.candidate.logical_split)
        if total != expected:
            raise ValueError(
                f"candidates[{i}].logical_split: sums to {total} layers, "
                f"expected {expected} like candidates[0]"
            )
    return expected


def _single_devices(parsed: list[_ParsedCandidate]) -> dict[str, Device]:
    """Each Mac's record, from its one single-device candidate, in the order they appear."""
    singles: dict[str, Device] = {}
    for i, entry in enumerate(parsed):
        if len(entry.stage_devices) != 1:
            continue
        device = entry.stage_devices[0]
        if device.device_id in singles:
            raise ValueError(
                f"candidates[{i}]: second single-device candidate for {device.device_id!r}"
            )
        singles[device.device_id] = device
    appearing = dict.fromkeys(d.device_id for entry in parsed for d in entry.stage_devices)
    for device_id in appearing:
        if device_id not in singles:
            raise ValueError(f"candidates: no single-device candidate for {device_id!r}")
    return singles


def _check_devices_agree(parsed: list[_ParsedCandidate], singles: dict[str, Device]) -> None:
    """A Mac has one budget and one set of fixed costs, so every candidate must state the same."""
    for i, entry in enumerate(parsed):
        for j, device in enumerate(entry.stage_devices):
            known = singles[device.device_id]
            for attr, where in _DEVICE_FIELDS:
                got, want = getattr(device, attr), getattr(known, attr)
                if got != want:
                    raise ValueError(
                        f"candidates[{i}].assignments[{j}].{where}: {got} differs from {want} "
                        f"in the single-device candidate for {device.device_id!r}"
                    )


def read_candidate_set(raw: Any) -> CandidateSet:
    """Read the dict from weave.scheduler.candidates.generate once, into immutable records.

    Fails closed with a ValueError that starts with the offending field's path. The input is
    never mutated, and nothing is returned that still points into it.
    """
    top = _object(raw, "candidate_set")
    _check_version(top, "")
    _flag(top, "simulation", True, "")
    selected = _get(top, "selected_plan_id", "")
    if selected is not None:
        raise ValueError(f"selected_plan_id: expected null, got {selected!r}")
    snapshot_id = _text(top, "source_snapshot_id", "")
    raw_candidates = _get(top, "candidates", "")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"candidates: expected a non-empty list, got {raw_candidates!r}")

    parsed: list[_ParsedCandidate] = []
    plan_ids: set[str] = set()
    for i, item in enumerate(raw_candidates):
        entry = _parse_candidate(item, i, snapshot_id)
        plan_id = entry.candidate.plan_id
        if plan_id in plan_ids:
            raise ValueError(f"candidates[{i}].plan_id: duplicate {plan_id!r}")
        plan_ids.add(plan_id)
        parsed.append(entry)

    total_layers = _check_layer_totals(parsed)
    singles = _single_devices(parsed)
    _check_devices_agree(parsed, singles)
    return CandidateSet(
        source_snapshot_id=snapshot_id,
        total_layers=total_layers,
        devices=tuple(singles.values()),
        candidates=tuple(entry.candidate for entry in parsed),
    )
