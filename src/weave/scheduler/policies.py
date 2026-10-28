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


# Policies. Each computes the plan it wants, finds it among the candidates and reports the
# candidate's own memory verdict; none falls back to another plan (spec 0002).

FEASIBLE = "FEASIBLE"
INFEASIBLE = "INFEASIBLE"
NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"

BEST_SINGLE_DEVICE = "best_single_device"
EQUAL_SPLIT = "equal_split"
MEMORY_PROPORTIONAL = "memory_proportional"


@dataclass(frozen=True)
class PolicyChoice:
    """What one policy chose from a candidate set, and why.

    `plan` is None only for NO_FEASIBLE_PLAN. A choice is not executable and not runtime-verified;
    it is a static pick on an estimate. `reason` is deterministic text, so a later decision record
    and golden comparison can carry it verbatim.
    """

    policy: str
    outcome: str
    plan: Candidate | None
    reason: str


def best_single_device(candidate_set: CandidateSet) -> PolicyChoice:
    """The "don't distribute" baseline: the single Mac with the most headroom that fits.

    Guide section 14 lists single-node MLX on the best single device as a baseline, so it must
    always exist for distribution to have something to beat. For each Mac in device order this
    looks up its single-device plan, keeps those that pass the memory estimate and picks the one
    with the largest headroom. A tie goes to the earlier Mac, because a later plan replaces the
    pick only when strictly greater. When no Mac fits alone the outcome is NO_FEASIBLE_PLAN and no
    plan is chosen: a missing baseline is reported, never replaced by a plan that does not fit.
    """
    best: Candidate | None = None
    for device in candidate_set.devices:
        plan = candidate_set.find((device.device_id,), (candidate_set.total_layers,))
        if not plan.memory_feasible:
            continue
        if best is None or plan.headroom_bytes[0] > best.headroom_bytes[0]:
            best = plan
    if best is None:
        return PolicyChoice(
            BEST_SINGLE_DEVICE,
            NO_FEASIBLE_PLAN,
            None,
            "no single-device plan fits its Mac's budget",
        )
    reason = (
        f"{best.device_order[0]} alone has the most headroom of the single-device plans "
        f"that fit: {best.headroom_bytes[0]} bytes"
    )
    return PolicyChoice(BEST_SINGLE_DEVICE, FEASIBLE, best, reason)


def equal_split_counts(total_layers: int, device_count: int) -> tuple[int, ...]:
    """Layers per Mac in device order, shared as evenly as whole layers allow.

    Every Mac gets total_layers // device_count and the first total_layers % device_count Macs get
    one more. The remainder goes to the earliest stages so that "the equal split" names one plan: a
    fixed, written rule, with no choice left about which Macs carry the extra layer. Public so the
    rule can be tested for any number of Macs without a candidate set.
    """
    layers = _positive_int(total_layers, "total_layers")
    devices = _positive_int(device_count, "device_count")
    if layers < devices:
        raise ValueError(
            f"total_layers: {layers} layers cannot give each of {devices} Macs at least one"
        )
    base, extra = divmod(layers, devices)
    return tuple(base + 1 if i < extra else base for i in range(devices))


def equal_split(candidate_set: CandidateSet) -> PolicyChoice:
    """The weakest common baseline: the same number of layers on every Mac.

    Guide section 14 and the README evaluation ladder start at the equal split. It spreads the
    layers over every Mac in device order, with one fixed order and no search. It reports the plan
    even when it fails the memory estimate: a baseline that does not fit is a result to count, and
    it is never replaced by a fallback plan.
    """
    order = tuple(device.device_id for device in candidate_set.devices)
    counts = equal_split_counts(candidate_set.total_layers, len(order))
    plan = candidate_set.find(order, counts)
    if plan.memory_feasible:
        outcome = FEASIBLE
        reason = f"equal split {list(counts)} on {list(order)} passes the memory estimate"
    else:
        outcome = INFEASIBLE
        failing = ", ".join(r["device_id"] for r in plan.rejection_reasons)
        reason = (
            f"equal split {list(counts)} on {list(order)} fails the memory estimate on {failing}"
        )
    return PolicyChoice(EQUAL_SPLIT, outcome, plan, reason)


def memory_proportional_counts(total_layers: int, rooms: Sequence[int]) -> tuple[int, ...]:
    """Layers per Mac in device order, shared in proportion to each Mac's room for layers.

    All steps use exact integer arithmetic, so a rounding never depends on float error:

    1. Validate. A str is a sequence, so only a list or tuple of non-negative ints is accepted,
       and bool is refused because it is an int in Python. Each Mac needs at least one layer, so
       fewer layers than Macs cannot be shared.
    2. If no Mac has any room, every quota is equal, so the answer is the equal split.
    3. Largest remainder. Each Mac first gets the floor of its quota, total_layers * room // sum
       of rooms. The layers left over then go one each to the Macs with the largest remainders,
       and a tie goes to the earlier Mac, so one input names one split.
    4. At least one layer per Mac. A Mac with almost no room can floor to zero, and a stage cannot
       be empty. Each such Mac, in order, takes one layer from the Mac with the most layers (the
       earliest on a tie). With total_layers >= len(rooms), that Mac always has two or more.

    Public so the rule can be tested for any number of Macs without a candidate set.
    """
    layers = _positive_int(total_layers, "total_layers")
    if not isinstance(rooms, (list, tuple)) or not rooms:
        raise ValueError(f"rooms: expected a non-empty sequence, got {rooms!r}")
    for i, room in enumerate(rooms):
        if type(room) is not int or room < 0:
            raise ValueError(f"rooms[{i}]: expected a non-negative integer, got {room!r}")
    if layers < len(rooms):
        raise ValueError(
            f"total_layers: {layers} layers cannot give each of {len(rooms)} Macs at least one"
        )
    total_room = sum(rooms)
    if total_room == 0:
        return equal_split_counts(layers, len(rooms))
    counts = [layers * room // total_room for room in rooms]
    remainders = [layers * room % total_room for room in rooms]
    leftover = layers - sum(counts)
    by_remainder = sorted(range(len(rooms)), key=lambda i: (-remainders[i], i))
    for i in by_remainder[:leftover]:
        counts[i] += 1
    for i in range(len(counts)):
        if counts[i] == 0:
            donor = counts.index(max(counts))
            counts[donor] -= 1
            counts[i] += 1
    return tuple(counts)


def memory_proportional(candidate_set: CandidateSet) -> PolicyChoice:
    """The hardware-aware static baseline: layers shared in proportion to each Mac's room.

    Guide section 14 lists deterministic memory-based placement as a baseline. A Mac's room for
    layers is its safe process budget minus the bytes every stage pays regardless of layers
    (Device.fixed_stage_bytes), floored at zero. Proportion is by layer count, which matches bytes
    only while layers weigh the same. It reports the plan even when it fails the memory estimate:
    a baseline that does not fit is a result to count, and it is never replaced by a fallback.
    """
    order = tuple(device.device_id for device in candidate_set.devices)
    rooms = tuple(
        max(device.safe_process_budget_bytes - device.fixed_stage_bytes, 0)
        for device in candidate_set.devices
    )
    counts = memory_proportional_counts(candidate_set.total_layers, rooms)
    plan = candidate_set.find(order, counts)
    if plan.memory_feasible:
        outcome = FEASIBLE
        reason = (
            f"memory-proportional split {list(counts)} on {list(order)} from room {list(rooms)} "
            "bytes passes the memory estimate"
        )
    else:
        outcome = INFEASIBLE
        failing = ", ".join(r["device_id"] for r in plan.rejection_reasons)
        reason = (
            f"memory-proportional split {list(counts)} on {list(order)} from room {list(rooms)} "
            f"bytes fails the memory estimate on {failing}"
        )
    return PolicyChoice(MEMORY_PROPORTIONAL, outcome, plan, reason)
