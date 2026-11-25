"""What one MLX pipeline launch needs, derived from a logical plan, and checks on what ran.

Logical stage 0 runs the earliest layers, but MLX's PipelineMixin gives rank 0 the *last* layers
(guide section 7.1). The mapping has one implementation, weave.scheduler.candidates.assignments;
this module reorders its output by rank and compares it with what each launched rank reports, so
a reversed split or hosts started in the wrong order stop the run instead of producing plausible
numbers from the wrong layers. Nothing here imports MLX, so the checks run on Intel and in CI.
"""

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weave.contracts.reading import (
    as_object,
    bool_field,
    field_path,
    positive_int,
    required,
    text_field,
)
from weave.scheduler.candidates import assignments

# A draft bring-up record, deliberately outside the 2.0.0 evidence contract: the validator does not
# accept it, and the P2-C/P2-D evidence format is agreed only after a two-Mac run exists.
RECORD = "weave-phase2-bring-up"
RECORD_VERSION = "draft-1"

# The installed sharded_load (mlx-lm 0.32.2) finds a tensor's layer this way; sanitize can rename
# keys, so it matches the layer number rather than a full prefix.
LAYER_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class Stage:
    """One logical stage and the launch rank that runs it in this launch only."""

    device_id: str
    logical_stage: int
    mlx_rank: int
    start_layer: int
    end_layer_exclusive: int


@dataclass(frozen=True)
class LaunchPlan:
    """A logical plan plus the rank-ordered view an MLX launch needs (guide section 7.1)."""

    device_order: tuple[str, ...]
    logical_split: tuple[int, ...]
    stages: tuple[Stage, ...]

    @property
    def size(self) -> int:
        return len(self.stages)

    @property
    def total_layers(self) -> int:
        return sum(self.logical_split)

    @property
    def mlx_split_by_rank(self) -> tuple[int, ...]:
        """The `split` argument of PipelineMixin.pipeline: logical_split reversed."""
        return tuple(reversed(self.logical_split))

    @property
    def device_id_by_mlx_rank(self) -> tuple[str, ...]:
        """The host order mlx.launch must use: device_order reversed, never device_order."""
        return tuple(reversed(self.device_order))

    def stage_for_rank(self, rank: int) -> Stage:
        for stage in self.stages:
            if stage.mlx_rank == rank:
                return stage
        raise ValueError(f"rank: expected 0 to {self.size - 1}, got {rank!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "device_order": list(self.device_order),
            "logical_split": list(self.logical_split),
            "mlx_split_by_rank": list(self.mlx_split_by_rank),
            "device_id_by_mlx_rank": list(self.device_id_by_mlx_rank),
        }


def installed_range(mlx_split_by_rank: Sequence[int], rank: int) -> tuple[int, int]:
    """The half-open layer range the installed PipelineMixin gives `rank` (guide section 7.1).

    Rank r runs the layers after those of ranks r+1 ... size-1. Kept separate from
    `assignments` so the two derivations check each other.
    """
    if not 0 <= rank < len(mlx_split_by_rank):
        raise ValueError(f"rank: expected 0 to {len(mlx_split_by_rank) - 1}, got {rank!r}")
    start = sum(mlx_split_by_rank[rank + 1 :])
    return start, start + mlx_split_by_rank[rank]


def launch_plan(device_order: Sequence[str], logical_split: Sequence[int]) -> LaunchPlan:
    """Build the launch view of a logical plan, and refuse it if the two mappings disagree."""
    rows = assignments(device_order, logical_split)
    stages = tuple(
        Stage(
            device_id=row["device_id"],
            logical_stage=row["logical_stage"],
            mlx_rank=row["mlx_rank"],
            start_layer=row["start_layer"],
            end_layer_exclusive=row["end_layer_exclusive"],
        )
        for row in rows
    )
    plan = LaunchPlan(tuple(device_order), tuple(logical_split), stages)
    for stage in stages:
        expected = (stage.start_layer, stage.end_layer_exclusive)
        if installed_range(plan.mlx_split_by_rank, stage.mlx_rank) != expected:
            raise RuntimeError(f"rank mapping disagrees with the installed formula for {stage}")
    return plan


def check_rank(
    plan: LaunchPlan, *, rank: int, size: int, device_id: str, start_idx: int, end_idx: int
) -> Stage:
    """Compare what a launched rank reports with the plan; any difference stops the run.

    Reversing the counts without reversing the host order is the mistake this catches: the
    layer ranges would still cover the model, but on the wrong Macs.
    """
    if size != plan.size:
        raise RuntimeError(f"launched group has {size} ranks, plan has {plan.size} stages")
    stage = plan.stage_for_rank(rank)
    if device_id != stage.device_id:
        raise RuntimeError(
            f"rank {rank} is running on {device_id!r}, plan puts {stage.device_id!r} there; "
            "check the hostfile order (device_id_by_mlx_rank)"
        )
    expected = (stage.start_layer, stage.end_layer_exclusive)
    if (start_idx, end_idx) != expected:
        raise RuntimeError(
            f"rank {rank} loaded layers [{start_idx}, {end_idx}), plan expects "
            f"[{expected[0]}, {expected[1]})"
        )
    return stage


def local_weight_files(
    weight_map: Mapping[str, str], start: int, end: int, total_layers: int
) -> list[str]:
    """The checkpoint files a rank running layers [start, end) needs, in sorted order.

    Mirrors the installed sharded_load: a file is needed if it holds a tensor of a local layer or
    any non-layer tensor (embeddings, final norm, head), which every stage keeps. Layers at or past
    `total_layers` are dropped by sanitize. A checkpoint missing one of the model's layers is
    refused, because a stage would silently run with uninitialized weights.
    """
    if not 0 <= start < end <= total_layers:
        raise ValueError(f"layer range [{start}, {end}) is not inside [0, {total_layers})")
    seen: set[int] = set()
    needed: set[str] = set()
    for key, file in weight_map.items():
        match = LAYER_KEY.search(key)
        if match is None:
            needed.add(file)
            continue
        layer = int(match.group(1))
        seen.add(layer)
        if start <= layer < end:
            needed.add(file)
    missing = sorted(set(range(total_layers)) - seen)
    if missing:
        raise ValueError(f"checkpoint has no weights for layers {missing}")
    return sorted(needed)


def hostfile(plan: LaunchPlan, hosts: Mapping[str, Any]) -> list[Any]:
    """Host entries in MLX rank order, from a map of device_id to that Mac's entry.

    Entries pass through unchanged: their format belongs to the pinned mlx.launch (check
    `mlx.launch --help` there), and this function only fixes their order. Extra devices in the
    map are allowed, so one inventory file serves every plan.
    """
    missing = [d for d in plan.device_id_by_mlx_rank if d not in hosts]
    if missing:
        raise ValueError(f"hosts: no entry for {missing}")
    return [hosts[device_id] for device_id in plan.device_id_by_mlx_rank]


def first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    """The first token index where two generations differ, or None if they are identical."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _tokens(record: dict[str, Any], key: str, parent: str) -> list[int]:
    """A non-empty list of token IDs; 0 is a real token ID, so only negatives are refused."""
    raw = required(record, key, parent)
    path = field_path(parent, key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list")
    return [_nonnegative(token, f"{path}[{i}]") for i, token in enumerate(raw)]


def _nonnegative(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path}: expected a non-negative integer, got {value!r}")
    return value


def _device_ids(plan: dict[str, Any], parent: str) -> list[str]:
    raw = required(plan, "device_order", parent)
    path = field_path(parent, "device_order")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list")
    for i, device_id in enumerate(raw):
        if not isinstance(device_id, str) or not device_id:
            raise ValueError(f"{path}[{i}]: expected a non-empty string, got {device_id!r}")
    return raw


def _counts(record: dict[str, Any], key: str, parent: str) -> list[int]:
    raw = required(record, key, parent)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field_path(parent, key)}: expected a non-empty list")
    return [positive_int(n, f"{field_path(parent, key)}[{i}]") for i, n in enumerate(raw)]


def read_record(raw: Any, parent: str) -> dict[str, Any]:
    """Check one rank's bring-up record far enough to compare it with the others."""
    record = as_object(raw, parent)
    if required(record, "record", parent) != RECORD:
        raise ValueError(f"{field_path(parent, 'record')}: expected {RECORD!r}")
    if required(record, "record_version", parent) != RECORD_VERSION:
        raise ValueError(f"{field_path(parent, 'record_version')}: expected {RECORD_VERSION!r}")
    if required(record, "status", parent) != "success":
        error = record.get("error")
        raise ValueError(f"{field_path(parent, 'status')}: rank did not finish: {error!r}")
    run_id = text_field(record, "run_id", parent)
    simulated = bool_field(record, "simulation", parent)
    if simulated != run_id.startswith("SIMULATED-"):
        raise ValueError(f"{parent}: simulation flag and SIMULATED- run ID disagree")
    plan = as_object(required(record, "plan", parent), field_path(parent, "plan"))
    return {
        "run_id": run_id,
        "simulation": simulated,
        "rank": _nonnegative(required(record, "rank", parent), field_path(parent, "rank")),
        "size": positive_int(required(record, "size", parent), field_path(parent, "size")),
        "device_id": text_field(record, "device_id", parent),
        "model_revision": text_field(record, "model_revision", parent),
        "start_idx": _nonnegative(
            required(record, "start_idx", parent), field_path(parent, "start_idx")
        ),
        "end_idx": positive_int(required(record, "end_idx", parent), field_path(parent, "end_idx")),
        "device_order": _device_ids(plan, field_path(parent, "plan")),
        "logical_split": _counts(plan, "logical_split", field_path(parent, "plan")),
        "prompt_tokens": _tokens(record, "prompt_tokens", parent),
        "tokens": _tokens(record, "tokens", parent),
    }


def check_bring_up(
    records: Sequence[dict[str, Any]], reference: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Gate checks P2-C/P2-D can make from the rank records alone (guide section 14).

    Every rank of one launch must agree on the run and the plan; each rank must be the Mac the
    plan puts there with the layers the plan gives it; the ranks together must load every layer
    exactly once; and every rank must have generated the same tokens. With a single-device
    reference (a size-1 record made by the same script on the same prompt), the tokens must also
    match it exactly at temperature 0. A divergence is reported with its index, never hidden.
    """
    if not records:
        raise ValueError("records: expected at least one rank record")
    first = records[0]
    for key in ("run_id", "simulation", "size", "model_revision", "device_order", "logical_split"):
        for record in records[1:]:
            if record[key] != first[key]:
                raise ValueError(
                    f"rank records disagree on {key}: {first[key]!r} vs {record[key]!r}"
                )
    plan = launch_plan(first["device_order"], first["logical_split"])
    ranks = sorted(record["rank"] for record in records)
    if ranks != list(range(plan.size)) or first["size"] != plan.size:
        raise ValueError(f"expected one record per rank 0 to {plan.size - 1}, got ranks {ranks}")
    covered: list[int] = []
    for record in sorted(records, key=lambda r: r["rank"]):
        check_rank(
            plan,
            rank=record["rank"],
            size=record["size"],
            device_id=record["device_id"],
            start_idx=record["start_idx"],
            end_idx=record["end_idx"],
        )
        covered.extend(range(record["start_idx"], record["end_idx"]))
    if sorted(covered) != list(range(plan.total_layers)):
        raise ValueError("ranks do not load every layer exactly once")
    prompts = {tuple(record["prompt_tokens"]) for record in records}
    if len(prompts) != 1:
        raise ValueError("ranks were given different prompt tokens")
    rank0 = next(record for record in records if record["rank"] == 0)
    disagreeing = {
        record["rank"]: first_divergence(rank0["tokens"], record["tokens"])
        for record in records
        if record["tokens"] != rank0["tokens"]
    }
    result: dict[str, Any] = {
        "run_id": first["run_id"],
        "simulation": first["simulation"],
        "plan": plan.to_json(),
        "layers_loaded_once": True,
        "ranks_agree": not disagreeing,
        "rank_divergence": disagreeing,
        "generated_tokens": len(rank0["tokens"]),
        "matches_reference": None,
        "reference_divergence": None,
    }
    if reference is not None:
        if reference["size"] != 1:
            raise ValueError("reference: expected a single-device record (size 1)")
        if reference["simulation"] != first["simulation"]:
            raise ValueError("reference: simulated and measured records cannot be compared")
        if reference["model_revision"] != first["model_revision"]:
            raise ValueError("reference: different model revision")
        if reference["prompt_tokens"] != rank0["prompt_tokens"]:
            raise ValueError("reference: different prompt tokens")
        divergence = first_divergence(reference["tokens"], rank0["tokens"])
        result["matches_reference"] = divergence is None
        result["reference_divergence"] = divergence
    result["passed"] = result["ranks_agree"] and result["matches_reference"] is not False
    return result


def _load_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc


def _write_new(path: str, payload: Any) -> None:
    """Write JSON to a file that must not exist yet; an earlier file is never replaced."""
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    show = sub.add_parser("plan", help="print the launch view of a logical plan")
    hosts = sub.add_parser("hostfile", help="write host entries in MLX rank order")
    for parser in (show, hosts):
        parser.add_argument("--device-order", nargs="+", required=True)
        parser.add_argument("--split", nargs="+", type=int, required=True)
    hosts.add_argument("--hosts", required=True, help="JSON object: device_id -> host entry")
    hosts.add_argument("--output", required=True)
    check = sub.add_parser("check-run", help="check the rank records of one bring-up run")
    check.add_argument("records", nargs="+", help="rank-<r>.json files, one per rank")
    check.add_argument("--reference", help="single-device rank-0.json from the same script")
    args = p.parse_args(argv)

    if args.command == "plan":
        print(json.dumps(launch_plan(args.device_order, args.split).to_json(), indent=2))
        return 0
    if args.command == "hostfile":
        plan = launch_plan(args.device_order, args.split)
        entries = hostfile(plan, as_object(_load_json(args.hosts), args.hosts))
        _write_new(args.output, entries)
        print(json.dumps(plan.to_json(), indent=2))
        return 0
    records = [read_record(_load_json(path), path) for path in args.records]
    reference = read_record(_load_json(args.reference), args.reference) if args.reference else None
    result = check_bring_up(records, reference)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
