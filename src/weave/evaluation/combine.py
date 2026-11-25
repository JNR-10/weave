"""Combine per-run result records into per-cell summaries under one plan (spec 0003).

A record is summarized only if it names the plan by ID and by the SHA-256 of its bytes, so results
can never be attached to a plan edited after they were produced. Simulated and measured records are
never combined, and measured records are refused against a draft plan. A failed or out-of-memory
request counts as infinitely slow in every latency quantile instead of being dropped, so a policy
cannot look fast by failing.
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from weave.contracts.reading import (
    as_object,
    bool_field,
    field_path,
    positive_field,
    required,
    text_field,
)
from weave.evaluation.plan import VERSION, Plan, cells, load_plan

SOURCES = ("simulated", "measured")
REQUEST_KINDS = ("warmup", "measured")
STATUSES = ("success", "failed", "oom")
SIMULATED_PREFIX = "SIMULATED-"

# Verdicts of a candidate arm against the reference arm in one scenario.
BETTER = "BETTER"
WORSE = "WORSE"
NO_FINDING = "NO_FINDING"
INCOMPLETE = "INCOMPLETE"


def nearest_rank(values: Sequence[float], q: float) -> float:
    """The nearest-rank q-quantile: the ceil(q * n)-th smallest value (1-based).

    It always returns an observed value, so there is no interpolation choice to argue about, and an
    infinite value (a failed request) stays infinite instead of being averaged away.
    """
    if not values:
        raise ValueError("a quantile of no values is undefined")
    if not 0 < q <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {q!r}")
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    status: str
    generation_tokens: int
    ttft_ms: float
    end_to_end_ms: float
    observed_tpot_ms: float | None


@dataclass(frozen=True)
class RunResult:
    """One repeat of one cell; `requests` holds the measured requests only (warmup excluded)."""

    run_id: str
    simulation: bool
    scenario_id: str
    arm: str
    repeat: int
    repeat_wall_ms: float
    requests: tuple[RequestResult, ...]

    @property
    def cell_id(self) -> str:
        return f"{self.scenario_id}/{self.arm}"


@dataclass(frozen=True)
class CellSummary:
    cell_id: str
    scenario_id: str
    arm: str
    repeats: int
    measured_requests: int
    successful: int
    failed: int
    oom: int
    end_to_end_ms_p95: float
    end_to_end_ms_median: float
    end_to_end_ms_p95_by_repeat: tuple[float, ...]
    ttft_ms_p95: float
    ttft_ms_median: float
    observed_tpot_ms_median: float | None
    throughput_tokens_per_s_median: float
    complete: bool


@dataclass(frozen=True)
class Combined:
    plan_id: str
    plan_sha256: str
    simulation: bool | None
    runs: int
    cells: tuple[CellSummary, ...]
    missing: tuple[str, ...]
    incomplete: tuple[str, ...]


def _number(obj: dict[str, Any], key: str, parent: str) -> float:
    value = required(obj, key, parent)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_path(parent, key)}: expected a positive number, got {value!r}")
    return float(value)


def _choice(obj: dict[str, Any], key: str, parent: str, allowed: Sequence[str]) -> str:
    value = required(obj, key, parent)
    if value not in allowed:
        raise ValueError(
            f"{field_path(parent, key)}: expected one of {list(allowed)}, got {value!r}"
        )
    return value


def _request(raw: Any, parent: str) -> tuple[str, RequestResult]:
    item = as_object(raw, parent)
    request_id = text_field(item, "request_id", parent)
    kind = _choice(item, "request_kind", parent, REQUEST_KINDS)
    status = _choice(item, "status", parent, STATUSES)
    if status != "success":
        # A failed request has no trustworthy timings; it counts as infinitely slow.
        return kind, RequestResult(request_id, status, 0, math.inf, math.inf, None)
    tokens = positive_field(item, "generation_tokens", parent)
    end_to_end = _number(item, "end_to_end_ms", parent)
    ttft = _number(item, "ttft_ms", parent)
    if ttft > end_to_end:
        raise ValueError(f"{parent}.ttft_ms: {ttft} is after end_to_end_ms {end_to_end}")
    tpot = required(item, "observed_tpot_ms", parent)
    if tpot is not None:
        tpot = _number(item, "observed_tpot_ms", parent)
    return kind, RequestResult(request_id, status, tokens, ttft, end_to_end, tpot)


def _read_result(raw: Any, plan: Plan) -> RunResult:
    run = as_object(raw, "record")
    version = required(run, "schema_version", "")
    if version != VERSION:
        raise ValueError(
            f"schema_version: unsupported {version!r}; migrate explicitly to {VERSION}"
        )
    source = _choice(run, "source", "", SOURCES)
    simulation = bool_field(run, "simulation", "")
    if simulation != (source == "simulated"):
        raise ValueError(f"simulation: {simulation} contradicts source {source!r}")
    run_id = text_field(run, "run_id", "")
    if run_id.startswith(SIMULATED_PREFIX) != simulation:
        raise ValueError(
            f"run_id: {run_id!r}: a run ID starts with {SIMULATED_PREFIX!r} exactly when the run "
            "is simulated"
        )
    if text_field(run, "plan_id", "") != plan.plan_id:
        raise ValueError(f"plan_id: expected {plan.plan_id!r}, got {run['plan_id']!r}")
    if text_field(run, "plan_sha256", "") != plan.sha256:
        raise ValueError(
            f"plan_sha256: {run['plan_sha256']!r} is not the plan's {plan.sha256}; "
            "the plan changed after this run"
        )
    scenario_id = text_field(run, "scenario_id", "")
    if scenario_id not in {s.scenario_id for s in plan.scenarios}:
        raise ValueError(f"scenario_id: {scenario_id!r} is not declared in the plan")
    arm = text_field(run, "arm", "")
    if arm not in plan.arms:
        raise ValueError(f"arm: {arm!r} is not declared in the plan")
    repeat = positive_field(run, "repeat", "")
    if repeat > plan.repeats_per_cell:
        raise ValueError(f"repeat: the plan declares {plan.repeats_per_cell} repeats, got {repeat}")
    wall = _number(run, "repeat_wall_ms", "")
    raw_requests = required(run, "requests", "")
    if not isinstance(raw_requests, list):
        raise ValueError(f"requests: expected a list, got {type(raw_requests).__name__}")
    seen: set[str] = set()
    warmups = 0
    measured: list[RequestResult] = []
    for i, raw_request in enumerate(raw_requests):
        kind, result = _request(raw_request, f"requests[{i}]")
        if result.request_id in seen:
            raise ValueError(f"requests[{i}].request_id: {result.request_id!r} appears twice")
        seen.add(result.request_id)
        if kind == "warmup":
            warmups += 1
        else:
            measured.append(result)
    # Every declared request must be listed, failed ones included, so none can be dropped.
    if warmups != plan.warmup_requests_per_repeat:
        raise ValueError(
            f"requests: expected {plan.warmup_requests_per_repeat} warmup requests, got {warmups}"
        )
    if len(measured) != plan.measured_requests_per_repeat:
        raise ValueError(
            f"requests: expected {plan.measured_requests_per_repeat} measured requests, "
            f"got {len(measured)}"
        )
    return RunResult(run_id, simulation, scenario_id, arm, repeat, wall, tuple(measured))


def result_from_bytes(data: bytes, source: str, plan: Plan) -> RunResult:
    """Read one run record; every error names the file, then the field."""
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: not valid JSON ({exc})") from None
    try:
        return _read_result(raw, plan)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from None


def _summarize(cell_id: str, runs: list[RunResult], plan: Plan) -> CellSummary:
    runs = sorted(runs, key=lambda r: r.repeat)
    requests = [q for run in runs for q in run.requests]
    successes = [q for q in requests if q.status == "success"]
    tpots = [q.observed_tpot_ms for q in successes if q.observed_tpot_ms is not None]
    throughputs = [
        sum(q.generation_tokens for q in run.requests if q.status == "success")
        / (run.repeat_wall_ms / 1000)
        for run in runs
    ]
    end_to_end = [q.end_to_end_ms for q in requests]
    ttft = [q.ttft_ms for q in requests]
    return CellSummary(
        cell_id=cell_id,
        scenario_id=runs[0].scenario_id,
        arm=runs[0].arm,
        repeats=len(runs),
        measured_requests=len(requests),
        successful=len(successes),
        failed=sum(q.status == "failed" for q in requests),
        oom=sum(q.status == "oom" for q in requests),
        end_to_end_ms_p95=nearest_rank(end_to_end, 0.95),
        end_to_end_ms_median=nearest_rank(end_to_end, 0.5),
        end_to_end_ms_p95_by_repeat=tuple(
            nearest_rank([q.end_to_end_ms for q in run.requests], 0.95) for run in runs
        ),
        ttft_ms_p95=nearest_rank(ttft, 0.95),
        ttft_ms_median=nearest_rank(ttft, 0.5),
        observed_tpot_ms_median=nearest_rank(tpots, 0.5) if tpots else None,
        throughput_tokens_per_s_median=nearest_rank(throughputs, 0.5),
        complete=len(runs) == plan.repeats_per_cell,
    )


def combine(plan: Plan, results: Sequence[RunResult]) -> Combined:
    """Group records by cell and summarize each; report cells with no or too few repeats."""
    kinds = {r.simulation for r in results}
    if len(kinds) > 1:
        raise ValueError("simulated and measured records are never combined")
    simulation = kinds.pop() if kinds else None
    if simulation is False and plan.status != "frozen":
        raise ValueError(
            f"measured records need a frozen plan; {plan.plan_id} is a {plan.status}, so these "
            "runs cannot count"
        )
    run_ids: set[str] = set()
    by_cell: dict[str, list[RunResult]] = {}
    for result in results:
        if result.run_id in run_ids:
            raise ValueError(f"run {result.run_id!r} appears twice")
        run_ids.add(result.run_id)
        group = by_cell.setdefault(result.cell_id, [])
        if any(r.repeat == result.repeat for r in group):
            raise ValueError(f"{result.cell_id} repeat {result.repeat} appears twice")
        group.append(result)
    summaries: list[CellSummary] = []
    missing: list[str] = []
    for cell in cells(plan):
        if cell.cell_id in by_cell:
            summaries.append(_summarize(cell.cell_id, by_cell[cell.cell_id], plan))
        else:
            missing.append(cell.cell_id)
    return Combined(
        plan_id=plan.plan_id,
        plan_sha256=plan.sha256,
        simulation=simulation,
        runs=len(results),
        cells=tuple(summaries),
        missing=tuple(missing),
        incomplete=tuple(s.cell_id for s in summaries if not s.complete),
    )


def _beats(plan: Plan, a: CellSummary, b: CellSummary) -> bool:
    rule = plan.decision_rule
    if rule.require_non_overlapping_repeat_ranges and not (
        max(a.end_to_end_ms_p95_by_repeat) < min(b.end_to_end_ms_p95_by_repeat)
    ):
        return False
    return (
        a.end_to_end_ms_p95 < b.end_to_end_ms_p95
        and a.end_to_end_ms_p95 <= (1 - rule.min_relative_difference) * b.end_to_end_ms_p95
    )


def compare_cells(plan: Plan, reference: CellSummary, candidate: CellSummary) -> str:
    """The plan's decision rule on the primary metric (lower p95 end to end is better).

    The candidate is BETTER only when both cells are complete, every per-repeat p95 of the candidate
    is below every per-repeat p95 of the reference, and its pooled p95 is lower by at least the
    plan's margin; WORSE is the same rule the other way round. Anything less is NO_FINDING: a
    difference smaller than the run-to-run spread is not a result (experiments/README.md).
    """
    if reference.scenario_id != candidate.scenario_id:
        raise ValueError(
            f"{reference.cell_id} and {candidate.cell_id} are different scenarios; "
            "only arms of one scenario are compared"
        )
    if not (reference.complete and candidate.complete):
        return INCOMPLETE
    if _beats(plan, candidate, reference):
        return BETTER
    if _beats(plan, reference, candidate):
        return WORSE
    return NO_FINDING


def verdicts(plan: Plan, combined: Combined) -> dict[str, str]:
    """Each summarized cell's verdict against the reference arm of its scenario."""
    reference_arm = plan.decision_rule.reference_arm
    references = {s.scenario_id: s for s in combined.cells if s.arm == reference_arm}
    result = {}
    for summary in combined.cells:
        if summary.arm == reference_arm:
            result[summary.cell_id] = "reference"
        elif summary.scenario_id not in references:
            result[summary.cell_id] = "no reference"
        else:
            result[summary.cell_id] = compare_cells(plan, references[summary.scenario_id], summary)
    return result


def _ms(value: float | None) -> str:
    if value is None:
        return "-"
    return "inf" if math.isinf(value) else f"{value:.1f}"


def markdown(plan: Plan, combined: Combined) -> str:
    """The per-cell table every report starts from; unsummarized cells are listed after it."""
    kind = {True: "Simulated", False: "Measured", None: "No"}[combined.simulation]
    lines = [
        f"Evaluation plan {plan.plan_id} ({plan.status}), sha256 {plan.sha256}.",
        f"{kind} results: {combined.runs} runs. Latencies in ms, nearest-rank quantiles; a failed "
        "or OOM request counts as inf.",
    ]
    if combined.simulation:
        lines.append("Simulated: no number here was measured on hardware.")
    lines += [
        "",
        "| cell | repeats | requests | failed | OOM | e2e p95 | e2e p95 by repeat | e2e median "
        f"| TTFT p95 | TPOT median | tokens/s median | vs {plan.decision_rule.reference_arm} |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    verdict = verdicts(plan, combined)
    for s in combined.cells:
        by_repeat = ", ".join(_ms(v) for v in s.end_to_end_ms_p95_by_repeat)
        lines.append(
            f"| {s.cell_id} | {s.repeats}/{plan.repeats_per_cell} | {s.measured_requests} "
            f"| {s.failed} | {s.oom} | {_ms(s.end_to_end_ms_p95)} | {by_repeat} "
            f"| {_ms(s.end_to_end_ms_median)} | {_ms(s.ttft_ms_p95)} "
            f"| {_ms(s.observed_tpot_ms_median)} | {s.throughput_tokens_per_s_median:.1f} "
            f"| {verdict[s.cell_id]} |"
        )
    lines += [
        "",
        f"Missing cells ({len(combined.missing)}): {', '.join(combined.missing) or 'none'}",
        f"Incomplete cells ({len(combined.incomplete)}): "
        f"{', '.join(combined.incomplete) or 'none'}",
    ]
    return "\n".join(lines)


def _finite(value: Any) -> Any:
    # Strict JSON has no Infinity; an infinite quantile (a failed request) is written as null.
    if isinstance(value, float) and math.isinf(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def to_json(plan: Plan, combined: Combined) -> dict[str, Any]:
    """The summary file's content; infinite quantiles become null, the failure counts say why."""
    verdict = verdicts(plan, combined)
    return {
        "schema_version": VERSION,
        "simulation": combined.simulation,
        "plan_id": plan.plan_id,
        "plan_sha256": plan.sha256,
        "plan_status": plan.status,
        "quantile_method": "nearest_rank",
        "runs": combined.runs,
        "cells": [
            {
                **{k: _finite(v) for k, v in asdict(s).items()},
                "verdict_vs_reference": verdict[s.cell_id],
            }
            for s in combined.cells
        ],
        "missing_cells": list(combined.missing),
        "incomplete_cells": list(combined.incomplete),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine run records under a predeclared evaluation plan (spec 0003)."
    )
    parser.add_argument("--plan", required=True, help="the plan file the records name")
    parser.add_argument("--output", help="also write the JSON summary here (never overwritten)")
    parser.add_argument("records", nargs="+", help="one JSON record per run")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
    except ValueError as exc:
        print(f"invalid plan: {exc}", file=sys.stderr)
        return 1
    try:
        results = [result_from_bytes(Path(p).read_bytes(), p, plan) for p in args.records]
        combined = combine(plan, results)
    except ValueError as exc:
        print(f"invalid results: {exc}", file=sys.stderr)
        return 1
    print(markdown(plan, combined))
    if args.output:
        text = json.dumps(to_json(plan, combined), indent=2, allow_nan=False) + "\n"
        try:
            with open(args.output, "x") as f:
                f.write(text)
        except FileExistsError:
            print(f"refusing to overwrite {args.output}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
