"""The predeclared evaluation plan (spec 0003): read it strictly and list the cells it implies.

The plan fixes, before any comparison run, which arms run on which workloads, how often, by which
metric and under which rule. Results name the plan by the SHA-256 of its bytes as read, so a plan
edited after a run (even reformatted) can never be matched to that run's results.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
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
from weave.scheduler.candidates import VERSION

STATUSES = ("draft", "frozen")
# The live signals of the adaptive policy (experiments/README.md ladder); one ablation turns off one.
SIGNALS = ("live_memory_headroom", "context_length", "concurrency", "observed_stage_timings")
QUANTILE_METHOD = "nearest_rank"
FAILED_REQUEST_LATENCY = "infinite"
PRIMARY_METRIC = "end_to_end_ms_p95"
SECONDARY_METRICS = (
    "end_to_end_ms_median",
    "ttft_ms_p95",
    "ttft_ms_median",
    "observed_tpot_ms_median",
    "throughput_tokens_per_s_median",
    "failed_requests",
    "oom_requests",
)
MEMORY_PRESSURE = ("none", "induced")
COMPUTE_CONTENTION = ("none", "induced")
LINK_CONDITION = ("baseline", "degraded")
# With nearest rank the p95 of n values is the ceil(0.95 n)-th smallest, which is the maximum for
# every n below 20. Fewer requests per repeat would make "p95" mean "slowest request".
MIN_MEASURED_REQUESTS = 20
# Cell IDs join a scenario and an arm with "/", so neither may contain one.
_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
_REVISION = re.compile(r"[0-9a-f]{40}")

_TOP_FIELDS = (
    "schema_version",
    "plan_id",
    "status",
    "model_id",
    "model_revision",
    "repeats_per_cell",
    "warmup_requests_per_repeat",
    "measured_requests_per_repeat",
    "quantile_method",
    "failed_request_latency",
    "primary_metric",
    "secondary_metrics",
    "decision_rule",
    "policies",
    "ablations",
    "scenarios",
)
_RULE_FIELDS = ("reference_arm", "min_relative_difference", "require_non_overlapping_repeat_ranges")
_POLICY_FIELDS = ("policy", "label", "available", "may_use_multiple_macs", "spec")
_ABLATION_FIELDS = ("ablation_id", "base_policy", "disables_signal", "available")
_SCENARIO_FIELDS = (
    "scenario_id",
    "purpose",
    "prompt_tokens_upper_bound",
    "max_new_tokens",
    "concurrency",
    "memory_pressure",
    "compute_contention",
    "link_condition",
    "needs_two_macs",
    "pilot_required",
)


@dataclass(frozen=True)
class Policy:
    policy: str
    label: str
    available: bool
    may_use_multiple_macs: bool
    spec: str | None


@dataclass(frozen=True)
class Ablation:
    """The base policy with exactly one live signal turned off."""

    ablation_id: str
    base_policy: str
    disables_signal: str
    available: bool


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    purpose: str
    prompt_tokens_upper_bound: int
    max_new_tokens: int
    concurrency: int
    memory_pressure: str
    compute_contention: str
    link_condition: str
    needs_two_macs: bool
    pilot_required: bool


@dataclass(frozen=True)
class DecisionRule:
    """When one arm beats another on the primary metric; anything less is "no finding"."""

    reference_arm: str
    min_relative_difference: float
    require_non_overlapping_repeat_ranges: bool


@dataclass(frozen=True)
class Plan:
    plan_id: str
    status: str
    model_id: str
    model_revision: str
    repeats_per_cell: int
    warmup_requests_per_repeat: int
    measured_requests_per_repeat: int
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    decision_rule: DecisionRule
    policies: tuple[Policy, ...]
    ablations: tuple[Ablation, ...]
    scenarios: tuple[Scenario, ...]
    sha256: str

    @property
    def arms(self) -> tuple[str, ...]:
        """Policies in declared order, then ablations: the column order of every report."""
        return tuple(p.policy for p in self.policies) + tuple(a.ablation_id for a in self.ablations)


@dataclass(frozen=True)
class Cell:
    """One scenario run under one arm; the unit that gets `repeats_per_cell` runs."""

    cell_id: str
    scenario_id: str
    arm: str
    needs_two_macs: bool
    available: bool


def _only(obj: dict[str, Any], allowed: Sequence[str], parent: str) -> None:
    # A typo in a predeclared plan must fail loudly, not leave a default in force.
    for key in obj:
        if key not in allowed:
            raise ValueError(f"{field_path(parent, key)}: unknown field")


def _identifier(obj: dict[str, Any], key: str, parent: str) -> str:
    value = text_field(obj, key, parent)
    if not _ID.fullmatch(value):
        raise ValueError(
            f"{field_path(parent, key)}: expected lowercase letters, digits, '-' or '_', "
            f"got {value!r}"
        )
    return value


def _choice(obj: dict[str, Any], key: str, parent: str, allowed: Sequence[str]) -> str:
    value = required(obj, key, parent)
    if value not in allowed:
        raise ValueError(
            f"{field_path(parent, key)}: expected one of {list(allowed)}, got {value!r}"
        )
    return value


def _non_negative_field(obj: dict[str, Any], key: str, parent: str) -> int:
    value = required(obj, key, parent)
    if type(value) is not int or value < 0:
        raise ValueError(
            f"{field_path(parent, key)}: expected a non-negative integer, got {value!r}"
        )
    return value


def _list(obj: dict[str, Any], key: str, parent: str, *, non_empty: bool) -> list[Any]:
    value = required(obj, key, parent)
    if not isinstance(value, list):
        raise ValueError(f"{field_path(parent, key)}: expected a list, got {type(value).__name__}")
    if non_empty and not value:
        raise ValueError(f"{field_path(parent, key)}: expected at least one entry")
    return value


def _parse_rule(raw: Any, arms: Sequence[str]) -> DecisionRule:
    parent = "decision_rule"
    rule = as_object(raw, parent)
    _only(rule, _RULE_FIELDS, parent)
    reference = text_field(rule, "reference_arm", parent)
    if reference not in arms:
        raise ValueError(f"{parent}.reference_arm: {reference!r} is not a declared arm")
    margin = required(rule, "min_relative_difference", parent)
    if type(margin) not in (int, float) or not 0 < margin < 1:
        raise ValueError(
            f"{parent}.min_relative_difference: expected a number between 0 and 1, got {margin!r}"
        )
    return DecisionRule(
        reference_arm=reference,
        min_relative_difference=float(margin),
        require_non_overlapping_repeat_ranges=bool_field(
            rule, "require_non_overlapping_repeat_ranges", parent
        ),
    )


def _parse_policy(raw: Any, parent: str) -> Policy:
    entry = as_object(raw, parent)
    _only(entry, _POLICY_FIELDS, parent)
    spec = required(entry, "spec", parent)
    if spec is not None and (not isinstance(spec, str) or not spec):
        raise ValueError(f"{parent}.spec: expected null or a non-empty string, got {spec!r}")
    return Policy(
        policy=_identifier(entry, "policy", parent),
        label=text_field(entry, "label", parent),
        available=bool_field(entry, "available", parent),
        may_use_multiple_macs=bool_field(entry, "may_use_multiple_macs", parent),
        spec=spec,
    )


def _parse_ablation(raw: Any, parent: str, policies: dict[str, Policy]) -> Ablation:
    entry = as_object(raw, parent)
    _only(entry, _ABLATION_FIELDS, parent)
    ablation_id = _identifier(entry, "ablation_id", parent)
    base = text_field(entry, "base_policy", parent)
    if base not in policies:
        raise ValueError(f"{parent}.base_policy: {base!r} is not a declared policy")
    available = bool_field(entry, "available", parent)
    if available and not policies[base].available:
        raise ValueError(
            f"{parent}.available: an ablation of {base!r} cannot run before {base!r} is available"
        )
    return Ablation(
        ablation_id=ablation_id,
        base_policy=base,
        disables_signal=_choice(entry, "disables_signal", parent, SIGNALS),
        available=available,
    )


def _parse_scenario(raw: Any, parent: str) -> Scenario:
    entry = as_object(raw, parent)
    _only(entry, _SCENARIO_FIELDS, parent)
    scenario = Scenario(
        scenario_id=_identifier(entry, "scenario_id", parent),
        purpose=text_field(entry, "purpose", parent),
        prompt_tokens_upper_bound=positive_field(entry, "prompt_tokens_upper_bound", parent),
        max_new_tokens=positive_field(entry, "max_new_tokens", parent),
        concurrency=positive_field(entry, "concurrency", parent),
        memory_pressure=_choice(entry, "memory_pressure", parent, MEMORY_PRESSURE),
        compute_contention=_choice(entry, "compute_contention", parent, COMPUTE_CONTENTION),
        link_condition=_choice(entry, "link_condition", parent, LINK_CONDITION),
        needs_two_macs=bool_field(entry, "needs_two_macs", parent),
        pilot_required=bool_field(entry, "pilot_required", parent),
    )
    if scenario.link_condition != "baseline" and not scenario.needs_two_macs:
        # A link exists only between two Macs, so a link scenario cannot run on one.
        raise ValueError(
            f"{parent}.needs_two_macs: a {scenario.link_condition} link needs two Macs"
        )
    return scenario


def _parse_secondary(raw: dict[str, Any]) -> tuple[str, ...]:
    metrics = _list(raw, "secondary_metrics", "", non_empty=False)
    seen: list[str] = []
    for i, metric in enumerate(metrics):
        where = f"secondary_metrics[{i}]"
        if metric not in SECONDARY_METRICS:
            raise ValueError(f"{where}: expected one of {list(SECONDARY_METRICS)}, got {metric!r}")
        if metric in seen:
            raise ValueError(f"{where}: {metric!r} is listed twice")
        seen.append(metric)
    return tuple(seen)


def _parse(raw: Any, sha256: str) -> Plan:
    plan = as_object(raw, "plan")
    _only(plan, _TOP_FIELDS, "")
    version = required(plan, "schema_version", "")
    if version != VERSION:
        raise ValueError(
            f"schema_version: unsupported {version!r}; migrate explicitly to {VERSION}"
        )
    revision = text_field(plan, "model_revision", "")
    if not _REVISION.fullmatch(revision):
        raise ValueError(f"model_revision: expected 40 lowercase hex characters, got {revision!r}")
    measured = positive_field(plan, "measured_requests_per_repeat", "")
    if measured < MIN_MEASURED_REQUESTS:
        raise ValueError(
            f"measured_requests_per_repeat: expected at least {MIN_MEASURED_REQUESTS}, so that the "
            f"nearest-rank p95 is not the slowest request; got {measured}"
        )
    status = _choice(plan, "status", "", STATUSES)
    _choice(plan, "quantile_method", "", (QUANTILE_METHOD,))
    _choice(plan, "failed_request_latency", "", (FAILED_REQUEST_LATENCY,))
    _choice(plan, "primary_metric", "", (PRIMARY_METRIC,))
    secondary = _parse_secondary(plan)

    policies: dict[str, Policy] = {}
    for i, raw_policy in enumerate(_list(plan, "policies", "", non_empty=True)):
        policy = _parse_policy(raw_policy, f"policies[{i}]")
        if policy.policy in policies:
            raise ValueError(f"policies[{i}].policy: {policy.policy!r} is declared twice")
        policies[policy.policy] = policy
    ablations: list[Ablation] = []
    for i, raw_ablation in enumerate(_list(plan, "ablations", "", non_empty=False)):
        where = f"ablations[{i}]"
        ablation = _parse_ablation(raw_ablation, where, policies)
        if ablation.ablation_id in policies or any(
            a.ablation_id == ablation.ablation_id for a in ablations
        ):
            raise ValueError(f"{where}.ablation_id: {ablation.ablation_id!r} is declared twice")
        if any(a.disables_signal == ablation.disables_signal for a in ablations):
            raise ValueError(
                f"{where}.disables_signal: {ablation.disables_signal!r} is already turned off "
                "by another ablation"
            )
        ablations.append(ablation)
    arms = list(policies) + [a.ablation_id for a in ablations]
    rule = _parse_rule(required(plan, "decision_rule", ""), arms)

    scenarios: list[Scenario] = []
    for i, raw_scenario in enumerate(_list(plan, "scenarios", "", non_empty=True)):
        scenario = _parse_scenario(raw_scenario, f"scenarios[{i}]")
        if any(s.scenario_id == scenario.scenario_id for s in scenarios):
            raise ValueError(
                f"scenarios[{i}].scenario_id: {scenario.scenario_id!r} is declared twice"
            )
        scenarios.append(scenario)

    if status == "frozen":
        # Counted runs start only against a frozen plan, so it must be runnable as written.
        for i, scenario in enumerate(scenarios):
            if scenario.pilot_required:
                raise ValueError(
                    f"scenarios[{i}].pilot_required: a frozen plan cannot wait for the pilot"
                )
        for i, policy in enumerate(policies.values()):
            if not policy.available:
                raise ValueError(
                    f"policies[{i}].available: a frozen plan cannot include an arm that is not "
                    "available"
                )
        for i, ablation in enumerate(ablations):
            if not ablation.available:
                raise ValueError(
                    f"ablations[{i}].available: a frozen plan cannot include an arm that is not "
                    "available"
                )

    return Plan(
        plan_id=_identifier(plan, "plan_id", ""),
        status=status,
        model_id=text_field(plan, "model_id", ""),
        model_revision=revision,
        repeats_per_cell=positive_field(plan, "repeats_per_cell", ""),
        warmup_requests_per_repeat=_non_negative_field(plan, "warmup_requests_per_repeat", ""),
        measured_requests_per_repeat=measured,
        primary_metric=PRIMARY_METRIC,
        secondary_metrics=secondary,
        decision_rule=rule,
        policies=tuple(policies.values()),
        ablations=tuple(ablations),
        scenarios=tuple(scenarios),
        sha256=sha256,
    )


def plan_from_bytes(data: bytes, source: str) -> Plan:
    """Parse a plan from the exact bytes whose SHA-256 results will name."""
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: not valid JSON ({exc})") from None
    return _parse(raw, hashlib.sha256(data).hexdigest())


def load_plan(path: str | Path) -> Plan:
    """Read and check a plan file."""
    return plan_from_bytes(Path(path).read_bytes(), str(path))


def cells(plan: Plan) -> tuple[Cell, ...]:
    """Every scenario under every arm, scenarios outer, arms in declared order."""
    multi = {p.policy: p.may_use_multiple_macs for p in plan.policies}
    available = {p.policy: p.available for p in plan.policies}
    for ablation in plan.ablations:
        multi[ablation.ablation_id] = multi[ablation.base_policy]
        available[ablation.ablation_id] = ablation.available
    return tuple(
        Cell(
            cell_id=f"{scenario.scenario_id}/{arm}",
            scenario_id=scenario.scenario_id,
            arm=arm,
            needs_two_macs=scenario.needs_two_macs or multi[arm],
            available=available[arm],
        )
        for scenario in plan.scenarios
        for arm in plan.arms
    )


def describe(plan: Plan, *, list_cells: bool = False) -> str:
    """The plan's identity and size, as `--check` prints it."""
    all_cells = cells(plan)
    runs = len(all_cells) * plan.repeats_per_cell
    per_run = plan.warmup_requests_per_repeat + plan.measured_requests_per_repeat
    waiting = [s.scenario_id for s in plan.scenarios if s.pilot_required]
    lines = [
        f"Plan {plan.plan_id} ({plan.status}), sha256 {plan.sha256}",
        f"Model {plan.model_id} at {plan.model_revision}",
        f"{len(all_cells)} cells ({len(plan.scenarios)} scenarios x {len(plan.arms)} arms), "
        f"{runs} runs, {runs * per_run} requests ({plan.warmup_requests_per_repeat} warmup + "
        f"{plan.measured_requests_per_repeat} measured per run)",
        f"Cells needing two Macs: {sum(c.needs_two_macs for c in all_cells)}; "
        f"cells whose arm is not available yet: {sum(not c.available for c in all_cells)}",
        f"Scenarios waiting for the pilot: {', '.join(waiting) if waiting else 'none'}",
    ]
    if plan.status == "draft":
        lines.append("Draft plan: measured results are refused until a frozen version replaces it.")
    if list_cells:
        lines.append("Cells:")
        lines.extend(
            f"  {c.cell_id}  {'two-Macs' if c.needs_two_macs else 'one-Mac'}  "
            f"{'available' if c.available else 'not-available'}"
            for c in all_cells
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a predeclared evaluation plan (spec 0003).")
    parser.add_argument("--check", required=True, metavar="PLAN", help="plan file to check")
    parser.add_argument("--cells", action="store_true", help="also list every cell")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.check)
    except ValueError as exc:
        print(f"invalid plan: {exc}", file=sys.stderr)
        return 1
    print(describe(plan, list_cells=args.cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
