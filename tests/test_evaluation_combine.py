"""Spec 0003: run records are combined into per-cell summaries only when they belong to the plan.

Every record summarized here is simulated (SIMULATED- run IDs, simulation true) and every number
is made up for the test. A record labelled measured appears only to prove it is refused.
"""

import copy
import json
import math
from pathlib import Path

import pytest

from weave.evaluation import combine
from weave.evaluation import plan as evaluation_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/plans/evaluation-plan-v1.json"
INF = math.inf


@pytest.fixture
def plan() -> evaluation_plan.Plan:
    return evaluation_plan.load_plan(PLAN)


def request(i: int, e2e, kind: str = "measured") -> dict:
    """One Phase 1-shaped request item; e2e "failed" or "oom" makes it a failed request."""
    if e2e in ("failed", "oom"):
        return {"request_id": f"req-{i:04d}", "request_kind": kind, "status": e2e}
    return {
        "request_id": f"req-{i:04d}",
        "request_kind": kind,
        "status": "success",
        "generation_tokens": 10,
        "ttft_ms": e2e / 4,
        "end_to_end_ms": e2e,
        "observed_tpot_ms": 2.0,
        "generated_text": "simulated",  # extra fields pass, as in a Phase 1 summary
    }


def record(plan, arm="equal_split", repeat=1, e2e=None, scenario="short-c1", **overrides) -> dict:
    """A simulated run record: 1 warmup and 20 measured requests (100..119 ms by default)."""
    e2e = list(range(100, 120)) if e2e is None else e2e
    raw = {
        "schema_version": "2.0.0",
        "source": "simulated",
        "simulation": True,
        "run_id": f"SIMULATED-{scenario}-{arm}-r{repeat}",
        "plan_id": plan.plan_id,
        "plan_sha256": plan.sha256,
        "scenario_id": scenario,
        "arm": arm,
        "repeat": repeat,
        "repeat_wall_ms": 4000.0,
        "requests": [request(0, 500.0, "warmup")]
        + [request(i + 1, value) for i, value in enumerate(e2e)],
    }
    raw.update(overrides)
    return raw


def read(plan, raw, name="run.json") -> combine.RunResult:
    return combine.result_from_bytes(json.dumps(raw).encode(), name, plan)


def five(plan, arm="equal_split", offset=0, scenario="short-c1") -> list[combine.RunResult]:
    """Five repeats whose measured latencies are offset + 100..119 ms shifted by the repeat."""
    return [
        read(plan, record(plan, arm, r, [offset + v + r for v in range(100, 120)], scenario))
        for r in range(1, 6)
    ]


def test_nearest_rank_quantiles():
    values = list(range(1, 21))
    assert combine.nearest_rank(values, 0.95) == 19  # not the maximum once n reaches 20
    assert combine.nearest_rank(values, 0.5) == 10
    assert combine.nearest_rank(list(range(1, 101)), 0.95) == 95
    assert combine.nearest_rank(list(range(1, 20)), 0.95) == 19  # n = 19: p95 is the maximum
    assert combine.nearest_rank([3.0, INF, 1.0], 0.5) == 3.0
    with pytest.raises(ValueError):
        combine.nearest_rank([], 0.95)


def test_one_complete_cell_is_summarized(plan):
    combined = combine.combine(plan, five(plan))
    (cell,) = combined.cells
    assert cell.cell_id == "short-c1/equal_split"
    assert (cell.repeats, cell.measured_requests, cell.successful) == (5, 100, 100)
    assert (cell.failed, cell.oom, cell.complete) == (0, 0, True)
    # Repeat r has latencies 100 + r .. 119 + r; its p95 (19th of 20) is 118 + r.
    assert cell.end_to_end_ms_p95_by_repeat == (119, 120, 121, 122, 123)
    pooled = sorted(v + r for r in range(1, 6) for v in range(100, 120))
    assert cell.end_to_end_ms_p95 == pooled[94]
    assert cell.end_to_end_ms_median == pooled[49]
    assert cell.ttft_ms_p95 == pooled[94] / 4
    assert cell.observed_tpot_ms_median == 2.0
    # 20 successful requests x 10 tokens in 4000 ms.
    assert cell.throughput_tokens_per_s_median == 50.0
    assert combined.simulation is True


def test_warmup_requests_never_enter_the_numbers(plan):
    raw = record(plan)
    raw["requests"][0]["end_to_end_ms"] = 1e9
    cell = combine.combine(plan, [read(plan, raw)]).cells[0]
    assert cell.measured_requests == 20
    assert cell.end_to_end_ms_p95 == 118


def test_a_failed_request_counts_as_infinitely_slow(plan):
    one_failure = read(plan, record(plan, e2e=list(range(100, 119)) + ["oom"]))
    cell = combine.combine(plan, [one_failure]).cells[0]
    assert (cell.successful, cell.failed, cell.oom) == (19, 0, 1)
    assert cell.end_to_end_ms_p95 == 118  # one failure in 20 sits above the 19th value
    two_failures = read(plan, record(plan, e2e=list(range(100, 118)) + ["failed", "oom"]))
    cell = combine.combine(plan, [two_failures]).cells[0]
    assert cell.end_to_end_ms_p95 == INF  # 10% failed: the p95 is a failure
    assert cell.throughput_tokens_per_s_median == 18 * 10 / 4.0


def test_missing_and_incomplete_cells_are_reported(plan):
    combined = combine.combine(plan, five(plan)[:3])
    assert combined.incomplete == ("short-c1/equal_split",)
    assert len(combined.missing) == 53
    assert combined.missing[0] == "short-c1/best_single_device"
    assert not combined.cells[0].complete


def test_cells_come_out_in_plan_order(plan):
    results = five(plan, "memory_proportional") + five(plan, "best_single_device")
    combined = combine.combine(plan, results)
    assert [c.arm for c in combined.cells] == ["best_single_device", "memory_proportional"]


def _cell(plan, arm, offset, repeats=5):
    return combine.combine(plan, five(plan, arm, offset)[:repeats]).cells[0]


def test_a_clear_win_beyond_the_margin_is_a_finding(plan):
    reference = _cell(plan, "profiled_static", 0)  # per-repeat p95 119..123
    faster = _cell(plan, "equal_split", -50)  # per-repeat p95 69..73
    assert combine.compare_cells(plan, reference, faster) == combine.BETTER
    assert combine.compare_cells(plan, faster, reference) == combine.WORSE


def test_overlapping_repeat_ranges_are_no_finding(plan):
    reference = _cell(plan, "profiled_static", 0)  # 119..123
    close = _cell(plan, "equal_split", -3)  # 116..120 overlaps
    assert combine.compare_cells(plan, reference, close) == combine.NO_FINDING


def test_a_gap_smaller_than_the_margin_is_no_finding(plan):
    reference = _cell(plan, "profiled_static", 0)  # pooled p95 near 122
    slightly = _cell(plan, "equal_split", -5)  # ranges do not overlap, but under 5% faster
    assert max(slightly.end_to_end_ms_p95_by_repeat) < min(reference.end_to_end_ms_p95_by_repeat)
    assert combine.compare_cells(plan, reference, slightly) == combine.NO_FINDING


def test_incomplete_cells_get_no_verdict(plan):
    reference = _cell(plan, "profiled_static", 0, repeats=4)
    faster = _cell(plan, "equal_split", -50)
    assert combine.compare_cells(plan, reference, faster) == combine.INCOMPLETE


def test_two_cells_that_both_fail_are_no_finding(plan):
    failing = ["failed"] * 20
    results = [read(plan, record(plan, "profiled_static", r, failing)) for r in range(1, 6)]
    results += [read(plan, record(plan, "equal_split", r, failing)) for r in range(1, 6)]
    reference, candidate = sorted(
        combine.combine(plan, results).cells, key=lambda c: c.arm != "profiled_static"
    )
    assert combine.compare_cells(plan, reference, candidate) == combine.NO_FINDING


def test_cells_from_different_scenarios_are_not_compared(plan):
    reference = _cell(plan, "profiled_static", 0)
    other = combine.combine(plan, five(plan, "equal_split", 0, "long-c1")).cells[0]
    with pytest.raises(ValueError, match="different scenarios"):
        combine.compare_cells(plan, reference, other)


def _mutate(path, value):
    def edit(raw):
        target = raw
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return edit


_DELETE = object()


def _drop_measured(raw):
    raw["requests"].pop()


def _extra_warmup(raw):
    raw["requests"].insert(0, request(99, 500.0, "warmup"))


def _repeat_request_id(raw):
    raw["requests"][2]["request_id"] = raw["requests"][1]["request_id"]


INVALID_RECORDS = [
    ("version", _mutate(["schema_version"], "1.0.0"), "schema_version"),
    ("source", _mutate(["source"], "replayed"), "source"),
    ("source disagrees", _mutate(["simulation"], False), "simulation"),
    ("simulated id", _mutate(["run_id"], "20261125T000000Z-run"), "run_id"),
    ("plan id", _mutate(["plan_id"], "evaluation-plan-v0"), "plan_id"),
    ("plan hash", _mutate(["plan_sha256"], "0" * 64), "plan_sha256"),
    ("scenario", _mutate(["scenario_id"], "huge-c9"), "scenario_id"),
    ("arm", _mutate(["arm"], "random_split"), "arm"),
    ("repeat zero", _mutate(["repeat"], 0), "repeat"),
    ("repeat too high", _mutate(["repeat"], 6), "repeat"),
    ("wall time", _mutate(["repeat_wall_ms"], 0), "repeat_wall_ms"),
    ("wall time bool", _mutate(["repeat_wall_ms"], True), "repeat_wall_ms"),
    ("no requests", _mutate(["requests"], {}), "requests"),
    ("dropped request", _drop_measured, "requests: expected 20 measured"),
    ("extra warmup", _extra_warmup, "requests: expected 1 warmup"),
    ("request id twice", _repeat_request_id, "requests[2].request_id"),
    ("request kind", _mutate(["requests", 1, "request_kind"], "probe"), "requests[1].request_kind"),
    ("request status", _mutate(["requests", 1, "status"], "ok"), "requests[1].status"),
    ("no ttft", _mutate(["requests", 1, "ttft_ms"], _DELETE), "requests[1].ttft_ms"),
    ("ttft after end", _mutate(["requests", 1, "ttft_ms"], 1e6), "requests[1].ttft_ms"),
    ("nan latency", _mutate(["requests", 1, "end_to_end_ms"], "NaN"), "requests[1].end_to_end"),
    ("tokens", _mutate(["requests", 1, "generation_tokens"], 0), "requests[1].generation_tokens"),
    ("tpot", _mutate(["requests", 1, "observed_tpot_ms"], -1), "requests[1].observed_tpot_ms"),
]


@pytest.mark.parametrize("name,edit,field", INVALID_RECORDS, ids=[c[0] for c in INVALID_RECORDS])
def test_invalid_record_is_refused_naming_the_file_and_field(plan, name, edit, field):
    raw = record(plan)
    edit(raw)
    pattern = "^run.json: " + field.replace("[", r"\[").replace("]", r"\]")
    with pytest.raises(ValueError, match=pattern):
        read(plan, raw)


def test_record_that_is_not_json_names_the_file(plan):
    with pytest.raises(ValueError, match="^broken.json: not valid JSON"):
        combine.result_from_bytes(b"{", "broken.json", plan)


def test_the_same_cell_and_repeat_twice_is_refused(plan):
    first = read(plan, record(plan))
    again = read(plan, record(plan, run_id="SIMULATED-other-run"))
    with pytest.raises(ValueError, match="short-c1/equal_split repeat 1 appears twice"):
        combine.combine(plan, [first, again])


def test_the_same_run_twice_is_refused(plan):
    first = read(plan, record(plan))
    again = read(plan, record(plan, repeat=2, run_id=first.run_id))
    with pytest.raises(ValueError, match="appears twice"):
        combine.combine(plan, [first, again])


def _measured(plan, **overrides):
    raw = record(plan, source="measured", simulation=False, run_id="20261125T000000Z-run")
    raw.update(overrides)
    return raw


def test_simulated_and_measured_records_are_never_combined(plan):
    frozen_raw = json.loads(PLAN.read_text())
    frozen_raw["status"] = "frozen"
    for entry in frozen_raw["policies"] + frozen_raw["ablations"]:
        entry["available"] = True
    for scenario in frozen_raw["scenarios"]:
        scenario["pilot_required"] = False
    frozen = evaluation_plan.plan_from_bytes(json.dumps(frozen_raw).encode(), "frozen")
    simulated = read(frozen, record(frozen))
    labelled_measured = read(frozen, _measured(frozen, repeat=2))
    with pytest.raises(ValueError, match="simulated and measured"):
        combine.combine(frozen, [simulated, labelled_measured])


def test_measured_records_need_a_frozen_plan(plan):
    labelled_measured = read(plan, _measured(plan))
    with pytest.raises(ValueError, match="frozen plan"):
        combine.combine(plan, [labelled_measured])


def test_no_records_combine_to_all_missing(plan):
    combined = combine.combine(plan, [])
    assert combined.cells == ()
    assert len(combined.missing) == 54
    assert combined.simulation is None


def test_markdown_names_the_plan_and_the_verdicts(plan):
    combined = combine.combine(plan, five(plan, "profiled_static") + five(plan, "equal_split", -50))
    text = combine.markdown(plan, combined)
    assert plan.sha256 in text
    assert "Simulated" in text
    assert "| short-c1/equal_split |" in text
    assert "BETTER" in text
    assert "reference" in text
    assert "Missing cells (52)" in text


def write_records(tmp_path, plan, results_raw):
    paths = []
    for i, raw in enumerate(results_raw):
        path = tmp_path / f"run-{i}.json"
        path.write_text(json.dumps(raw))
        paths.append(str(path))
    return paths


def test_command_prints_and_writes_nothing(tmp_path, plan, capsys):
    paths = write_records(tmp_path, plan, [record(plan, repeat=r) for r in range(1, 6)])
    before = sorted(p.name for p in tmp_path.iterdir())
    assert combine.main(["--plan", str(PLAN), *paths]) == 0
    assert "| short-c1/equal_split |" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_command_writes_the_summary_on_request(tmp_path, plan):
    failing = record(plan, repeat=1, e2e=["failed"] * 20)
    paths = write_records(tmp_path, plan, [failing])
    out = tmp_path / "summary.json"
    assert combine.main(["--plan", str(PLAN), "--output", str(out), *paths]) == 0
    summary = json.loads(out.read_text())  # strict JSON: no Infinity
    assert summary["plan_sha256"] == plan.sha256
    assert summary["simulation"] is True
    (cell,) = summary["cells"]
    assert cell["end_to_end_ms_p95"] is None  # infinite: every request failed
    assert cell["failed"] == 20
    assert summary["incomplete_cells"] == ["short-c1/equal_split"]


def test_command_refuses_to_overwrite(tmp_path, plan, capsys):
    paths = write_records(tmp_path, plan, [record(plan)])
    out = tmp_path / "summary.json"
    out.write_text("earlier summary")
    assert combine.main(["--plan", str(PLAN), "--output", str(out), *paths]) == 1
    assert out.read_text() == "earlier summary"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_command_reports_a_bad_record_and_exits_nonzero(tmp_path, plan, capsys):
    raw = record(plan)
    raw["plan_sha256"] = "0" * 64
    paths = write_records(tmp_path, plan, [raw])
    assert combine.main(["--plan", str(PLAN), *paths]) == 1
    assert "invalid results" in capsys.readouterr().err


def test_record_fixture_is_not_mutated_between_reads(plan):
    raw = record(plan)
    snapshot = copy.deepcopy(raw)
    read(plan, raw)
    assert raw == snapshot
