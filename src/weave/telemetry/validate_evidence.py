"""Strict acceptance validator for successful single-process Phase 1 evidence."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def validate(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    s = json.loads((root / "run_summary.json").read_text())
    w = json.loads((root / "workload.json").read_text())
    rows = list(csv.DictReader((root / "trace.csv").open()))
    require(s["schema_version"] == "2.0.0" and s["status"] == "success", "run did not succeed")
    require(s["simulation"] is False and s["source"] == "measured", "not measured evidence")
    require(bool(rows), "trace is empty")
    require([int(r["sequence"]) for r in rows] == list(range(len(rows))), "sequence gap/duplicate")
    times = [int(r["monotonic_ns"]) for r in rows]
    require(times == sorted(times), "monotonic timestamp regression")
    for r in rows:
        for field in (
            "run_id",
            "device_id",
            "workload_id",
            "model_revision",
            "policy",
            "partition_version",
        ):
            require(r[field] == s[field], f"identity mismatch: {field}")
        require(
            r["schema_version"] == "2.0.0" and r["source"] == "measured",
            "trace version/source mismatch",
        )
        for field in (
            "process_rss_bytes",
            "system_available_bytes",
            "mlx_active_bytes",
            "mlx_cache_bytes",
            "mlx_peak_bytes",
        ):
            require(r[field] == "" or int(r[field]) >= 0, f"negative memory: {field}")
    global_events = [r["event"] for r in rows if not r["request_id"]]
    require(
        global_events == ["process_start", "model_load_start", "model_load_end", "process_end"],
        "bad process lifecycle",
    )
    load_end = next(int(r["monotonic_ns"]) for r in rows if r["event"] == "model_load_end")
    process_end = int(rows[-1]["monotonic_ns"])
    groups = defaultdict(list)
    for r in rows:
        if r["request_id"]:
            groups[r["request_id"]].append(r)
    requests = s["requests"]
    require(len(requests) == w["warmup_requests"] + w["measured_requests"], "wrong request count")
    require(len({q["request_id"] for q in requests}) == len(requests), "duplicate summary request")
    require(set(groups) == {q["request_id"] for q in requests}, "request IDs differ")
    require(
        sum(q["request_kind"] == "measured" for q in requests) == w["measured_requests"],
        "wrong measured count",
    )
    previous_end = -1
    for q in requests:
        rs = groups[q["request_id"]]
        events = [r["event"] for r in rs]
        require(
            set(events)
            <= {"request_start", "first_token", "token", "first_visible_text", "request_end"},
            "unknown request event",
        )
        for name in ("request_start", "first_token", "request_end", "first_visible_text"):
            require(events.count(name) == 1, f"{q['request_id']}: missing/duplicate {name}")
        require(
            events[0] == "request_start" and events[-1] == "request_end", "bad request boundaries"
        )
        require(all(r["request_kind"] == q["request_kind"] for r in rs), "warmup/measurement mixed")
        by_event = {r["event"]: r for r in rs if r["event"] != "token"}
        start, first, end = (
            int(by_event[x]["monotonic_ns"])
            for x in ("request_start", "first_token", "request_end")
        )
        require(
            max(previous_end, load_end) <= start <= first <= end <= process_end,
            "request order/overlap",
        )
        previous_end = end
        token_counts = [int(r["token_count"]) for r in rs if r["event"] == "token"]
        mode = s.get("token_trace", "full")
        require(mode in {"full", "boundary"}, "unknown token trace mode")
        expected_counts = list(range(1, q["generation_tokens"] + 1)) if mode == "full" else []
        require(token_counts == expected_counts, "token counter gap/duplicate or wrong trace mode")
        require(
            bool(q["generated_text"].strip()) and q["status"] == "success", "empty/failed output"
        )
        require(0 < q["prompt_tokens"] <= w["prompt_max_tokens"], "prompt outside envelope")
        require(
            type(q["generation_tokens"]) is int
            and 0 < q["generation_tokens"] <= w["max_new_tokens"],
            "generation count outside envelope",
        )
        for boundary in ("request_start", "request_end"):
            for field in (
                "process_rss_bytes",
                "system_available_bytes",
                "mlx_active_bytes",
                "mlx_cache_bytes",
                "mlx_peak_bytes",
            ):
                require(by_event[boundary][field] != "", f"missing boundary memory: {field}")
        for key, expected in (
            ("ttft_ms", (first - start) / 1e6),
            ("end_to_end_ms", (end - start) / 1e6),
        ):
            require(
                math.isfinite(q[key]) and abs(q[key] - expected) < 1e-6,
                f"summary/trace mismatch: {key}",
            )
        require(q["request_peak_mlx_bytes"] > 0, "missing MLX peak")
    return {
        "run_id": s["run_id"],
        "trace_rows": len(rows),
        "requests": len(requests),
        "status": "PASS",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    print(json.dumps(validate(p.parse_args().directory), indent=2))
