"""Strict acceptance validator for successful single-process Phase 1 evidence."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from weave.profiler.model_profile import profile_id


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def close(value: Any, expected: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and abs(value - expected) < 1e-6


def validate(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    s = json.loads((root / "run_summary.json").read_text())
    w = json.loads((root / "workload.json").read_text())
    lock = json.loads((root / "model-lock.json").read_text())
    rows = list(csv.DictReader((root / "trace.csv").open()))
    require(s["schema_version"] == "2.0.0" and s["status"] == "success", "run did not succeed")
    require(s["simulation"] is False and s["source"] == "measured", "not measured evidence")
    # resolve() so "." or a trailing "/." still names the run directory.
    require(s["run_id"] == root.resolve().name, "run_id differs from the run directory name")
    # The copied inputs define what ran, and counts below come from the workload copy, so an
    # edit after the run must not pass silently.
    for name, key in (
        ("workload.json", "workload_sha256"),
        ("model-lock.json", "model_lock_sha256"),
    ):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        require(digest == s.get(key), f"{name} differs from its recorded hash")
    require(
        w.get("schema_version") == "2.0.0" and w.get("workload_id") == s["workload_id"],
        "workload copy does not match the run",
    )
    require(
        lock.get("model_id") == s["model_id"] and lock.get("model_revision") == s["model_revision"],
        "model lock copy does not match the run",
    )
    require(
        isinstance(s.get("git_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", s["git_commit"]) is not None,
        "missing source commit",
    )
    require(s.get("git_dirty") is False, "run used uncommitted source; commit, then rerun")
    env = s.get("environment") or {}
    packages = env.get("packages") or {}
    require(
        env.get("system") == "Darwin"
        and env.get("machine") == "arm64"
        and bool(packages.get("mlx"))
        and bool(packages.get("mlx-lm")),
        "environment not recorded as native Apple Silicon with MLX",
    )
    # The measured profile is the hand-off to the estimator, so it must describe this run's model.
    require((root / "model_profile.json").is_file(), "model_profile.json is missing")
    profile = json.loads((root / "model_profile.json").read_text())
    require(isinstance(profile, dict), "model profile is not a JSON object")
    require(
        profile.get("schema_version") == "2.0.0"
        and profile.get("source") == "measured"
        and profile.get("measurement_kind") == "materialized_parameter_nbytes"
        and profile.get("model_id") == s["model_id"]
        and profile.get("model_revision") == s["model_revision"]
        and profile.get("config_sha256") == lock.get("config_sha256")
        and profile.get("source_run_id") == s["run_id"],
        "model profile does not describe this run's model",
    )
    layers = profile.get("layer_weight_bytes")
    tensors = profile.get("replicated_tensor_bytes")
    require(
        isinstance(layers, list)
        and len(layers) == (lock.get("config") or {}).get("num_hidden_layers")
        and all(type(n) is int and n > 0 for n in layers)
        and isinstance(tensors, dict)
        and bool(tensors)
        and all(type(n) is int and n > 0 for n in tensors.values())
        and profile.get("replicated_weight_bytes") == sum(tensors.values()),
        "model profile byte tallies incomplete",
    )
    require(
        profile.get("profile_id") == profile_id(profile) == s.get("profile_id"),
        "model profile ID does not match its content or the summary",
    )
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
    # Request order is checked against time below, so this also fixes warmups before measurement.
    require(
        [q["request_kind"] for q in requests]
        == ["warmup"] * w["warmup_requests"] + ["measured"] * w["measured_requests"],
        "warmup/measured counts or order differ from the workload",
    )
    require(len({q["request_id"] for q in requests}) == len(requests), "duplicate summary request")
    require(set(groups) == {q["request_id"] for q in requests}, "request IDs differ")
    mode = s.get("token_trace", "full")
    require(mode in {"full", "boundary"}, "unknown token trace mode")
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
        visible = int(by_event["first_visible_text"]["monotonic_ns"])
        require(start <= visible <= end, "first visible text outside its request")
        # Boundary mode has no token rows, so these two rows are its only token-count evidence.
        require(by_event["first_token"]["token_count"] == "1", "first_token is not token 1")
        require(
            by_event["request_end"]["token_count"] == str(q["generation_tokens"]),
            "request_end token count differs from summary",
        )
        token_rows = [r for r in rs if r["event"] == "token"]
        token_counts = [int(r["token_count"]) for r in token_rows]
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
            ("first_visible_text_ms", (visible - start) / 1e6),
        ):
            require(close(q.get(key), expected), f"summary/trace mismatch: {key}")
        # The last token time is only in the trace in full mode; boundary mode cannot check this.
        if mode == "full":
            n = q["generation_tokens"]
            last = int(token_rows[-1]["monotonic_ns"])
            tpot = q.get("observed_tpot_ms")
            require(
                tpot is None if n == 1 else close(tpot, (last - first) / (n - 1) / 1e6),
                "summary/trace mismatch: observed_tpot_ms",
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
