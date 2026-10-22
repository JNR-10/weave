"""Fake production-shaped rows in pytest temp dirs; never experiment evidence."""

import csv
import json

import pytest

from weave.telemetry.baseline import FIELDS
from weave.telemetry.validate_evidence import validate


def write_fake(root):
    identity = {
        "run_id": "unit-test-only",
        "device_id": "fake",
        "workload_id": "test",
        "model_revision": "a" * 40,
        "policy": "single_device",
        "partition_version": "test",
    }
    names = [
        "process_start",
        "model_load_start",
        "model_load_end",
        "request_start",
        "first_token",
        "token",
        "first_visible_text",
        "request_end",
        "request_start",
        "first_token",
        "token",
        "first_visible_text",
        "request_end",
        "process_end",
    ]
    rows = []
    for i, name in enumerate(names):
        row = dict.fromkeys(FIELDS, "")
        row.update(
            identity,
            schema_version="2.0.0",
            sequence=i,
            source="measured",
            event=name,
            monotonic_ns=i * 1000000,
        )
        if 3 <= i <= 7 or 8 <= i <= 12:
            row["request_id"] = "warmup" if i <= 7 else "measured"
            row["request_kind"] = row["request_id"]
        if name == "token":
            row["token_count"] = 1
        if name in {"request_start", "request_end"}:
            for field in (
                "process_rss_bytes",
                "system_available_bytes",
                "mlx_active_bytes",
                "mlx_cache_bytes",
                "mlx_peak_bytes",
            ):
                row[field] = 1
        rows.append(row)
    summary = {
        **identity,
        "schema_version": "2.0.0",
        "source": "measured",
        "simulation": False,
        "status": "success",
        "requests": [
            {
                "request_id": k,
                "request_kind": k,
                "status": "success",
                "generation_tokens": 1,
                "prompt_tokens": 3,
                "ttft_ms": 1.0,
                "end_to_end_ms": 4.0,
                "generated_text": "test",
                "request_peak_mlx_bytes": 1,
            }
            for k in ("warmup", "measured")
        ],
    }
    (root / "run_summary.json").write_text(json.dumps(summary))
    (root / "workload.json").write_text(
        json.dumps(
            {
                "warmup_requests": 1,
                "measured_requests": 1,
                "prompt_max_tokens": 512,
                "max_new_tokens": 64,
            }
        )
    )
    with (root / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary, rows


def test_valid_fake_evidence(tmp_path):
    write_fake(tmp_path)
    assert validate(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize(
    "change", ["metric", "failure", "identity", "missing_token", "zero_tokens", "missing_memory"]
)
def test_corruption_fails(tmp_path, change):
    summary, rows = write_fake(tmp_path)
    if change == "metric":
        summary["requests"][1]["ttft_ms"] = 7
    elif change == "failure":
        summary["status"] = "failed"
    elif change == "identity":
        rows[4]["device_id"] = "wrong-device"
    elif change == "zero_tokens":
        summary["requests"][1]["generation_tokens"] = 0
    elif change == "missing_memory":
        rows[7]["mlx_peak_bytes"] = ""
    else:
        rows[4]["event"] = "unexpected"
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    with (tmp_path / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        validate(tmp_path)
