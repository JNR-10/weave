"""Fake production-shaped rows in pytest temp dirs; never experiment evidence."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

from weave.profiler.model_profile import profile_from_tensor_bytes
from weave.telemetry import baseline
from weave.telemetry.baseline import FIELDS
from weave.telemetry.validate_evidence import validate

ROOT = Path(__file__).resolve().parents[1]
MEMORY = (
    "process_rss_bytes",
    "system_available_bytes",
    "mlx_active_bytes",
    "mlx_cache_bytes",
    "mlx_peak_bytes",
)
HASH_KEYS = {"workload.json": "workload_sha256", "model-lock.json": "model_lock_sha256"}


def save(root, summary, rows):
    (root / "run_summary.json").write_text(json.dumps(summary))
    with (root / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_fake(root):
    """Two requests of two tokens each, so TTFT, visible text and TPOT are all checkable."""
    inputs = {
        "workload.json": {
            "schema_version": "2.0.0",
            "workload_id": "test",
            "warmup_requests": 1,
            "measured_requests": 1,
            "prompt_max_tokens": 512,
            "max_new_tokens": 64,
        },
        "model-lock.json": {
            "schema_version": "2.0.0",
            "model_id": "fake/model",
            "model_revision": "a" * 40,
            "config_sha256": "e" * 64,
            "config": {"num_hidden_layers": 2},
        },
    }
    hashes = {}
    for name, data in inputs.items():
        raw = json.dumps(data).encode()
        (root / name).write_bytes(raw)
        hashes[HASH_KEYS[name]] = hashlib.sha256(raw).hexdigest()
    profile = profile_from_tensor_bytes(
        [("model.layers.0.w", 8), ("model.layers.1.w", 8), ("model.embed_tokens.weight", 4)],
        model_id="fake/model",
        model_revision="a" * 40,
        config_sha256="e" * 64,
        total_layers=2,
    )
    profile["source_run_id"] = root.name
    (root / "model_profile.json").write_text(json.dumps(profile))
    identity = {
        "run_id": root.name,
        "device_id": "fake",
        "workload_id": "test",
        "model_revision": "a" * 40,
        "policy": "single_device",
        "partition_version": "test",
    }

    def request(kind):
        return [
            ("request_start", kind, ""),
            ("first_token", kind, 1),
            ("token", kind, 1),
            ("first_visible_text", kind, 1),
            ("token", kind, 2),
            ("request_end", kind, 2),
        ]

    events = [
        ("process_start", "", ""),
        ("model_load_start", "", ""),
        ("model_load_end", "", ""),
        *request("warmup"),
        *request("measured"),
        ("process_end", "", ""),
    ]
    rows = []
    for i, (name, kind, count) in enumerate(events):
        row = dict.fromkeys(FIELDS, "")
        row.update(
            identity,
            schema_version="2.0.0",
            sequence=i,
            source="measured",
            event=name,
            monotonic_ns=i * 1000000,
            request_id=kind,
            request_kind=kind,
            token_count=count,
        )
        if name in {"request_start", "request_end"}:
            row.update(dict.fromkeys(MEMORY, 1))
        rows.append(row)
    summary = {
        **identity,
        **hashes,
        "model_id": "fake/model",
        "schema_version": "2.0.0",
        "source": "measured",
        "simulation": False,
        "status": "success",
        "token_trace": "full",
        "git_commit": "b" * 40,
        "git_dirty": False,
        "profile_id": profile["profile_id"],
        "environment": {
            "python": "3.11.17",
            "system": "Darwin",
            "machine": "arm64",
            "macos": "26.0",
            "packages": {"mlx": "0.32.4", "mlx-lm": "0.32.2"},
        },
        "requests": [
            {
                "request_id": k,
                "request_kind": k,
                "status": "success",
                "generation_tokens": 2,
                "prompt_tokens": 3,
                "ttft_ms": 1.0,
                "end_to_end_ms": 5.0,
                "first_visible_text_ms": 3.0,
                "observed_tpot_ms": 3.0,
                "generated_text": "test",
                "request_peak_mlx_bytes": 1,
            }
            for k in ("warmup", "measured")
        ],
    }
    save(root, summary, rows)
    return summary, rows


def to_boundary(summary, rows):
    summary["token_trace"] = "boundary"
    rows[:] = [r for r in rows if r["event"] != "token"]
    for i, r in enumerate(rows):
        r["sequence"] = i


def edit_input(root, summary, name, rehash=False, **changes):
    """Edit a copied input; rehash=True models a run that really used the edited file."""
    data = json.loads((root / name).read_text())
    data.update(changes)
    raw = json.dumps(data).encode()
    (root / name).write_bytes(raw)
    if rehash:
        summary[HASH_KEYS[name]] = hashlib.sha256(raw).hexdigest()


def relabel(summary, rows, mapping):
    for item in [*summary["requests"], *rows]:
        if item["request_kind"]:
            item["request_kind"] = mapping[item["request_kind"]]


def test_valid_fake_evidence(tmp_path):
    write_fake(tmp_path)
    assert validate(tmp_path)["status"] == "PASS"


def test_valid_boundary_evidence(tmp_path):
    summary, rows = write_fake(tmp_path)
    to_boundary(summary, rows)
    save(tmp_path, summary, rows)
    assert validate(tmp_path)["status"] == "PASS"


CORRUPTIONS = {
    "metric": (
        lambda root, s, r: s["requests"][1].update(ttft_ms=7.0),
        "summary/trace mismatch: ttft_ms",
    ),
    "failure": (lambda root, s, r: s.update(status="failed"), "run did not succeed"),
    "identity": (
        lambda root, s, r: r[4].update(device_id="wrong-device"),
        "identity mismatch: device_id",
    ),
    "unknown_event": (lambda root, s, r: r[4].update(event="unexpected"), "unknown request event"),
    "zero_tokens": (
        lambda root, s, r: s["requests"][1].update(generation_tokens=0),
        "request_end token count differs",
    ),
    "over_envelope": (
        lambda root, s, r: edit_input(root, s, "workload.json", rehash=True, max_new_tokens=1),
        "generation count outside envelope",
    ),
    "missing_memory": (
        lambda root, s, r: r[8].update(mlx_peak_bytes=""),
        "missing boundary memory: mlx_peak_bytes",
    ),
    "sequence_gap": (lambda root, s, r: r[5].update(sequence=99), "sequence gap"),
    "time_regression": (lambda root, s, r: r[6].update(monotonic_ns=0), "timestamp regression"),
    "lifecycle": (lambda root, s, r: r[1].update(event="model_load_end"), "bad process lifecycle"),
    "duplicate_first_token": (
        lambda root, s, r: r[5].update(event="first_token"),
        "missing/duplicate first_token",
    ),
    "end_token_count": (
        lambda root, s, r: r[8].update(token_count=3),
        "request_end token count differs",
    ),
    "visible_text_time": (
        lambda root, s, r: s["requests"][1].update(first_visible_text_ms=0.5),
        "summary/trace mismatch: first_visible_text_ms",
    ),
    "tpot": (
        lambda root, s, r: s["requests"][1].update(observed_tpot_ms=0.5),
        "summary/trace mismatch: observed_tpot_ms",
    ),
    "request_kind": (
        lambda root, s, r: relabel(s, r, {"warmup": "bogus", "measured": "measured"}),
        "counts or order differ",
    ),
    "warmup_after_measured": (
        lambda root, s, r: relabel(s, r, {"warmup": "measured", "measured": "warmup"}),
        "counts or order differ",
    ),
    "run_id": (lambda root, s, r: s.update(run_id="another-run"), "run_id differs"),
    "workload_edited": (
        lambda root, s, r: edit_input(root, s, "workload.json", max_new_tokens=63),
        "workload.json differs from its recorded hash",
    ),
    "lock_edited": (
        lambda root, s, r: edit_input(root, s, "model-lock.json", model_revision="c" * 40),
        "model-lock.json differs from its recorded hash",
    ),
    "old_workload_schema": (
        lambda root, s, r: edit_input(
            root, s, "workload.json", rehash=True, schema_version="1.0.0"
        ),
        "workload copy does not match",
    ),
    "dirty_source": (lambda root, s, r: s.update(git_dirty=True), "uncommitted source"),
    "no_commit": (lambda root, s, r: s.update(git_commit=None), "missing source commit"),
    "rosetta_or_intel": (
        lambda root, s, r: s["environment"].update(machine="x86_64"),
        "native Apple Silicon",
    ),
    "no_environment": (lambda root, s, r: s.pop("environment"), "native Apple Silicon"),
    "no_profile": (lambda root, s, r: (root / "model_profile.json").unlink(), "is missing"),
    "profile_edited": (
        lambda root, s, r: edit_input(root, s, "model_profile.json", layer_weight_bytes=[8, 9]),
        "profile ID does not match",
    ),
    "profile_other_model": (
        lambda root, s, r: edit_input(root, s, "model_profile.json", model_revision="c" * 40),
        "does not describe this run's model",
    ),
    "profile_missing_layer": (
        lambda root, s, r: edit_input(root, s, "model_profile.json", layer_weight_bytes=[8]),
        "byte tallies incomplete",
    ),
    "profile_from_other_run": (
        lambda root, s, r: edit_input(root, s, "model_profile.json", source_run_id="other-run"),
        "does not describe this run's model",
    ),
    "profile_negative_tensor": (
        lambda root, s, r: edit_input(
            root,
            s,
            "model_profile.json",
            replicated_tensor_bytes={"a": 5, "b": -1},
            replicated_weight_bytes=4,
        ),
        "byte tallies incomplete",
    ),
    "profile_tensors_not_a_mapping": (
        lambda root, s, r: edit_input(root, s, "model_profile.json", replicated_tensor_bytes=[4]),
        "byte tallies incomplete",
    ),
}


def test_run_directory_given_as_dot(tmp_path, monkeypatch):
    write_fake(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert validate(".")["status"] == "PASS"


def git_repo(root):
    def git(*argv):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv],
            cwd=root,
            check=True,
            capture_output=True,
        )

    (root / "src").mkdir(parents=True)
    (root / "src/tracked.py").write_bytes(b"x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    return root


def test_clean_checkout_is_not_dirty(tmp_path):
    repo, out = git_repo(tmp_path / "repo"), tmp_path / "out"
    out.mkdir()
    (repo / "notes.md").write_text("untracked outside src does not run")
    p = baseline.git_provenance(repo, out)
    assert re.fullmatch(r"[0-9a-f]{40}", p["git_commit"])
    assert p["git_dirty"] is False and not (out / "git-diff.patch").exists()


def test_untracked_source_counts_as_dirty(tmp_path):
    repo, out = git_repo(tmp_path / "repo"), tmp_path / "out"
    out.mkdir()
    (repo / "src/new_helper.py").write_text("y = 2\n")
    p = baseline.git_provenance(repo, out)
    assert p["git_dirty"] is True and p["git_untracked_source"] == ["src/new_helper.py"]


def test_non_utf8_change_is_recorded_not_raised(tmp_path):
    repo, out = git_repo(tmp_path / "repo"), tmp_path / "out"
    out.mkdir()
    (repo / "src/tracked.py").write_bytes(b"x = '\xff\xfe'\n")
    p = baseline.git_provenance(repo, out)
    assert p["git_dirty"] is True
    assert b"\xff\xfe" in (out / "git-diff.patch").read_bytes()


def test_installed_copy_has_unknown_provenance(tmp_path):
    assert baseline.git_provenance(None, tmp_path) == {"git_commit": None, "git_dirty": None}
    assert baseline.source_checkout() == ROOT  # this suite runs from the editable checkout


@pytest.mark.parametrize("change", CORRUPTIONS)
def test_corruption_fails(tmp_path, change):
    mutate, message = CORRUPTIONS[change]
    summary, rows = write_fake(tmp_path)
    mutate(tmp_path, summary, rows)
    save(tmp_path, summary, rows)
    with pytest.raises(ValueError, match=message):
        validate(tmp_path)


def test_boundary_token_count_must_match_trace(tmp_path):
    summary, rows = write_fake(tmp_path)
    to_boundary(summary, rows)
    summary["requests"][1]["generation_tokens"] = 50
    save(tmp_path, summary, rows)
    with pytest.raises(ValueError, match="request_end token count differs"):
        validate(tmp_path)


def test_runner_records_exact_inputs_and_source(tmp_path, monkeypatch):
    """Forces the non-Apple path, so this never loads MLX, even on the M4."""
    workload = ROOT / "benchmarks/workloads/phase1-smoke.json"
    lock = ROOT / "experiments/phase1/model-lock.json"
    out = tmp_path / "run"
    monkeypatch.setattr(baseline.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline",
            "--workload",
            str(workload),
            "--model-lock",
            str(lock),
            "--output-dir",
            str(out),
            "--device-id",
            "test",
        ],
    )
    assert baseline.main() == 1
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["status"] == "failed"
    for name, source in (("workload.json", workload), ("model-lock.json", lock)):
        copied = (out / name).read_bytes()
        assert copied == source.read_bytes()
        assert hashlib.sha256(copied).hexdigest() == summary[HASH_KEYS[name]]
    if summary["git_commit"] is None:
        assert summary["git_dirty"] is None
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", summary["git_commit"])
        assert (out / "git-diff.patch").exists() is summary["git_dirty"]
    env = summary["environment"]
    assert env["python"] == platform.python_version()
    assert env["packages"]["pytest"] == importlib.metadata.version("pytest")
