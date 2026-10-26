"""The simulated planning fixture (spec 0001, step 7): synthetic, labeled so, and reproducible.

Byte-for-byte reproduction of the committed fixture is pinned by the golden tests.
"""

import json
import sys
from pathlib import Path

import jsonschema
import pytest

from weave.profiler import simulated_fixture
from weave.profiler.simulated_fixture import make_state

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "experiments/phase1/schemas/v2/cluster_state_simulated.schema.json"
FIXTURE = ROOT / "experiments/phase1/cluster-state-SIMULATED.json"


def run_command(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["simulated_fixture", *argv])
    simulated_fixture.main()


def test_every_identity_is_labeled_simulated():
    state = make_state()
    identities = [
        state["snapshot_id"],
        state["model"]["model_revision"],
        state["model"]["profile_id"],
        *(d["device_id"] for d in state["devices"]),
    ]
    assert all(i.startswith("SIMULATED-") for i in identities), identities


def test_every_source_is_marked_simulated():
    state = make_state()
    sources = [
        state["simulation"],
        state["model"]["source"],
        *(d["source"] for d in state["devices"]),
    ]
    assert sources == [True, "simulated", "simulated", "simulated"]


def test_fixture_matches_the_committed_schema():
    jsonschema.validate(make_state(), json.loads(SCHEMA.read_text()))


def test_each_call_returns_a_fresh_state():
    # Callers (and these tests) mutate the result; that must never leak into the next call.
    first = make_state()
    first["model"]["layer_weight_bytes"][0] = 1
    first["devices"][0]["safe_process_budget_bytes"] = 1
    assert make_state() == json.loads(FIXTURE.read_text())


def test_fixture_command_creates_missing_folders(tmp_path, monkeypatch):
    out = tmp_path / "new" / "folder" / "cluster-state-SIMULATED.json"
    run_command(monkeypatch, "--output", str(out))
    assert json.loads(out.read_text()) == make_state()


def test_fixture_command_refuses_to_overwrite(tmp_path, monkeypatch):
    out = tmp_path / "cluster-state-SIMULATED.json"
    out.write_text("earlier fixture")
    with pytest.raises(FileExistsError):
        run_command(monkeypatch, "--output", str(out))
