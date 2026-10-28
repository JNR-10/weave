"""The committed placement-decision schema is the contract; code must not drift from it silently."""

import json
import sys
from pathlib import Path

import pytest

from weave.contracts import policy_schemas

COMMITTED = Path(__file__).resolve().parents[1] / "experiments/policies/schemas/v2"


def run_main(monkeypatch, output_dir):
    monkeypatch.setattr(sys, "argv", ["policy_schemas", "--output-dir", str(output_dir)])
    policy_schemas.main()


def test_generated_schemas_match_committed(tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path)
    generated = sorted(p.name for p in tmp_path.iterdir())
    assert generated == sorted(p.name for p in COMMITTED.iterdir())
    for name in generated:
        assert json.loads((tmp_path / name).read_text()) == json.loads(
            (COMMITTED / name).read_text()
        )


def test_refusal_writes_nothing(tmp_path, monkeypatch):
    (tmp_path / "placement_decision.schema.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_main(monkeypatch, tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["placement_decision.schema.json"]
