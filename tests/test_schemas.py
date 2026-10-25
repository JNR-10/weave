"""The committed 2.0.0 schemas are the contract; code must not drift from them silently."""

import json
import sys
from pathlib import Path

import pytest

from weave.contracts import schemas

COMMITTED = Path(__file__).resolve().parents[1] / "experiments/phase1/schemas/v2"


def run_main(monkeypatch, output_dir):
    monkeypatch.setattr(sys, "argv", ["schemas", "--output-dir", str(output_dir)])
    schemas.main()


def test_generated_schemas_match_committed(tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path)
    generated = sorted(p.name for p in tmp_path.iterdir())
    assert generated == sorted(p.name for p in COMMITTED.iterdir())
    for name in generated:
        assert json.loads((tmp_path / name).read_text()) == json.loads(
            (COMMITTED / name).read_text()
        )


def test_refusal_writes_nothing(tmp_path, monkeypatch):
    (tmp_path / "workload.schema.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_main(monkeypatch, tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["workload.schema.json"]
