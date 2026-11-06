"""Shared setup for the spec 0002 policy tests.

The golden candidate file is the policies' input; tests/test_phase1_golden.py holds it to the
generator byte for byte. Each test gets a fresh copy to edit.
"""

import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).resolve().parent / "golden/phase1-candidate-plans.json"


@pytest.fixture
def golden_set():
    """A fresh dict of the golden candidate set, safe to edit."""
    return json.loads(GOLDEN.read_text())


@pytest.fixture
def make_unfit():
    """A function that makes one raw candidate fail the memory estimate on one stage, in place."""

    def make_unfit(candidate: dict, stage: int = 0, excess: int = 1) -> dict:
        assignment = candidate["assignments"][stage]
        candidate["memory_feasible"] = False
        candidate["rejection_reasons"] = [
            {
                "code": "MEMORY_BUDGET_EXCEEDED",
                "device_id": assignment["device_id"],
                "excess_bytes": excess,
            }
        ]
        assignment["headroom_bytes"] = -excess
        return candidate

    return make_unfit
