"""Golden reference for the Phase 1 planner rebuild (spec 0001).

The golden candidate file was captured from the guide's reference implementation before the rebuild,
so it is independent of the rebuilt code. The rebuilt planner must reproduce it byte for byte; a
deliberate change regenerates it in the same commit and names the change in the message.
"""

import json
import sys
from pathlib import Path

from weave.profiler import simulated_fixture
from weave.scheduler import candidates

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments/phase1/cluster-state-SIMULATED.json"
GOLDEN = ROOT / "tests/golden/phase1-candidate-plans.json"


def run_command(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_candidate_command_reproduces_the_golden_file(tmp_path, monkeypatch):
    out = tmp_path / "candidate_plans.json"
    run_command(monkeypatch, candidates, "--state", str(FIXTURE), "--output", str(out))
    assert out.read_bytes() == GOLDEN.read_bytes(), (
        "Candidate output differs from tests/golden/phase1-candidate-plans.json. If the change is "
        "deliberate, regenerate the golden file in the same commit and name the change in its message."
    )


def test_fixture_command_reproduces_the_committed_fixture(tmp_path, monkeypatch):
    out = tmp_path / "cluster-state-SIMULATED.json"
    run_command(monkeypatch, simulated_fixture, "--output", str(out))
    assert out.read_bytes() == FIXTURE.read_bytes()


def test_golden_has_56_candidates_of_which_15_fit():
    # Guide section 11: two single-Mac plans plus 27 cuts in each of two orders. Fitting: the
    # 24 GiB Mac alone, plus 1 to 7 layers on the 3 GiB Mac in either order, since each layer there
    # costs 128 MiB of weights plus 2,359,296 KV bytes against 1 GiB left after fixed costs.
    golden = json.loads(GOLDEN.read_text())
    assert (golden["candidate_count"], golden["memory_feasible_count"]) == (56, 15)


def test_golden_rejects_logical_10_18_on_the_pressure_mac_by_its_worked_excess():
    # Section 10 worked by hand for 10 layers on the 3 GiB Mac:
    # 10 x 128 MiB + 1 GiB replicated + 2*10*1*(512+64)*4*128*4 KV + 2 x 512 MiB reserves
    # = 3,513,253,888 bytes against 3,221,225,472, an excess of 292,028,416.
    golden = json.loads(GOLDEN.read_text())
    plan = next(
        p
        for p in golden["candidates"]
        if p["logical_split"] == [10, 18]
        and p["assignments"][0]["device_id"] == "SIMULATED-m4-pressure"
    )
    assert (plan["mlx_split_by_rank"], plan["rejection_reasons"]) == (
        [18, 10],
        [
            {
                "code": "MEMORY_BUDGET_EXCEEDED",
                "device_id": "SIMULATED-m4-pressure",
                "excess_bytes": 292028416,
            }
        ],
    )
