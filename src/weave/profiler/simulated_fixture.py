"""Make a wholly synthetic memory-pressure case, never measured device evidence."""

import argparse
import json
from pathlib import Path
from typing import Any


def make_state() -> dict[str, Any]:
    mib = 2**20
    return {
        "schema_version": "2.0.0",
        "snapshot_id": "SIMULATED-qwen7-pressure-v2",
        "simulation": True,
        "model": {
            "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "model_revision": "SIMULATED-NOT-A-MODEL-LOCK",
            "profile_id": "SIMULATED-weight-profile-v2",
            "total_layers": 28,
            "layer_weight_bytes": [128 * mib] * 28,
            "replicated_weight_bytes": 1024 * mib,
            "kv_heads": 4,
            "head_dim": 128,
            "kv_bytes_per_element": 4,
            "source": "simulated",
        },
        "request_envelope": {
            "prompt_tokens_upper_bound": 512,
            "max_new_tokens": 64,
            "concurrency": 1,
        },
        "devices": [
            {
                "device_id": name,
                "source": "simulated",
                "budget_basis": "fresh_process_absolute",
                "safe_process_budget_bytes": budget * mib,
                "workspace_reserve_bytes": 512 * mib,
                "safety_margin_bytes": 512 * mib,
            }
            for name, budget in [("SIMULATED-m4-pressure", 3072), ("SIMULATED-second-mac", 24576)]
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(make_state(), f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
