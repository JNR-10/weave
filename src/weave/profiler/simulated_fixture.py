"""Make a wholly synthetic memory-pressure case, never measured device evidence.

Every number below is invented planning input (guide section 11), labeled SIMULATED wherever it
carries an identity so it can never join measured evidence. The output must stay byte-identical
to the committed fixture, which the candidate golden file was generated from.
"""

import argparse
import json
from pathlib import Path
from typing import Any

MIB = 2**20

# Qwen2.5-7B geometry (guide section 2): 28 transformer layers, 4 KV heads of dimension 128.
LAYERS = 28
KV_HEADS = 4
HEAD_DIM = 128

# Round synthetic weight sizes, not the measured profile: a fixed 128 MiB per layer, plus 1 GiB of
# embeddings, final norm and output head that every pipeline rank holds.
LAYER_WEIGHT_BYTES = 128 * MIB
REPLICATED_WEIGHT_BYTES = 1024 * MIB

# A deliberate overestimate: MLX's real cache stores bf16 (2 bytes) in 256-token steps.
# Calibrating to that is Phase 2 work; the fixture keeps the conservative 4 bytes.
KV_BYTES_PER_ELEMENT = 4

# The Phase 1 smoke envelope: one request of at most 512 prompt tokens plus 64 new ones.
PROMPT_TOKENS_UPPER_BOUND = 512
MAX_NEW_TOKENS = 64

# Illustrative budgets, not hardware claims. The 3 GiB Mac is a constrained case chosen so that
# splits like logical [10, 18] are rejected; it does not mean Mohit's Air has 3 GiB free. The
# 24 GiB Mac is likewise invented, not a specification of the future second Mac.
DEVICE_BUDGETS = (
    ("SIMULATED-m4-pressure", 3072 * MIB),
    ("SIMULATED-second-mac", 24576 * MIB),
)
WORKSPACE_RESERVE_BYTES = 512 * MIB
SAFETY_MARGIN_BYTES = 512 * MIB


def _device(device_id: str, budget_bytes: int) -> dict[str, Any]:
    return {
        "device_id": device_id,
        "source": "simulated",
        "budget_basis": "fresh_process_absolute",
        "safe_process_budget_bytes": budget_bytes,
        "workspace_reserve_bytes": WORKSPACE_RESERVE_BYTES,
        "safety_margin_bytes": SAFETY_MARGIN_BYTES,
    }


def make_state() -> dict[str, Any]:
    """The committed planning fixture, built fresh on every call so callers may mutate it."""
    return {
        "schema_version": "2.0.0",
        "snapshot_id": "SIMULATED-qwen7-pressure-v2",
        "simulation": True,
        "model": {
            "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "model_revision": "SIMULATED-NOT-A-MODEL-LOCK",
            "profile_id": "SIMULATED-weight-profile-v2",
            "total_layers": LAYERS,
            "layer_weight_bytes": [LAYER_WEIGHT_BYTES] * LAYERS,
            "replicated_weight_bytes": REPLICATED_WEIGHT_BYTES,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "kv_bytes_per_element": KV_BYTES_PER_ELEMENT,
            "source": "simulated",
        },
        "request_envelope": {
            "prompt_tokens_upper_bound": PROMPT_TOKENS_UPPER_BOUND,
            "max_new_tokens": MAX_NEW_TOKENS,
            "concurrency": 1,
        },
        "devices": [_device(device_id, budget) for device_id, budget in DEVICE_BUDGETS],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:  # exclusive: never replace an earlier fixture
        json.dump(make_state(), f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
