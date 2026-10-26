"""Emit the exact 2.0.0 input and output schemas used by this guide."""

import argparse
import json
from pathlib import Path
from typing import Any

STRING = {"type": "string", "minLength": 1}
POS = {"type": "integer", "minimum": 1}
NONNEG = {"type": "integer", "minimum": 0}


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": True,
    }


def build_schemas() -> dict[str, Any]:
    version = {"const": "2.0.0"}
    workload = obj(
        {
            "schema_version": version,
            "workload_id": STRING,
            "prompt": STRING,
            "prompt_max_tokens": POS,
            "max_new_tokens": POS,
            "concurrency": {"const": 1},
            "warmup_requests": POS,
            "measured_requests": POS,
            "temperature": {"const": 0},
            "seed": {"type": "integer"},
        }
    )
    model = obj(
        {
            "model_id": STRING,
            "model_revision": STRING,
            "profile_id": STRING,
            "total_layers": POS,
            "layer_weight_bytes": {"type": "array", "minItems": 1, "items": POS},
            "replicated_weight_bytes": POS,
            "kv_heads": POS,
            "head_dim": POS,
            "kv_bytes_per_element": POS,
            "source": {"const": "simulated"},
        }
    )
    device = obj(
        {
            "device_id": STRING,
            "source": {"const": "simulated"},
            "budget_basis": {"const": "fresh_process_absolute"},
            "safe_process_budget_bytes": POS,
            "workspace_reserve_bytes": POS,
            "safety_margin_bytes": POS,
        }
    )
    state = obj(
        {
            "schema_version": version,
            "snapshot_id": STRING,
            "simulation": {"const": True},
            "model": model,
            "request_envelope": obj(
                {
                    "prompt_tokens_upper_bound": POS,
                    "max_new_tokens": POS,
                    "concurrency": {"const": 1},
                }
            ),
            "devices": {"type": "array", "minItems": 1, "maxItems": 2, "items": device},
        }
    )
    assignment = obj(
        {
            "device_id": STRING,
            "logical_stage": NONNEG,
            "mlx_rank": NONNEG,
            "start_layer": NONNEG,
            "end_layer_exclusive": POS,
            "layer_count": POS,
            "memory_components": obj(
                {
                    k: NONNEG
                    for k in (
                        "layer_weight_bytes",
                        "replicated_weight_bytes",
                        "kv_upper_bound_bytes",
                        "workspace_reserve_bytes",
                        "safety_margin_bytes",
                    )
                }
            ),
            "predicted_memory_bytes": POS,
            "safe_process_budget_bytes": POS,
            "headroom_bytes": {"type": "integer"},
        }
    )
    candidate = obj(
        {
            "schema_version": version,
            "plan_id": STRING,
            "source_snapshot_id": STRING,
            "model_id": STRING,
            "model_revision": STRING,
            "profile_id": STRING,
            "simulation": {"const": True},
            "selected": {"const": False},
            "policy": {"const": "candidate_only"},
            "mode": {"enum": ["single_device", "pipeline"]},
            "logical_split": {"type": "array", "minItems": 1, "maxItems": 2, "items": POS},
            "mlx_split_by_rank": {"type": "array", "items": POS},
            "device_id_by_mlx_rank": {"type": "array", "items": STRING},
            "assignments": {"type": "array", "minItems": 1, "maxItems": 2, "items": assignment},
            "memory_feasible": {"type": "boolean"},
            "rejection_reasons": {"type": "array", "items": {"type": "object"}},
            "runtime_verified": {"const": False},
            "executable": {"const": False},
        }
    )
    candidate_set = obj(
        {
            "schema_version": version,
            "simulation": {"const": True},
            "source_snapshot_id": STRING,
            "selected_plan_id": {"type": "null"},
            "candidate_count": POS,
            "memory_feasible_count": NONNEG,
            "candidates": {"type": "array", "items": candidate},
        }
    )
    run_summary = obj(
        {
            "schema_version": version,
            "run_id": STRING,
            "device_id": STRING,
            "model_id": STRING,
            "model_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
            "workload_id": STRING,
            "source": {"const": "measured"},
            "simulation": {"const": False},
            "status": {"enum": ["success", "failed"]},
            "policy": {"const": "single_device"},
            "partition_version": STRING,
            "requests": {"type": "array", "items": {"type": "object"}},
        }
    )
    return {
        "workload": workload,
        "cluster_state_simulated": state,
        "candidate_set": candidate_set,
        "run_summary": run_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/phase1/schemas/v2")
    root = Path(parser.parse_args().output_dir)
    schemas = build_schemas()
    # Check every target first so a refused run leaves no partial schema set behind.
    existing = [str(p) for name in schemas if (p := root / f"{name}.schema.json").exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing}; use a fresh --output-dir")
    root.mkdir(parents=True, exist_ok=True)
    for name, schema in schemas.items():
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
        with (root / f"{name}.schema.json").open("x") as f:
            json.dump(schema, f, indent=2)
    print(root)


if __name__ == "__main__":
    main()
