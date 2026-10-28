"""Emit the 2.0.0 placement-decision schema (spec 0002) into its own folder.

The Phase 1 contract folder, experiments/phase1/schemas/v2/, is frozen and hashed into evidence, so
the decision schema is generated here, into experiments/policies/schemas/v2/, and never touches it.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from weave.contracts.schemas import STRING
from weave.scheduler.candidates import VERSION
from weave.scheduler.policies import (
    BEST_SINGLE_DEVICE,
    EQUAL_SPLIT,
    FEASIBLE,
    INFEASIBLE,
    MEMORY_PROPORTIONAL,
    NO_FEASIBLE_PLAN,
)


def build_schemas() -> dict[str, Any]:
    properties = {
        "schema_version": {"const": VERSION},
        "simulation": {"const": True},
        "policy": {"enum": [BEST_SINGLE_DEVICE, EQUAL_SPLIT, MEMORY_PROPORTIONAL]},
        "source_snapshot_id": STRING,
        "candidate_set_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "chosen_plan_id": {"type": ["string", "null"]},
        "outcome": {"enum": [FEASIBLE, INFEASIBLE, NO_FEASIBLE_PLAN]},
        "reason": STRING,
        "rejection_reasons": {"type": "array", "items": {"type": "object"}},
        "executable": {"const": False},
        "runtime_verified": {"const": False},
    }
    placement_decision = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
        # chosen_plan_id is null exactly when no plan was chosen.
        "allOf": [
            {
                "if": {"properties": {"outcome": {"const": NO_FEASIBLE_PLAN}}},
                "then": {"properties": {"chosen_plan_id": {"type": "null"}}},
                "else": {"properties": {"chosen_plan_id": STRING}},
            }
        ],
    }
    return {"placement_decision": placement_decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/policies/schemas/v2")
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
