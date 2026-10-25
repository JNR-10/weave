"""The smoke-workload rules the measured runner enforces.

Moved here unchanged in intent from the guide's reference planner tests, which also checked them.
"""

import jsonschema
import pytest

from weave.contracts.schemas import build_schemas
from weave.telemetry.baseline import validate_workload


def smoke_workload(**changes):
    workload = {
        "schema_version": "2.0.0",
        "workload_id": "test",
        "prompt": "test",
        "prompt_max_tokens": 512,
        "max_new_tokens": 64,
        "concurrency": 1,
        "warmup_requests": 1,
        "measured_requests": 3,
        "temperature": 0,
        "seed": 0,
    }
    return {**workload, **changes}


def test_smoke_workload_is_accepted_and_matches_the_schema():
    workload = smoke_workload()
    validate_workload(workload)
    jsonschema.validate(workload, build_schemas()["workload"])


def test_workload_with_concurrency_two_is_rejected():
    with pytest.raises(ValueError):
        validate_workload(smoke_workload(concurrency=2))
