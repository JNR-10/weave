"""Measure materialized parameter bytes without importing MLX on portable paths."""

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

# source and measurement_kind are part of the identity so a synthetic or estimated profile with
# the same byte counts can never share an ID with a measured one.
IDENTITY = (
    "schema_version",
    "source",
    "measurement_kind",
    "model_id",
    "model_revision",
    "config_sha256",
    "layer_weight_bytes",
    "replicated_tensor_bytes",
)


def profile_id(profile: dict[str, Any]) -> str:
    """Content-addressed, so equal measurements share an ID and plans citing it stay comparable."""
    identity = json.dumps({k: profile[k] for k in IDENTITY}, sort_keys=True)
    return "profile-" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def profile_from_tensor_bytes(
    tensor_bytes: Iterable[tuple[str, int]],
    *,
    model_id: str,
    model_revision: str,
    config_sha256: str,
    total_layers: int,
) -> dict[str, Any]:
    """Retain non-layer weights separately because Qwen replicates them per rank.

    They are kept per tensor because a later adapter may drop the embedding or head from ranks
    that never use them, and the estimator then needs to know which bytes those were.
    """
    layers, replicated = [0] * total_layers, {}
    for name, nbytes in tensor_bytes:
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match:
            index = int(match.group(1))
            if not 0 <= index < total_layers:
                raise ValueError("parameter layer index disagrees with locked geometry")
            layers[index] += nbytes
        else:
            replicated[name] = replicated.get(name, 0) + nbytes
    if not all(layers):
        raise ValueError("weight profile did not match every layer")
    profile = {
        "schema_version": "2.0.0",
        "model_id": model_id,
        "model_revision": model_revision,
        "config_sha256": config_sha256,
        "source": "measured",
        "measurement_kind": "materialized_parameter_nbytes",
        "layer_weight_bytes": layers,
        "replicated_weight_bytes": sum(replicated.values()),
        "replicated_tensor_bytes": dict(sorted(replicated.items())),
        "limitation": "parameter bytes only; excludes KV, workspace, allocator, process/OS memory",
    }
    profile["profile_id"] = profile_id(profile)
    return profile


def build_parameter_profile(
    parameters: Any,
    *,
    model_id: str,
    model_revision: str,
    config_sha256: str,
    total_layers: int,
) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    return profile_from_tensor_bytes(
        ((name, int(array.nbytes)) for name, array in tree_flatten(parameters)),
        model_id=model_id,
        model_revision=model_revision,
        config_sha256=config_sha256,
        total_layers=total_layers,
    )
