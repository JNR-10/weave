"""Measure materialized parameter bytes without importing MLX on portable paths."""

import re
from typing import Any


def build_parameter_profile(
    parameters: Any,
    *,
    model_id: str,
    model_revision: str,
    total_layers: int,
) -> dict[str, Any]:
    """Retain non-layer weights separately because Qwen replicates them per rank."""
    from mlx.utils import tree_flatten

    layers, replicated = [0] * total_layers, 0
    for name, array in tree_flatten(parameters):
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match:
            index = int(match.group(1))
            if not 0 <= index < total_layers:
                raise ValueError("parameter layer index disagrees with locked geometry")
            layers[index] += int(array.nbytes)
        else:
            replicated += int(array.nbytes)
    if not all(layers):
        raise ValueError("weight profile did not match every layer")
    return {
        "schema_version": "2.0.0",
        "model_id": model_id,
        "model_revision": model_revision,
        "source": "measured",
        "measurement_kind": "materialized_parameter_nbytes",
        "layer_weight_bytes": layers,
        "replicated_weight_bytes": replicated,
        "limitation": "parameter bytes only; excludes KV, workspace, allocator, process/OS memory",
    }
