"""Conservative absolute process-memory accounting, not a runtime capacity proof."""

from collections.abc import Sequence


def estimate_stage_memory(
    *,
    layer_weight_bytes: Sequence[int],
    replicated_weight_bytes: int,
    total_tokens: int,
    kv_heads: int,
    head_dim: int,
    kv_bytes_per_element: int,
    concurrency: int,
    workspace_reserve_bytes: int,
    safety_margin_bytes: int,
) -> dict[str, int]:
    """Keep each cost explicit so a feasible estimate cannot hide an omitted term."""
    values = {
        "replicated_weight_bytes": replicated_weight_bytes,
        "total_tokens": total_tokens,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "kv_bytes_per_element": kv_bytes_per_element,
        "concurrency": concurrency,
        "workspace_reserve_bytes": workspace_reserve_bytes,
        "safety_margin_bytes": safety_margin_bytes,
    }
    if not layer_weight_bytes or any(type(n) is not int or n <= 0 for n in layer_weight_bytes):
        raise ValueError("assigned layer bytes must be nonempty positive integers")
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    kv_bytes = (
        2
        * len(layer_weight_bytes)
        * total_tokens
        * kv_heads
        * head_dim
        * kv_bytes_per_element
        * concurrency
    )
    return {
        "layer_weight_bytes": sum(layer_weight_bytes),
        "replicated_weight_bytes": replicated_weight_bytes,
        "kv_upper_bound_bytes": kv_bytes,
        "workspace_reserve_bytes": workspace_reserve_bytes,
        "safety_margin_bytes": safety_margin_bytes,
    }
