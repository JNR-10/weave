"""Conservative absolute process-memory accounting, not a runtime capacity proof."""

from collections.abc import Sequence
from typing import Any


def _positive_int(value: Any, name: str) -> int:
    # bool is a subclass of int in Python, so `type(...) is int` keeps true/false out.
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}: expected a positive integer, got {value!r}")
    return value


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
    """One pipeline stage's predicted memory, term by term (guide section 10), in integer bytes.

    Every term stays visible in the result, so a feasible estimate can never hide an omitted
    cost, and no term has a default or may be zero. Replicated weights are charged in full to
    every stage, because the current Qwen pipeline keeps the embeddings, final norm and output
    head on every rank.

    The KV bound holds a key and a value (the 2) for every assigned layer, concurrent request and
    token, with no allocation rounding. MLX actually allocates its cache in 256-token steps, at
    the cache's own dtype; fitting that is Phase 2 calibration, not a Phase 1 fix.
    """
    if not layer_weight_bytes:
        raise ValueError("layer_weight_bytes: expected at least one layer")
    layers = [
        _positive_int(n, f"layer_weight_bytes[{i}]") for i, n in enumerate(layer_weight_bytes)
    ]
    for name, value in (
        ("replicated_weight_bytes", replicated_weight_bytes),
        ("total_tokens", total_tokens),
        ("kv_heads", kv_heads),
        ("head_dim", head_dim),
        ("kv_bytes_per_element", kv_bytes_per_element),
        ("concurrency", concurrency),
        ("workspace_reserve_bytes", workspace_reserve_bytes),
        ("safety_margin_bytes", safety_margin_bytes),
    ):
        _positive_int(value, name)
    kv_upper_bound = (
        2 * len(layers) * concurrency * total_tokens * kv_heads * head_dim * kv_bytes_per_element
    )
    return {
        "layer_weight_bytes": sum(layers),
        "replicated_weight_bytes": replicated_weight_bytes,
        "kv_upper_bound_bytes": kv_upper_bound,
        "workspace_reserve_bytes": workspace_reserve_bytes,
        "safety_margin_bytes": safety_margin_bytes,
    }
