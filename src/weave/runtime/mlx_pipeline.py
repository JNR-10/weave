"""Split-aware loading and lock-step generation on Apple Silicon (spec 0005, T14 and T15).

The installed sharded_load (mlx-lm 0.32.2) calls `pipeline(group)` twice and never passes a
split, so it can only run MLX's even split. This loader follows the simple path in guide
section 7.3 instead: every Mac holds the full locked snapshot, the model is loaded lazily, and
`pipeline(group, split=...)` is called exactly once on a fresh model before any weight is
materialized, so only the local layers are read. Calling `pipeline` again on a truncated model
would destroy the layer-count basis, so a model object is never reused for another plan.

Every rank runs the same generation loop with the same prompt tokens and greedy sampling. The
installed Qwen2Model all-gathers the last hidden state, so every rank computes the same logits
and makes the same stopping decision; skipping generation on a rank would deadlock its peers.

MLX is imported inside each function so `import weave.runtime` works on Intel and in CI.
UNVERIFIED: none of this has run on two Macs yet; gates P2-C and P2-D decide whether it works.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weave.runtime.launch import LaunchPlan, check_rank, local_weight_files


@dataclass(frozen=True)
class LoadedStage:
    """A loaded model for one rank, with what the rank actually reported after loading."""

    model: Any
    tokenizer: Any
    rank: int
    size: int
    start_idx: int
    end_idx: int
    weight_files: tuple[str, ...]
    load_ms: float
    load_peak_bytes: int


def _needed_files(model_dir: Path, start: int, end: int, total_layers: int) -> tuple[str, ...]:
    """The local checkpoint files this rank reads; refuses a snapshot missing any of them."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = local_weight_files(weight_map, start, end, total_layers)
    else:
        files = ["model.safetensors"]
    absent = [name for name in files if not (model_dir / name).is_file()]
    if absent:
        raise FileNotFoundError(f"{model_dir}: missing checkpoint files {absent}")
    return tuple(files)


def init_group(backend: str) -> Any:
    """Join the launched group; mlx.launch decides the rank, so it is read, never assumed."""
    import mlx.core as mx

    return mx.distributed.init(backend=backend)


def load_stage(model_dir: Path, plan: LaunchPlan, group: Any | None, device_id: str) -> LoadedStage:
    """Load this rank's layers of `plan` and check the rank against the plan before using it.

    A one-stage plan takes the ordinary `mlx_lm.load` path with no group (guide section 7.2),
    never a two-rank launch with an idle rank.
    """
    import mlx.core as mx
    from mlx_lm import load

    mx.synchronize()
    mx.reset_peak_memory()
    started = time.perf_counter_ns()
    if plan.size == 1:
        if group is not None and group.size() != 1:
            raise RuntimeError("a single-device plan must run in one process, without a group")
        check_rank(
            plan, rank=0, size=1, device_id=device_id, start_idx=0, end_idx=plan.total_layers
        )
        files = _needed_files(model_dir, 0, plan.total_layers, plan.total_layers)
        model, tokenizer = load(str(model_dir))
        if len(model.layers) != plan.total_layers:
            raise RuntimeError(
                f"model has {len(model.layers)} layers, plan has {plan.total_layers}"
            )
        rank, size, start, end = 0, 1, 0, plan.total_layers
    else:
        if group is None:
            raise RuntimeError(f"a {plan.size}-stage plan needs a launched group")
        rank, size = group.rank(), group.size()
        if size != plan.size:
            raise RuntimeError(f"launched group has {size} ranks, plan has {plan.size} stages")
        # Refuse a wrong host before reading any weights; the planned range passes trivially here.
        planned = plan.stage_for_rank(rank)
        check_rank(
            plan,
            rank=rank,
            size=size,
            device_id=device_id,
            start_idx=planned.start_layer,
            end_idx=planned.end_layer_exclusive,
        )
        files = _needed_files(
            model_dir, planned.start_layer, planned.end_layer_exclusive, plan.total_layers
        )
        model, tokenizer = load(str(model_dir), lazy=True)
        inner = model.model
        if not hasattr(inner, "pipeline"):
            raise RuntimeError("installed model does not support pipeline parallelism")
        if len(inner.layers) != plan.total_layers:
            raise RuntimeError(
                f"model has {len(inner.layers)} layers, plan has {plan.total_layers}"
            )
        inner.pipeline(group, split=list(plan.mlx_split_by_rank))
        start, end = int(inner.start_idx), int(inner.end_idx)
        # The readback after the call is the real check; the one above only refuses a bad host.
        check_rank(plan, rank=rank, size=size, device_id=device_id, start_idx=start, end_idx=end)
    mx.eval(model.parameters())
    mx.synchronize()
    if plan.size > 1:
        # The same barrier sharded_load uses, so a slow loader cannot time out its peers.
        mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    return LoadedStage(
        model=model,
        tokenizer=tokenizer,
        rank=rank,
        size=size,
        start_idx=start,
        end_idx=end,
        weight_files=files,
        load_ms=(time.perf_counter_ns() - started) / 1e6,
        load_peak_bytes=int(mx.get_peak_memory()),
    )


def generate(stage: LoadedStage, prompt: str, max_new_tokens: int, seed: int) -> dict[str, Any]:
    """Greedy generation on this rank; every rank of the launch must call this identically.

    Times come from this process's perf_counter_ns and are comparable only within this rank.
    """
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    prompt_tokens = stage.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    mx.synchronize()
    mx.reset_peak_memory()
    start = time.perf_counter_ns()
    first, tokens, pieces = None, [], []
    for response in stream_generate(
        stage.model,
        stage.tokenizer,
        prompt_tokens,
        max_tokens=max_new_tokens,
        sampler=make_sampler(temp=0),
    ):
        now = time.perf_counter_ns()
        count = int(response.generation_tokens)
        if count < len(tokens) or count > len(tokens) + 1:
            raise ValueError("unexpected stream token counter; review installed API")
        if count > len(tokens):
            tokens.append(int(response.token))
            if first is None:
                first = now
        pieces.append(response.text)
    mx.synchronize()
    end = time.perf_counter_ns()
    if first is None or not tokens:
        raise ValueError("no tokens generated; an empty output is not a successful run")
    return {
        "prompt_tokens": [int(t) for t in prompt_tokens],
        "tokens": tokens,
        "text": "".join(pieces),
        "ttft_ms": (first - start) / 1e6,
        "end_to_end_ms": (end - start) / 1e6,
        "generation_peak_bytes": int(mx.get_peak_memory()),
    }
