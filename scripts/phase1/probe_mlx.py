"""M4-only source/API probe; source inspection is not a two-Mac runtime test."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("run this probe on native Apple Silicon macOS")
    import mlx.core as mx
    from mlx_lm import utils
    from mlx_lm.models import qwen2
    from mlx_lm.models.pipeline import PipelineMixin

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    signature = inspect.signature(PipelineMixin.pipeline)
    if "split" not in signature.parameters:
        raise RuntimeError("installed PipelineMixin does not expose split")
    # The plan rests on Qwen2 using the shared pipeline hook; check it rather than assume it.
    if not issubclass(qwen2.Qwen2Model, PipelineMixin):
        raise RuntimeError("installed Qwen2Model does not inherit PipelineMixin")
    sources = {
        "pipeline.py": inspect.getsource(PipelineMixin),
        "sharded_load.py": inspect.getsource(utils.sharded_load),
        "qwen2.py": inspect.getsource(qwen2),
    }
    for name, source in sources.items():
        (out / name).write_text(source)
    report = {
        "schema_version": "2.0.0",
        "machine": platform.machine(),
        "macos_version": platform.mac_ver()[0],
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "device_info": dict(mx.device_info()),
        "pipeline_signature": str(signature),
        "sharded_load_signature": str(inspect.signature(utils.sharded_load)),
        "qwen2_inherits_pipeline_mixin": True,
        # Guide section 7: stock sharded_load calls pipeline twice (shard selection and final
        # model) without a split, so a custom split must be threaded through both.
        "sharded_load_pipeline_calls": sources["sharded_load.py"].count(".pipeline("),
        "source_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
        "runtime_two_device_verified": False,
    }
    (out / "compatibility.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
