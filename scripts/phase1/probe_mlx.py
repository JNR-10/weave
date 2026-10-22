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
    from mlx_lm.models.pipeline import PipelineMixin

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    signature = inspect.signature(PipelineMixin.pipeline)
    if "split" not in signature.parameters:
        raise RuntimeError("installed PipelineMixin does not expose split")
    sources = {
        "pipeline.py": inspect.getsource(PipelineMixin),
        "sharded_load.py": inspect.getsource(utils.sharded_load),
    }
    for name, source in sources.items():
        (out / name).write_text(source)
    report = {
        "schema_version": "2.0.0",
        "machine": platform.machine(),
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "device_info": dict(mx.device_info()),
        "pipeline_signature": str(signature),
        "sharded_load_signature": str(inspect.signature(utils.sharded_load)),
        "source_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
        "runtime_two_device_verified": False,
    }
    (out / "compatibility.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
