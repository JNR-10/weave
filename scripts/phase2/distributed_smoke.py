"""T13 / gate P2-B: prove two launched ranks really talk before any model is loaded.

From guide section 14. Every rank adds a vector of ones with all_sum; with N ranks each must get
N everywhere. One process printing ones proves nothing, so the group size is asserted. Each rank
also writes its own record, because P2-B keeps the rank logs of *both* machines as evidence.

UNVERIFIED: written on an Intel Mac; it has not yet run on two Macs. Launch it with mlx.launch
only after both Macs have the same commit and pinned environment (experiments/phase2/README.md).
"""

import argparse
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="ring", help="MLX backend; JACCL only after its own gate")
    p.add_argument("--expect-size", type=int, default=2)
    p.add_argument("--device-id-file", default=str(Path.home() / ".weave-device-id"))
    p.add_argument("--output-dir", required=True, help="new results/<run-id> on each Mac")
    args = p.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("run on native Apple Silicon macOS")
    import mlx.core as mx

    device_id = Path(args.device_id_file).read_text().strip()
    world = mx.distributed.init(backend=args.backend)
    rank, size = world.rank(), world.size()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "record": "weave-phase2-transport-smoke",
        "record_version": "draft-1",
        "run_id": out.name,
        "source": "measured",
        "simulation": False,
        "wall_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "device_id": device_id,
        "backend": args.backend,
        "rank": rank,
        "size": size,
        "mlx_version": mx.__version__,
        "python": sys.version.split()[0],
        "status": "error",
    }
    path = out / f"rank-{rank}.json"
    try:
        if size != args.expect_size:
            raise RuntimeError(f"expected {args.expect_size} launched ranks, got {size}")
        total = mx.distributed.all_sum(mx.ones(4))
        mx.eval(total)
        record["sum"] = total.tolist()
        if not bool(mx.all(total == size).item()):
            raise RuntimeError(f"collective failed: {record['sum']}")
        record["status"] = "success"
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Exclusive create: a rerun needs a new run directory, never an overwritten record.
        with path.open("x") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    print(json.dumps({k: record[k] for k in ("rank", "size", "device_id", "sum")}))


if __name__ == "__main__":
    main()
