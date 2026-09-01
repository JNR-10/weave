"""Runtime Monitoring — what actually happened.

Records per-stage execution time, which stage is throttling the pipeline, per-node memory in
use, communication cost, and out-of-memory events during inference.

This closes the loop: measurements taken here feed back into the scheduler's next decision,
and are also the raw material for the evaluation (TTFT, inter-token latency, throughput, P95
latency, OOM count, scheduler decision time, partition-change cost).
"""
