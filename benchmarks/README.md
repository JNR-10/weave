# Benchmarks

Workload definitions and generators used to drive the cluster under controlled conditions.

The point is not a single tokens-per-second number. It is to vary the conditions that the
scheduler is supposed to react to, and see whether reacting helps.

## Dimensions varied

| Dimension | Range |
|---|---|
| Context length | short / medium / long (e.g. 2K → 32K+) |
| Concurrency | single request → several simultaneous requests |
| Machine state | idle, controlled background compute load, induced memory pressure |
| Communication | JACCL over Thunderbolt 5, RING over TCP/Ethernet |
| Model | 2–3 MLX-supported models; at least one too large for a single machine |

## Layout

```
workloads/    workload configurations (context, output length, arrival pattern, concurrency)
```

Generators must be deterministic given a seed — a workload that cannot be replayed cannot be
used to compare two schedulers.
