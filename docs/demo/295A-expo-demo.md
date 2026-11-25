# CMPE 295A expo demo runbook (T19)

About eight minutes, three speakers. Every step says what kind of evidence it shows, because the
guide (section 12.2) requires the demo to say which commands ran on real Apple hardware and which
are portable or simulated:

- **REAL:** measured on Apple Silicon (Mohit's M4).
- **REAL DATA, CHECKED HERE:** measured on the M4 earlier, checked live on any laptop.
- **SIMULATED:** synthetic inputs with `SIMULATED-` IDs. Never present these as measurements.

Update the last step to the furthest two-Mac gate actually reached by expo day; never show a step
that has not passed.

## Before the expo

On the demo laptop (Intel or Apple Silicon), from a clean checkout of the commit the team agrees
on:

```sh
git status --short                 # must print nothing
make setup && make check           # all tests pass; the one M4-only test skips on Intel
shasum -a 256 donotcheckin/weave-phase1-m4-evidence-20261025T224835Z.zip
# expect 81cfedc907de3a7c237e052bab64d31670595377097b7fa9707261ab62ec15fe
DEMO="$(mktemp -d)" && unzip -q donotcheckin/weave-phase1-m4-evidence-20261025T224835Z.zip -d "$DEMO"
R="$DEMO/weave-phase1-m4-evidence-20261025T224835Z/results"
```

Rehearse every command once; the whole sequence takes under a minute to run. If the zip is
missing, download it from the team Drive folder and check the checksum first.

## 1. The question (30 s, slide, Jainil)

Splitting a model across Macs is solved; choosing the split is not. The best split changes with
context length, concurrency and memory pressure. RQ1: does live workload information beat equal
and static hardware-based splits?

## 2. The model really runs on one Mac (90 s, Mohit) — REAL

Optional live run on the M4, from the Phase 1 README:

```sh
mlx_lm.generate --model models/qwen7b \
  --prompt "Explain model partitioning in two short sentences." --max-tokens 64
```

Then the accepted measurement, `20261025T224855Z-m4air-b0-qwen7b-v2`: Qwen2.5-7B 4-bit on a
16 GB M4 MacBook Air, median TTFT 360.4 ms, median end to end 1511.7 ms, median 42.0 ms per
output token, request peak 4,441,795,936 bytes (4.14 GiB). Logging every token costs −0.05%
end to end (below the 5% trigger).

## 3. The evidence cannot be faked (60 s, Urmi) — REAL DATA, CHECKED HERE

```sh
python -m weave.telemetry.validate_evidence "$R/20261025T224855Z-m4air-b0-qwen7b-v2"
python -m weave.telemetry.validate_evidence "$R/TAMPER-CHECK/20261025T224855Z-m4air-b0-qwen7b-v2"
```

The first prints `"status": "PASS"`. The second is the same run with one TTFT raised by 1 ms;
it fails with `summary/trace mismatch: ttft_ms`. The validator also rejects runs from a checkout
with uncommitted changes.

## 4. The planner and the memory estimate (90 s, Jainil) — SIMULATED

```sh
python -m weave.scheduler.checkpoint
```

It prints 56 candidate placements, 15 of which pass the memory estimate; one single-Mac plan; an
accepted split `[1, 27]`; the rejected `[10, 18]`, which needs 3.27 GiB on a Mac with 3.00 GiB;
and the rank mapping. Say clearly: the two Macs here are invented (a 3 GiB "pressure" Mac and a
24 GiB second Mac), and "fits" means "passes the stated estimate", not "ran".

## 5. Three static baselines on the same input (60 s, Urmi) — SIMULATED

```sh
python -m weave.scheduler.policies \
  --candidates "$(ls -d results/SIMULATED-*-checkpoint | tail -1)/candidate_plans.json" --compare
```

Best single Mac picks the second Mac alone. Equal split `[14, 14]` is over the pressure Mac's
budget by 838,336,512 bytes. Memory-proportional picks `[1, 27]`, which fits. That is RQ1's
point in miniature: a fixed rule that ignores the workload can pick an unsafe split.

## 6. Logical order versus MLX rank order (30 s, Jainil) — portable

```sh
python -m weave.runtime.launch plan --device-order m4 second --split 10 18
```

Logical `[10, 18]` on `[m4, second]` launches as MLX split `[18, 10]` on ranks
`[second, m4]`, because MLX rank 0 runs the last layers. Every two-Mac run checks this on each
rank before and after loading.

## 7. What is next (60 s, all)

The two-Mac gates in order: second-Mac inventory (P2-A), transport (P2-B), equal split (P2-C),
unequal split (P2-D), per-stage measurements (P2-E). The tooling for P2-B to P2-D is written and
tested on Intel (spec 0005) and waits for the second Mac. Spring: the profiled static baseline
B3, the adaptive policy, and the evaluation plan (spec 0003).

If a command fails on the day, show its committed golden output instead
(`tests/golden/policy-comparison.md`, `tests/golden/phase1-candidate-plans.json`) and say so.
