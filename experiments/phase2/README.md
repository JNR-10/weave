# Phase 2 bring-up on two Macs (T12–T15, gates P2-A to P2-D)

Spec 0005. **Nothing on this page has run yet.** The scripts were written on an Intel Mac; their
portable logic is tested in CI, and their MLX calls copy the Phase 1 runner and the installed
mlx-lm 0.32.2 source recorded in Mohit's M4 probe. A gate passes only with the records below
from real Macs. Raw output goes in `results/` (gitignored); zip each run with its SHA-256 and back
it up to the team Drive folder (CONTRIBUTING.md, Experiments).

Placeholders: `<m4>` and `<second>` are the two Macs' stable device IDs (for example
`mohit-m4-air`), and `<repo>` and `<venv>` are absolute paths that must exist on both Macs.

## P2-A: second-Mac inventory (T12, Mohit)

On the second Mac, from the same commit as the M4 and with no uncommitted changes, follow "M4
setup" in `experiments/phase1/README.md`: `make setup-m4`, `pip check`, the same model snapshot
as the committed `experiments/phase1/model-lock.json` (checked with
`prepare_model.py --verify-report`, never a new lock written over the old one), and the MLX
probe. Record the real chip, memory in bytes (`sysctl -n hw.memsize`), macOS version, and how the
Macs are connected. Never reuse the simulated fixture's 24 GiB Mac.

On each Mac, write its device ID once, so every script can check which Mac it is on:

```sh
echo "<m4>" > ~/.weave-device-id        # on the M4; "<second>" on the other Mac
```

## Hosts

`mlx.launch` reads a hostfile whose entries must be in **MLX rank order**, which is the logical
order reversed. Do not order it by hand. Keep one entry per Mac in a JSON object keyed by device
ID, copying the entry format from `mlx.launch --help` in the pinned environment (do not invent
addresses):

```json
{"<m4>": <the M4's host entry>, "<second>": <the second Mac's host entry>}
```

Then let the plan order it:

```sh
python -m weave.runtime.launch hostfile --device-order <m4> <second> --split 10 18 \
  --hosts experiments/phase2/hosts.json --output results/<id>/hostfile.json
```

`python -m weave.runtime.launch plan --device-order <m4> <second> --split 10 18` prints the
mapping without writing anything: logical `[10, 18]` on `[<m4>, <second>]` launches as MLX split
`[18, 10]` on ranks `[<second>, <m4>]`.

## P2-B: transport (T13, Jainil)

RING over the network first; JACCL only after its own hardware gate.

```sh
mlx.launch --backend ring --hostfile results/<id>/hostfile.json -- \
  <venv>/bin/python <repo>/scripts/phase2/distributed_smoke.py --output-dir <repo>/results/<id>
```

Pass: each Mac's `results/<id>/rank-<r>.json` shows `status: success`, `size: 2` and
`sum: [2.0, 2.0, 2.0, 2.0]`, with the device IDs in rank order. Keep both records and the
launcher output. One process printing ones proves nothing.

## Single-device reference

The two-Mac tokens are compared with this run, made by the same script on the same prompt
(`benchmarks/workloads/phase1-smoke.json`, greedy):

```sh
python scripts/phase2/pipeline_generate.py --device-order <m4> --split 28 \
  --output-dir results/<id>-single-<m4>
```

## P2-C: equal split `[14, 14]` (T14, Jainil)

```sh
mlx.launch --backend ring --hostfile results/<id>/hostfile.json -- \
  <venv>/bin/python <repo>/scripts/phase2/pipeline_generate.py \
  --device-order <m4> <second> --split 14 14 --output-dir <repo>/results/<id>
```

Copy the other Mac's `rank-<r>.json` next to this Mac's, then check the run:

```sh
python -m weave.runtime.launch check-run results/<id>/rank-0.json results/<id>/rank-1.json \
  --reference results/<id>-single-<m4>/rank-0.json
```

Pass: exit 0, meaning each rank ran on its planned Mac with its planned layers, the ranks loaded
every layer exactly once, every rank generated the same tokens, and they match the single-device
reference exactly. A divergence is printed with its token index; report it, do not rerun until
it goes away. Each record also holds its Mac's load and generation memory peaks. A run that
hangs past `--timeout-s` writes an error record and exits 3.

## P2-D: unequal split `[10, 18]` (T15, Jainil)

Same commands with `--split 10 18`, in both device orders (`<m4> <second>` and
`<second> <m4>`), each in a new run directory with its own hostfile. Every run loads a fresh
model, so no truncated model is reused. Pass: `check-run` exits 0 for both orders, and the
single-device command above still runs unchanged.
