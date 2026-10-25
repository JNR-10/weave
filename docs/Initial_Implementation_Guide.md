# **Revised Phase 1 Implementation Guide (v2)**

Workload-adaptive model partitioning across heterogeneous Apple Silicon Macs

Prepared for Mohit Manoj Barade, Urmi Manishkumar Shah, and Jainil Narendrakumar Rana. Revised October 20, 2026\. Phase 1 checkpoint: October 23, 2026\. This is an implementation runbook, not a report of completed hardware experiments.

Repository alignment: [JNR-10/weave at 8ac1104](https://github.com/JNR-10/weave/tree/8ac11046aa181d9122bb362f94b4cabb85e238e5), verified October 20, 2026\. The appendix files are proposed additions to this checkout, not files already merged into main. No GitHub push or merge was performed while preparing this guide. The tab may be named “Jainil's Revision”; the guide remains v2 and the interface version remains 2.0.0.

## **1\. What the team must finish**

Phase 1 establishes the pieces that can be verified with the hardware currently available: Mohit's M4 MacBook Air, with 16 GB unified memory and 512 GB storage. Urmi and Jainil can develop the portable Python components on their Intel Macs. The future second Apple Silicon Mac is not an available execution device in this phase. Its specifications, usable memory, network behavior, and measured speed remain unverified.

By the checkpoint, the team must have a reproducible Qwen2.5-7B single-device baseline; a model revision and environment lock; validated timing and memory evidence; a measured parameter-byte profile; installed-source checks for the MLX pipeline interface; a candidate generator that includes single-device fallback and both two-device orders; versioned interfaces; passing portable tests; and an explicit list of integration blockers. A simulated candidate set is a software result, not a two-Mac inference result.

The preserved first tab remains the historical guide. This tab governs the revised Phase 1 implementation wherever the two differ. Document version v2 and interface version 2.0.0 are separate identifiers. Existing 1.0.0 evidence must remain intact.

| Required now | Deferred until a later gate |
| ----- | ----- |
| M4 environment, model lock, serial single-device execution | Real two-Mac equal and unequal pipeline execution |
| One warmup and three measured requests for the smoke workload | Large benchmark matrix, concurrency, statistically credible p95 comparisons |
| Source/API probe and unequal rank-mapping contract test | Production split-aware loader and verified distributed token loop |
| Parameter-byte profile and explicit conservative memory terms | Calibrated KV/workspace predictor across workloads and both Macs |
| Offline, labeled simulated candidates and rejection reasons | Cost ranking, automatic selection, KEEP/SWITCH controller |
| Unit tests, trace validation, handoffs, clean reproduction | Draining, reload/reconstruction, health checks, rollback, node-loss recovery |

No speedup, optimal split, safe online switching, or distributed correctness result is an acceptance requirement for this phase. Completing Phase 1 does not establish the research contribution. The full project must later test whether closed-loop workload/runtime-aware placement earns back its switching cost against strong static and single-device baselines.

## **2\. Decisions that replace the original assumptions**

| Topic | Revised decision | Practical consequence |
| ----- | ----- | ----- |
| Bring-up model | mlx-community/Qwen2.5-7B-Instruct-4bit | Use one dense Qwen2 implementation with the required pipeline hook; freeze the actual downloaded commit. |
| Layer count | Qwen2.5-7B has 28 transformer blocks | The planner generates 27 positive cuts per device order, not 31\. Geometry comes from config.json. |
| Later evaluation model | Qwen2.5-14B-Instruct-4bit is a candidate after feasibility checks; 48 layers | Do not download or run it as a Phase 1 dependency. A larger model does not automatically justify distribution. |
| Small rescue smoke | Llama-3.2-3B-Instruct-4bit can diagnose installation/resource problems; its 3B configuration has 28 layers | Label it separately. A successful standalone run does not establish the intended Qwen pipeline path. |
| Logical stage versus rank | Logical stage 0 executes the earliest layers; current MLX pipeline rank 0 executes the latest layers | Store logical\_stage and mlx\_rank separately. Reverse logical counts when constructing the native split list. |
| Loading | Stock sharded\_load does not expose a scheduler split in the inspected source | The later adapter must use the same custom split during shard discovery and final construction, or use a validated local-snapshot path. |
| Fixed weights | Current Qwen pipeline retains embeddings, final norm, and output head on every rank | Count this replicated cost on every participating device, including rank 0 and the first logical stage. |
| Feasibility | memory\_feasible means passing a stated estimate | It does not mean executable, runtime-verified, fastest, or selected. |
| Fallback | Enumerate each one-device placement independently | Do not represent fallback as a two-device split containing zero layers. Launch one process through the ordinary load path. |
| Units | Integer bytes and monotonic nanoseconds internally | Use GiB \= bytes / 2^30 for display. Keep library GB values explicitly separate. |
| Interfaces | 2.0.0, because rank semantics, intervals, and feasibility fields change | Reject 1.0.0 inputs rather than silently reinterpreting them. |

These compatibility findings are source-level observations, checked on October 20, 2026\. A moving main branch is not an environment lock. Mohit must inspect and test the installed package before freezing it. If the installed source differs, stop the affected integration, preserve the source/hash, and update the compatibility decision together.

## **3\. Phase 1 authorship, long-term ownership, and handoffs**

README.md names the long-term component owners. The September 23 meeting note records specific early implementation tasks, which the October guide follows. Retain those Phase 1 tasks, with the long-term owner's review made explicit. This is temporary authorship across components, not a reassignment of project roles. CONTRIBUTING.md says an owner is responsible for an area, not its sole author. Record this distinction in the interface-freeze note.

| Person | Long-term responsibility in README.md | Phase 1 author/operator work | Review and handoff |
| ----- | ----- | ----- | ----- |
| Mohit | Cluster profiler, telemetry, KV-cache/memory estimator | Native M4 environment; model lock; scripts/phase1/prepare\_model.py and probe\_mlx.py; hardware execution of the baseline and installed mapping test | Jainil reviews execution/rank mapping. Mohit reviews the profiler, estimator, and telemetry code authored by teammates. |
| Urmi | Placement policies, how-many-Macs decision, ablation design | Smoke workload; weave.telemetry.baseline and validate\_evidence; measurement definitions; schema packaging and reproduction instructions | Mohit reviews telemetry/profile correctness. Jainil reviews workload/execution behavior. Urmi reviews candidate semantics as the future policy consumer. |
| Jainil | MLX Distributed integration, pipeline/backends, partition application/switching, workload generator | weave.scheduler.candidates; weave.profiler.simulated\_fixture; weave.estimator.memory; candidate tests; logical-layer/native-rank contract | Urmi reviews scheduling boundaries; Mohit reviews memory assumptions. Jainil retains the later unequal-split adapter and application/switching responsibility. |
| All three | Shared reproduction, integration, experiments, analysis, and report | Contract 2.0.0; shared package setup; independent reproduction; blocker notes | At least one teammate reviews each PR before merge, preferably the owner of the affected area. |

Hardware access does not transfer every module's authorship to Mohit. Each author uses their own branch/account and shares reviews. Jainil's immediate prototype enumerates valid candidates; it does not yet select an adaptive optimum. Urmi's evidence work does not claim completed distributed experiments.

Phase 1 is a hardware-limited bring-up checkpoint feeding repository milestones M1/M2. It does not complete M1's multi-Mac/Exo reproduction, M2's full profiler, or M3's placement optimizer just because portable tests pass.

## **4\. Start in parallel after a short contract freeze**

Hold a 30–45 minute setup meeting. Record the selected model, schema version, interval convention, byte units, trace event names, workload envelope, file owners, and the model-lock format in meetings/phase1-interface-freeze.md. Agree that measured evidence and simulated fixtures remain separate. Do not wait for all MLX work to finish before starting the portable modules.

| Sequence | Mohit | Urmi | Jainil | Dependency resolved |
| ----- | ----- | ----- | ----- | ----- |
| Start, October 20 | Hardware/Python inventory; create native arm64 environment | Freeze workload/trace contract; implement portable evidence validator | Freeze state/plan contract; implement fixture and candidates | Shared names, units, and semantics |
| Next independent block | Pin model; CLI smoke; capture pipeline source | Implement runner; unit-test malformed evidence on Intel | Test single-node cases, both device orders, exact layer cover, memory rejection | Three components can be developed independently |
| Integration, October 21–22 | Execute Urmi's runner on M4; inspect weight profile; run installed-source test | Validate real trace; fix measurement issues with Mohit | Run fixture through schema and candidate generator; review measured profile | One trustworthy single-device evidence package |
| Reproduction, October 22 | Recreate environment in a fresh venv using the lock | Have a teammate validate a new run directory | Have a teammate regenerate the same simulated candidates | Documentation matches executable code |
| Checkpoint, October 23 | Demonstrate model output and installed API evidence | Demonstrate trace/summary validation and retained failure records | Demonstrate accepted/rejected candidates and mapping | Phase 1 sign-off and explicit Phase 2 blockers |

The real sequence is: freeze contracts → work in parallel → integrate the runner on M4 → validate evidence → cross-reproduce. Two-Mac integration depends on the second Mac and a working adapter. Cost-model fitting depends on calibrated measurements. Automatic switching depends on both. Calendar dates do not override a failed gate.

## **5\. Repository structure, working agreement, and environment**

Use [JNR-10/weave](https://github.com/JNR-10/weave). At audited commit 8ac1104, src/weave contains the four component scaffolds; pyproject.toml and the implementation/test files below are new additions. Add them deliberately. Preserve the existing package initializers and unrelated changes. Do not create another top-level phase1 Python package or install the unrelated package named weave from PyPI. The supplied local distribution is weave-inference; the Python import is weave.

If there is no checkout yet:

```sh
git clone https://github.com/JNR-10/weave.git
cd weave
```

From an existing checkout's root:

```sh
git rev-parse --show-toplevel
git status --short --branch
git remote get-url origin
git rev-parse HEAD
```

Confirm origin points to JNR-10/weave. If main has advanced beyond 8ac1104, inspect the changes and merge settings into any existing files; do not overwrite them. Start from an agreed clean main or an isolated worktree, without reset/clean or discarding work.

Use CONTRIBUTING.md's branch convention, \<type\>/\<area\>-\<short-description\>. When the working tree is clean:

```sh
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feat/runtime-phase1-mohit
# Urmi's checkout: git switch -c feat/telemetry-phase1-urmi
# Jainil's checkout: git switch -c feat/scheduler-phase1-jainil
```

main stays working; all merges go through focused PRs with at least one teammate review, normally squash-merged. Each member pushes their own working branch under their own account. The repository requires review before merge; it does not require permission for every local edit or feature-branch push. Honor any separately agreed restriction on a specific action. Stage explicit paths and inspect the diff, especially when raw evidence exists.

### **5.1 File map and shared bootstrap**

Copy the complete appendix files into these paths before running installation/tests:

| Path | Purpose and author/reviewer |
| ----- | ----- |
| scripts/phase1/prepare\_model.py; probe\_mlx.py | Model lock and installed-source probe; Mohit, with Jainil reviewing runtime semantics |
| src/weave/profiler/model\_profile.py | Materialized layer/replicated bytes; Urmi's runner handoff, reviewed by Mohit |
| src/weave/profiler/simulated\_fixture.py | Synthetic cluster input; Jainil, reviewed by Mohit |
| src/weave/estimator/memory.py | Explicit weight/KV/workspace/safety accounting; Jainil, reviewed by Mohit |
| src/weave/scheduler/candidates.py | Complete unranked candidates; Jainil, reviewed by Urmi |
| src/weave/telemetry/baseline.py; validate\_evidence.py | Instrumented single-device runner and validator; Urmi, reviewed by Mohit |
| `src/weave/contracts/__init__.py`; schemas.py | New shared versioned interface package; team review |
| benchmarks/workloads/phase1-smoke.json | Frozen smoke workload; Urmi, reviewed by Jainil |
| experiments/phase1/ | Committed inputs, model/environment locks, schemas, and reproduction README |
| tests/ | Portable tests plus the M4-only installed-pipeline contract test |
| pyproject.toml | New src-layout packaging, optional dev/M4 dependencies, pytest/Ruff setup |
| results/\<unique\_run\_id\>/ | Ignored raw trace, summary, profile, copied inputs, logs, errors, and per-run README |
| results/SIMULATED-\<unique\_id\>/ | Ignored synthetic outputs; never measured inference evidence |

Add the shared package setup and contracts in a small bootstrap PR before dependent branches integrate. Owners can develop locally against the same reviewed bootstrap patch meanwhile. The estimator/profile helpers must exist before the candidate/runner imports pass. Avoid three divergent pyproject.toml files.

```sh
mkdir -p scripts/phase1 src/weave/contracts benchmarks/workloads
mkdir -p experiments/phase1 tests results
```

Append the final appendix's .gitignore additions; preserve all existing rules. The current ignore file covers JSON/CSV/log extensions under results but misses raw text and copied source. The added /results/\* rule ignores entire run trees while retaining results/.gitkeep. Keep environment logs, error.txt, copied package source, and per-run README under results. Ignore .venv-reproduce/, secrets, model weights, and caches.

Commit source, tests, frozen inputs, environment/model locks, schemas, and experiments/phase1/README.md with commands and evidence locations. Raw outputs remain under results, as CONTRIBUTING.md and experiments/README.md require. An ignored local directory is not a durable evidence archive; record the team's agreed archive location when one is created. Use repository-relative paths in model locks, never personal tokens or home-directory paths.

### **5.2 Portable environment: Urmi and Jainil**

Use native Python 3.11+; 3.11 is the common starting point. First add all appendix files. Editable installation makes src-layout imports work without PYTHONPATH changes:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[dev]"
python -m pip check
python -c "import weave; print(weave.__file__)"
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest -q tests/test_candidates.py tests/test_evidence.py
python -m pip freeze --all --exclude-editable > experiments/phase1/requirements-portable-lock.txt
```

The printed package path must be this checkout's src/weave. Portable modules do not import MLX. New public source functions are annotated, and supplied files are Ruff formatted. Use ruff format while authoring and \--check during verification. If python3.11 is unavailable, use an installed native 3.11+ and record its exact version.

Freeze after passing checks. \--exclude-editable avoids committing a machine-specific editable path; \--all retains packaging tools. Reinstall the source checkout separately during reproduction. Do not install M4 extras or its platform-specific lock on Intel.

### **5.3 M4 environment: Mohit**

Record hardware/OS/Python first. The reviewed MLX guidance requires native Apple Silicon Python and macOS 14.0+; MLX-LM metadata requires Python 3.11+. Recheck release requirements when installing. Do not measure under a Rosetta x86\_64 Python.

```sh
uname -m
sw_vers
system_profiler SPHardwareDataType
sysctl -n hw.memsize
python3.11 --version
which python3.11
df -h .
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[dev,m4]"
python -m pip check
```

Save environment-preinstall.txt and environment-install.txt under a fresh results/ENV-\<unique-id\>/ directory. Record power/low-power settings, relevant background applications, known thermal conditions, and cached weights. Do not infer thermals from a speed change alone. Replace every \<unique-id\> placeholder.

Bootstrap resolves an official release, not a claimed pretested M4 combination. After probe, smoke, runner, and tests pass, freeze the exact environment:

```sh
python scripts/phase1/probe_mlx.py --output-dir results/ENV-<unique-id>/compatibility
python -m pytest -q tests/test_installed_pipeline.py
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pip freeze --all --exclude-editable > experiments/phase1/requirements-m4-lock.txt
python -m pip check
```

The probe creates a fresh compatibility directory rather than overwriting evidence. It records versions, device info, source hashes/copies, and a split signature. The installed test checks unequal \[18, 10\] counts on a dummy layer list; it does not initialize two ranks or prove distributed inference.

## **6\. Pin and verify the model**

Run the supplied prepare\_model module on M4. It resolves main once to an immutable 40-character model commit, downloads exactly that snapshot to a fresh directory, records the config hash, and checks the 7B Qwen2 geometry and 4-bit conversion. A repo name ending in 4bit is not enough by itself.

```sh
python scripts/phase1/prepare_model.py \
  --repo mlx-community/Qwen2.5-7B-Instruct-4bit \
  --model-dir models/qwen7b \
  --manifest experiments/phase1/model-lock.json
mlx_lm.generate \
  --model models/qwen7b \
  --prompt "Explain model partitioning in two short sentences." \
  --max-tokens 64
```

Save the CLI terminal log. The first download is setup time; do not include network download time in an inference latency. The model lock is immutable for this experiment. Use a new directory/manifest for a new revision rather than relabeling old evidence. For an offline repeat on a second machine, transfer or download the same immutable snapshot and verify the manifest/config before execution. Weight-file checksums are recommended for the final reproducibility archive; the current script records the HF revision and config checksum and does not claim to hash every downloaded weight.

Published 7B geometry is 28 layers, hidden size 3584, 28 attention heads, four KV heads, and head dimension 128\. The 14B configuration has 48 layers, hidden size 5120, 40 attention heads, eight KV heads, and head dimension 128\. Always inspect the converted snapshot's own config rather than transplanting geometry from another model.

On the 16 GB Air, begin with the short 7B workload and close unrelated memory-heavy applications. Do not set a larger wired-memory limit or disable system protections as a routine Phase 1 step. If the run is killed, memory pressure is severe, or the environment cannot load 7B, retain the failure and use the smaller 3B model only to isolate the installation issue. Record this as a blocked 7B baseline, not a completed equivalent experiment.

## **7\. MLX pipeline compatibility and layer mapping**

The project uses contiguous pipeline parallelism first. Tensor parallelism divides tensors within a layer and has different communication/memory behavior; a tensor-sharding hook does not establish a compatible pipeline loader. JACCL/RDMA is not a Phase 1 dependency. The M4 Air is not an assumed Thunderbolt 5 RDMA node. Later RING/TCP bring-up is the practical fallback on supported links; JACCL requires its own verified hardware, OS, topology, and backend gate.

The inspected Qwen2Model inherits PipelineMixin. Its pipeline method accepts split. Current sharded\_load calls pipeline during shard-file selection and again on the final model, without a custom split parameter. Changing only the second call can select the wrong checkpoint shard files; calling pipeline again on an already truncated model can also destroy the original layer-count basis. A two-line late-stage patch is not a complete adapter.

### **7.1 Canonical mapping**

All stored layer ranges use a half-open interval \[start\_layer, end\_layer\_exclusive). The last included layer is end\_layer\_exclusive \- 1\. This matches Python slicing and MLX's start\_idx/end\_idx. Do not reuse the old inclusive end\_layer field.

For logical counts \[10, 18\] and execution order \[A, B\]:

| Device | Logical stage | Layers | Current MLX rank | Native split entry |
| ----- | ----- | ----- | ----- | ----- |
| A | 0 | \[0, 10), i.e. 0–9 | 1 | split\[1\] \= 10 |
| B | 1 | \[10, 28), i.e. 10–27 | 0 | split\[0\] \= 18 |

The launch host order by native rank must therefore be \[B, A\], and mlx\_split\_by\_rank must be \[18, 10\]. The adapter must verify the actual rank/device identity after launch; reversing counts without reversing the rank-to-device map is wrong. Ranks belong to a particular launch group and must not be stored as a permanent property of a physical Mac.

The installed mapping to verify is start\_idx \= sum(native\_split\[rank \+ 1:\]); end\_idx \= start\_idx \+ native\_split\[rank\]. Use a fresh model instance for each plan. Once pipeline has removed other layers, that object must not be reused as the full source for another split.

### **7.2 Single-device fallback**

A one-device candidate assigns \[0, L) to one Mac and has one logical stage and one native rank, both zero. Execute it in one process using ordinary mlx\_lm.load. Do not launch a two-rank group and give an inactive rank zero layers. Both Macs get independent one-device candidates when both are available; the larger/faster Mac may be the right answer.

### **7.3 Later adapter experiment, explicitly outside Phase 1 acceptance**

After the communication gate passes, one simple research path is a full local immutable snapshot on each participating machine, lazy model load, exactly one custom pipeline call, then materialization. The following is an API experiment to validate on real hardware, not a verified production loader or a memory-peak guarantee:

```py
def load_local_split(model_dir, world, logical_counts):
    import mlx.core as mx
    from mlx_lm import load
    native_split = list(reversed(logical_counts))
    if len(native_split) != world.size():
        raise ValueError("split/group mismatch")
    model, tokenizer = load(model_dir, lazy=True)
    if sum(native_split) != len(model.model.layers):
        raise ValueError("layer count mismatch")
    model.model.pipeline(world, split=native_split)
    expected_start = sum(native_split[world.rank() + 1:])
    expected_end = expected_start + native_split[world.rank()]
    if (model.model.start_idx, model.model.end_idx) != (expected_start, expected_end):
        raise RuntimeError("installed pipeline mapping differs")
    mx.eval(model.parameters())
    mx.synchronize()
    return model, tokenizer
```

Lazy loading can still map/read complete checkpoint files or allocate temporary objects. Profile load-time peak separately from steady-state parameters. This path trades disk duplication for implementation simplicity. A more efficient adapter can vendor a small reviewed copy of the pinned loader and thread the split through both shard-discovery and final construction paths; test both indexed multi-file and single-file checkpoints. Avoid a global monkey-patch.

A later distributed generation loop must make all ranks execute matching collectives, identical prompt tokens, compatible cache behavior, deterministic sampling or token synchronization, and identical stopping decisions. Printing only on rank 0 is fine; skipping generation on the other ranks is not. A layer-count contract test alone proves none of these runtime properties.

## **8\. Versioned interfaces and evidence rules**

Use schema\_version 2.0.0 on every new JSON artifact and every CSV row. Generate the JSON Schemas with the supplied module, then commit the generated files after reviewing them. JSON Schema validates shape; Python validators and tests enforce cross-field invariants such as layer-cover equality, identifier agreement, and trace order.

```sh
python -m weave.contracts.schemas --output-dir experiments/phase1/schemas/v2
```

| Artifact | Producer → consumer | Required meaning |
| ----- | ----- | ----- |
| model-lock.json | Mohit → runner/teammates | Immutable model revision, local snapshot location, config and its hash |
| workload.json | Urmi → runner | Exact prompt, token bounds, concurrency 1, warmup/measured counts, greedy sampler, seed |
| compatibility.json | Mohit → team | Installed package versions, source hashes/signatures, runtime verification explicitly false |
| model\_profile.json | Runner → Urmi/Jainil | Materialized parameter-byte tally, by layer plus replicated tensors; excludes KV/workspace/OS |
| trace.csv | Runner → validator/analysis | Ordered process/request/token events with consistent identity and measured source |
| run\_summary.json | Runner → validator/team | Status, per-request metrics, exact model/workload identity, errors if failed |
| cluster-state-SIMULATED.json | Jainil → candidate generator | Wholly synthetic planning fixture, explicit workload envelope and absolute fresh-process budgets |
| candidate\_plans.json | Jainil → Mohit/team | All candidates, logical/native mapping, memory components, rejection reasons, no selected plan |

Generate schemas into the committed configuration directory only on first addition. On later checkouts, retain those schemas; regenerate into a fresh results/SCHEMA-CHECK-\<unique-id\>/ directory using \--output-dir and compare before proposing a change. The generator deliberately refuses to overwrite files.

Every run gets a unique directory. Code uses exclusive creation to prevent accidental overwrites. run\_id is the directory name; request\_id is unique inside that run; device\_id is a stable physical identifier; partition\_version links metrics to placement; source\_snapshot\_id/profile\_id link plans to their planning inputs. Give the synthetic identities SIMULATED prefixes to prevent accidental joins with measured evidence.

All memory estimates must say whether they are measured, derived from measured inputs, configured reserves, simulated, or unknown. In Phase 1, the candidate command intentionally accepts only a wholly simulated state and always writes simulation=true, executable=false, runtime\_verified=false, and selected\_plan\_id=null. This is a safety boundary against presenting an uncalibrated offline estimate as a measured deployment. Extending it to mixed real/synthetic state requires a separately reviewed interface and per-component provenance; simply changing simulation to false is not permitted.

Use explicit controlled error codes such as MEMORY\_BUDGET\_EXCEEDED, UNSUPPORTED\_SCHEMA, MISSING\_PROFILE, MODEL\_REVISION\_MISMATCH, or STALE\_STATE in later interfaces. The current prototype emits a structured memory-rejection code and raises clear validation errors for invalid inputs. It fails closed instead of silently substituting zero for missing memory terms.

### **8.1 Trace contract**

Every row contains schema\_version, sequence, run\_id, device\_id, source, wall\_time\_utc, monotonic\_ns, request\_id, request\_kind, event, policy, partition\_version, workload\_id, model\_revision, token\_count, and the five memory columns listed in the runner. request\_id/request\_kind are blank for process-wide events. request\_kind is warmup or measured for request events. Empty memory cells mean not sampled; they do not mean zero bytes.

The successful process lifecycle is process\_start → model\_load\_start → model\_load\_end → the serial request groups → process\_end. Each successful request has request\_start, first\_token, ordered token events, first\_visible\_text, and request\_end. Full token tracing is the default acceptance mode. The explicitly recorded boundary mode omits per-token rows only for the matched overhead experiment. error terminates a failed attempt and is not a valid successful lifecycle. The sequence number disambiguates events with the same monotonic timestamp.

Monotonic clocks are comparable only within the same process/clock domain. Wall-clock UTC is useful for approximate alignment across devices but is not proof of synchronized clocks. Later distributed telemetry needs clock-offset checks or coordinator-relative measurements, plus device/process/rank identifiers. Do not subtract independent Macs' perf\_counter\_ns values.

## **9\. Urmi's measurement implementation and verification**

Save the supplied workload at benchmarks/workloads/phase1-smoke.json. It uses one warmup request and three measured requests. This small repeat count checks reproducibility; it is not adequate for a p95 or a performance significance claim. Exclude warmup from measured aggregates, preserve each request result, and keep model-load timing separate from request timing.

The supplied runner imports MLX only after creating its evidence directory and process-start event. It copies inputs, checks the model lock/config, loads the model once, creates a parameter-byte profile, runs serial requests with fresh default prompt caches, and writes the summary in a finally block. Exceptions retain an error file and failed summary. An OS-level kill can prevent finally from running; retain that incomplete directory and report it as incomplete, never success.

```sh
source .venv/bin/activate
PHASE1_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m4air-b0-qwen7b-v2"
python -m weave.telemetry.baseline \
  --workload benchmarks/workloads/phase1-smoke.json \
  --model-lock experiments/phase1/model-lock.json \
  --output-dir "results/$PHASE1_RUN_ID" \
  --device-id mohit-m4-air
python -m weave.telemetry.validate_evidence "results/$PHASE1_RUN_ID"
```

If a directory already exists, use a new identifier. Do not delete the old attempt to rerun. Copy environment logs and the exact requirements lock into or alongside the ignored results run directory and reference them in its README.md. Keep committed reproduction instructions separately at experiments/phase1/README.md. The runner records the git commit when available, workload/model-lock hashes, and the M4 requirements hash when the lock file exists. Acceptance requires those provenance records, even though the code can run before the first local commit.

### **9.1 Metric definitions**

| Field | Definition | Limit |
| ----- | ----- | ----- |
| ttft\_ms | request\_start to first yielded token-bearing stream callback | Includes prompt tokenization and instrumentation; it is observable callback TTFT, not GPU kernel timing. |
| first\_visible\_text\_ms | request\_start to first nonempty decoded text segment | May differ from TTFT because token text can be empty until detokenization advances. |
| end\_to\_end\_ms | request\_start to completion after synchronization | Excludes model load/download; includes tokenization, decode, finalization, and instrumentation. |
| observed\_tpot\_ms | First-to-last distinct token callback divided by token intervals | Null for fewer than two tokens; terminal-token treatment follows the recorded library count. |
| library\_prompt\_tps / library\_generation\_tps | The library's reported metrics | Preserve their definitions; do not relabel them as independently measured throughput. |
| request\_peak\_mlx\_bytes | MLX peak since reset immediately before that request | Contains resident model memory plus request allocations; it is not incremental KV memory alone. |
| model\_load\_peak\_bytes | MLX peak during model loading/materialization | Needed because load/reconfiguration can peak above steady-state use. |

The inspected streaming implementation can emit an empty text segment and a finalization response. Do not count decoded characters as tokens or blindly increment one token per response. The runner tracks the library generation counter and ignores a repeated final counter. Its recorded token count may include terminal EOS. Empty-output runs fail acceptance. Recheck these semantics against the installed package after any upgrade.

### **9.2 Memory metrics and overhead**

Record MLX active, cache, and peak bytes, process RSS, and system available bytes. These overlap and must not be added together as if independent pools. Parameter nbytes describes model tensors; it does not describe complete process or unified-memory pressure. Record system swap/memory-pressure observations separately where available. A successful short run does not establish safety at a long context or higher concurrency.

The runner samples memory at lifecycle boundaries and buffers token rows; it avoids a full memory sample and flush for every token. Instrumentation still changes latency. Phase 1 must include a matched token-logging overhead check using the supplied \--token-trace full and \--token-trace boundary modes. Boundary mode retains request timing, memory samples, identity, and counters but omits per-token CSV rows. Alternate order and keep the same frozen workload; each run contains three measured trials after warmup. For example, run full, boundary, boundary, full in four distinct directories, and validate every run. Record raw results and the median latency difference. A difference above 5% is an investigation trigger, not a promised bound. This comparison isolates per-token logging overhead; it does not measure all instrumentation cost. A CLI run with extensive console printing is not a matched uninstrumented control.

```sh
# Repeat with full, boundary, boundary, full; use a fresh ID each time.
PHASE1_OVERHEAD_ID="$(date -u +%Y%m%dT%H%M%SZ)-m4air-overhead-boundary"
python -m weave.telemetry.baseline \
  --workload benchmarks/workloads/phase1-smoke.json --model-lock experiments/phase1/model-lock.json \
  --output-dir "results/$PHASE1_OVERHEAD_ID" --device-id mohit-m4-air \
  --token-trace boundary
python -m weave.telemetry.validate_evidence "results/$PHASE1_OVERHEAD_ID"
```

### **9.3 Validator acceptance and failure tests**

The validator checks consistent identities, versions, source labels, contiguous sequence numbers, nondecreasing timestamps, process lifecycle, serial request boundaries, warmup/measurement counts, exactly one first-token/text event, token counts, nonempty output, nonnegative memory fields, and summary/trace timing agreement. It rejects failed/incomplete runs. Passing shape validation alone is insufficient.

Run portable corruption tests first. Then verify one real successful M4 run, one intentionally invalid input such as concurrency 2, and one copied trace whose identity or TTFT has been changed. Keep corruption copies under tests/ or a clearly marked validation-fixtures directory; never include their fake values in experiments or charts.

## **10\. Jainil's memory feasibility model**

For each participating device i, require:

predicted\_memory\_i \= assigned\_layer\_weight\_bytes\_i \+ replicated\_weight\_bytes\_i \+ KV\_upper\_bound\_i \+ workspace\_reserve\_i \+ safety\_margin\_i ≤ safe\_process\_budget\_i.

All terms use integer bytes. Replicated Qwen tensors must be counted on every rank. Use actual packed quantized parameter nbytes, including scales/biases, rather than dividing a floating-point parameter count by eight and assuming all overhead disappears.

For a dense attention model with homogeneous cache geometry, a conservative KV term is:

KV\_upper\_bound\_i \= 2 × assigned\_layers\_i × concurrency × (prompt\_tokens\_upper\_bound \+ max\_new\_tokens) × KV\_heads × head\_dim × KV\_bytes\_per\_element.

The factor two is for keys and values. Weight quantization does not imply a 4-bit KV cache. The synthetic fixture uses four bytes per KV element as a conservative planning assumption, not a measurement of MLX's cache dtype. Later, inspect the real cache dtype and allocation behavior. Add cache block rounding, alignment, retained/prefix caches, and extra buffers where applicable. This simple equation does not cover hybrid attention/state-space models, quantized-cache overhead, or model-specific cache sharing; those require separate validated profiles.

workspace\_reserve must cover activations, prefill/decode workspace, allocator effects, and relevant runtime allocations not counted elsewhere. safety\_margin is an additional guard for prediction error. Neither may silently default to zero. A configured reserve is not measured proof; calibrate it over the supported request envelope. Capture both load peak and inference peak. A runtime implementation must check the transient maximum needed while reconstructing/reloading a plan, not only the final shard size.

safe\_process\_budget is an absolute model-process budget for a fresh process, after an explicit OS/background reserve. It is not the machine's advertised RAM and not simply current system\_available\_bytes measured while the existing model is already resident. Comparing a full predicted footprint to post-load available memory double-counts the resident model; adding all current RSS back without checking reclaimability can be unsafe. Phase 1 avoids that live-state ambiguity by using a plainly synthetic fresh-process fixture. Later telemetry must define resident, reclaimable, external pressure, and budget freshness before driving execution.

For M4 pilot runs, document a conservative initial policy cap, for example 8 GiB for the inference process, and reduce it if observed background pressure requires it. This example is a planning cap, not a guarantee or a measured capacity. Do not derive a real M4 layer profile from the single end-to-end peak alone. Use model\_profile.json for tensor bytes, then calibrate the other terms independently.

## **11\. Jainil's candidate generator and worked checks**

The fixture generator creates a wholly synthetic pressure case with 28 equally sized layer entries, a replicated fixed cost, KV assumptions, workspace/safety reserves, and two artificial absolute budgets. Its 3 GiB first-device budget represents an illustrative constrained scenario; it does not claim that Mohit's Air only has 3 GiB available. The second-device budget is also invented test input, not a future M5 Pro specification. All identities and output paths are labeled SIMULATED.

```sh
# First addition only; reuse this committed input on later checkouts.
python -m weave.profiler.simulated_fixture --output experiments/phase1/cluster-state-SIMULATED.json
PHASE1_SIM_ID="SIMULATED-$(date -u +%Y%m%dT%H%M%SZ)-candidates-v2"
mkdir -p "results/$PHASE1_SIM_ID"
python -m weave.scheduler.candidates \
  --state experiments/phase1/cluster-state-SIMULATED.json \
  --output "results/$PHASE1_SIM_ID/candidate_plans.json"
```

If the fixture already exists, skip its creation. To verify regeneration, create a copy under a fresh results/SIMULATED-\<unique-id\>/ directory and compare the JSON. The candidate command only reads the frozen input.

Expected output for the supplied fixture is 56 candidates and 15 memory-feasible estimates. This count is a deterministic fixture check, not a measured performance result. There are two single-device candidates plus 2 × 27 ordered two-device cuts. Only one device in the input produces one single-device candidate. If the model has one layer, two devices still produce only the two one-device candidates; there is no positive two-way cut.

For every candidate, verify unique participating devices, positive counts, sum(counts) \= L, ordered contiguous intervals covering each layer exactly once, consistent logical/native ordering, all memory terms, and structured rejection reasons. The output never selects a winner. A nonzero feasible count cannot establish the best plan without measured compute, network, workload, and switching-cost inputs.

The supplied \[10, 18\] pressure-case assignment to the first/second devices predicts more than 3 GiB on the first synthetic device and is rejected. Reducing its allocated layer count can pass. A single-device candidate on the large synthetic device can also pass. Inspect these side by side to demonstrate that fallback and rejection are intentional behavior.

```sh
python - <<'PY'
import json
from pathlib import Path
from jsonschema import validate
validate(json.loads(Path("experiments/phase1/cluster-state-SIMULATED.json").read_text()),
         json.loads(Path("experiments/phase1/schemas/v2/cluster_state_simulated.schema.json").read_text()))
print("PASS: synthetic state shape")
PY
python -m pytest -q tests/test_candidates.py tests/test_evidence.py
```

Validate the generated candidate JSON against experiments/phase1/schemas/v2/candidate\_set.schema.json as well, using the actual output path from the preceding command. Tests verify the cross-field rules the schema cannot express by itself. Candidate IDs are deterministic for a fixed snapshot/profile/order/counts. A changed profile or input snapshot needs a new ID, not an edited historical plan.

## **12\. Integration gates and definition of done**

| Gate | Owner | Evidence and acceptance |
| ----- | ----- | ----- |
| G0: shared contract | All | Repository commit, shared bootstrap, author/long-term reviewer map, and version 2.0.0 reviewed; byte units, half-open intervals, identifiers, warmup rules and provenance agreed; original 1.0.0 artifacts preserved. |
| G1: hardware/environment | Mohit | arm64 macOS, actual physical bytes recorded, compatible Python/packages; pip check passes; exact working M4 lock saved. |
| G2: model/API | Mohit | Immutable Qwen7B revision/config lock; CLI produces nonempty text; probe records split hook; installed unequal mapping test passes. This is source compatibility only. |
| G3: measured evidence | Mohit \+ Urmi | One warmup plus three measured requests in full token-trace mode in a new run directory; successful validator; outputs, memory/profile, identities, environment and errors retained; matched token-logging overhead documented; no fabricated metrics. |
| G4: portable planner | Jainil | 56/15 fixture check; exact-cover tests; both orders; each single-device case; one-device input; memory boundary/rejection; invalid inputs rejected; schemas validate; every plan unselected/unverified. |
| G5: reproduction/review | All | Editable install; Ruff lint/format and portable tests pass; public functions annotated; raw evidence ignored; a teammate repeats commands from a clean environment; owner-reviewed PR before merge; evidence labels correct; checkpoint note lists real successes and remaining blockers. |

Phase 1 is complete only when all required gates pass. A source probe is not a substitute for G3, and a synthetic test is not a substitute for a real M4 run. If a gate is blocked, record the blocker, observed error, owner, next action, and dependent work that remains blocked. Complete the independent work without making up the missing result.

The final evidence package for Mohit includes environment logs, requirements lock, compatibility report/source hashes, model lock, and CLI log. Urmi's package includes copied workload/model lock, trace.csv, run\_summary.json, model\_profile.json, validator output, overhead comparison, and a short reproduction README. Jainil's package includes the explicitly synthetic state, candidate JSON, tests, validator/schema results, and mapping examples. The team package includes the interface-freeze note, review note, test summary, and Phase 2 dependency list.

### **12.1 Clean reproduction and git hygiene**

Use the agreed source commit and Python minor version. A checkout cannot execute the guide until its appendix additions are present in the agreed feature branch/PR. Record git rev-parse HEAD; dependencies alone do not identify the source.

On M4, install the M4 lock into a fresh environment and then install this checkout. Reuse the immutable model snapshot but state that weights are cached. A fresh process is not a cold filesystem cache.

```sh
python3.11 -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r experiments/phase1/requirements-m4-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -c "import weave; print(weave.__file__)"
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest -q
# Repeat baseline and validator with a NEW results/ run ID.
```

For Intel reproduction, substitute experiments/phase1/requirements-portable-lock.txt and run the portable test files. An M4-only module skipped without MLX is not a passed hardware gate.

Check ignored raw evidence types and the staged diff:

```sh
git check-ignore results/IGNORE-CHECK/trace.csv results/IGNORE-CHECK/error.txt
git check-ignore results/IGNORE-CHECK/compatibility/pipeline.py results/IGNORE-CHECK/README.md
git diff --check
git status --short
# After staging explicit intended source/config paths:
git diff --cached --check
git diff --cached --stat
```

All four results paths must be reported ignored. Keep raw runs, weights, source snapshots, and secrets out of the staged change. Commit intended code/configuration, push the working branch under its author's account, and obtain teammate review before a focused PR merges into main. No GitHub push or merge is performed as part of preparing this guide.

### **12.2 Checkpoint demonstration**

Mohit shows actual hardware, frozen environment, model commit, CLI output, and pipeline source/mapping checks. Urmi runs the instrumented baseline and validator, shows one measured request's TTFT reconstruction, excludes warmup, and explains the memory/profile limits. Jainil regenerates the fixture candidates, shows one single-device candidate, one accepted two-device estimate, one rejected estimate, and the \[10,18\] logical → \[18,10\] native mapping. The team finishes with the missing second-device/backend/adapter gates and the approved next step. The demonstration must say which commands ran on real Apple hardware and which ran as portable synthetic tests.

## **13\. Risk controls and stop conditions**

| Risk / early signal | Required response | Owner |
| ----- | ----- | ----- |
| Package/source differs from the reviewed split API | Save installed source/hash, fail the mapping gate, choose and freeze a working official release before dependent integration. | Mohit |
| M4 pressure, swap growth, kill, or severe slowdown | Preserve failure/incomplete attempt; shorten within the declared workload envelope, reduce background pressure, or diagnose with 3B; do not mark Qwen7B complete. | Mohit \+ Urmi |
| Model geometry/quantization/revision mismatch | Reject input; repair or create a new lock/profile/run. Never patch old evidence labels. | Mohit |
| Logical stage and launch rank confused | Require unequal-count mapping test and rank/device readback before distributed execution. | Jainil \+ Mohit |
| Memory estimate omits replicated weights or cache | Reject an incomplete profile; keep parameter, KV, workspace and safety terms visible. | Jainil \+ Urmi |
| Post-load available memory treated as full-process budget | Stop live feasibility integration; define resident/reclaimable/external memory accounting first. | Urmi \+ Jainil |
| Empty stream/final response counted as ordinary text tokens | Check library counter and separate first token from first text; keep empty-output runs failed. | Urmi |
| Trace appears successful after an exception/kill | Require successful summary, complete lifecycle and validator; absent final evidence is incomplete. | Urmi |
| Synthetic state leaks into measured charts | Separate paths/IDs/source fields and show provenance at every result boundary. | All |
| Second Mac or network hardware delayed | Finish portable planner and real single-device evidence; keep distributed claims blocked. | All |
| Full reload dominates future benefit | Measure downtime/load peak and keep switching cost in the decision rule; staying on one device/static plan is valid. | Mohit \+ Urmi \+ Jainil |
| Phase expands into a new runtime or model support effort | Retain one dense model and a thin MLX adapter; postpone discovery UI, tensor parallelism, custom quantization, and optional systems. | All |

A further modeling concern: for one autoregressive request at concurrency 1, sequential pipeline stages do not automatically overlap. A first latency approximation is the sum of stage compute plus transfer/collective costs, not max(stage\_time). A bottleneck/max-stage throughput expression requires an actually overlapped pipeline workload. Account for Qwen's observed collective behavior, not only a single idealized boundary transfer. Fit separate prefill/decode models later from measurements.

## **14\. Phase 2 and later: dependency order**

The P2–P5 labels below are gates in this guide, not replacements for README.md's M1–M7 milestones. Multi-Mac reproduction is M1; full profiling is M2; static placement is M3; adaptation is M4; experiments/evaluation are M5–M6. General node-loss fault tolerance and stall recovery remain stretch goals. A bounded abort/rollback during a planned partition change is a narrower execution-safety requirement, not a promise of cluster fault tolerance.

Jainil owns later execution/backend/partition application; Mohit owns profiling/telemetry/estimator calibration; Urmi owns policy ranking and comparisons/ablations. The early prototype handoffs do not change these README responsibilities.

| Next gate | Must be true first | Required evidence before advancing |
| ----- | ----- | ----- |
| P2-A: second-node inventory | Second Apple Silicon Mac available | Real hardware/OS/memory, matching model commit/software, rank/device identities; no assumed future-Mac specs. |
| P2-B: transport | Verified SSH/topology and supported RING/TCP backend | Two-process all\_sum prints group size 2 and numeric \[2,2,…\] on both machines; rank logs retained. One process producing ones is not distributed verification. |
| P2-C: equal split | Transport and deterministic multi-rank generation loop | Correct \[14,14\] layer cover; model output/token agreement with single-device reference under declared tolerances; no deadlock; per-rank memory evidence. |
| P2-D: unequal split | Split-aware loader passes shard-selection and rank tests | A non-symmetric split such as logical \[10,18\] runs correctly; fresh reconstruction; native rank/device mapping recorded; single-device fallback launches correctly. |
| P2-E: calibration/baselines | Real model/runtime path on both Macs | Actual per-stage prefill/decode, network and KV/workspace measurements; baseline reproducibility; transient load peak. |
| P3: scheduler ranking | Calibrated inputs and feasible complete candidate sets | Cost prediction error, static comparisons, explicit single-device alternative, no hidden failures. |
| P4: safe adaptation | Correct ranking \+ runtime manager | Drain at safe request boundary → reconstruct/reload → health check → publish partition version, with timeout/rollback and measured downtime. No mid-request KV migration assumption. |
| P5: research evaluation | Reliable execution and switching | Predeclared baseline matrix, ablations, sufficient repeats, tail metrics, failure/OOM accounting, measured switching break-even, reproducibility archive. |

The communication check can be as small as scripts/phase2/distributed\_smoke.py, supplied below as a later-phase example. It asserts group size 2 so a silent one-process no-op cannot pass:

```py
import mlx.core as mx
world = mx.distributed.init(backend="ring")
if world.size() != 2:
    raise RuntimeError("expected two launched ranks")
x = mx.distributed.all_sum(mx.ones(4))
mx.eval(x)
if not bool(mx.all(x == 2).item()):
    raise RuntimeError("collective failed")
print({"rank": world.rank(), "size": world.size(), "sum": x.tolist()})
```

Run this only after both machines have the same script and verified launcher/hostfile. Inspect mlx.launch \--help in the pinned environment before using launcher arguments. An illustrative later command is:

```sh
mlx.launch --help
mlx.launch --backend ring --hostfile experiments/phase2/hosts-ring.json -- \
  /absolute/path/to/the/pinned/venv/bin/python scripts/phase2/distributed_smoke.py
```

The absolute interpreter/script path must exist on each host, or launcher configuration must explicitly map per-host paths. Do not copy invented IP addresses or a synthetic device order into a real hostfile. Retain launched rank logs and compare them with device\_id\_by\_mlx\_rank. JACCL remains optional until its own prerequisites are met.

Later baseline definitions must stay aligned with Workbook 1: single-node MLX on each feasible Mac/best single device; equal splitting; deterministic memory-based/static hardware-aware placement; a frozen profiled static baseline; and the adaptive policy. exo and dnet are conditional comparison systems only when the same model/workload/hardware settings can be retained. The main research comparison is against the declared profiled static baseline with failures/OOMs accounted for, not a weak arbitrary split.

Evaluate short and long contexts only after pilot feasibility; distinguish prefill/decode, actual generated token counts, warm/cold process state, concurrency and network topology. Three Phase 1 requests cannot support the later p95 claim. Reconfiguration evaluation must include downtime, temporary memory peak, failed requests, cooldown/hysteresis, and cases where KEEP or single-device execution wins.

## **15\. Source references and validation status**

Repository alignment was checked against 8ac11046aa181d9122bb362f94b4cabb85e238e5 on October 20, 2026\. This is the audited baseline, not an instruction to discard subsequent work. Inspect upstream changes before later application. Retained MLX findings are source-level observations; the installed probe/hash and hardware tests determine the team's frozen runtime compatibility.

1. [Audited repository](https://github.com/JNR-10/weave/tree/8ac11046aa181d9122bb362f94b4cabb85e238e5); [README](https://github.com/JNR-10/weave/blob/8ac11046aa181d9122bb362f94b4cabb85e238e5/README.md), [CONTRIBUTING](https://github.com/JNR-10/weave/blob/8ac11046aa181d9122bb362f94b4cabb85e238e5/CONTRIBUTING.md), [experiment rules](https://github.com/JNR-10/weave/blob/8ac11046aa181d9122bb362f94b4cabb85e238e5/experiments/README.md), and [benchmark rules](https://github.com/JNR-10/weave/blob/8ac11046aa181d9122bb362f94b4cabb85e238e5/benchmarks/README.md).  
2. [September 23 tasks](https://github.com/JNR-10/weave/blob/8ac11046aa181d9122bb362f94b4cabb85e238e5/meetings/2026-09-23-team.md) and [Workbook 1](https://docs.google.com/document/d/11jah0T4t5GaMup-dEpVh6-Q2XpYC3UjBI1kbMWTiGik/edit). Section 3 explains the authorship/long-term ownership distinction.  
3. [MLX installation](https://ml-explore.github.io/mlx/build/html/install.html).  
4. [MLX-LM source](https://github.com/ml-explore/mlx-lm) and [package metadata](https://github.com/ml-explore/mlx-lm/blob/main/pyproject.toml).  
5. [Qwen2 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py) and [PipelineMixin](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/pipeline.py).  
6. [Loading utilities](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py) and [streaming generation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py).  
7. [Distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html).  
8. [Converted 7B config](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit/blob/main/config.json), [official 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json), and [official 14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json).

This revision's authoring checks used an isolated checkout of the audited repository with the proposed additions. Editable installation, package/CLI imports, Ruff lint/format, portable tests, schema generation, deterministic 56-candidate/15-feasible synthetic output, and git-ignore behavior were checked there. The suite has 22 passing portable tests; the M4-only installed-pipeline module skips on Linux. This supersedes testing only a standalone top-level phase1 scaffold. It does not mean the proposed additions are merged into GitHub.

The audited main branch contains scaffold packages, not a completed implementation. MLX inference, real GPU memory measurements, two-Mac communication/equivalence, and online switching were not executed during authoring. They remain hardware gates. No repository push, PR, or merge was performed.

## **16\. How to use the executable appendices**

Copy every complete block to its stated repository path, preserving indentation, straight quotes, and backslashes. Add all files before the editable-install/test commands. The new contracts initializer is supplied; preserve the existing src/weave and four component initializers. The .gitignore appendix is an addition, not a replacement. If a newer checkout already contains a file, merge/review the intended change.

Run package commands from the root after editable installation; run setup scripts through scripts/phase1 paths. New public source functions are annotated and files Ruff formatted. The future-phase loader/communication examples are separate unverified experiments, not required Phase 1 files. Passing portable checks does not replace the real M4 baseline, second-Mac integration, or teammate review.

## **Appendix A. Mohit: immutable model snapshot and lock**

Save as `scripts/phase1/prepare_model.py`.

```py
"""Resolve one immutable HF revision, then download exactly that snapshot."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    p.add_argument("--revision", default="main")
    p.add_argument("--model-dir", default="models/qwen7b")
    p.add_argument("--manifest", default="experiments/phase1/model-lock.json")
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    out, model_dir = Path(args.manifest), Path(args.model_dir)
    if out.exists() or model_dir.exists():
        raise FileExistsError("choose new paths; never overwrite an existing model lock/snapshot")
    info = HfApi().model_info(args.repo, revision=args.revision)
    revision = info.sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("expected immutable 40-character model commit")
    snapshot_download(repo_id=args.repo, revision=revision, local_dir=str(model_dir))
    config_bytes = (model_dir / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("model_type") != "qwen2" or config.get("num_hidden_layers") != 28:
        raise ValueError("this Phase 1 profile requires Qwen2.5-7B geometry")
    quant = config.get("quantization", {})
    if quant.get("bits") != 4:
        raise ValueError("expected 4-bit converted model")
    manifest = {
        "schema_version": "2.0.0",
        "model_id": args.repo,
        "model_revision": revision,
        "model_dir": str(model_dir),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": config,
        "source": "downloaded_metadata",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(manifest, f, indent=2)
    print(
        json.dumps(
            {
                "model_id": args.repo,
                "model_revision": revision,
                "manifest": str(out),
                "model_dir": str(model_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
```

## **Appendix B. Mohit: installed MLX compatibility probe**

Save as `scripts/phase1/probe_mlx.py`.

```py
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
```

## **Appendix C. Urmi: exact smoke workload**

Save as `benchmarks/workloads/phase1-smoke.json`.

```json
{
  "schema_version": "2.0.0",
  "workload_id": "qwen7b-smoke-v2",
  "prompt": "Explain model partitioning in two short sentences.",
  "prompt_max_tokens": 512,
  "max_new_tokens": 64,
  "concurrency": 1,
  "warmup_requests": 1,
  "measured_requests": 3,
  "temperature": 0,
  "seed": 0
}
```

## **Appendix D. Urmi: complete single-device evidence runner**

Save as `src/weave/telemetry/baseline.py`.

```py
"""Apple Silicon only. Observed callback timings, serial requests, fresh KV per request."""

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weave.profiler.model_profile import build_parameter_profile

VERSION = "2.0.0"
FIELDS = [
    "schema_version",
    "sequence",
    "run_id",
    "device_id",
    "source",
    "wall_time_utc",
    "monotonic_ns",
    "request_id",
    "request_kind",
    "event",
    "policy",
    "partition_version",
    "workload_id",
    "model_revision",
    "token_count",
    "process_rss_bytes",
    "system_available_bytes",
    "mlx_active_bytes",
    "mlx_cache_bytes",
    "mlx_peak_bytes",
]


def validate_workload(w: dict[str, Any]) -> None:
    if (
        w.get("schema_version") != VERSION
        or type(w.get("concurrency")) is not int
        or w["concurrency"] != 1
    ):
        raise ValueError("requires schema 2.0.0 and concurrency=1")
    for k in ("prompt_max_tokens", "max_new_tokens", "warmup_requests", "measured_requests"):
        if type(w.get(k)) is not int or w[k] <= 0:
            raise ValueError(f"{k} must be a positive integer")
    if not isinstance(w.get("prompt"), str) or not w["prompt"].strip():
        raise ValueError("prompt must be nonempty")
    if not w.get("workload_id") or w.get("temperature") != 0 or type(w.get("seed")) is not int:
        raise ValueError("requires workload_id, temperature=0, integer seed")


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--model-lock", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device-id", required=True)
    p.add_argument("--token-trace", choices=["full", "boundary"], default="full")
    args = p.parse_args()
    w = json.loads(Path(args.workload).read_text())
    validate_workload(w)
    lock = json.loads(Path(args.model_lock).read_text())
    if lock.get("schema_version") != VERSION or not re.fullmatch(
        r"[0-9a-f]{40}", lock.get("model_revision", "")
    ):
        raise ValueError("requires a real immutable model lock")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "workload.json").write_text(json.dumps(w, indent=2))
    (out / "model-lock.json").write_text(json.dumps(lock, indent=2))
    summary = {
        "schema_version": VERSION,
        "run_id": out.name,
        "device_id": args.device_id,
        "source": "measured",
        "simulation": False,
        "status": "failed",
        "policy": "single_device",
        "partition_version": "single-device-v2",
        "model_id": lock["model_id"],
        "model_revision": lock["model_revision"],
        "workload_id": w["workload_id"],
        "requests": [],
        "timing_definition": "instrumented stream callback wall time; includes tokenization; excludes model load",
        "token_count_definition": "MLX-LM generation_tokens, may include terminal EOS",
        "token_trace": args.token_trace,
        "workload_sha256": sha(args.workload),
        "model_lock_sha256": sha(args.model_lock),
    }
    try:
        summary["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        summary["git_commit"] = None
    if Path("experiments/phase1/requirements-m4-lock.txt").exists():
        summary["requirements_sha256"] = sha("experiments/phase1/requirements-m4-lock.txt")
    trace = (out / "trace.csv").open("x", newline="")
    writer = csv.DictWriter(trace, fieldnames=FIELDS)
    writer.writeheader()
    sequence, request_id, request_kind, memory = 0, "", "", None

    def event(name, token_count="", sample_memory=True, timestamp=None):
        nonlocal sequence
        stamp = time.perf_counter_ns() if timestamp is None else timestamp
        row = dict.fromkeys(FIELDS, "")
        row.update(
            {
                "schema_version": VERSION,
                "sequence": sequence,
                "run_id": out.name,
                "device_id": args.device_id,
                "source": "measured",
                "wall_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "monotonic_ns": stamp,
                "request_id": request_id,
                "request_kind": request_kind,
                "event": name,
                "policy": "single_device",
                "partition_version": "single-device-v2",
                "workload_id": w["workload_id"],
                "model_revision": lock["model_revision"],
                "token_count": token_count,
            }
        )
        if sample_memory and memory is not None:
            row.update(memory())
        writer.writerow(row)
        sequence += 1
        if name != "token":
            trace.flush()
        return stamp

    exit_code = 1
    try:
        event("process_start")
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            raise RuntimeError("run measured inference on native Apple Silicon macOS")
        import mlx.core as mx
        import psutil
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        process = psutil.Process()

        def memory():
            return {
                "process_rss_bytes": process.memory_info().rss,
                "system_available_bytes": psutil.virtual_memory().available,
                "mlx_active_bytes": int(mx.get_active_memory()),
                "mlx_cache_bytes": int(mx.get_cache_memory()),
                "mlx_peak_bytes": int(mx.get_peak_memory()),
            }

        model_dir = Path(lock["model_dir"])
        if sha(model_dir / "config.json") != lock["config_sha256"]:
            raise ValueError("model config differs from the lock")
        mx.random.seed(w["seed"])
        mx.reset_peak_memory()
        event("model_load_start")
        model, tokenizer = load(str(model_dir))
        mx.eval(model.parameters())
        mx.synchronize()
        event("model_load_end")
        summary["model_load_peak_bytes"] = int(mx.get_peak_memory())
        profile = build_parameter_profile(
            model.parameters(),
            model_id=lock["model_id"],
            model_revision=lock["model_revision"],
            total_layers=lock["config"]["num_hidden_layers"],
        )
        (out / "model_profile.json").write_text(json.dumps(profile, indent=2))
        for index in range(w["warmup_requests"] + w["measured_requests"]):
            request_kind = "warmup" if index < w["warmup_requests"] else "measured"
            request_id = f"req-{index:04d}"
            mx.synchronize()
            mx.reset_peak_memory()
            start = event("request_start")
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": w["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if not 0 < len(tokens) <= w["prompt_max_tokens"]:
                raise ValueError("actual tokenized prompt exceeds the frozen workload envelope")
            if len(tokens) + w["max_new_tokens"] > lock["config"]["max_position_embeddings"]:
                raise ValueError("request exceeds model context limit")
            first, first_text, last_token, previous, pieces, last = None, None, None, 0, [], None
            for response in stream_generate(
                model,
                tokenizer,
                tokens,
                max_tokens=w["max_new_tokens"],
                sampler=make_sampler(temp=0),
            ):
                now = time.perf_counter_ns()
                count = int(response.generation_tokens)
                if count < previous or count > previous + 1:
                    raise ValueError("unexpected stream token counter; review installed API")
                if count > previous:
                    if first is None:
                        first = event("first_token", count, False, now)
                    if args.token_trace == "full":
                        event("token", count, False, now)
                    last_token, previous = now, count
                if response.text and first_text is None:
                    first_text = event("first_visible_text", count, False)
                pieces.append(response.text)
                last = response
            mx.synchronize()
            end = event("request_end", previous)
            text = "".join(pieces)
            if first is None or last is None or not text.strip():
                raise ValueError("empty/no-token output is not a successful baseline")
            item = {
                "request_id": request_id,
                "request_kind": request_kind,
                "status": "success",
                "prompt_tokens": len(tokens),
                "generation_tokens": previous,
                "ttft_ms": (first - start) / 1e6,
                "end_to_end_ms": (end - start) / 1e6,
                "first_visible_text_ms": None if first_text is None else (first_text - start) / 1e6,
                "observed_tpot_ms": (last_token - first) / (previous - 1) / 1e6
                if previous > 1
                else None,
                "library_prompt_tps": float(last.prompt_tps),
                "library_generation_tps": float(last.generation_tps),
                "request_peak_mlx_bytes": int(mx.get_peak_memory()),
                "generated_text": text,
            }
            summary["requests"].append(item)
        request_id, request_kind = "", ""
        event("process_end")
        summary["status"], exit_code = "success", 0
    except Exception as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        (out / "error.txt").write_text(traceback.format_exc())
        event("error", sample_memory=False)
    finally:
        trace.close()
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"run_id": out.name, "status": summary["status"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
```

## **Appendix E. Urmi: semantic evidence validator**

Save as `src/weave/telemetry/validate_evidence.py`.

```py
"""Strict acceptance validator for successful single-process Phase 1 evidence."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def validate(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    s = json.loads((root / "run_summary.json").read_text())
    w = json.loads((root / "workload.json").read_text())
    rows = list(csv.DictReader((root / "trace.csv").open()))
    require(s["schema_version"] == "2.0.0" and s["status"] == "success", "run did not succeed")
    require(s["simulation"] is False and s["source"] == "measured", "not measured evidence")
    require(bool(rows), "trace is empty")
    require([int(r["sequence"]) for r in rows] == list(range(len(rows))), "sequence gap/duplicate")
    times = [int(r["monotonic_ns"]) for r in rows]
    require(times == sorted(times), "monotonic timestamp regression")
    for r in rows:
        for field in (
            "run_id",
            "device_id",
            "workload_id",
            "model_revision",
            "policy",
            "partition_version",
        ):
            require(r[field] == s[field], f"identity mismatch: {field}")
        require(
            r["schema_version"] == "2.0.0" and r["source"] == "measured",
            "trace version/source mismatch",
        )
        for field in (
            "process_rss_bytes",
            "system_available_bytes",
            "mlx_active_bytes",
            "mlx_cache_bytes",
            "mlx_peak_bytes",
        ):
            require(r[field] == "" or int(r[field]) >= 0, f"negative memory: {field}")
    global_events = [r["event"] for r in rows if not r["request_id"]]
    require(
        global_events == ["process_start", "model_load_start", "model_load_end", "process_end"],
        "bad process lifecycle",
    )
    load_end = next(int(r["monotonic_ns"]) for r in rows if r["event"] == "model_load_end")
    process_end = int(rows[-1]["monotonic_ns"])
    groups = defaultdict(list)
    for r in rows:
        if r["request_id"]:
            groups[r["request_id"]].append(r)
    requests = s["requests"]
    require(len(requests) == w["warmup_requests"] + w["measured_requests"], "wrong request count")
    require(len({q["request_id"] for q in requests}) == len(requests), "duplicate summary request")
    require(set(groups) == {q["request_id"] for q in requests}, "request IDs differ")
    require(
        sum(q["request_kind"] == "measured" for q in requests) == w["measured_requests"],
        "wrong measured count",
    )
    previous_end = -1
    for q in requests:
        rs = groups[q["request_id"]]
        events = [r["event"] for r in rs]
        require(
            set(events)
            <= {"request_start", "first_token", "token", "first_visible_text", "request_end"},
            "unknown request event",
        )
        for name in ("request_start", "first_token", "request_end", "first_visible_text"):
            require(events.count(name) == 1, f"{q['request_id']}: missing/duplicate {name}")
        require(
            events[0] == "request_start" and events[-1] == "request_end", "bad request boundaries"
        )
        require(all(r["request_kind"] == q["request_kind"] for r in rs), "warmup/measurement mixed")
        by_event = {r["event"]: r for r in rs if r["event"] != "token"}
        start, first, end = (
            int(by_event[x]["monotonic_ns"])
            for x in ("request_start", "first_token", "request_end")
        )
        require(
            max(previous_end, load_end) <= start <= first <= end <= process_end,
            "request order/overlap",
        )
        previous_end = end
        token_counts = [int(r["token_count"]) for r in rs if r["event"] == "token"]
        mode = s.get("token_trace", "full")
        require(mode in {"full", "boundary"}, "unknown token trace mode")
        expected_counts = list(range(1, q["generation_tokens"] + 1)) if mode == "full" else []
        require(token_counts == expected_counts, "token counter gap/duplicate or wrong trace mode")
        require(
            bool(q["generated_text"].strip()) and q["status"] == "success", "empty/failed output"
        )
        require(0 < q["prompt_tokens"] <= w["prompt_max_tokens"], "prompt outside envelope")
        require(
            type(q["generation_tokens"]) is int
            and 0 < q["generation_tokens"] <= w["max_new_tokens"],
            "generation count outside envelope",
        )
        for boundary in ("request_start", "request_end"):
            for field in (
                "process_rss_bytes",
                "system_available_bytes",
                "mlx_active_bytes",
                "mlx_cache_bytes",
                "mlx_peak_bytes",
            ):
                require(by_event[boundary][field] != "", f"missing boundary memory: {field}")
        for key, expected in (
            ("ttft_ms", (first - start) / 1e6),
            ("end_to_end_ms", (end - start) / 1e6),
        ):
            require(
                math.isfinite(q[key]) and abs(q[key] - expected) < 1e-6,
                f"summary/trace mismatch: {key}",
            )
        require(q["request_peak_mlx_bytes"] > 0, "missing MLX peak")
    return {
        "run_id": s["run_id"],
        "trace_rows": len(rows),
        "requests": len(requests),
        "status": "PASS",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    print(json.dumps(validate(p.parse_args().directory), indent=2))
```

## **Appendix F. Jainil: labeled synthetic cluster fixture**

Save as `src/weave/profiler/simulated_fixture.py`.

```py
"""Make a wholly synthetic memory-pressure case, never measured device evidence."""

import argparse
import json
from pathlib import Path
from typing import Any


def make_state() -> dict[str, Any]:
    mib = 2**20
    return {
        "schema_version": "2.0.0",
        "snapshot_id": "SIMULATED-qwen7-pressure-v2",
        "simulation": True,
        "model": {
            "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "model_revision": "SIMULATED-NOT-A-MODEL-LOCK",
            "profile_id": "SIMULATED-weight-profile-v2",
            "total_layers": 28,
            "layer_weight_bytes": [128 * mib] * 28,
            "replicated_weight_bytes": 1024 * mib,
            "kv_heads": 4,
            "head_dim": 128,
            "kv_bytes_per_element": 4,
            "source": "simulated",
        },
        "request_envelope": {
            "prompt_tokens_upper_bound": 512,
            "max_new_tokens": 64,
            "concurrency": 1,
        },
        "devices": [
            {
                "device_id": name,
                "source": "simulated",
                "budget_basis": "fresh_process_absolute",
                "safe_process_budget_bytes": budget * mib,
                "workspace_reserve_bytes": 512 * mib,
                "safety_margin_bytes": 512 * mib,
            }
            for name, budget in [("SIMULATED-m4-pressure", 3072), ("SIMULATED-second-mac", 24576)]
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(make_state(), f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
```

## **Appendix G. Jainil: complete candidate generator**

Save as `src/weave/scheduler/candidates.py`.

```py
"""Offline Phase 1 candidates. This module never starts MLX or selects a winner."""

import argparse
import hashlib
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weave.estimator.memory import estimate_stage_memory

VERSION = "2.0.0"


def positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name}: expected {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def validate_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("schema_version") != VERSION:
        raise ValueError("unsupported schema_version; migrate explicitly to 2.0.0")
    if state.get("simulation") is not True:
        raise ValueError("Phase 1 fixture runner accepts simulation=true only")
    if not isinstance(state.get("snapshot_id"), str) or not state["snapshot_id"]:
        raise ValueError("snapshot_id is required")
    model, req = state["model"], state["request_envelope"]
    for key in ("model_id", "model_revision", "profile_id"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ValueError(f"model.{key} is required")
    total = positive_int(model["total_layers"], "total_layers")
    weights = model["layer_weight_bytes"]
    if not isinstance(weights, list) or len(weights) != total:
        raise ValueError("layer_weight_bytes must contain one entry per layer")
    for n in weights:
        positive_int(n, "layer_weight_bytes entry")
    for key in ("replicated_weight_bytes", "kv_heads", "head_dim", "kv_bytes_per_element"):
        positive_int(model[key], key)
    if model["source"] != "simulated":
        raise ValueError("fixture model profile must be labeled simulated")
    for key in ("prompt_tokens_upper_bound", "max_new_tokens", "concurrency"):
        positive_int(req[key], key)
    if req["concurrency"] != 1:
        raise ValueError("Phase 1 supports concurrency=1 only")
    devices = state["devices"]
    if not 1 <= len(devices) <= 2:
        raise ValueError("Phase 1 supports one or two candidate devices")
    ids = [d["device_id"] for d in devices]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("device_id values must be nonempty and unique")
    for d in devices:
        if d["source"] != "simulated":
            raise ValueError("every fixture device must be labeled simulated")
        if d["budget_basis"] != "fresh_process_absolute":
            raise ValueError("unsupported budget basis")
        for key in ("safe_process_budget_bytes", "workspace_reserve_bytes", "safety_margin_bytes"):
            positive_int(d[key], key)
    return state


def assignments(device_order: Sequence[str], logical_counts: Sequence[int]) -> list[dict[str, Any]]:
    if len(device_order) != len(logical_counts) or not logical_counts:
        raise ValueError("device/count length mismatch")
    if len(set(device_order)) != len(device_order):
        raise ValueError("duplicate device assignment")
    start, rows = 0, []
    for stage, (device_id, count) in enumerate(zip(device_order, logical_counts)):
        positive_int(count, "layer_count")
        rows.append(
            {
                "device_id": device_id,
                "logical_stage": stage,
                "mlx_rank": len(logical_counts) - 1 - stage,
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "layer_count": count,
            }
        )
        start += count
    return rows


def evaluate(order: Sequence[str], counts: Sequence[int], state: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    model, req = state["model"], state["request_envelope"]
    if sum(counts) != model["total_layers"]:
        raise ValueError("counts must sum to total_layers")
    devices = {d["device_id"]: d for d in state["devices"]}
    rows, reasons = assignments(order, counts), []
    tokens = req["prompt_tokens_upper_bound"] + req["max_new_tokens"]
    for row in rows:
        d = devices[row["device_id"]]
        parts = estimate_stage_memory(
            layer_weight_bytes=model["layer_weight_bytes"][
                row["start_layer"] : row["end_layer_exclusive"]
            ],
            replicated_weight_bytes=model["replicated_weight_bytes"],
            total_tokens=tokens,
            kv_heads=model["kv_heads"],
            head_dim=model["head_dim"],
            kv_bytes_per_element=model["kv_bytes_per_element"],
            concurrency=req["concurrency"],
            workspace_reserve_bytes=d["workspace_reserve_bytes"],
            safety_margin_bytes=d["safety_margin_bytes"],
        )
        predicted = sum(parts.values())
        row.update(
            {
                "memory_components": parts,
                "predicted_memory_bytes": predicted,
                "safe_process_budget_bytes": d["safe_process_budget_bytes"],
                "headroom_bytes": d["safe_process_budget_bytes"] - predicted,
            }
        )
        if predicted > d["safe_process_budget_bytes"]:
            reasons.append(
                {
                    "code": "MEMORY_BUDGET_EXCEEDED",
                    "device_id": d["device_id"],
                    "excess_bytes": predicted - d["safe_process_budget_bytes"],
                }
            )
    identity = {
        "order": list(order),
        "counts": list(counts),
        "snapshot": state["snapshot_id"],
        "profile": model["profile_id"],
    }
    plan_id = (
        "candidate-"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    )
    return {
        "schema_version": VERSION,
        "plan_id": plan_id,
        "source_snapshot_id": state["snapshot_id"],
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "profile_id": model["profile_id"],
        "simulation": True,
        "selected": False,
        "policy": "candidate_only",
        "mode": "single_device" if len(counts) == 1 else "pipeline",
        "logical_split": list(counts),
        "mlx_split_by_rank": list(reversed(counts)),
        "device_id_by_mlx_rank": list(reversed(order)),
        "assignments": rows,
        "memory_feasible": not reasons,
        "rejection_reasons": reasons,
        "runtime_verified": False,
        "executable": False,
    }


def generate(state: dict[str, Any]) -> dict[str, Any]:
    validate_state(state)
    total = state["model"]["total_layers"]
    ids = [d["device_id"] for d in state["devices"]]
    plans = [evaluate([d], [total], state) for d in ids]
    if len(ids) == 2:
        for order in itertools.permutations(ids):
            for left in range(1, total):
                plans.append(evaluate(order, [left, total - left], state))
    return {
        "schema_version": VERSION,
        "simulation": True,
        "source_snapshot_id": state["snapshot_id"],
        "selected_plan_id": None,
        "candidate_count": len(plans),
        "memory_feasible_count": sum(p["memory_feasible"] for p in plans),
        "candidates": plans,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = generate(json.loads(Path(args.state).read_text()))
    with Path(args.output).open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
```

## **Appendix H. Team: versioned JSON Schema generator**

Save as `src/weave/contracts/schemas.py`.

```py
"""Emit the exact 2.0.0 input and output schemas used by this guide."""

import argparse
import json
from pathlib import Path
from typing import Any

STRING = {"type": "string", "minLength": 1}
POS = {"type": "integer", "minimum": 1}
NONNEG = {"type": "integer", "minimum": 0}


def obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": True,
    }


def build_schemas() -> dict[str, Any]:
    version = {"const": "2.0.0"}
    workload = obj(
        {
            "schema_version": version,
            "workload_id": STRING,
            "prompt": STRING,
            "prompt_max_tokens": POS,
            "max_new_tokens": POS,
            "concurrency": {"const": 1},
            "warmup_requests": POS,
            "measured_requests": POS,
            "temperature": {"const": 0},
            "seed": {"type": "integer"},
        }
    )
    model = obj(
        {
            "model_id": STRING,
            "model_revision": STRING,
            "profile_id": STRING,
            "total_layers": POS,
            "layer_weight_bytes": {"type": "array", "minItems": 1, "items": POS},
            "replicated_weight_bytes": POS,
            "kv_heads": POS,
            "head_dim": POS,
            "kv_bytes_per_element": POS,
            "source": {"const": "simulated"},
        }
    )
    device = obj(
        {
            "device_id": STRING,
            "source": {"const": "simulated"},
            "budget_basis": {"const": "fresh_process_absolute"},
            "safe_process_budget_bytes": POS,
            "workspace_reserve_bytes": POS,
            "safety_margin_bytes": POS,
        }
    )
    state = obj(
        {
            "schema_version": version,
            "snapshot_id": STRING,
            "simulation": {"const": True},
            "model": model,
            "request_envelope": obj(
                {
                    "prompt_tokens_upper_bound": POS,
                    "max_new_tokens": POS,
                    "concurrency": {"const": 1},
                }
            ),
            "devices": {"type": "array", "minItems": 1, "maxItems": 2, "items": device},
        }
    )
    assignment = obj(
        {
            "device_id": STRING,
            "logical_stage": NONNEG,
            "mlx_rank": NONNEG,
            "start_layer": NONNEG,
            "end_layer_exclusive": POS,
            "layer_count": POS,
            "memory_components": obj(
                {
                    k: NONNEG
                    for k in (
                        "layer_weight_bytes",
                        "replicated_weight_bytes",
                        "kv_upper_bound_bytes",
                        "workspace_reserve_bytes",
                        "safety_margin_bytes",
                    )
                }
            ),
            "predicted_memory_bytes": POS,
            "safe_process_budget_bytes": POS,
            "headroom_bytes": {"type": "integer"},
        }
    )
    candidate = obj(
        {
            "schema_version": version,
            "plan_id": STRING,
            "source_snapshot_id": STRING,
            "model_id": STRING,
            "model_revision": STRING,
            "profile_id": STRING,
            "simulation": {"const": True},
            "selected": {"const": False},
            "policy": {"const": "candidate_only"},
            "mode": {"enum": ["single_device", "pipeline"]},
            "logical_split": {"type": "array", "minItems": 1, "maxItems": 2, "items": POS},
            "mlx_split_by_rank": {"type": "array", "items": POS},
            "device_id_by_mlx_rank": {"type": "array", "items": STRING},
            "assignments": {"type": "array", "minItems": 1, "maxItems": 2, "items": assignment},
            "memory_feasible": {"type": "boolean"},
            "rejection_reasons": {"type": "array", "items": {"type": "object"}},
            "runtime_verified": {"const": False},
            "executable": {"const": False},
        }
    )
    candidate_set = obj(
        {
            "schema_version": version,
            "simulation": {"const": True},
            "source_snapshot_id": STRING,
            "selected_plan_id": {"type": "null"},
            "candidate_count": POS,
            "memory_feasible_count": NONNEG,
            "candidates": {"type": "array", "items": candidate},
        }
    )
    run_summary = obj(
        {
            "schema_version": version,
            "run_id": STRING,
            "device_id": STRING,
            "model_id": STRING,
            "model_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
            "workload_id": STRING,
            "source": {"const": "measured"},
            "simulation": {"const": False},
            "status": {"enum": ["success", "failed"]},
            "policy": {"const": "single_device"},
            "partition_version": STRING,
            "requests": {"type": "array", "items": {"type": "object"}},
        }
    )
    return {
        "workload": workload,
        "cluster_state_simulated": state,
        "candidate_set": candidate_set,
        "run_summary": run_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/phase1/schemas/v2")
    root = Path(parser.parse_args().output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, schema in build_schemas().items():
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
        with (root / f"{name}.schema.json").open("x") as f:
            json.dump(schema, f, indent=2)
    print(root)


if __name__ == "__main__":
    main()
```

## **Appendix I. Portable scheduler and contract tests**

Save as `tests/test_candidates.py`.

```py
import jsonschema
import pytest

from weave.contracts.schemas import build_schemas
from weave.profiler.simulated_fixture import make_state
from weave.scheduler.candidates import assignments, evaluate, generate, validate_state
from weave.telemetry.baseline import validate_workload


@pytest.mark.parametrize("total", [1, 2, 3, 28])
def test_complete_candidates_and_exact_cover(total):
    s = make_state()
    s["model"]["total_layers"] = total
    s["model"]["layer_weight_bytes"] = [2**27] * total
    result = generate(s)
    assert result["candidate_count"] == 2 + 2 * (total - 1)
    assert len({p["plan_id"] for p in result["candidates"]}) == result["candidate_count"]
    for p in result["candidates"]:
        covered = [
            i for a in p["assignments"] for i in range(a["start_layer"], a["end_layer_exclusive"])
        ]
        assert covered == list(range(total))
        assert p["mlx_split_by_rank"] == p["logical_split"][::-1]
        assert [a["mlx_rank"] for a in p["assignments"]] == list(
            range(len(p["assignments"]) - 1, -1, -1)
        )
        assert p["selected"] is False and p["executable"] is False
    jsonschema.validate(result, build_schemas()["candidate_set"])


def test_unequal_rank_mapping_and_single_device_candidates():
    assert assignments(["A", "B"], [10, 18]) == [
        {
            "device_id": "A",
            "logical_stage": 0,
            "mlx_rank": 1,
            "start_layer": 0,
            "end_layer_exclusive": 10,
            "layer_count": 10,
        },
        {
            "device_id": "B",
            "logical_stage": 1,
            "mlx_rank": 0,
            "start_layer": 10,
            "end_layer_exclusive": 28,
            "layer_count": 18,
        },
    ]
    plans = generate(make_state())["candidates"]
    assert sum(p["mode"] == "single_device" for p in plans) == 2
    assert {tuple(p["device_id_by_mlx_rank"]) for p in plans if p["mode"] == "pipeline"} == {
        ("SIMULATED-second-mac", "SIMULATED-m4-pressure"),
        ("SIMULATED-m4-pressure", "SIMULATED-second-mac"),
    }


def test_budget_boundary_and_rejection_explanation():
    s = make_state()
    order = [d["device_id"] for d in s["devices"]]
    p = evaluate(order, [10, 18], s)
    required = p["assignments"][0]["predicted_memory_bytes"]
    s["devices"][0]["safe_process_budget_bytes"] = required
    assert evaluate(order, [10, 18], s)["memory_feasible"]
    s["devices"][0]["safe_process_budget_bytes"] -= 1
    p = evaluate(order, [10, 18], s)
    assert not p["memory_feasible"]
    assert p["rejection_reasons"][0]["excess_bytes"] == 1


def test_memory_grows_with_context_and_single_node_state():
    s = make_state()
    s["devices"] = s["devices"][:1]
    before = generate(s)["candidates"][0]["assignments"][0]["predicted_memory_bytes"]
    s["request_envelope"]["prompt_tokens_upper_bound"] *= 2
    result = generate(s)
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["assignments"][0]["predicted_memory_bytes"] > before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(schema_version="1.0.0"),
        lambda s: s.update(simulation=False),
        lambda s: s["model"].update(total_layers=True),
        lambda s: s["model"].update(layer_weight_bytes=[1]),
        lambda s: s["devices"][0].update(workspace_reserve_bytes=0),
        lambda s: s["devices"][1].update(device_id=s["devices"][0]["device_id"]),
        lambda s: s["request_envelope"].update(concurrency=2),
    ],
)
def test_invalid_state_fails_closed(mutation):
    s = make_state()
    mutation(s)
    with pytest.raises(ValueError):
        validate_state(s)


def test_invalid_split_and_workload_rejected():
    s = make_state()
    ids = [d["device_id"] for d in s["devices"]]
    for split in ([0, 28], [-1, 29], [10, 17], [True, 27]):
        with pytest.raises(ValueError):
            evaluate(ids, split, s)
    w = {
        "schema_version": "2.0.0",
        "workload_id": "test",
        "prompt": "test",
        "prompt_max_tokens": 512,
        "max_new_tokens": 64,
        "concurrency": 1,
        "warmup_requests": 1,
        "measured_requests": 3,
        "temperature": 0,
        "seed": 0,
    }
    validate_workload(w)
    jsonschema.validate(w, build_schemas()["workload"])
    w["concurrency"] = 2
    with pytest.raises(ValueError):
        validate_workload(w)
```

## **Appendix J. Portable evidence corruption tests**

Save as `tests/test_evidence.py`.

```py
"""Fake production-shaped rows in pytest temp dirs; never experiment evidence."""

import csv
import json

import pytest

from weave.telemetry.baseline import FIELDS
from weave.telemetry.validate_evidence import validate


def write_fake(root):
    identity = {
        "run_id": "unit-test-only",
        "device_id": "fake",
        "workload_id": "test",
        "model_revision": "a" * 40,
        "policy": "single_device",
        "partition_version": "test",
    }
    names = [
        "process_start",
        "model_load_start",
        "model_load_end",
        "request_start",
        "first_token",
        "token",
        "first_visible_text",
        "request_end",
        "request_start",
        "first_token",
        "token",
        "first_visible_text",
        "request_end",
        "process_end",
    ]
    rows = []
    for i, name in enumerate(names):
        row = dict.fromkeys(FIELDS, "")
        row.update(
            identity,
            schema_version="2.0.0",
            sequence=i,
            source="measured",
            event=name,
            monotonic_ns=i * 1000000,
        )
        if 3 <= i <= 7 or 8 <= i <= 12:
            row["request_id"] = "warmup" if i <= 7 else "measured"
            row["request_kind"] = row["request_id"]
        if name == "token":
            row["token_count"] = 1
        if name in {"request_start", "request_end"}:
            for field in (
                "process_rss_bytes",
                "system_available_bytes",
                "mlx_active_bytes",
                "mlx_cache_bytes",
                "mlx_peak_bytes",
            ):
                row[field] = 1
        rows.append(row)
    summary = {
        **identity,
        "schema_version": "2.0.0",
        "source": "measured",
        "simulation": False,
        "status": "success",
        "requests": [
            {
                "request_id": k,
                "request_kind": k,
                "status": "success",
                "generation_tokens": 1,
                "prompt_tokens": 3,
                "ttft_ms": 1.0,
                "end_to_end_ms": 4.0,
                "generated_text": "test",
                "request_peak_mlx_bytes": 1,
            }
            for k in ("warmup", "measured")
        ],
    }
    (root / "run_summary.json").write_text(json.dumps(summary))
    (root / "workload.json").write_text(
        json.dumps(
            {
                "warmup_requests": 1,
                "measured_requests": 1,
                "prompt_max_tokens": 512,
                "max_new_tokens": 64,
            }
        )
    )
    with (root / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary, rows


def test_valid_fake_evidence(tmp_path):
    write_fake(tmp_path)
    assert validate(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize(
    "change", ["metric", "failure", "identity", "missing_token", "zero_tokens", "missing_memory"]
)
def test_corruption_fails(tmp_path, change):
    summary, rows = write_fake(tmp_path)
    if change == "metric":
        summary["requests"][1]["ttft_ms"] = 7
    elif change == "failure":
        summary["status"] = "failed"
    elif change == "identity":
        rows[4]["device_id"] = "wrong-device"
    elif change == "zero_tokens":
        summary["requests"][1]["generation_tokens"] = 0
    elif change == "missing_memory":
        rows[7]["mlx_peak_bytes"] = ""
    else:
        rows[4]["event"] = "unexpected"
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    with (tmp_path / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        validate(tmp_path)
```

## **Appendix K. M4-only installed pipeline mapping contract test**

Save as `tests/test_installed_pipeline.py`.

```py
"""Installed-source contract check on the M4; not a distributed runtime test."""

import pytest

pytest.importorskip("mlx.core")
PipelineMixin = pytest.importorskip("mlx_lm.models.pipeline").PipelineMixin


@pytest.mark.parametrize("rank,start,end", [(0, 10, 28), (1, 0, 10)])
def test_installed_unequal_split(rank, start, end):
    class Group:
        def rank(self):
            return rank

        def size(self):
            return 2

    class Layers(PipelineMixin):
        def __init__(self):
            super().__init__()
            self.layers = list(range(28))

    model = Layers()
    model.pipeline(Group(), split=[18, 10])
    assert (model.start_idx, model.end_idx) == (start, end)
    assert model.pipeline_layers == list(range(start, end))
```

## **Appendix L. Estimator: explicit stage memory components**

Save as `src/weave/estimator/memory.py`.

```py
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
```

## **Appendix M. Profiler: materialized parameter-byte measurements**

Save as `src/weave/profiler/model_profile.py`.

```py
"""Measure materialized parameter bytes without importing MLX on portable paths."""

import re
from typing import Any


def build_parameter_profile(
    parameters: Any,
    *,
    model_id: str,
    model_revision: str,
    total_layers: int,
) -> dict[str, Any]:
    """Retain non-layer weights separately because Qwen replicates them per rank."""
    from mlx.utils import tree_flatten

    layers, replicated = [0] * total_layers, 0
    for name, array in tree_flatten(parameters):
        match = re.match(r"model\.layers\.(\d+)\.", name)
        if match:
            index = int(match.group(1))
            if not 0 <= index < total_layers:
                raise ValueError("parameter layer index disagrees with locked geometry")
            layers[index] += int(array.nbytes)
        else:
            replicated += int(array.nbytes)
    if not all(layers):
        raise ValueError("weight profile did not match every layer")
    return {
        "schema_version": "2.0.0",
        "model_id": model_id,
        "model_revision": model_revision,
        "source": "measured",
        "measurement_kind": "materialized_parameter_nbytes",
        "layer_weight_bytes": layers,
        "replicated_weight_bytes": replicated,
        "limitation": "parameter bytes only; excludes KV, workspace, allocator, process/OS memory",
    }
```

## **Appendix N. Team: editable src-layout installation and quality checks**

Save as `pyproject.toml`.

```
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "weave-inference"
version = "0.0.0"
description = "Runtime-adaptive placement for distributed LLM inference on Apple Silicon"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest", "jsonschema", "ruff"]
m4 = ["mlx-lm", "psutil", "huggingface-hub"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
```

## **Appendix O. Team: new shared contracts package initializer**

Save as `src/weave/contracts/__init__.py`.

```py
"""Versioned interfaces shared by execution, telemetry, and scheduling."""
```

## **Appendix P. Team: append these ignore rules**

Append to the existing .gitignore; preserve all current content.

```
# Phase 1: raw evidence includes text, source snapshots, and logs, not just JSON/CSV.
/results/*
!/results/.gitkeep
.venv-reproduce/
.env
.env.*
!.env.example
```

End of revised Phase 1 guide. The appendix code was checked in an isolated repository checkout. Real M4 and distributed gates remain to be completed by the team; no repository changes were pushed or merged.
