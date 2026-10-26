# Spec 0001: Phase 1 planner rebuild and deliverables

- **Status:** implemented (plan steps 1–9 done, step 9 and pre-merge review fixes included); Jainil
  approved the merge on 2026-10-25, after the pre-merge review
- **Owner:** Jainil
- **Branch:** `feat/jainilr-phase1-schedular`; merges into `feat/runtime-phase1-mohit` only when
  Jainil says so
- **Governs:** guide v2 sections 7, 8, 10, 11 and 12 (gate G4 and the section 12.2 demo), under
  interface 2.0.0 as recorded in the draft interface-freeze note

## Problem Statement

My Phase 1 planner (the candidate generator, the memory estimator and the simulated cluster fixture)
exists in the repository only as a verbatim copy of the reference code from my own implementation
guide's appendices, committed from Mohit's branch. It produces the numbers the guide promises, but it
was written to illustrate the plan, not engineered as a foundation:

- Input checks are scattered dictionary lookups; malformed input has already crashed with the wrong
  error type.
- The Phase 1 limits (at most two Macs, one request at a time, simulated input only) are woven into
  the planning logic, so Phase 2 would need a rewrite to lift them.
- Every candidate re-validates the whole input.
- Two contract rules that look like bugs (what a plan ID ignores, and the fixed KV bytes per element)
  are undocumented, so someone may "fix" them and silently change the contract.
- Git history shows the code under Mohit's account, so my contribution to my own gate is not visible.

My Phase 1 gate (G4) also still lacks the evidence package and the checkpoint demonstration the guide
requires.

## Solution

Rebuild the three planner modules test-first under my authorship, rule by rule, so that every
behaviour traces to a test. Lock today's output in as a golden reference first: for the committed
fixture, the rebuilt planner must produce a byte-identical candidate file, with the same function
names and command-line flags, unless a difference is a deliberate, named change. Read and validate the
input once into typed records, keep the planning core general (any number of Macs, any concurrency),
and enforce the Phase 1 limits only where the input is read.

Then add one checkpoint command that prints the four items the guide's section 12.2 demonstration
needs and writes my evidence package into a fresh, gitignored, simulated results directory from the
same run.

## User Stories

1. As the planner's author, I want the candidate generator rebuilt test-first, so that every rule it enforces traces to a test I wrote and understand.
2. As the planner's author, I want the rebuilt code committed under my own account, so that git history records my Phase 1 contribution.
3. As a teammate consuming candidates, I want the candidate file for the committed fixture to stay byte-identical, so that nothing downstream changes because of the rebuild.
4. As a teammate, I want the same function names and command-line flags, so that the commands in the guide, the README and the Phase 1 instructions keep working.
5. As a reviewer, I want any output difference to be a deliberate, named change, so that I can tell a regression from a decision.
6. As the planner's author, I want the cluster state read and validated once into typed, immutable records, so that the planning logic never touches raw dictionaries.
7. As a user feeding a malformed state (a missing field, a wrong JSON type, an unknown device, a non-positive count), I want a `ValueError` that names the offending field, so that I can fix the input instead of reading a stack trace.
8. As a safety reviewer, I want the planner to refuse any state that is not wholly simulated, so that an uncalibrated estimate is never presented as a measured deployment.
9. As a reviewer, I want any schema version other than 2.0.0 rejected rather than reinterpreted, so that old semantics never leak in.
10. As the planner's author, I want the Phase 1 limits enforced only where the input is read, so that Phase 2 can lift them without touching the planning core.
11. As a Phase 2 developer, I want the planning core to handle any number of Macs and any concurrency, so that three-Mac and concurrent planning need only a boundary change.
12. As a teammate, I want one single-device candidate for each available Mac, so that "don't distribute" is always considered and the larger Mac can win.
13. As an execution owner, I want single-device fallback represented as one stage at MLX rank 0, never a multi-rank group with an empty stage, so that it launches through the ordinary load path.
14. As a teammate, I want every two-Mac cut with positive layer counts enumerated in both device orders, so that the candidate set is complete.
15. As a teammate, I want a one-Mac input to produce exactly one candidate and a one-layer model to produce only single-device candidates, so that degenerate inputs behave predictably.
16. As a reviewer, I want each candidate's layer ranges to be half-open, contiguous, ordered and covering every layer exactly once, so that no layer is dropped or duplicated.
17. As an execution owner, I want each stage to record its logical stage and its MLX rank separately, with MLX rank 0 running the last layers, so that the launch order is unambiguous.
18. As an execution owner, I want the split by MLX rank to be the logical split reversed and the devices by MLX rank to be the device order reversed, so that the logical [10, 18] plan launches as [18, 10].
19. As the memory estimator's owner, I want each stage's predicted memory to be the sum of its own layer bytes, the replicated weight bytes, the KV upper bound, the workspace reserve and the safety margin, so that no cost is hidden.
20. As the estimator's owner, I want replicated weights counted on every participating Mac, so that the per-rank embedding and output-head cost of the current Qwen pipeline is never missed.
21. As the estimator's owner, I want the KV upper bound computed exactly by the guide's section 10 formula, so that Phase 1 stays faithful to the guide until calibration.
22. As the estimator's owner, I want any zero, negative, boolean or missing memory term rejected, so that a feasible estimate can never come from an omitted cost.
23. As a teammate, I want a plan marked memory-feasible only when every stage's prediction is within its safe process budget, with equality allowed, so that the boundary is exact.
24. As a teammate, I want each rejected plan to carry a structured MEMORY_BUDGET_EXCEEDED reason naming the Mac and the excess bytes, so that I can see why it failed and by how much.
25. As the placement-policy owner, I want every candidate marked simulated, not selected, not executable and not runtime-verified, and the set's selected plan ID left null, so that nobody mistakes the list for a decision.
26. As a reviewer, I want plan IDs deterministic for a given snapshot, profile, device order and layer counts, so that regenerating the same input reproduces the same IDs.
27. As a maintainer, I want the two guide-defined quirks documented next to the code (plan IDs ignore the model revision and the budgets; the KV estimate uses four bytes per value with no block rounding), so that nobody "fixes" them without a contract change.
28. As a teammate, I want the simulated fixture generator to reproduce the committed fixture byte for byte, so that the planning input is reproducible from code.
29. As a reviewer, I want every synthetic identity labeled SIMULATED, so that synthetic data can never join measured evidence.
30. As the planner's author, I want the candidate file validated against the committed 2.0.0 candidate-set schema, so that the contract is checked mechanically.
31. As a teammate, I want the candidate and fixture commands to refuse to overwrite an existing file, so that an earlier output is never silently replaced.
32. As a teammate on an Intel Mac or in CI, I want the planner modules to import without MLX, so that they run wherever the portable checks run.
33. As the planner's author, I want one checkpoint command that prints the guide's section 12.2 items, so that I can demonstrate my gate in one step.
34. As a checkpoint reviewer, I want to see one single-device plan, one accepted two-Mac plan, the rejected [10, 18] plan with its excess bytes, and the logical-to-MLX rank mapping, so that I can confirm fallback, feasibility, rejection and rank semantics at a glance.
35. As the planner's author, I want the same command to write my evidence package into a fresh simulated results directory, so that the demo and the evidence always come from one run.
36. As a teammate, I want the evidence package to hold the candidate file, a copy of the fixture with its hash, the schema-validation result with the hashes of the committed schema files used, the planner test results, the source commit and any uncommitted source, a machine-readable summary and a short README, so that I can review the gate without rerunning anything.
37. As a teammate, I want the checkpoint command to refuse to reuse an existing results directory, so that every evidence package is a new attempt.
38. As a coding agent, I want the specs folder listed in AGENTS.md, so that I read the plan before changing the planner.
39. As a reviewer, I want the rebuild landed in small commits that each pass `make check`, so that I can review it rule by rule and `main` stays green when it lands.
40. As the planner's author, I want the rebuilt planner reviewed on two axes (standards and spec) before it merges into the Phase 1 integration branch, and merged only when I say so, so that the candidate semantics others build on are sound.

## Implementation Decisions

- **Modules rebuilt in place:** `weave.scheduler.candidates`, `weave.estimator.memory` and
  `weave.profiler.simulated_fixture`, replaced piece by piece rather than in one swap. **New module:**
  `weave.scheduler.checkpoint` for the demo and the evidence package.
- **Public interface unchanged:** `validate_state`, `assignments`, `evaluate`, `generate`,
  `estimate_stage_memory` and `make_state` keep their names, arguments and return shapes; the
  candidate command keeps `--state` and `--output`, and the fixture command keeps `--output`. Other
  helpers become private.
- **Parse once:** the cluster state is read once into immutable typed records (standard-library
  dataclasses): the cluster state, the model profile, the request envelope and each device's budget.
  Every validation failure raises `ValueError` naming the field. The planning logic receives only
  parsed records: the state is parsed and checked once per call to `generate` or `evaluate`, never
  once per candidate. The public helpers `assignments` and `estimate_stage_memory` still check their
  own arguments, so each stays safe to call on its own.
- **Limits at the boundary:** the Phase 1 limits (wholly simulated input, at most two Macs,
  concurrency 1, fresh-process absolute budgets) form one explicit check on the parsed state, separate
  from parsing. The schema version is a contract check made while parsing, not a Phase 1 limit:
  lifting the limits must never admit a version other than 2.0.0. The planning core supports any
  number of Macs and any concurrency; raising the public device-limit setting is enough to plan
  three Macs.
- **Enumeration order is part of the output:** single-device candidates in device order, then each
  ordered pair of Macs in permutation order, with the first Mac's layer count rising from 1 to L − 1.
  The general core's enumeration must reduce to exactly this order for two Macs.
- **Mapping (guide section 7):** layer ranges are half-open. Logical stage 0 runs the earliest
  layers; MLX rank = (number of stages − 1 − logical stage). The split by MLX rank is the logical split
  reversed, and the devices by MLX rank are the device order reversed. `assignments` also rejects an
  empty or non-string device name.
- **Memory (guide section 10):** predicted bytes per stage = assigned layer bytes + replicated bytes +
  KV upper bound + workspace reserve + safety margin, all integer bytes. KV upper bound = 2 × assigned
  layers × concurrency × (prompt tokens upper bound + max new tokens) × KV heads × head dimension × KV
  bytes per element. Every term must be a positive integer; booleans are not integers. The component
  names and their order in the output stay as today.
- **Feasibility:** a stage fits when predicted bytes ≤ its safe process budget. A plan is
  memory-feasible only when every stage fits; each stage that doesn't adds a MEMORY_BUDGET_EXCEEDED
  reason with the device and the excess bytes.
- **Safety flags fixed:** every candidate is simulated, not selected, policy "candidate_only", not
  executable and not runtime-verified; the set's selected plan ID is null.
- **Guide-defined quirks kept and documented:** plan ID = "candidate-" plus the first 16 hex digits of
  the SHA-256 of the sorted-key JSON of (device order, layer counts, snapshot ID, profile ID); it
  ignores the model revision and the budgets, so a changed input needs a new snapshot ID. The fixture's
  4 bytes per KV element is a conservative planning assumption; calibration to MLX's real cache (bf16,
  256-token allocation steps) is Phase 2 work.
- **Serialization unchanged:** 2-space-indented JSON with today's key order and no trailing newline,
  written with exclusive creation (never overwriting).
- **Checkpoint command:** `python -m weave.scheduler.checkpoint` prints the four section 12.2 items.
  By default it reads the committed fixture and writes under the repository's gitignored results
  folder, both resolved from the checkout the module was imported from, never the working directory.
  Outside a source checkout it refuses before writing anything, because the package records that
  checkout's planner tests and committed schema files. Each run creates a new
  `SIMULATED-<UTC timestamp>-checkpoint` directory, refusing an existing one and any name without the
  SIMULATED- prefix (guide sections 5.1 and 8), so simulated evidence is always labeled. It holds a
  byte copy of the input state, the candidate file, a machine-readable summary (hashes, counts, check
  results, the source commit and whether it had uncommitted changes, the demo's plan IDs), the planner
  test output (the new suites and the guide's reference tests) and a README; when the source was
  dirty, also the tracked diff and the untracked source (guide section 12.1 asks for the source
  commit). The copy keeps the input's file name, so an input named like one of the package's own
  files is refused before the directory is created. Schema validation checks the files as written
  against the committed 2.0.0 schema files and records each schema file's hash. For an input without a
  rejected logical [10, 18], the demo shows another rejected split. It needs the dev environment
  (`make setup`) for `jsonschema`, and exits non-zero if the planner tests or the schema validation
  fail.
- **AGENTS.md** gains a pointer to the specs folder.
- **Pre-commit:** the whitespace and end-of-file fixers skip the golden-reference folder, whose files
  must stay byte-exact.
- **Reference tests kept:** the guide's reference planner tests stay alongside the new suites;
  removing them would be a file move, which AGENTS.md says to ask about first.
- **Authorship:** commits are made under Jainil's account with no AI co-author trailers.

## Testing Decisions

- **A good test** drives a public interface with realistic input and asserts on what a caller can
  observe: the returned values, the written file and the raised error. No private helpers, and no
  mocks except at a system boundary: the checkpoint's test-suite subprocess may be replaced by a fake,
  including a failing one to prove the non-zero exit, and its checkout lookup may report no checkout
  to prove the refusal. A test may point the checkpoint's schema folder at an empty directory (a real
  missing contract file, not a fake) and may raise the public Phase 1 device-limit
  setting to exercise the general core through `generate`.
- **Seams (confirmed):**
  1. `generate` and the candidate command, for the whole output: the golden comparison, 56
     candidates and 15 memory-feasible, exact layer cover, the safety flags, and schema validation.
  2. `assignments`, for the mapping rules: half-open ranges, logical stage versus MLX rank, and
     [10, 18] → [18, 10].
  3. `estimate_stage_memory`, for the section 10 formula and its "no term may be zero" rule.
  4. `validate_state`, for every malformed input, each raising `ValueError` that names the field.
- **Golden reference:** before any code changes, today's candidate file for the committed fixture is
  saved with the tests. A test compares the rebuilt output to it byte for byte. A deliberate change
  updates the golden file in the same commit and says so in the message.
- **Exhaustive checks** replace sampling: every cut for every layer count from 1 to 28, in both device
  orders, plus single-device and one-device inputs. No new dependency (no property-testing library).
- **Budget boundary:** a budget exactly equal to the prediction passes; one byte less fails and
  reports an excess of exactly 1.
- **Invalid-input table:** one parametrized case per malformed state, asserting the field named in the
  error.
- **Fixture reproduction:** the generated fixture equals the committed one byte for byte.
- **General core:** with the device limit raised to three, `generate` gives 2,271 candidates, worked
  by hand (3 alone + 6 ordered pairs × 27 cuts + 6 orders of three × C(27, 2)).
- **Checkpoint command:** run against temporary directories and from outside the repository; assert
  the printed items, that the recorded hashes match the written files and the committed schema files,
  that the default folder is the repository's own results folder (checked without writing into it),
  that a failing planner suite or a missing schema file exits non-zero, that an existing or unlabeled
  directory, an input named like a package file and a run outside a checkout are refused, and that
  the recorded test run covers every test file of the planner modules. No test writes into the
  repository's results folder.
- **Prior art:** the existing candidate tests (parametrized exact-cover checks), the schema-drift
  test (comparing generated artifacts with committed ones), the evidence corruption table (one case
  per failure, matched on the message), and the repo-rule tests (failure messages that say how to
  fix).

## Out of Scope

- Measured model profiles as planner input (needs a separately reviewed interface with
  per-component provenance, guide section 8).
- Lifting the Phase 1 limits at the boundary (three Macs, concurrency above 1).
- Cost ranking, plan selection, and KEEP/SWITCH decisions (Phase 3 and later).
- Calibrating the KV estimate or workspace reserve against MLX (Phase 2).
- MLX execution, the split-aware loader, launching, and switching (Phase 2 and later).
- Any change to interface 2.0.0 or to the plan-identity rule.
- Issue-tracker setup and the repository restructure (both deferred).
- Changes to the telemetry runner, the validator, or the M4 setup scripts.

## Further Notes

- The reference implementation comes from my guide's appendices F, G, I and L. Mohit copied them
  verbatim into the repository on 2026-10-22; the 2026-10-25 audit fixed its malformed-input errors.
  This rebuild replaces that code with an engineered version under my authorship.
- Reference numbers for the committed fixture: 56 candidates and 15 memory-feasible (the second Mac
  alone, plus 1 to 7 layers on the 3 GiB pressure Mac in each order). The [10, 18] plan with the
  pressure Mac first is rejected by 292,028,416 bytes. The reference candidate file is 121,884 bytes.
- The draft interface-freeze note governs the semantics. Anything here that would change it is out
  of scope until the team signs off.
- Step 9's two-axis review (standards and spec) led to these fixes: the checkpoint's paths no longer
  depend on the working directory, its evidence is validated against the committed schema files, and
  three rules gained tests (plan IDs ignore the model revision, the candidate-file hash, three Macs).
  The spec was amended to record the deliberate choices above.
- The pre-merge review (standards and spec, against Mohit's branch) led to five more fixes: no test
  writes into the real results folder any more, an input named like a package file is refused, the
  recorded test run includes the guide's reference tests, the command refuses to run outside a
  checkout, and a missing schema file has a test. The review's naming, duplication and
  shared-constant suggestions were left as judgement calls.

## Plan

Each step is test-first, passes `make check`, and is committed and pushed on its own. Steps 1–9 and
the pre-merge review are done; Jainil approved step 10 on 2026-10-25.

1. **Spec:** add this spec and the AGENTS.md pointer.
2. **Golden reference:** save today's candidate file with the tests; add the byte-for-byte golden test
   and the fixture reproduction test against the current code.
3. **Input reading:** typed records, parsing and the Phase 1 boundary check behind `validate_state`,
   with the full invalid-input table.
4. **Layer mapping:** rebuild `assignments` with the exhaustive cover and rank-mapping tests.
5. **Memory estimate:** rebuild `estimate_stage_memory` with the formula and positive-term tests.
6. **Candidates:** rebuild `evaluate` and `generate` on the typed records with the general core; the
   golden test stays byte-identical.
7. **Fixture:** rebuild `make_state`; the committed fixture is reproduced byte for byte.
8. **Checkpoint:** add the checkpoint command, its evidence package and its tests.
9. **Review:** run the code-review skill on the branch and fix its findings.
10. **Merge:** merge into `feat/runtime-phase1-mohit` with a merge commit, only when Jainil says so.
