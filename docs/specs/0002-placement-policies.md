# Spec 0002: Static placement policies over the candidate set

- **Status:** draft. The implementation decisions below are proposed; Urmi confirms them before
  step 2, and Jainil and Mohit sign off on the decision record (step 6) before it is built
- **Owner:** Urmi
- **Branch:** `feat/urmi-scheduler-placement-policies`, merged into `main` through a pull request
  reviewed by Jainil and Mohit
- **Governs:** README milestone M3 (static placement) and the baseline list in guide v2 section 14,
  under interface 2.0.0 as recorded in the draft interface-freeze note

## Problem Statement

The Phase 1 planner (spec 0001) lists every way to place the model on the available Macs and says
which ones pass the memory estimate, but it never chooses one: every candidate is unselected and the
set's `selected_plan_id` is fixed to null by the committed schema. Choosing is my area (placement
policies, the how-many-Macs decision and the ablations), and nothing in the repository does it yet.

The research question (RQ1) is whether live workload information beats equal and static
hardware-based splitting. That comparison needs the static baselines to exist first, defined
precisely enough that "Weave beat the equal split" means one thing. Guide section 14 lists them:
the best single Mac, an equal split, a deterministic memory-based split, a frozen profiled static
split, and the adaptive policy. The first three need only what the candidate set already holds. The
profiled split needs measured per-layer speed from two real Macs, and the adaptive policy needs a
calibrated cost model; neither exists yet.

## Solution

Add the three baselines that can be defined today as policies that read Jainil's candidate set and
choose one of its plans by ID. A policy never builds its own layer layout and never re-estimates
memory: it computes the split it wants, finds that plan among the candidates, and reports the
candidate's own memory verdict. That keeps one implementation of layer cover, rank mapping and the
memory estimate.

A policy reports its plan even when the plan fails the memory estimate, because a baseline that
does not fit is a result the evaluation must count, not something to hide behind a fallback.

Each choice is written as a separate, simulated decision record that points at the candidate set
by hash and at the chosen plan by ID, and a comparison command runs every policy on one candidate
set. This is the first piece of the ablation harness.

## User Stories

1. As the policy owner, I want each baseline defined by a written, deterministic rule, so that "the equal split" names one plan for a given candidate set.
2. As the policy owner, I want policies to choose among the generator's candidates by plan ID, so that layer cover, rank mapping and memory feasibility have one implementation.
3. As a teammate, I want the candidate set read once into typed records and checked, so that a policy never works from a malformed or partly selected set.
4. As a user feeding a malformed candidate set, I want a `ValueError` that names the offending field, so that I can fix the input instead of reading a stack trace.
5. As a safety reviewer, I want any candidate set that is not wholly simulated, or that already has a selected plan, refused, so that a policy decision is never stacked on another decision or on measured data.
6. As a reviewer, I want any schema version other than 2.0.0 refused rather than reinterpreted, so that old semantics never leak in.
7. As a teammate, I want a best-single-Mac policy that picks the memory-feasible single-device plan with the most headroom, so that "don't distribute" is always a baseline.
8. As a teammate, I want the best-single-Mac policy to say explicitly when no single Mac fits, so that a missing baseline is visible.
9. As a teammate, I want an equal-split policy that spreads the layers as evenly as possible over every Mac, so that the weakest common baseline is defined.
10. As a teammate, I want a memory-proportional policy that gives each Mac layers in proportion to the memory it has left after the costs every stage pays, so that the hardware-aware static baseline is defined.
11. As an evaluator, I want a policy whose plan fails the memory estimate to report that plan with its rejection reasons, never to fall back silently, so that baseline failures are counted (guide section 14).
12. As a reviewer, I want ties broken by a written rule, so that the same input always gives the same choice.
13. As a Phase 2 developer, I want the policies to work for any number of Macs, so that a third Mac needs no policy change.
14. As a teammate, I want each choice written as a decision record that names the policy, the chosen plan, the candidate set's hash, the outcome and the reason, so that a decision can be traced to its input.
15. As a safety reviewer, I want every decision record marked simulated, not executable and not runtime-verified, so that a static choice on an uncalibrated estimate is never mistaken for a deployment.
16. As a teammate, I want the decision command to refuse to overwrite an existing file, so that an earlier decision is never silently replaced.
17. As the ablation designer, I want one command that runs every policy on one candidate set and prints a comparison table, so that baselines are compared on identical input.
18. As a reviewer, I want the comparison output for the committed fixture locked in a golden file, so that any change to a policy's choice is deliberate and named.
19. As a teammate on an Intel Mac or in CI, I want the policies to import without MLX, so that they run wherever the portable checks run.
20. As the policy owner, I want the work landed in small commits under my own account that each pass `make check`, so that git history records my contribution and `main` stays green.

## Implementation Decisions

Proposed; confirm or change before step 2.

- **New module:** `weave.scheduler.policies`. No change to `weave.scheduler.candidates`, the
  estimator, the fixture or any committed schema in steps 2–5 and 7.
- **Input is the candidate set alone.** Everything a policy needs is already in it: the device order
  and each Mac's budget and fixed costs come from the single-device candidates, which the generator
  emits one per Mac in the state's device order (part of its golden output); the layer count is the
  sum of any candidate's `logical_split`.
- **Read once:** the candidate set is parsed into immutable typed records (standard-library
  dataclasses). It is refused, with a `ValueError` naming the field, when the schema version is not
  2.0.0; the set or any candidate is not simulated; `selected_plan_id` is not null; any candidate is
  selected, executable, runtime-verified or has a policy other than `candidate_only`; a candidate's
  `memory_feasible` contradicts its rejection reasons; plan IDs repeat;
  candidates disagree on the snapshot ID or the layer total; or there is not exactly one
  single-device candidate per Mac.
- **Device order:** the order of the single-device candidates (the state's device order). A
  multi-Mac policy places its stages in that order; it does not search the other orders, because a
  static baseline has one fixed layout.
- **Best single Mac:** among memory-feasible single-device plans, the one with the largest headroom;
  ties go to the earlier Mac. When none fits, the outcome is `NO_FEASIBLE_PLAN` and no plan is
  chosen.
- **Equal split:** every Mac gets ⌊L / N⌋ layers and the first L mod N Macs in device order get one
  more, so 28 layers on two Macs is `[14, 14]` and on three is `[10, 9, 9]`.
- **Memory-proportional split:** each Mac's room for layers is its safe process budget minus the
  costs every stage pays regardless of layers (replicated weights, workspace reserve, safety
  margin), floored at zero. Layers are shared in proportion to that room by largest remainder: each
  Mac gets the floor of its quota, and the leftover layers go to the largest fractional parts, ties
  to the earlier Mac. Every Mac gets at least one layer, taken from the Mac with the most layers; if
  every Mac has zero room the quotas are equal. Proportion is by layer count, which matches bytes
  only while layers weigh the same; the fixture's layers do, and measured profiles are out of scope.
- **Reporting a plan that does not fit:** equal and memory-proportional always choose their plan;
  the outcome is `FEASIBLE` or `INFEASIBLE` from the candidate's own `memory_feasible`, and the
  candidate's rejection reasons are carried through. No fallback.
- **Missing plan:** if the computed split is not among the candidates, the policy raises
  `ValueError`; it means the candidate set does not match the generator.
- **Decision record (decided by Urmi on 2026-10-28; raised for review in the pull request):** a new
  artifact under 2.0.0 that adds files and changes no existing field or schema byte. Fields:
  `schema_version`, `simulation: true`, `policy` (`best_single_device`, `equal_split` or
  `memory_proportional`), `source_snapshot_id`, `candidate_set_sha256` (of the candidate file's
  bytes as read), `chosen_plan_id` (null exactly when the outcome is `NO_FEASIBLE_PLAN`), `outcome`,
  `reason`, `rejection_reasons`, `executable: false`, `runtime_verified: false`, and nothing else.
  It says "chosen", not "selected", so it cannot be confused with the candidate set's
  `selected_plan_id`, which stays null.
- **Decision schema location:** the Phase 1 contract folder `experiments/phase1/schemas/v2/` stays
  frozen. The decision schema is generated by a new module, `weave.contracts.policy_schemas`, into
  its own folder, `experiments/policies/schemas/v2/`, and committed there; a test holds the
  generator to the committed file. No existing schema, generator or checkpoint changes.
- **Commands:** `python -m weave.scheduler.policies --candidates <file> --policy <name> --output
  <file>` writes one decision record with exclusive creation; `--compare` prints the table for every
  policy. Same JSON style as the candidate file.
- **Phase limits:** none added. The candidate set already carries the Phase 1 limits; the policies
  themselves work for any number of Macs.
- **Authorship:** commits are made under Urmi's account with no AI co-author trailers.

## Testing Decisions

- **A good test** drives a public function or command with realistic input and asserts on what a
  caller observes: the chosen plan ID, the outcome, the written file and the raised error. No
  private helpers and no mocks. The input is the committed fixture's candidate set, read from
  `tests/golden/phase1-candidate-plans.json` (which `tests/test_phase1_golden.py` holds to the
  generator byte for byte) and edited per case. Policy tests do not import the planner modules:
  `tests/test_checkpoint.py` requires every test file that does to be part of the planner's
  checkpoint evidence, and policy tests are not.
- **Worked numbers for the committed fixture** (28 layers of 134,217,728 bytes; KV 2,359,296 bytes
  per layer; fixed costs 2,147,483,648 bytes per stage):
  - Best single Mac: `SIMULATED-second-mac` alone, headroom 19,798,163,456 bytes. The pressure Mac
    alone fails by 2,750,414,848 bytes.
  - Equal split: `[14, 14]` with the pressure Mac first, `INFEASIBLE`; the pressure Mac exceeds its
    budget by 838,336,512 bytes. This is the case that motivates the project.
  - Memory-proportional: room 1,073,741,824 and 23,622,320,128 bytes (1 : 22), quotas 1.217 and
    26.783, so `[1, 27]` with the pressure Mac first, `FEASIBLE`; headroom 937,164,800 and
    19,934,740,480 bytes.
- **Rule tests:** equal split for several layer and Mac counts; largest-remainder rounding and its
  tie rule; the at-least-one-layer rule; the all-zero-room rule; the best-single tie rule; no single
  Mac fitting.
- **Invalid-input table:** one parametrized case per refused candidate set, asserting the field in
  the error.
- **Golden reference:** the comparison output for the committed fixture, byte for byte.
- **Prior art:** `tests/test_planner_input.py` (invalid-input table), `tests/test_phase1_golden.py`
  (golden comparison) and `tests/test_schemas.py` (schema drift).

## Out of Scope

- The frozen profiled static policy: it needs measured per-stage speed from two real Macs
  (guide gate P2-E). Designing its input interface waits for that measurement.
- The adaptive policy, the cost model and the how-many-Macs decision (gate P3).
- KEEP or SWITCH decisions and partition changes (gate P4).
- Searching device orders, or any ranking beyond each policy's rule.
- Measured model profiles as input, and lifting the Phase 1 limits.
- Any change to the candidate generator, the estimator, the fixture or existing schemas.
- Running a chosen plan on hardware.

## Further Notes

- Decisions to confirm before step 2: the memory-proportional definition of room (budget minus
  fixed costs, not the raw budget); reporting infeasible baselines instead of falling back; one
  fixed device order; deferring the profiled policy.
- The decision record is an interface addition under AGENTS.md "Ask first". To avoid stalling,
  Urmi decided it on 2026-10-28 (additive under 2.0.0, its own schema folder, `chosen_plan_id`);
  the pull request asks Jainil and Mohit to confirm, and any change they ask for lands before merge.
- With the raw budget instead, the fixture's memory-proportional split would be `[3, 25]`.

## Plan

Each step is test-first, passes `make check`, and is committed and pushed on its own.

1. **Spec:** add this spec.
2. **Input reading:** typed candidate-set records and the invalid-input table.
3. **Best single Mac:** the policy, its tie and no-fit rules.
4. **Equal split:** the policy and its rounding rule; the fixture's `[14, 14]` is `INFEASIBLE`.
5. **Memory-proportional:** the policy and its rounding rules; the fixture gives `[1, 27]`.
6. **Decision record:** the schema in its own folder, the record and the `--policy` command; the
   team confirms the format in the pull request.
7. **Comparison:** the `--compare` command and its golden file.
8. **Review:** run the code-review skill against `main` and fix its findings.
9. **Pull request:** open it against `main` for Jainil's and Mohit's review.
