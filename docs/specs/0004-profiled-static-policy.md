# Spec 0004: Frozen profiled static placement (baseline B3)

- **Status:** draft for review (pull request #4). Spec only: no code until Mohit's T16 measurement
  format exists
- **Owner:** Urmi (placement policies). Written by Jainil's session at his request (schedule task
  T18, spec half), for Urmi's review; the stage profile in this spec is a proposal for Mohit
- **Branch:** `feat/jainilr-phase2-portable`, merged into `main` through pull request #4
- **Governs:** README milestone M3 ("measured-performance splits") and the "frozen profiled static
  baseline" of guide v2 section 14, under interface 2.0.0

## Problem Statement

RQ1 is judged against B3. Guide section 14 says so directly: "The main research comparison is
against the declared profiled static baseline with failures/OOMs accounted for, not a weak arbitrary
split", and spec 0003 makes `profiled_static` the reference arm of the evaluation plan. If Weave
only beats the equal split (B1) or the memory-proportional split (B2), that shows that speed
information helps, which is already known (prima.cpp, exo and dnet use it), not that *live*
information helps.

B0–B2 (spec 0002) need only the candidate set. B3 needs measured speed: how long each Mac takes per
layer and how long the Macs take to exchange activations. Spec 0002 deferred it for that reason, and
the measurement is gate P2-E (schedule task T16, Mohit), which needs two Macs.

## Solution

B3 reads the candidate set and a one-time measured **stage profile** taken at startup, predicts the
end-to-end latency of every memory-feasible candidate from the profile, and chooses the fastest. It
is **frozen**: chosen once from that profile and that candidate set, and kept for every scenario of
the evaluation, however the workload or the Macs change afterwards. That is exactly the difference
Weave's live signals have to justify.

Like B0–B2, B3 chooses a candidate by plan ID and never builds its own layout or re-estimates
memory, and it writes the spec-0002 decision record with policy name `profiled_static`.

## User Stories

1. As the policy owner, I want B3 to choose among the generator's memory-feasible candidates by plan ID, so that layer cover, rank mapping and the memory estimate keep one implementation.
2. As an evaluator, I want B3 chosen once from a startup profile and never re-evaluated, so that the comparison isolates what live information adds.
3. As a reviewer, I want the latency prediction written as a formula over named profile fields, so that a B3 choice can be recomputed by hand.
4. As the profiler owner, I want the profile fields B3 needs proposed now, so that T16 measures them once instead of twice.
5. As a reviewer, I want a single Mac to be a possible B3 choice, so that "do not distribute" competes on measured speed too (RQ2).
6. As a safety reviewer, I want B3 refused when the profile is missing, belongs to another model revision or another set of Macs, lacks a link it needs, or mixes simulated with measured data.
7. As a reviewer, I want ties broken by a written rule, so that one input always gives one choice.
8. As the ablation designer, I want B3's decision record in the spec-0002 format, so that `--compare` can show B0–B3 side by side.

## Implementation Decisions

- **Inputs:** the candidate set (spec 0001 generator output) and one stage profile. Only candidates
  with `memory_feasible: true` are eligible; when none is, the outcome is `NO_FEASIBLE_PLAN`, as for
  B0.
- **Stage profile (proposal for Mohit, T16 / P2-E):** one JSON file per profiling session, measured
  on the real Macs with the locked model, with:
  - identity: `schema_version`, `source`, `simulation`, `profile_id` (prefixed `SIMULATED-` exactly
    when simulated), `model_id`, `model_revision`, `git_commit`, the run IDs it was computed from;
  - the workload it was taken at: prompt tokens and generated tokens;
  - per device (`device_id`, as in the cluster state): `prefill_ms_per_layer` (prompt pass per
    layer at the profile's prompt length) and `decode_ms_per_layer` (per generated token per
    layer), each a median with its sample count and its p95 so the spread is visible;
  - per ordered device pair actually connected (`from_device_id`, `to_device_id`, `backend` RING or
    JACCL): `prefill_transfer_ms` and `decode_transfer_ms` for one boundary hand-off;
  - per device group: `decode_all_gather_ms` and `prefill_all_gather_ms`. The installed Qwen2 model
    (mlx-lm 0.32.2 source in the M4 evidence) ends every forward pass with an `all_gather` of the
    hidden state across all ranks, besides the send/receive between neighbours, so guide section
    13's "account for Qwen's observed collective behavior" is a measured term, not an assumption.
  - Per-layer times assume the model's layers cost the same, which holds for their shapes (every
    Qwen2.5-7B layer is 131,111,936 bytes in the measured M4 profile). P2-E should check it by
    timing at least two different layer counts per Mac (the `[14, 14]` and `[10, 18]` runs give 10,
    14 and 18).
- **Latency prediction at concurrency 1:** for a candidate with stages s on devices d(s) holding
  n(s) layers, and boundaries b between neighbouring stages,
  `prefill = Σ n(s)·prefill_ms_per_layer[d(s)] + Σ prefill_transfer_ms[b] + prefill_all_gather_ms`,
  `decode = Σ n(s)·decode_ms_per_layer[d(s)] + Σ decode_transfer_ms[b] + decode_all_gather_ms`,
  `end_to_end = prefill + (generated_tokens − 1)·decode`, at the profile's workload. Stage times
  are **summed**, not maximized: one autoregressive request does not overlap the stages (guide
  section 13). A single-device candidate has no transfer and no all-gather term.
- **Throughput objective (defined, not used yet):** for an actually overlapped pipeline (concurrency
  above 1 with requests in flight together), the bottleneck is the slowest stage including its
  outgoing transfer. B3 v1 does not use it, because the Phase 1 runner serves one request at a time
  and P2-E has not shown overlap; using it needs its own review.
- **Choice and ties:** the eligible candidate with the lowest predicted end-to-end latency, compared
  after rounding to whole microseconds. Ties go to fewer stages (distributing costs communication),
  then the larger smallest headroom, then the earlier candidate in the candidate file.
- **Frozen:** B3 is chosen once per (candidate set, profile) pair. The evaluation runs that one plan
  in every scenario; a plan that later fails or runs out of memory is counted, not replaced.
- **Decision record:** the spec-0002 record with `policy: "profiled_static"`, the chosen plan ID,
  the outcome and reason (the reason states the predicted end-to-end time and its two runners-up),
  plus the profile's ID and SHA-256 so the choice can be traced to its measurements.
- **Refusals** (`ValueError` naming the field): no profile; a profile whose `source`, `simulation`
  and ID prefix disagree; a simulated profile with a measured candidate set or the other way round
  (decided by the `SIMULATED-` prefix of the profile ID and the candidate set's snapshot ID); a
  model ID or revision different from the candidates'; a device in the candidate set missing from
  the profile; a link a candidate needs missing from the profile (never guessed from another link);
  a zero, negative or non-numeric timing.
- **Open questions for review (AGENTS.md "Ask first"):**
  1. The decision schema in `experiments/policies/schemas/v2/` fixes `policy` to B0–B2 and has no
     profile fields, and that folder is never edited in place. Adding `profiled_static` and the
     two profile fields is a reviewed regeneration under 2.0.0 or a new schema folder; the team
     decides which.
  2. The decision record and the candidate set both fix `simulation: true` (freeze note section 9),
     but a B3 choice from a measured profile rests on measured inputs. The team decides whether
     that needs a new field (for example `inputs_measured`) or a new record version.
  3. Mohit confirms or changes the profile fields above; B3 follows whatever T16 actually measures.

## Testing Decisions

- When code follows (spring, T20), tests drive the public policy function and the command with a
  simulated candidate set and a `SIMULATED-` stage profile whose numbers are made up and labelled
  so, and assert the chosen plan ID, the outcome, the reason and the written record.
- Worked examples by hand: a single Mac that wins because the link is slow; a split that wins
  because one Mac is much faster per layer; a tie broken by stage count; the summed (not maximum)
  stage time changing the choice relative to a bottleneck rule.
- One invalid-input table for the profile, one case per refusal, asserting the field.
- Prior art: `tests/test_policy_memory_proportional.py` and `tests/test_policy_input.py`.

## Out of Scope

- Measuring anything: the profile is T16 / P2-E (Mohit, two Macs).
- The profile reader and the policy code: they wait for T16's actual format, so this semester has
  the spec only.
- Choosing with real data and running the chosen plan (spring, T20 onward).
- The adaptive policy, the cost model's calibration and any re-evaluation (gates P3–P4).
- Searching device orders beyond the candidates the generator already lists.

## Plan

1. **Spec:** add this spec (this pull request).
2. **Profile format:** Mohit confirms the stage-profile fields when T16's measurement format is
   fixed; this spec is updated to match.
3. **Schema questions:** the team answers the two open questions above.
4. **Reader and policy (spring, T20):** test-first, under Urmi's account, after steps 2 and 3.
5. **Choice on real data (spring, T20):** run B3 on the measured profile and record the decision.
