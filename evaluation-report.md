# Weaver vs. White Paper — Detailed Evaluation (2025-11-27)

## How I evaluated (explicit process)
1) Read the white paper (requirements for schema, validation, analytics, discourse, evidence).  
2) Traced implementation files: `src/schema/*` (types + validators), `src/graph/*` (storage, guards, invariants), `src/analysis/*` (checks/reports), `src/tools/llm/graph_tools/*` (exposed analyses/commands), tests under `tests/`.  
3) Verified enforcement points by following code paths for node/edge inserts, updates, and global invariant checks.  
4) Cross-checked presence of each §6 validation (DAG, reachability, coverage, purity, examples, fadeability, practice, borrow-ahead, discourse continuity) and whether it is enforced, warning-only, or missing.  
5) Noted persistence/metadata alignment with paper (commit pinning, SourceRef policy).  
6) Summarized gaps, risks, and prioritized fixes.

## Compliance matrix (white paper ⇒ code)
- **Node ontology (Bloom + LO + Assessment + TeachingStep)**: Implemented (`KnowledgeType`, `TeachingStepNode`). Statements/source_refs required. ✔
- **Edge layers restricted to requires/supports/assesses + discourse precedes/anchors**: Implemented with endpoint guards and cycle checks. ✔
- **Requires DAG**: Enforced on insert/update; global invariant error. ✔
- **Supports fadeability (must not carry prerequisite load)**: Enforced by simulation before insert. ✔
- **Assesses claim = target LO**: Auto-normalized and validated. ✔
- **Case tags + coverage tags on supports (example richness)**: Case tag optional → partial compliance. ❌ (paper requires case_tag and variety)
- **Rubric coverage (observation_features cover LO rubric)**: Computed; only warns, not error. Partial. ⚠️
- **Constructive alignment reachability (FP → assessment → LO target)**: Reachability function exists but not enforced or surfaced in invariants. ❌
- **Purity/extraneous (construct-irrelevant demands)**: Tool available; not in invariants; not compared to CID list. ❌
- **Rubric drift re-check**: Missing. ❌
- **Borrow-ahead guardrail**: Analysis tool with severities; not enforced on insert. ⚠️
- **TeachingStep anchor/rationale requirement**: Not enforced. ⚠️
- **SourceRef policy (path+lines+commit hash)**: Structural validation only; no check that revision matches `course_commit`. ⚠️
- **Assessment items as sinks in requires**: Enforced by requires endpoint validation. ✔
- **Example minimums by knowledge type**: Implemented in `analysis::example_gaps` and run as warning; not error. ⚠️
- **Procedural practice (path to assessment w/ target LO)**: Checked; warning only. ⚠️
- **Keystone detection**: Implemented tool; advisory only. ✔ (optional in paper)
- **Discourse DAG per episode**: Enforced on precedes insert. ✔
- **Discourse borrow-ahead classification**: Implemented analysis; not enforced. ⚠️

## Evidence by code location
- Schema & validators: `src/schema/types.rs`; `src/schema/validate.rs` (requires/supports/assesses rules; case_tag not required).  
- Node/edge storage & guards: `src/graph/mod.rs` (EdgeSpec validations, fadeability simulation, normalize_assesses_claims, SourceRef presence checks).  
- Global invariants: `src/graph/mod.rs::validate_global_invariants` (DAG error; warnings for missing target assesses, missing coverage, example gaps, practice gaps; error for fadeability).  
- Analyses: `src/analysis/mod.rs` (reachability, coverage_report, extraneous_report, fadeability_issues, example_gaps, practice_gaps, borrow_ahead, keystone, discourse_orphans, unreachable assessments).  
- LLM tools exposing analyses: `src/tools/llm/graph_tools/analysis.rs` (alignment, coverage, gap summaries), `algorithms.rs` (cycles, feedback arcs), `commands.rs` (mutations).  
- Tests: `tests/analysis_props.rs` (requires DAG guard), `tests/analysis_unit.rs` (fadeability, borrow-ahead, example gaps, practice gaps).

## Detailed findings
1) **Constructive alignment not enforced**: No invariant ensures every LO has a target assessment reachable from first principles; unreachable assessments only checked, not LO reachability. Risk: LOs can pass validation without realizable evidence paths.  
2) **Rubric coverage is warning-only**: Missing LO criteria are logged as warnings; paper expects hard failure and drift re-check on rubric change.  
3) **Purity/Extraneous omitted from invariants**: Extraneous(A,L) exists but isn’t run; construct_irrelevant_demands not compared.  
4) **Example variety partially enforced**: `case_tag` optional; example gaps logged as warnings; procedural nodes may ship without edge/error cases.  
5) **Borrow-ahead & introduce anchors**: Analysis exists; no enforcement that in-course knowledge has an introduce anchor before use (Level 1–3 severities not enforced).  
6) **SourceRef revision integrity**: Validates format but not equality to graph `course_commit`, so snapshots may mix commits.  
7) **TeachingStep anchor/rationale**: Paper requires at least one anchor or rationale; code allows orphan steps except for precedes continuity.  
8) **Practice gap severity**: Procedural practice gaps produce warnings, not errors.  
9) **Case-tag + coverage-tags for intrinsic_load=high**: Recommended in paper; code only warns (part of example_gaps description).  
10) **Documentation gap**: No README mapping which white-paper rules are enforced vs. advisory; developers must read code to know what fails vs. warns.

## Risk assessment
- **High**: Misaligned curriculum accepted (unreachable LOs, impure assessments) → invalid claims of constructive alignment.  
- **Medium**: Missing edge/error examples leads to brittle procedural coverage.  
- **Medium**: SourceRefs not tied to commit can invalidate audit trail.  
- **Low**: Discourse borrow-ahead and anchor completeness are advisory only; narrative quality risk.

## Recommendations (ordered by impact)
1) **Promote constructive alignment to an error**: In `validate_global_invariants`, fail if any LO lacks a reachable target assessment from a first principle (use `analysis::lo_reachability`).  
2) **Make rubric coverage strict + drift-aware**: Turn missing criteria into errors; store a hash of rubric_criteria per LO and re-run coverage on change.  
3) **Require case_tag on supports; enforce typical+edge/error for procedural nodes as errors**; require coverage_tags for intrinsic_load=high.  
4) **Run purity check in invariants**: For each assessment→LO (scope=target), compute Extraneous vs intended; error if non-empty minus declared CID.  
5) **Enforce SourceRef revision = `GraphConfig.course_commit`** on all inserts/updates and snapshot loads; reject mismatches.  
6) **Discourse enforcement**: require at least one introduce anchor for in-course knowledge; block borrow-ahead severity ≥ CrossEpisode unless introduction_scope is Prior/External; require anchor or rationale per TeachingStep.  
7) **Elevate practice gaps to errors**: procedural nodes must reach an assessment with scope=target.  
8) **Document enforcement levels**: add README/CONTRIBUTING mapping white-paper rules to error/warning/omitted, with remediation steps.  
9) **Optional**: add invariant that assessments remain sinks in requires (already structurally enforced, but a sanity check helps).

## Quick pass/fail summary
- Pass: Edge set, DAG enforcement, fadeability guard, assesses claim normalization, discourse DAG per episode, keystone analysis availability.  
- Partial: Rubric coverage, example minimums/variety, borrow-ahead handling, SourceRef commit integrity, procedural practice, purity.  
- Fail/Missing: Constructive-alignment invariant, rubric drift guard, mandatory case_tag, anchor/rationale requirement for steps, commit-pin enforcement for SourceRefs.

Prepared by: ChatGPT (Codex) — based on code as of 2025-11-27.
