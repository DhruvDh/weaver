# Step 2 – Graph Stack Audit Against the Quality Bar (expanded)

This document applies the Step 1 quality bar to the “vibe-coded” graph stack. It is intentionally long-form so every misalignment is explicit, source-referenced, and tied to impact. Use it as the truth set for remediation work in Step 3.

## 0) Method

- Read code: `src/graph/{model.rs,service.rs,specs.rs,traversal.rs,persist.rs,manager.rs}`, `src/analysis/mod.rs`, `src/schema/{types.rs,validate.rs}`, `src/tools/llm/graph_tools/*`.
- Compared to bar dimensions: actor hygiene, schema/validation rigor, preview ergonomics, observability, executable invariants, reproducibility, safety defaults, cognitive economy, human-facing affordances.
- Cross-checked with white-paper rules: alignment predicate, fadeability, coverage, example minimums, grain audits, borrow-ahead, source refs, evidence discipline.

## 1) What matches the bar (credit where due)

1. **Typed ontology** – Node/edge payloads use enums (`KnowledgeType`, `SupportKind`, `AnchorImpact`, `AssessmentScope`, `GrainLevel`, `IntrinsicLoad`) and structured attrs (`RequiresAttrs`, `SupportsAttrs`, `AssessesAttrs`). This keeps pedagogy encoded in types, not strings.
2. **EdgeSpec gatekeeping** – Each edge kind validates endpoints/attrs centrally; requires cycles rejected; supports self-loops blocked; assesses claim must equal target slug. Good separation of concerns.
3. **Fadeability guard** – Supports validation simulates insertion and rejects scaffolds that carry prerequisite load—directly enforces the white paper’s fadeability rule.
4. **Existing audits** – `analysis::example_gaps`, `procedural_practice_gaps`, `coverage_report`, `fadeability_issues`, `lo_reachability`, `borrow_ahead`, and provenance checks (source_ref revision must match course commit) run inside `validate_global_invariants`.
5. **Preview/pagination + cache on analysis tools** – Graph analysis tools default to preview with `fetch_body=false`, byte hints, and pagination; results are memoized per `(graph_version, AnalysisKind)` via `AnalysisCache` and an admin clear tool.
6. **Rollback discipline** – Slug index rebuilt on load; duplicate slugs rejected; node/edge inserts roll back on invariant failure; renames keep assesses.claim aligned.
7. **Snapshot hygiene** – Save uses temp+fsync+rename; load checks snapshot version.
8. **Background observability (partial)** – Autosave/prune in `main.rs` log rerun scalars for duration/success; foreground graph ops still lack metrics.

These foundations should be preserved and instrumented, not rewritten.

## 2) Where it diverges (by bar dimension)

### 2.1 Observability & ergonomics

- **No metrics/rerun**: Graph mutations/validations log only at `debug`; no Rerun scalars, no counters for invariant failures, no latency histograms. Impact: invisible regressions; can’t correlate graph ops with LLM usage.
- **Partial preview/budgeting**: Analysis tools now default to preview with `fetch_body=false`, byte hints, and pagination, but mutating tools still return full bodies with no previews and none of the responses include graph metadata or remaining-budget info. Impact: token waste on writes and no context about graph version/course_commit.
- **Flat errors**: Tool errors are strings (`InvalidPayload`); no structured codes or remediation hints. Impact: LLM can’t self-correct effectively.

### 2.2 Missing or partial invariants

- **Orphan/unreachable assessments**: Functions exist but invariants ignore them; assessments may be unreachable from first principles or lack assesses edges.
- **Keystone/granularity**: No keystone/centrality alerts; no over-bundle/fragment or grain-level audits; intrinsic_load heuristics not enforced.
- **Full discourse DAG recheck**: Precedes acyclicity is only checked per insert; no global revalidation after other mutations.
- **Strict-quality flag is narrow**: It only promotes warnings to errors; it does not expand the set of checks. Critical omissions stay unchecked in strict mode.

### 2.3 Safety, boundary hygiene, reproducibility

- **Snapshot paths unsandboxed**: Graph snapshot tools accept arbitrary paths, no workspace confinement. Risk: writing outside workspace.
- **SourceRef path escapes unguarded**: Revision is enforced, but paths are not checked against the workspace root.
- **Blocking validations**: `validate_global_invariants` and supports fadeability simulation run synchronously in actor handlers; fadeability clones entire graph; rayon runs inside actor. Risk: mailbox stalls on large graphs.
- **Thin versioning**: Only `SNAPSHOT_VERSION`; no schema versioning, checksum, or commit verification on load.

### 2.4 Cognitive economy & surface design

- **CRUD-centric tools**: LLM must chain many calls (insert node, add requires, add supports, add assesses) to do one pedagogical action. No semantic macros (e.g., upsert concept with supports & assessments, align LO).
- **Algorithm leakage (reduced but present)**: Raw petgraph algorithms remain compiled; only `graph_requires_cycles` is curated but others still exist behind the scenes.
- **Analysis monolith**: `analysis/mod.rs` (~500+ lines) mixes structural, pedagogical, discourse logic; hard to reason/test; raises cognitive overhead.

### 2.5 Error handling & messaging

- `InvariantViolation` is `Vec<String>`; no codes, no IDs. Anchors/precedes errors don’t cite slugs; `GraphError::Schema` strings are opaque. Tool layer rewraps all as `InvalidPayload`.

### 2.6 Human-facing affordances

- Tool descriptions improved in `docs/graph_tools.md`, but outputs still lack metadata (`graph_version`, `course_commit`, `strict_quality`) and mutating tools do not provide previews or hints.

## 3) Evidence pointers (where issues live)

- Invariants run: `src/graph/service.rs:520-594` plus provenance check and purity/borrow-ahead/coverage/reachability/example/practice validations.
- Fadeability sim: `src/graph/specs.rs:63-118`.
- Orphan/unreachable assessments functions exist but are not called: `analysis::orphan_assessments`, `analysis::unreachable_assessments`.
- Snapshot path gap: `src/tools/llm/graph_tools/persist.rs`.
- Algorithm leakage: `src/tools/llm/graph_tools/algorithms.rs` (only cycles curated, others still compiled).
- Monolith: `src/analysis/mod.rs`.
- Blocking validations: mutation handlers call `validate_global_invariants` synchronously in `src/graph/manager.rs` and supports clone in `specs.rs`.

## 4) Impact and risk assessment (ordered)

1. **Observability gap** – No metrics/rerun for mutate/validate; responses lack graph metadata.
2. **Invariant coverage gap** – Orphan/unreachable assessments, keystone/granularity, and full precedes DAG checks are missing; strict mode misleads by only promoting warnings.
3. **Blocking actor handlers** – Latency spikes, potential mailbox starvation on real graphs.
4. **Safety gaps** – Path sandboxing absent; SourceRef path escape unchecked; unstructured errors.
5. **Low-level tool semantics** – Cognitive burden on LLM; more tokens/round-trips; higher failure rate.
6. **Schema/version rigor** – Drift risk; snapshots may mismatch course commit silently.
7. **Affordance/documentation thinness** – Outputs lack metadata/hints; mutating tools lack previews.

## 5) What to keep stable (do not regress)

- EdgeSpec architecture; typed ontology.
- Fadeability guard (but optimize and instrument it).
- Example/practice/coverage checks—extend, don’t remove.
- Slug index rebuild + rollback patterns; snapshot fsync flow.

## 6) Recommended remediation themes (preview to Step 3)

- **Telemetry & preview**: add rerun metrics and metadata; extend previews/byte hints to mutating tools.
- **Executable invariants**: wire orphan/unreachable assessments, keystone/grain, full precedes DAG; expand strict mode.
- **Async hygiene**: offload heavy validations, bound rayon; avoid graph clones in hot paths.
- **Semantic tools**: introduce upsert/align/fix-coverage/add-examples macros; de-curate raw algos; DRY parsing.
- **Safety & versioning**: sandbox snapshot paths; enforce workspace-contained SourceRef paths; add schema_version/hash and clearer commit drift handling.
- **UX**: structured error codes with IDs; outputs include graph metadata; richer descriptions.

## 7) Expanded gap details (for implementers)

### Orphans/unreachable assessments

- Missing: assessments with no assesses edges or unreachable from first principles are not flagged.
- Fix: run `orphan_assessments`, `unreachable_assessments`; treat as error in strict, warn otherwise.

### Borrow-ahead/discourse DAG

- Missing: per-episode precedes DAG revalidation and borrow-ahead severity wiring.
- Fix: add episode DAG check in invariants; include borrow-ahead results with severity; strict upgrades to errors except suppressed scopes.

### Keystone/granularity

- Missing: centrality alerts; over-bundle/fragment per white-paper thresholds; intrinsic_load cross-checks.
- Fix: add `keystone_warnings`, `grain_warnings`; warn default, error strict.

### Source-ref rigor

- Missing: revision vs course_commit match; evidence count on assesses edges.
- Fix: enforce revision equality; reject empty observation_features and empty evidence_refs (if stored) with structured code.

### Observability

- Missing: rerun scalars and tracing fields.
- Fix: log per-op latency, counts, invariant pass/fail, node/edge totals, graph_version; trace with structured fields.

### Preview ergonomics

- Missing: preview mode and pagination.
- Fix: add `fetch_body` flag, `limit`/`offset`, first-page previews with counts and hints; bodies carry byte_hint.

### Tool surface

- Missing: semantic macros and deprecation path.
- Fix: curate list to semantic tools; keep CRUD hidden/compat; remove raw algos from curated set.

### Concurrency and validation isolation

- Offloading validations requires snapshot isolation: you cannot hand actor-owned state to `spawn_blocking`. Safe patterns are clone-to-validate (likely fine at curriculum graph sizes) or shadow-graph apply+swap for batches. Avoid premature de-cloning unless metrics prove it is the bottleneck; dirty-flagging can reduce how often heavy checks run.

## 8) Closing synthesis

The graph stack’s foundations are solid but its safety envelope is thin: invariants are partial, previews/telemetry absent, errors unstructured, and tools overly granular. The fixes are mostly additive and mechanical—wiring existing analyses into invariants, adding metrics and previews, optimizing fadeability, and replacing CRUD/algorithm tools with semantic macros. With these changes, the graph side will meet the intentional architecture’s expectations for observability, cognitive economy, and contract enforcement.

## 9) Metrics and observability blueprint (to implement in Step 3)

- **Per-mutation scalars:** `graph/mutate/latency_ms{op}`, `graph/mutate/success{op}`, `graph/mutate/errors{op}`.
- **Validation:** `graph/validate/latency_ms`, `graph/validate/errors`, `graph/validate/warnings`, plus counts per invariant code (cycle, alignment_gap, coverage_gap, fadeability, purity, orphan_assessment, unreachable_assessment, borrow_ahead, keystone, grain, discourse_cycle).
- **Inventory gauges:** `graph/nodes`, `graph/edges`, `graph/version`, `graph/strict_quality` (0/1).
- **Prune/maintenance:** `graph/prune/edges_removed`, `graph/prune/latency_ms`; `graph/autosave/latency_ms`, `graph/autosave/success`.
- **Tool previews:** emit `approx_tokens`, `bytes_total`, `remaining_tokens`, `remaining_ratio` mirroring file tools.

## 10) Effort/risk map for remediation tasks

| Task | Effort | Risk | Notes |
| --- | --- | --- | --- |
| Add preview/body + pagination to tools | Medium | Low | Mostly plumbing; follows existing file-tool pattern |
| Add rerun metrics + tracing | Medium | Low | Needs thread-safe counters; minimal behavior change |
| Wire missing invariants (alignment/purity/orphans/borrow-ahead/keystone/grain) | Medium | Medium | Behavior change; gate with strict/preview |
| Optimize fadeability and offload validations | High | Medium | Must avoid regression; add timeouts and rollbacks |
| Sandbox snapshot paths + commit checks + schema_version/hash | Medium | Low | Clear acceptance criteria |
| Replace raw algo tools with semantic macros | Medium | Medium | Requires prompt/docs updates; keep CRUD behind flag |
| Split analysis module | Low | Low | Improves readability; minimal behavior change |

## 11) Acceptance tests to add/adjust

- Unit: alignment_gap detection when LO has target assesses but unreachable from first principles.
- Unit: purity check flags extraneous ancestor not in intended set for an assessment→LO pair.
- Unit: borrow-ahead severity classification (suppressed vs in-episode vs cross-episode vs no-intro).
- Unit: sandboxed snapshot path rejects `..` and absolute paths.
- Property: validate_global_invariants returns structured codes; strict mode expands set and upgrades severities.
- Integration: tool preview returns cost/hints; body requires `fetch_body=true`; pagination works.
- Performance guard: fadeability validation times out gracefully on synthetic large graph (bounded).

## 12) Migration/deprecation notes

- Keep CRUD tools for compatibility but mark deprecated; update system prompt to favor semantic macros.
- Retire raw algorithm tools from curated list; keep behind debug flag for internal use if required.
- Introduce schema_version in snapshots; provide one-step migration script if version mismatch occurs.

## 13) Staffing/ownership suggestion

- **Graph surface + tools:** 1 engineer to implement preview envelope, semantic macros, and DRY parsing.
- **Validation/invariants:** 1 engineer familiar with white-paper rules to wire missing checks and structured errors.
- **Perf/infra:** 1 engineer to offload validations, bound rayon, add metrics, and sandbox paths.

## 14) Timeline sketch (assuming 3-person parallel work)

- Week 1: Metrics scaffold; preview envelope; sandbox paths; de-curate algos; analysis split skeleton.
- Week 2: Structured error codes; invariant wiring (alignment, purity, orphans, borrow-ahead); semantic macro tools in preview-only mode.
- Week 3: Offload validations and optimize fadeability; enable strict-mode widening; docs/prompt updates; deprecate CRUD tools.

## 15) What “good” looks like post-fix

- Running `graph_gap_report` in preview shows coverage/alignment/borrow-ahead summaries with counts, byte/tokens, and hints; body only on opt-in.
- Rerun dashboard shows graph mutate/validate latencies and invariant counts alongside LLM token metrics.
- Strict mode fails graphs that violate any white-paper rule; default mode still surfaces all warnings.
- Snapshot loads fail fast on commit/schema mismatch; paths confined to workspace.
- Tool list presented to LLM is small and semantic; CRUD/algorithm tools no longer crowd the prompt budget.
