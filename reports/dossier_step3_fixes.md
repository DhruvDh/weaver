# Step 3 – Concrete Fixes to Bring the Graph Stack Up to the Quality Bar (expanded)

This document turns the Step 2 audit into an actionable, file-scoped plan. It preserves the intentional architecture (actors, existing analysis previews/pagination/cache) and applies the white-paper contract. Lengthy by design so nothing is implicit.

## 0) Design principles carried forward

- **Preview-first, opt-in bodies**: Every potentially large response defaults to preview with cost/hints; bodies require `fetch_body=true` and include `byte_hint`.
- **Observability everywhere**: Stable rerun metrics + structured tracing for mutations, validations, analyses, and maintenance.
- **Executable invariants**: All white-paper rules become code; strict mode both widens checks and hardens severities.
- **Actor hygiene**: Keep handlers short; push heavy work off-thread with bounded concurrency and timeouts.
- **Cognitive economy**: Prefer semantic macros over CRUD/graph-theory tools; small curated surface.
- **Safety & reproducibility**: Workspace sandboxing, commit-pinned source refs, versioned snapshots with hashes.
- **Structured feedback**: Machine-actionable error codes + offending slugs/ids for LLM self-correction.

## 1) Observability & preview scaffolding (do first)

**Files:** `src/graph/manager.rs`, `src/graph/service.rs`, `src/graph/specs.rs`, `src/tools/llm/graph_tools/*`, `src/rerun_sink.rs`

### 1.1 Graph metrics

- Add `GraphMetrics` (mirrors `GatewayMetrics`) with atomic counters/gauges: `total_mutations`, `total_validations`, `invariant_failures{code}`, `nodes_total`, `edges_total`, `graph_version`, `strict_quality`, per-op `last_latency_ms`.
- Inject optional `RerunSink` into `GraphManager` state; emit scalars on each mutation, validation, prune, snapshot save/load, and heavy analysis. Metric paths (flat):
  - `graph/mutate/latency_ms{op}`, `graph/mutate/success{op}`
  - `graph/validate/latency_ms`, `graph/validate/errors`, `graph/validate/warnings`, `graph/validate/{code}`
  - `graph/nodes`, `graph/edges`, `graph/version`, `graph/strict_quality`
  - `graph/prune/edges_removed`, `graph/prune/latency_ms`
  - `graph/autosave/latency_ms`, `graph/autosave/success`

### 1.2 Structured tracing

- Enrich tracing events with fields: `op`, `graph_version`, `course_commit`, `node_count`, `edge_count`, `strict`, `duration_ms`, `error_code`.

### 1.3 Preview/body envelope for graph tools

- Extend the existing preview/pagination/byte-hint helper (used by analysis tools) to **mutating** and large inspection tools; add remaining-budget fields.
- Ensure all list-producing tools use `paginate` with sane limits (many already do); cap defaults/max consistently.
- Every response includes `meta`: `{graph_version, course_commit, strict_quality}` (commands/admin included).

## 2) Structured errors & codes

**Files:** `src/graph/model.rs`, `src/tools/llm/graph_tools/common.rs`

- Replace `InvariantViolation(Vec<String>)` with `InvariantViolation(Vec<InvariantCode>)` where `InvariantCode` is an enum: `Cycle { path_slugs }`, `AlignmentGap { lo, missing_assessments }`, `CoverageGap { lo, missing_criteria }`, `Fadeability { assessment, supports }`, `Purity { assessment, extraneous }`, `BorrowAhead { step, target, severity }`, `OrphanAssessment { slug }`, `UnreachableAssessment { slug }`, `Keystone { slug, score }`, `Granularity { slug, kind }`, `DiscourseCycle { episode }`.
- Extend `GraphError::Schema` to include optional `slug`/`edge` fields for endpoint errors.
- Map errors in tool layer to structured payloads `{type:"error", code, details:{...}}`; keep user vs internal distinction.

## 3) Invariant widening & strict-mode strengthening

**Files:** `src/analysis/*` (after split), `src/graph/service.rs`

### 3.1 Split analysis into cohesive modules

- `analysis/structure.rs`: cycles, toposort, transitive reduction, reachability helpers.
- `analysis/pedagogy.rs`: example gaps, procedural practice gaps, fadeability.
- `analysis/alignment.rs`: lo_reachability, coverage_report, extraneous/purity, alignment_gap finder.
- `analysis/discourse.rs`: borrow_ahead, discourse_orphans, episode DAG check.
- `analysis/keystone.rs` (optional): keystone/centrality alerts.
- `mod.rs`: re-export typed reports.

### 3.2 New/expanded checks in `validate_global_invariants`

- Always run: requires DAG, fadeability, example/practice gaps, coverage gaps, alignment reachability (already present), purity (already present), borrow-ahead (already present), plus **orphan/unreachable assessments**, **keystone/granularity**, and a **full precedes DAG** check.
- Default severity: cycles/fadeability/alignment/discourse_cycle = errors; coverage/example/practice/borrow-ahead/keystone/granularity/orphans/unreachable = warnings.
- Strict mode: promotes warnings to errors and **expands** to include keystone/granularity/orphan/unreachable if not run in default; surface codes/severity.

### 3.3 Borrow-ahead/discourse

- Re-run per-episode precedes DAG over all edges (not just incremental) to catch cycles introduced indirectly.
- Borrow-ahead severities: suppress when introduction_scope is prior/external; otherwise classify InEpisode/CrossEpisode/NoIntro and emit issues accordingly.

### 3.4 Keystone/granularity

- Use existing reach counts to flag top-N keystones; require extra supports for high keystones (warn default, error strict).
- Grain rules per white paper: over-bundle (statement >2 sentences & indegree >=4) and fragment (statement <15 tokens & no assesses/supports). Emit `Granularity` codes.

## 4) Async hygiene & performance

**Files:** `src/graph/manager.rs`, `src/graph/service.rs`, `src/graph/specs.rs`

- Prefer **snapshot isolation** over sharing live state: clone the graph for validation/off-thread checks. For batch mutations, apply to a shadow clone, validate, then `mem::swap` into place on success—no complicated rollback bookkeeping.
- Run validation/fadeability on the clone inside `spawn_blocking` (bounded rayon) with a timeout (default ~2s) → emit `validation_timeout` on overrun.
- Keep the clone for safety; only optimize if metrics show it as a bottleneck. Add optional dirty flags to skip heavy families that a mutation cannot affect.
- Fadeability: use incremental reachability on the cloned graph; reserve full recompute for debug mode.
- Move maintenance (prune redundant requires, full validation sweeps) to scheduler workers; return ticket/promise IDs if needed; emit progress metrics.

## 5) Safety & reproducibility

**Files:** `src/tools/llm/graph_tools/persist.rs`, `src/tools/llm/mod.rs`, `src/graph/persist.rs`, `src/schema/validate.rs`

- Path sandboxing: resolve snapshot paths via `resolve_workspace_path`; reject escapes; preview before execute.
- Source refs: keep revision enforcement; add path containment to the workspace root.
- Snapshot metadata: add `schema_version`, `graph_hash` (blake3), `course_commit` (mandatory). Load verifies version/hash/commit and returns specific codes `schema_version_mismatch`, `commit_mismatch`, `hash_mismatch`. Default to fail-closed; expose an explicit force flag for drift if needed.

## 6) Semantic macro tools (cognitive economy)

**Files:** new module `src/tools/llm/graph_tools/semantic.rs` (or extend `commands.rs`), `src/graph/manager.rs`

- `graph_upsert_concept`: upsert knowledge node + batch edges (requires/supports/assesses) using a shadow-graph apply (clone → mutate → validate → swap). Preview by default, apply with `apply=true`; returns created/updated slugs and edge counts.
- `graph_align_lo`: run alignment predicate; report missing target assessments/reachability; optional `apply=true` to link a chosen assessment.
- `graph_fix_coverage`: report rubric vs observation gaps; with `apply`, add observation_features to selected assesses edges.
- `graph_add_examples`: ensure procedural node has typical + edge/error worked examples; attach supports in one go.
- Reads: keep analysis tools in three intent buckets instead of one mega-tool: `inspect_structure` (DAG/cycles/connectivity), `audit_pedagogy` (alignment/coverage/examples/practice/fadeability), `check_integrity` (source refs, commit/schema/hash drift). All are preview-first with pagination.
- Keep CRUD tools for compatibility but mark deprecated; remove raw algorithm tools from curated list (optionally keep behind `debug_allow_raw_algos`).

## 7) Tool hygiene & DRY parsing

**Files:** `src/tools/llm/graph_tools/common.rs`, `commands.rs`, `analysis.rs`, `inspection.rs`

- Reduce boilerplate with a small `define_graph_tool!` macro that generates the parser + `ToolPrototype` wiring (schema, `deny_unknown_fields`, trimming, preview envelope). Simpler and clearer than a deep generic trait stack.
- Apply `#[serde(deny_unknown_fields)]` to all remaining args; enforce non-empty slugs/titles/statements/rationales; cap collection sizes.
- Standardize response schema: `{meta, preview?, data?, pagination?}` across all tools.

## 8) Documentation & affordances

**Files:** `src/tools/llm/graph_tools/*` descriptions, `AGENTS.md`

- Update tool descriptions to instruct preview-first, fetch_body usage, path sandboxing, and strict-mode effects. Add a concise operator section in `AGENTS.md` listing semantic tools and invariants enforced in strict.
- Include metadata (graph_version, course_commit, strict_quality) in all tool outputs to expose drift.

## 9) Migration/deprecation plan

- Maintain CRUD tools for one release; mark them deprecated in descriptions; suggest semantic replacements.
- Update system prompt/tool list to remove raw algorithm tools and highlight semantic macros.
- Add a migration note for snapshots: new `schema_version` and `graph_hash`; provide a one-shot migrate tool or script for old snapshots.
- Snapshot load: default is warn-on-drift in preview, fail on apply unless `allow_commit_drift`/force flag is set. Provide a `graph_snapshot_resync` CLI or tool to bless a new commit hash to avoid lockout after repo moves.

## 10) Testing plan

- Unit: alignment_gap detection; purity extraneous detection; borrow-ahead severity; sandboxed path rejection; structured error codes emitted.
- Property: invariant validation preserves graph when rolling back failed mutations; fadeability check time-bounded.
- Integration: semantic tools run in preview then apply; pagination works; metrics/rerun entries emitted.
- Regression: strict vs default mode severity matrix; snapshot load failures on commit/schema mismatch.

## 11) Ordering (minimal risk path)

1. Telemetry + preview envelope (additive, low risk).
2. Structured errors + response schema standardization.
3. Invariant widening with strict-mode expansion (behavioral change; guard with good messages and metrics).
4. Safety hardening (sandbox paths, commit/schema/hash checks).
5. Async/offloading optimizations with timeouts.
6. Semantic macros + deprecation of CRUD/algorithm tools.
7. Docs/prompt updates; cleanup of deprecated entries after a grace period.

## 12) Definition of done (alignment with quality bar)

- All graph tools are preview-first, paginated, with cost hints; bodies require opt-in.
- Rerun/metrics show mutate/validate/prune/autosave latencies and invariant counts; tracing events carry graph metadata.
- `validate_global_invariants` covers the full white-paper rule set; strict mode widens and hardens checks.
- Heavy validations/fadeability run off-thread with bounded time; actor handlers stay fast.
- Snapshots are sandboxed, versioned, hashed, and commit-checked; source refs are pinned.
- Semantic macro tools are curated; CRUD/algorithm tools no longer pollute the prompt; responses are structured and consistent.
- Analysis code is modular; error codes are machine-actionable.

These fixes raise the graph stack to the same safety, observability, and cognitive-economy standard as the intentional components, while keeping changes localized and incremental.

## 13) Sample metric and payload shapes (for consistency)

- Mutation success event (tracing): `{ op:\"add_requires\", graph_version:42, course_commit:\"abc1234\", nodes:1200, edges:3800, strict:true, duration_ms:8 }`.
- Validation failure (rerun): path `graph/validate/cycle`, value 1; accompanying trace includes `code:\"cycle\"`, `path_slugs:[\"a\",\"b\",\"c\",\"a\"]`.
- Tool preview payload: `{ \"meta\":{ \"graph_version\":42,\"course_commit\":\"abc1234\",\"strict_quality\":true }, \"preview\":{ \"count\":120, \"first\": [...sample...], \"cost\":{ \"bytes_total\":20480, \"approx_tokens\":512, \"remaining_tokens\":80000, \"remaining_ratio\":0.6 }, \"hints\":[\"Set fetch_body=true to stream all\",\"Use limit/offset to page\"] }, \"pagination\":{ \"limit\":20,\"offset\":0,\"has_more\":true } }`.
- Error payload: `{ \"type\":\"error\", \"code\":\"coverage_gap\", \"details\":{ \"lo\":\"lo.functions.post\", \"missing\":[\"edge cases\",\"negative inputs\"], \"graph_version\":42 } }`.

## 14) Rollback and idempotency plan

- Prefer shadow-graph apply (clone → mutate → validate → swap) for multi-step mutations. If mutating in-place, record changed nodes/edges and restore them on failure.
- Semantic upsert commands must be idempotent: re-running identical payloads should yield a “no-op” result with zero new edges.
- Snapshot load should be staged: validate snapshot, then swap; on failure the in-memory graph stays untouched.

## 15) Risk mitigations and feature flags

- Env/CLI: `GRAPH_VALIDATE_TIMEOUT_MS`, `GRAPH_STRICT_QUALITY`, `GRAPH_ALLOW_COMMIT_DRIFT` (force flag), `GRAPH_DEBUG_ALLOW_RAW_ALGOS`, `GRAPH_VALIDATION_DIRTY_FLAGS`.
- Gate semantic macro `apply` behind `apply=true`; keep preview-only default.
- Keep deprecated CRUD tools callable for one release; emit deprecation warnings with suggested replacements.

## 16) Interface changes to communicate

- Update system prompt: semantic tools, preview-first, deprecated algos/CRUD, force flags for commit drift.
- `AGENTS.md` change log: metrics, strict-mode expansion, sandboxed paths, schema_version/hash, semantic tool IDs, deprecated IDs.
- Provide preview→apply examples for each semantic tool.

## 17) Acceptance criteria (binary)

- `graph_upsert_concept` preview → no mutation; `apply=true` mutates via shadow-graph swap; metrics show one mutation; no invariant errors.
- `validate_global_invariants` returns structured codes; strict mode fails alignment/purity; default mode warns.
- Snapshot save sandboxed; load enforces schema/commit/hash unless forced; on failure state unchanged.
- Rerun dashboard shows graph mutate/validate metrics; tool list is semantic-only.

## 18) Work split suggestion (who does what)

- **Observability + preview envelope**; **Invariant expansion + structured errors**; **Async/offload + fadeability optimization**; **Semantic tools + DRY parsing (macro)**; **Safety/versioning** (path sandbox, commit/hash/schema).

## 19) Sequencing with rollback safety

1. Metrics/tracing + preview helper.
2. Structured errors + response schema.
3. Invariant expansion (warn-only first), then strict hardening.
4. Safety: sandbox paths, schema_version/hash, commit drift handling/migration script.
5. Offload validations + fadeability timeout/metrics.
6. Semantic tools preview → apply; deprecate CRUD/algos after grace.

## 20) Size/complexity reduction targets (to align with Step 4 goals)

- Curated tool count ~8 (semantic macros + essential inspection/persist); commands LOC -40–60%.
- `analysis` split into ≤5 files <300 LOC each; `analysis/mod.rs` slim.
- Supports validation avoids extra clones; validation runs once per mutation with timeouts.

Meeting these criteria will align the graph stack with the intentional architecture’s safety, observability, and cognitive-economy standards while also setting up the simplifications outlined in Step 4.
