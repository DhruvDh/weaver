# Step 4 – Complexity & Bloat Reduction Plan (expanded)

Purpose: identify and reduce accidental complexity and code bloat in the graph/analysis/tooling stack, while preserving the safety, observability, and semantic improvements proposed in Steps 2–3. This is the longest dossier by design; it should stand alone as the roadmap for “make it lean.”

## 0) Principles for slimming down

- **Value density**: every exported tool or function must deliver semantic value (curriculum/quality insight), not implementation trivia.
- **Consolidate before creating**: merge overlapping tools, reports, and helpers before adding new knobs.
- **Surface minimalism**: keep the curated tool list small, semantic, and budget-aware; remove redundant or raw algorithm tools.
- **Code locality**: split god files into cohesive modules; keep each file under ~300 lines where practical.
- **Shared utilities**: centralize preview, pagination, parsing, and error-shaping; delete duplicate boilerplate.
- **Operational safety**: deprecate, don’t rip—provide migration paths and deprecation warnings.

## 1) Complexity map (hotspots to target)

1. **Tool layer duplication** – `src/tools/llm/graph_tools/commands.rs` contains near-identical parsers and structs per edge type.
2. **Analysis monolith** – `src/analysis/mod.rs` mixes structural, pedagogical, discourse, and centrality logic in one file.
3. **Algorithm leakage** – `graph_tools/algorithms.rs` exposes petgraph terms (bridges, articulation, pagerank) to the LLM.
4. **Persistence split & duplication** – snapshot logic lives in both `graph/persist.rs` and bespoke code paths in `graph/manager.rs`.
5. **Redundant validation paths** – multiple cycle checks and graph clones (supports fadeability) inflate cost and code size.
6. **Inconsistent payload shapes** – tools emit varied JSON shapes, requiring per-tool parsing and maintenance.

## 2) De-curate and delete low-value tools

**Goal:** shrink curated surface to semantic, budget-aware tools; hide or remove raw/duplicative ones.

- Remove from `CURATED_TOOL_IDS`: raw algorithm tools (`graph_requires_bridges`, pagerank, articulation), and any primitive inspection tools superseded by semantic reports. Keep them behind `debug_allow_raw_algos` if needed for developers.
- Deprecate CRUD tools (`graph_insert_knowledge`, `graph_add_requires`, etc.) once semantic macros are stable. Keep stubs emitting deprecation warnings plus a pointer to replacements.
- Consolidate analysis tools into a **small intent-based set** (not a single mega-tool): e.g., `inspect_structure` (DAG/cycles/connectivity), `audit_pedagogy` (alignment/coverage/examples/practice/fadeability/purity), `check_integrity` (source refs, commit/schema/hash drift). All use the shared preview/pagination envelope.
- Consolidate multiple analysis outputs into the intent-based trio (`inspect_structure`, `audit_pedagogy`, `check_integrity`) using the shared preview/pagination envelope; avoid one mega-tool that mixes too many flags.

## 3) DRY parsing and tool scaffolding

**Files:** `src/tools/llm/graph_tools/common.rs`, `commands.rs`, `analysis.rs`, `inspection.rs`

- Introduce a generic `GraphCommandMeta` + `GraphPreviewResponder` that: (a) derives schema, (b) enforces `deny_unknown_fields`, (c) trims strings, (d) wraps responses in the standard `{meta, preview?, data?, pagination?}` envelope, (e) supports `fetch_body`, `limit`, `offset` automatically.
- Replace all per-tool `parse_*` functions with instances of this generic. Expect 40–60% LOC reduction in tool layer.
- Centralize validation of slugs/titles/lines and list-size caps. Delete bespoke validations sprinkled across tools.
- If generics become noisy, prefer a small `define_graph_tool!` macro over a deep trait stack—readability first.

## 4) Collapse god modules into cohesive units

### 4.1 Analysis split

- Create `analysis/{structure, alignment, pedagogy, discourse, keystone}.rs` plus a thin `mod.rs` re-export. Each file owns one concern and stays small.
- Co-locate tests with each module to keep feedback local. Remove cross-cutting helper clutter.

### 4.2 Manager organization

- In `graph/manager.rs`, group handlers by domain (mutations, queries, persistence, maintenance). Extract shared helpers (logging/metrics/preview) to a small internal `ops` module. Goal: shorter file, fewer repeated patterns.

## 5) Collapse duplicate logic and expensive clones

- Unify cycle checks (requires/precedes) into a helper with edge filters.
- Replace supports fadeability full-graph clone with incremental reachability test (insert edge into view; use reachability deltas). Eliminates repeated large allocations and simplifies code paths.
- Remove repeated global validations inside mutation paths once a single, off-thread validation pipeline is in place (from Step 3). One validation per mutation; no silent double-checks.

## 6) Simplify persistence paths

- Ensure snapshot save/load go through `graph/persist.rs` only. Remove duplicate save/load code in `manager.rs` beyond orchestration.
- Keep one serialization format with `schema_version`/`graph_hash`; avoid parallel formats.

## 7) Standardize payload schema

- All graph tool responses use a common envelope:
  - `meta`: `{graph_version, course_commit, strict_quality}`
  - `preview?`: counts, samples, cost estimates, hints
  - `data?`: full body when `fetch_body=true`
  - `pagination?`: `{limit, offset, has_more}` when applicable
  - `errors?`: structured `{code, details}` list for partial failures
- This removes ad-hoc shapes and simplifies client parsing and docs.

## 8) Documentation and prompt diet

- Update `AGENTS.md` and tool descriptions once, reflecting the slimmed curated list and semantic macros. Remove mention of raw algorithm tools and deprecated CRUDs.
- Update system prompt to list only semantic tools, reducing prompt-token load and LLM confusion.

## 9) Tests: reduce overlap, increase focus

- Collapse overlapping property tests into parameterized sets; cap `PROPTEST_CASES` as already supported.
- Add golden tests for the new unified envelope; remove per-tool response-shape tests.
- Keep targeted unit tests for the refactored analysis modules; avoid duplicating coverage across modules.

## 10) Telemetry namespace hygiene

- Keep metric names flat and minimal (see Step 3): `graph/mutate/*`, `graph/validate/*`, `graph/nodes`, `graph/edges`, `graph/version`. Avoid per-tool metric explosions.

## 11) Concrete bloat-reduction targets

- **Tool count**: reduce curated tool IDs from ~20 to ~8 (semantic macros + core inspection/persistence/validation triggers).
- **Commands file**: shrink `commands.rs` by ≥50% via generic parser and deprecation of CRUD tools.
- **Analysis**: split into ≤5 files; each <300 LOC; remove monolithic `mod.rs` bulk.
- **Payload shapes**: one envelope pattern everywhere; delete bespoke schemas.
- **Cloning**: remove full-graph clone in supports validation; target 2–3× speed and lower memory footprint on large graphs.

## 12) Execution sequence (low-risk to high)

1. **De-curate surface**: remove raw algos from curated list; add deprecation warnings to CRUD tools; adjust prompt/docs.
2. **Shared tool envelope**: implement generic parser + preview wrapper; refactor existing tools to use it.
3. **Analysis split**: move functions to cohesive modules; re-export; fix imports; add focused tests.
4. **Payload standardization**: update all tools to common envelope; adjust tests/docs accordingly.
5. **Clone/duplication removal**: refactor fadeability and validation paths; remove duplicate cycle checks and redundant validations.
6. **Persistence path simplification**: ensure manager delegates to persist module only.
7. **Final surface prune**: remove deprecated CRUD tools from curated list after one release; clean docs/tests.

## 13) Risk mitigation

- Keep deprecated tools callable for one release; emit deprecation notices; provide mapping to replacements.
- Maintain debug flag to expose raw algos internally if needed for analysis; keep them out of curated surface to protect LLM prompt budget.
- Add feature flags for envelope preview defaults and pagination limits to allow rollback if clients struggle.
- Use comprehensive tests and rerun metrics to watch for performance regressions when removing clones.

## 14) Before/after snapshots (expected)

- **Prompt/tool list**: from a long list of CRUD/algorithm tools to a short semantic set; prompt token savings; clearer LLM behavior.
- **LOC**: commands/tool layer and analysis shrunk significantly; easier onboarding and code reviews.
- **Runtime**: faster supports validation (no full clone); fewer redundant checks; bounded workloads.
- **Docs**: shorter, sharper `AGENTS.md`; system prompt simpler.

## 15) Alignment with earlier steps

- Step 2 gaps (observability/preview, invariant coverage, blocking handlers, safety, low-level semantics) are addressed by Step 3; Step 4 prevents new bloat while consolidating. The slimming measures here ensure the stronger invariants and telemetry don’t explode the surface area.
- Step 3 semantic macros reduce tool count; Step 4 removes deprecated CRUD and raw algorithms, ensuring the curated interface remains lean.

## 16) Ownership and effort estimates

- **Tool surface & parsing DRY**: medium effort, low risk; good first task.
- **Analysis split**: low effort, low risk; mostly moving code and fixing imports/tests.
- **Fadeability/clone removal**: medium/high effort, medium risk; needs careful perf testing.
- **Payload standardization**: medium effort, medium risk (touches many tools/tests/docs).
- **De-curation/removal**: low effort, low risk, but needs comms and deprecation period.

## 17) Success criteria (binary checks)

- Curated tool list ≤8 semantic entries; raw algos absent; CRUD marked deprecated or removed after grace period.
- All tool responses share the common envelope; previews default with cost/hints; bodies require explicit opt-in; pagination works.
- Analysis code organized into cohesive modules; `analysis/mod.rs` slim.
- Supports validation no longer clones entire graph; validation pipeline runs once per mutation.
- Snapshot save/load code centralized in `graph/persist.rs`; no parallel serialization paths.
- Documentation and prompt reflect the slimmed surface; tests updated accordingly.

## 18) Communication plan

- Release notes: announce semantic tool set, deprecated/removed tools, new envelope, and standardized metrics; include migration examples.
- Update system prompt and any client scaffolding to point to semantic tools only.
- Add short “How to migrate” section to `AGENTS.md` linking old → new tool names and sample payloads.

## 19) Post-slim audit

- After refactor, run a simple audit: tool ID count, LOC of commands/tool modules, LOC per analysis module, memory/time of supports validation on sample large graph, prompt-token size for tool specs. Record before/after to ensure targets hit.

## 20) Final note

This slimming plan makes the strengthened graph system maintainable: fewer tools to reason about, fewer places to fix when invariants evolve, smaller prompts for the LLM, and reduced runtime cost. It locks in the gains from Steps 2–3 without letting complexity creep back.
