Objective
---------

Shrink boilerplate in LLM tools, make graph mutations safer, and speed analysis tooling via shared runners, macros, caching, and typed slug checks.

Execution checklist (update as you go)
--------------------------------------

- [x] ToolRunner + run_graph_action + paginate! helper in place; sample tools migrated.
- [x] graph_action_tool! / analysis_tool! / tool_ids! macros added; at least one mutation + one analysis tool migrated.
- [x] ValidationGuard wired into core graph mutations (add/update/remove, edges).
- [x] with_cached_analysis adopted by representative analysis tools; legacy cache boilerplate removed in those files.
- [x] Slug<T> resolver used in hot paths (extraneous, alignment, add_*). *Typed preflight added for add_* edges.*
- [x] CURATED_TOOL_IDS driven by tool_ids!; old hard-coded lists removed.
- [x] Sweep complete: remaining tools migrated; dead helpers deleted.
- [x] Macros everywhere: migrate all remaining mutation/analysis tools to graph_action_tool!/analysis_tool!, delete render_payload/finalize_summary_tool, update docs/graph_tools.md and CURATED_TOOL_IDS if needed, and extend typed Slug resolution to remaining edge/command hot paths.
- [x] Align CURATED_TOOL_IDS with the documented graph tool surface (first_principles views, gap/anchor views, extraneous, etc.); rerun fmt/clippy/tests after the change.
- [x] Restore debug-only access to requires_* algorithm tools when `WEAVER_DEBUG_GRAPH_ALGORITHMS=1`; currently filtered out by CURATED_TOOL_IDS.
- [x] Re-run `cargo fmt`, `cargo clippy --all-targets`, and `cargo test --all --locked` after the debug gating change.
- [x] Guard rename_node with ValidationGuard to keep mutations consistent.
- [x] Add a basic_tool! macro for non-graph tools and migrate filesystem + delegate_tasks onto it.
- [x] Make delegate_tasks use ToolRunner with preview/body (fetch_body=false by default) to avoid context blow-ups.

Follow-ups discovered (2025-12-01)
----------------------------------

- [x] Bring ops/admin tools onto ToolRunner for consistent cost/meta/pagination (`graph_redundant_requires`, `graph_analysis_cache_clear`).
- [x] Rerun `cargo fmt`, `cargo clippy --all-targets`, and `cargo test --all --locked` after the above changes.

In-flight notes
---------------

- All graph mutation tools (insert/update, edges, rename/remove, save/load snapshot) now use `graph_action_tool!`.
- Pagination helpers consolidated on `paginate!`; function variant removed.
- ToolRunner covers filesystem + graph tools; analysis tools share `with_cached_analysis`.
- Tests rerun after ToolRunner/admin sweep (2025-12-01): `cargo fmt`, `cargo clippy --all-targets`, `cargo test --all --locked` all pass.

Definition of done (per phase)
------------------------------

- Phase A: ToolRunner used by `list_directory` + `graph_save_now`; no direct render_payload in those tools.
- Phase B: Macros generate parse+execute for at least 2 tools; curated IDs sourced from macro output.
- Phase C: add/update/remove node/edge paths use ValidationGuard; duplicated rollback code gone.
- Phase D: `dag_check` + `keystone` (or equivalent) rely on with_cached_analysis; no manual cache plumbing there.
- Phase E: Slug<T> type enforces expected KnowledgeType in at least extraneous/alignment/add_requires flows.

  A. Tool-running & graph-command abstraction

  - Introduce a generic ToolRunner (in existing tools/llm/common.rs) that handles preview/body selection, cost estimation, meta attachment, and optional pagination hook. Each tool then only
    supplies input parsing and a compute() async closure.
  - Define a GraphAction trait (or just a struct) for mutations with apply flag, build_message, and success_payload. Add a helper run_graph_action(action, graph) that does preview vs apply,
    sends the message, formats the standard {type:"graph_command", status} response.
  - Refactor filesystem tools and graph tools to use ToolRunner; eliminate per-tool render_payload/finalize_summary/attach_meta boilerplate.

  B. Macros to erase repetition

  - Add graph_action_tool! macro (same file) that expands a tool ID, args type, builder, and success JSON into a complete tool implementation using run_graph_action.
  - Add analysis_tool! macro for preview/body analysis tools: parse args, load (graph, version), run with_cached_analysis!(key => compute), optional pagination, and wrap response.
  - Add paginate! macro that clamps limit/offset and returns (page, offset, limit, has_more) to replace duplicated logic.
  - Add tool_ids! macro to declare tool ID constants and populate CURATED_TOOL_IDS, keeping strings in sync.

  C. Validation & mutation hygiene

  - Implement a ValidationGuard (RAII) in graph/service.rs: marks dirty families, executes a closure, validates (targeted or full), rolls back on error, bumps version on success. Replace
    repeated mark/validate/rollback blocks in add/update/remove.
  - Add small helpers for provenance checks (Provenance::check) and path hints to reuse in validation and file tools, trimming repeated error text.

  D. Analysis caching unification

  - Create with_cached_analysis(key, version, cache, compute) helper that prunes per kind/version, stores sync/async, and returns payload only. Use it across analysis/*.rs, algorithms.rs,
    gaps.rs, discourse.rs, etc.
  - Standardize analysis tool shape: resolve inputs → with_cached_analysis → optional paginate! → ToolRunner for preview/body.

  E. Slug resolution & typing

  - Add a Slug<T> wrapper (phantom typed by expected KnowledgeType) plus resolve<T>(&GraphManager, Slug<T>) -> NodeId that enforces node kind at compile time. Replace scattered
    ensure_knowledge_type calls and stringly slug errors in analysis/extraneous/commands.

  Execution order (suggested)

  1. Implement ToolRunner + run_graph_action + paginate! helper.
  2. Introduce graph_action_tool! and analysis_tool! macros; refactor a couple of representative tools to prove the pattern (one mutation, one analysis).
  3. Add ValidationGuard and refactor GraphService mutations to use it.
  4. Add with_cached_analysis helper and unify analysis tools.
  5. Add Slug<T> resolver and migrate hot paths (extraneous, alignment, add_*).
  6. Add tool_ids! macro to dedupe string IDs and sync CURATED_TOOL_IDS.
  7. Sweep remaining tools to the new macros/helpers and delete dead boilerplate.

Guidance / approach
-------------------

- **ToolRunner & run_graph_action**: centralize preview vs body, cost estimation, meta attach, optional pagination; tools provide parsing + compute closure.
- **GraphAction trait**: carries apply flag, builds actor message, shapes {type:"graph_command",status} payload; run_graph_action handles preview/apply and error mapping.
- **Macros**: graph_action_tool! for mutations; analysis_tool! for preview/body analyses with with_cached_analysis + optional paginate!; paginate! clamps limit/offset; tool_ids! declares IDs and feeds CURATED_TOOL_IDS.
- **ValidationGuard** in graph/service.rs: RAII for marking dirty families, running mutation, validating (targeted/full), rollback on error, bump version on success; add provenance/path helpers.
- **with_cached_analysis**: helper to prune by version/kind, cache sync/async, and return payload; standard analysis tool shape: resolve inputs → with_cached_analysis → optional paginate! → ToolRunner.
- **Slug<T> resolver**: typed slug wrapper + resolve that enforces KnowledgeType and replaces ad hoc ensure_knowledge_type/string errors in extraneous/alignment/add_*.
- **Execution order**: 1) ToolRunner/run_graph_action/paginate! 2) macros + refactor sample tools 3) ValidationGuard refactor mutations 4) with_cached_analysis unify analyses 5) typed Slug resolver 6) tool_ids! sync IDs 7) sweep remaining tools and delete old boilerplate.

Open questions / risks
----------------------

- Macro visibility/export: ensure graph_action_tool! and analysis_tool! are usable across modules without `#[macro_use]` surprises.
- ValidationGuard scope: confirm it covers targeted vs full invariant runs without extra allocations.
- Slug<T> ergonomics: balance type safety with LLM tool arg JSON parsing (may need From<String> shim).

Validation gate
---------------

- Run before push: `cargo fmt`, `cargo clippy --all-targets`, `cargo test --all --locked`.

Full guide
----------

Here is a comprehensive implementation guide to refactor the `weaver` codebase according to the requested tasks.

---

### Phase 1: ToolRunner & Graph Action Abstraction (Task A)

**Goal:** Centralize the logic for cost estimation, preview generation, metadata attachment, and standard response formatting.

**File:** `src/tools/llm/common.rs`

1. **Define `ToolRunner`:**
    Create a struct that holds the context required to render a response (metrics, model, conversation_id, etc.).

    ```rust
    pub struct ToolRunner<'a> {
        pub state: &'a CallState,
        pub tool_id: &'static str,
    }

    impl<'a> ToolRunner<'a> {
        pub fn new(state: &'a CallState, tool_id: &'static str) -> Self {
            Self { state, tool_id }
        }

        pub fn run<F>(self, fetch_body: bool, payload_builder: F) -> Result<ToolOutput, ToolExecutionError>
        where
            F: FnOnce() -> serde_json::Value,
        {
            // 1. Calculate size of payload
            // 2. Estimate tokens using self.state.metrics
            // 3. If !fetch_body, build preview envelope
            // 4. Return ToolOutput
            // Refactor existing `render_payload` logic into this method.
        }
    }
    ```

2. **Define `GraphAction`:**
    Create a trait or structure to standardize mutations.

    ```rust
    // In src/tools/llm/graph_tools/common.rs

    pub trait GraphAction {
        type Args: Send + Sync;
        type Message: kameo::Message<Reply = Result<Self::Success, crate::graph::GraphError>> + Send;
        type Success: Send;

        fn build_message(args: &Self::Args) -> Self::Message;
        fn success_payload(args: &Self::Args, reply: Self::Success) -> serde_json::Value;
    }

    pub async fn run_graph_action<A>(
        tool_id: &'static str,
        args: A::Args,
        apply: bool,
        state: &CallState,
    ) -> Result<ToolOutput, ToolExecutionError>
    where
        A: GraphAction,
        // Bounds for Args to allow cloning/serializing for preview
    {
        if !apply {
            // Return standard preview JSON
        }
        
        let msg = A::build_message(&args);
        let reply = state.graph.ask(msg).await...; // Handle error mapping
        let payload = A::success_payload(&args, reply);
        
        // Wrap in standard {type:"graph_command", status:"ok", ...}
        // Use ToolRunner to finalize output
    }
    ```

### Phase 2: Analysis Caching Unification (Task D)

**Goal:** Remove repetitive cache lookup code in `algorithms.rs`, `alignment.rs`, etc.

**File:** `src/tools/llm/graph_tools/common.rs`

1. **Implement `with_cached_analysis`:**

    ```rust
    pub async fn with_cached_analysis<F, Fut>(
        tool_id: &'static str,
        state: &CallState,
        cache_kind: AnalysisKind,
        compute: F,
    ) -> Result<ToolOutput, ToolExecutionError>
    where
        F: FnOnce(Arc<CurriculumGraph>) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<serde_json::Value>> + Send,
    {
        // 1. Load graph & version via actor
        // 2. Construct AnalysisCacheKey
        // 3. state.analysis_cache.get_or_try_insert_with_async(...)
        // 4. Attach graph_meta
        // 5. Use ToolRunner to return output
    }
    ```

### Phase 3: Macros for Boilerplate Erasure (Task B)

**Goal:** Replace manual struct definition and `impl ToolInstance` with declarative macros.

**File:** `src/tools/llm/macros.rs` (create new or add to `mod.rs`)

1. **`tool_ids!` Macro:**

    ```rust
    macro_rules! tool_ids {
        ($($name:ident = $str:literal),* $(,)?) => {
            $(pub const $name: &str = $str;)*
            
            pub fn curated_tool_ids() -> &'static [&'static str] {
                &[$($str),*]
            }
        };
    }
    ```

2. **`graph_action_tool!` Macro:**
    Should accept the Tool ID, the Args struct, and the Logic closure. It expands into the `parse_...` function and the `ToolInstance` implementation that calls `run_graph_action`.

    ```rust
    macro_rules! graph_action_tool {
        ($tool_id:ident, $args_type:ty, $msg_builder:expr, $success_builder:expr) => {
            // Generates struct, parse function, and ToolInstance impl
            // calling run_graph_action inside execute()
        };
    }
    ```

3. **`paginate!` Macro:**
    Refactor existing `paginate` function in `common.rs` into a macro if it needs to handle different types loosely, or keep it as a generic function if simpler.

### Phase 4: Validation & Mutation Hygiene (Task C)

**Goal:** Safe graph mutations with automatic rollback and invariant checking.

**File:** `src/graph/service.rs`

1. **Implement `MutationGuard` (Transaction-like):**

    ```rust
    impl GraphService {
        pub fn mutate<F, R>(&mut self, families: InvariantFamilies, mutation: F) -> Result<R, GraphError>
        where
            F: FnOnce(&mut CurriculumGraph) -> Result<R, GraphError>,
        {
            // 1. Snapshot critical state (graph pointer, version, slugs)
            // 2. Mark dirty families
            // 3. Run mutation closure
            // 4. If result is Err -> restore snapshot, return Err
            // 5. Run validate_invariants(families)
            // 6. If validation fails -> restore snapshot, return Err
            // 7. If success -> bump_version(), refresh_rubric_hashes, return Ok
        }
    }
    ```

2. **Refactor Methods:**
    Update `add_knowledge_node`, `add_edge`, etc., to use `self.mutate(...)` instead of the manual "mark dirty -> try -> validate -> rollback" blocks.

### Phase 5: Slug Resolution & Typing (Task E)

**Goal:** Compile-time or centralized runtime checking of node kinds.

**File:** `src/tools/llm/graph_tools/common.rs` or `src/graph/mod.rs`

1. **Define Typed Slug:**

    ```rust
    use std::marker::PhantomData;

    pub struct Slug<K>(pub String, PhantomData<K>);

    // Marker traits or structs
    pub struct AssessmentKind;
    pub struct LOKind;
    pub struct AnyKind;
    ```

2. **Implement Resolver:**

    ```rust
    pub async fn resolve_slug<K>(
        graph_actor: &ActorRef<GraphManager>,
        slug: Slug<K>,
    ) -> Result<NodeId, ToolExecutionError> {
        // 1. Resolve string to NodeId via actor
        // 2. Check NodeKind matches K (helper trait required on K to map to KnowledgeType)
        // 3. Return NodeId
    }
    ```

3. **Update Tools:**
    Change tool implementations (like `extraneous` or `alignment`) to use `Slug<AssessmentKind>` in their internal logic, even if the Input Args still use raw Strings (parsing happens at the edge).

---

### Execution Order

1. **Core Abstraction (A):** Implement `ToolRunner` and `run_graph_action` in `tools/llm/common.rs`. Refactor 2-3 simple tools (e.g., `list_directory`, `graph_save_now`) to use them.
2. **Validation (C):** Refactor `GraphService` to use the `mutate` closure pattern. This cleans up the massive `service.rs` file significantly.
3. **Caching & Analysis (D):** Implement `with_cached_analysis`. Convert `dag_check` and `keystone` to use it.
4. **Typing (E):** Introduce `Slug<T>` and refactor `extraneous` and `alignment` analysis tools to use it.
5. **Macros (B):** Implement `tool_ids!` and `graph_action_tool!`. Convert all remaining mutation tools (`add_requires`, `update_knowledge`, etc.) to use the macro.
6. **Cleanup:** Remove the old boilerplate functions (`render_payload`, manual validations) once all tools are migrated.
