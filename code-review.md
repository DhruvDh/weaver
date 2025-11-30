I can see all 5/5 context chunks and all the files they referenced; nothing looks missing relative to your first header. 👍

I’ll treat the graph stack (graph service, analyses, and `graph_tools/*`) as the “vibe-coded” area and compare it against the quality bar implied by your actor system, tool framework, preview handshake, metrics, etc.

I’ll go in this order:

1. What your **quality bar** looks like from the “intentional” parts.
2. How the **graph stack** compares, layer by layer.
3. **Concrete, local change suggestions** by file / area.
4. Where the code has **bloated** and how to trim it without a rewrite.

---

## 1. Inferred “quality bar” from the intentional parts

From the pieces you clearly cared about (actor system, `CallState`, tool prototypes, preview handshake, metrics, file reader, etc.), the bar looks roughly like:

1. **Thin, layered design**

   * Actors own state; tools are thin wrappers; core logic lives in pure modules (`analysis`, `traversal`, etc.).
   * You avoid stuffing “policy” into the LLM tool surface.

2. **Strong typing & schemas**

   * Enums for discrete things (e.g., `SupportKind`, `AssessmentScope`), not raw strings.
   * `serde` + `schemars` with good `description` metadata.
   * Clear separation of “tool input schema” vs “graph-internal types”.

3. **Uniform tool ergonomics**

   * Every tool has a `ToolPrototype` with id, description, schema, and a small `parse_*` function.
   * Parse functions validate required fields with `require_string`.
   * LLM-visible JSON payloads are structured and predictable (`type`, `tool`, etc.).

4. **Cost-aware preview handshake**

   * For potentially large payloads, you:

     * Compute `approx_bytes`.
     * Use `prepare_payload_estimates` → `build_cost_preview` → `apply_preview_cost`.
     * Gate full bodies behind `fetch_body = true` and `ToolPayloadMode`.

5. **Error shaping & logging**

   * `GraphError` mapped to `ToolExecutionError::Input` vs `Internal` clearly.
   * Messages are human-readable and tests assert on meaningful substrings.
   * `tracing::info!` with structured fields (`tool`, `offset`, `limit`, etc.).

6. **Compact, reusable plumbing**

   * Things like `GraphCommandTool`, `parse_graph_command`, `resolve_slug(s)`, `paginate` reduce boilerplate.
   * You prefer a small number of sharp primitives over bespoke wrappers per tool.

7. **Curated LLM surface**

   * `graph_tools::mod` whitelists tool IDs; the LLM doesn’t get direct access to every internal knob.

That’s the standard I’m using to judge the graph stack.

---

## 2. How the graph stack compares to that bar

### 2.1 Core graph & invariants (GraphService / GraphManager / analysis / traversal)

**What’s strong / matches the bar**

* **GraphService is principled**:

  * Explicit `GraphError` variants: `MissingSlug`, `Schema`, `RequiresCycle`, `InvariantViolation`.
  * `GraphManagerState` snapshot+restore keeps `graph_version` and invariants intact (tested in `graph_manager_state_round_trip_preserves_version`).
  * `set_strict_quality(true)` and `validate_global_invariants` enforce:

    * No stranded LOs.
    * No procedural practice gaps.
    * Coverage against LO rubric criteria.
    * Borrow-ahead violations.
    * Discourse orphans.
    * Extraneous-purity violations.
  * All of that is exercised in your tests, which is solid for a vibe-coded area.

* **Analysis functions are conceptually cohesive**:

  * `example_gaps`, `fadeability_issues`, `procedural_practice_gaps`, `coverage_report`, `lo_reachability`, `extraneous_report`, `borrow_ahead`, `discourse_orphans`, etc. all line up with your measurement / instructional design philosophy.
  * Tests in `analysis_unit.rs` make sure each concept has at least one precise, non-trivial scenario validating it.

* **Traversal utilities match the invariants**

  * `requires_transitive_reduction` + `requires_path_exists` are tested; you’re explicitly using them to identify redundant requires without breaking reachability.

So despite being “vibe-coded”, the **core semantics are surprisingly high quality** and strongly anchored by tests.

**Where it diverges / feels “heavier”**

Most of the divergence is not conceptual, it’s about **shape and reuse**:

* Invariants and analyses are somewhat **scattered**:

  * Some invariants are enforced as *guards* on edge creation, others via `validate_global_invariants`, others exposed via separate `analysis::*` functions, and then again via LLM tools.
  * That’s fine, but it makes it harder to see “these 3 things are the constructive-alignment policy” vs “these 2 are discourse policy”, etc.

* Some helpers are “almost generic” but not fully factored:

  * The patterns GraphService uses (e.g., borrow-ahead, purity, fadeability, coverage) are reused by tools manually, instead of via a small set of shared helper functions.

I *don’t* think you need a big redesign here; with a few local refactors, the graph side will feel as intentional as the rest.

---

### 2.2 LLM surface for the graph (`graph_tools/*`)

Overall: the tools are much closer to your desired bar than your “vibe-coded” description suggests. But there are a few rough edges / bloat spots.

#### Good / aligned:

* **Commands (`commands.rs`)**

  * All graph mutations go through `GraphCommandTool` and `parse_graph_command`.
  * Error mapping uses `map_send_err`.
  * Success payloads have a uniform structure via `command_ok`.
  * Enums (`Strength`, `SupportKind`, `IntendedEffect`, `AssessmentScope`, `AnchorImpact`) are used instead of strings.
  * Tool descriptions are rich and explain the intended semantics (Assessable Atom, fadeable supports, etc.), which matches your earlier work.

* **Inspection (`inspection.rs`)**

  * `graph_neighbors` and `graph_get_node` are classic, thin inspection tools.
  * They delegate everything to `GraphManager` and return clean JSON like `{ neighbor_slug, edge_kind, direction }` or detailed node payloads.

* **Analysis tools (`analysis.rs`)**

  * They are read-only and thin over `crate::analysis`.
  * Heavy ones (`graph_first_principles_summary`, `graph_lo_alignment_summary`, `graph_gap_summary`, `graph_assessment_gaps`) correctly use the **preview handshake**.
  * Everything is paginated where cardinality can grow.
  * Logging is consistent and structured.

* **Persist (`persist.rs`)**

  * Wraps `SaveSnapshot` and `LoadSnapshot` with clean, small tools and nice error shaping (e.g., treating missing files or snapshot version mismatches as input errors instead of internal).

* **Curated registration (`graph_tools/mod.rs`)**

  * Graph tools are aggressively curated via `CURATED_TOOL_IDS`, so the LLM sees a clean surface.

So the LLM-side really is on brand.

#### Divergences / rough spots:

These are mainly the “vibe” parts:

1. **`bon::Builder` derives everywhere but not actually used**

   * Many argument structs (`InsertKnowledgeArgs`, `InsertTeachingArgs`, `AddRequiresArgs`, `AddSupportsArgs`, etc., plus a bunch in `analysis.rs` and `inspection.rs`) derive `Builder` and even include per-field `#[builder(with = ...)]` validators…
   * …but the parsers all do `serde_json::from_value` and then call `require_string` manually, never actually using the builder type.
   * That’s **dead weight** and confusing: it violates your usual “one clear way to do things” vibe.

2. **Stringly-typed “direction” in `NeighborsArgs`**

   * `NeighborsArgs.direction: Option<String>` with manual validation against `"incoming" | "outgoing" | "both"`.
   * Everywhere else, you use enums (`SupportKind`, `IntendedEffect`, etc.). This field sticks out as “not like the others”.

3. **Repetitive preview-handshake boilerplate**

   * `FirstPrinciplesSummaryTool`, `LoAlignmentTool`, `GapSummaryTool`, and `AlignmentGapsTool` all repeat:

     * Compute `approx_bytes`.
     * Call `prepare_payload_estimates`.
     * Switch on `ToolPayloadMode`.
     * Build hints, preview, and apply preview cost.
   * The code is nearly identical, with only the hint string differing slightly.

4. **Repeated “resolve → get graph → ensure type” boilerplate**

   * Several tools (LO reach, coverage, alignment, LO views, extraneous, etc.) repeat:

     ```rust
     let lo = resolve_slug(&self.graph, self.args.lo_slug.clone(), TOOL).await?;
     let graph = self.graph.ask(GetGraph).await.map_err(map_send_err_inf)?;
     ensure_knowledge_type(&graph, lo, &self.args.lo_slug, KnowledgeType::LearningOutcome, TOOL)?;
     ```
   * That’s a prime candidate for a small helper; your non-graph code tends to factor this kind of pattern.

5. **One-off deviations from helper utilities**

   * `RedundantRequiresTool`:

     * Stores full `CallState` instead of just `ActorRef<GraphManager>`, while only using `state.graph`.
     * Does manual `limit`/`offset` logic instead of `paginate` (with good reason re: different bounds, but it still looks “off” next to other tools).

6. **Tests duplicate helpers**

   * `analysis_props.rs` and `analysis_unit.rs` both define `mk_kn`, both define inline “add_knowledge_node with conceptual type” patterns, etc.
   * Nothing wrong with that, but it’s more code to maintain than needed.

---

## 3. Concrete, local changes to bring graph code up to your bar

I’ll keep these local and incremental—things you could realistically apply piece by piece.

### 3.1 Kill or actually use `bon::Builder` on tool args

**Files:** `graph_tools/commands.rs`, `graph_tools/analysis.rs`, `graph_tools/inspection.rs`, `graph_tools/persist.rs` (for snapshot args).

Right now:

* `#[derive(Builder)]` is widespread.
* Many fields have `#[builder(with = ...)]` validation that **never runs**, because you never construct the builder.

**Option A (simplest + trim code): drop `Builder` where unused**

1. For any arg struct where you never use `FooArgsBuilder`:

   * Remove `#[derive(Builder)]`.
   * Remove `use bon::Builder;` from that module if it becomes unused.
   * Remove `#[builder(with = ...)]` attributes that are effectively dead.

2. Keep the existing `require_string(...)` calls in the `parse_*` functions as the single source of validation.

That:

* Reduces compile time.
* Reduces cognitive load for anyone reading the code.
* Aligns with your existing non-graph tools that also use “plain struct + `require_string`”.

**Option B (if you *want* the builder pattern): actually use it**

If you prefer the builder style (and use it elsewhere), create a helper like:

```rust
fn parse_args_with_builder<Args, B>(raw: Value, tool: &'static str) -> ToolInputResult<Args>
where
    Args: Clone,
    B: Default + bon::Builder<Args = Args>,
    B::Error: std::fmt::Display,
{
    let args: Args = serde_json::from_value(raw).map_err(|err| ToolInputError::InvalidPayload {
        tool,
        message: err.to_string(),
    })?;
    Ok(args)
}
```

…or simply commit to *not* having the `#[builder(with = ...)]` layer and leave validation in `parse_*`.

**Recommendation:** Option A feels more in line with “make graph stuff match the rest” and reduce bloat.

---

### 3.2 Strong-typing `direction` in `graph_neighbors`

**File:** `graph_tools/inspection.rs`

Right now:

```rust
pub struct NeighborsArgs {
    pub slug: String,
    pub edge_kind: Option<EdgeKindFilter>,
    pub direction: Option<String>, // "incoming" | "outgoing" | "both"
    ...
}
```

And you manually validate the string in `parse_neighbors`.

**Local change:**

1. Introduce a small enum for tool-facing direction:

```rust
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum NeighborDirectionArg {
    Incoming,
    Outgoing,
    Both,
}
```

2. Change `NeighborsArgs` to:

```rust
pub struct NeighborsArgs {
    pub slug: String,
    #[serde(default)]
    pub edge_kind: Option<EdgeKindFilter>,
    #[serde(default)]
    pub direction: Option<NeighborDirectionArg>,
    ...
}
```

3. Simplify the mapping in `execute`:

```rust
let direction = match self.args.direction {
    Some(NeighborDirectionArg::Incoming) => Some(NeighborDirection::Incoming),
    Some(NeighborDirectionArg::Outgoing) => Some(NeighborDirection::Outgoing),
    _ => Some(NeighborDirection::Both),
};
```

4. Delete the manual `if let Some(dir) = args.direction.as_deref() && ...` validation in `parse_neighbors`.

That:

* Removes a stringly-typed one-off.
* Matches the rest of your tool argument style (enums for discrete options).
* Is fully backwards compatible at the JSON level if you keep the same `serde(rename_all)`.

---

### 3.3 Factor the summary + preview-handshake pattern

**Files:** `graph_tools/analysis.rs` (FirstPrinciplesSummaryTool, LoAlignmentTool, GapSummaryTool, AlignmentGapsTool).

They all have this skeleton:

```rust
let approx_bytes = payload_size_bytes(&payload);
let estimates = prepare_payload_estimates(&self.metrics, self.model.as_str(), approx_bytes);
let mode = ToolPayloadMode::from_fetch_flag(self.args.fetch_body);
info!(tool = TOOL, mode = mode.as_str(), approx_bytes, "...");

match mode {
    ToolPayloadMode::Preview => {
        let hints = vec![format!(
            "Summary is ~{} bytes; set fetch_body=true to retrieve it.",
            approx_bytes
        )];
        let mut preview = build_cost_preview(
            TOOL,
            approx_bytes,
            estimates.safe_tokens,
            hints,
        );
        let preview_bytes = payload_size_bytes(&preview);
        let preview_tokens = estimate_tokens_from_characters(preview_bytes as usize);
        apply_preview_cost(
            &mut preview,
            &self.metrics,
            self.model.as_str(),
            self.conversation_id.as_str(),
            preview_tokens,
            estimates.safe_tokens,
        );
        Ok(ToolOutput::with_byte_hint(preview, preview_bytes))
    }
    ToolPayloadMode::Body => Ok(ToolOutput::with_byte_hint(payload.clone(), approx_bytes)),
}
```

**Local refactor:**

Add a helper in `graph_tools/analysis.rs` or `graph_tools/common.rs`:

```rust
fn finalize_summary_tool(
    tool: &'static str,
    payload: serde_json::Value,
    fetch_body: bool,
    metrics: &crate::llm_gateway::GatewayMetrics,
    model: &str,
    conversation_id: &str,
    hint_prefix: &str,
) -> Result<ToolOutput, ToolExecutionError> {
    let approx_bytes = payload_size_bytes(&payload);
    let estimates = prepare_payload_estimates(metrics, model, approx_bytes);
    let mode = ToolPayloadMode::from_fetch_flag(fetch_body);

    info!(
        tool = tool,
        mode = mode.as_str(),
        approx_bytes,
        "graph summary tool"
    );

    match mode {
        ToolPayloadMode::Preview => {
            let hints = vec![format!(
                "{} ~{} bytes; set fetch_body=true to retrieve it.",
                hint_prefix, approx_bytes
            )];
            let mut preview =
                build_cost_preview(tool, approx_bytes, estimates.safe_tokens, hints);
            let preview_bytes = payload_size_bytes(&preview);
            let preview_tokens = estimate_tokens_from_characters(preview_bytes as usize);
            apply_preview_cost(
                &mut preview,
                metrics,
                model,
                conversation_id,
                preview_tokens,
                estimates.safe_tokens,
            );
            Ok(ToolOutput::with_byte_hint(preview, preview_bytes))
        }
        ToolPayloadMode::Body => Ok(ToolOutput::with_byte_hint(payload, approx_bytes)),
    }
}
```

Then in each summary tool’s `execute`:

* Replace the duplicated snippet with:

```rust
finalize_summary_tool(
    FIRST_PRINCIPLES_SUMMARY,
    payload,
    self.args.fetch_body,
    &self.metrics,
    &self.model,
    &self.conversation_id,
    "Summary is",
)
```

…and similarly for LO alignment, gap summary, and alignment gaps.

Result:

* The important bits in each tool’s `execute` become “what we’re summarizing”, not “how preview works”.
* The graph side now looks like other parts of the system where you’ve already extracted small framework helpers.

---

### 3.4 Factor slug+graph+type resolution for LOs (and maybe other types)

**File:** primarily `graph_tools/analysis.rs`, plus `graph_tools/commands.rs` for similar patterns.

You repeat:

```rust
let lo = resolve_slug(&self.graph, self.args.lo_slug.clone(), TOOL).await?;
let graph = self
    .graph
    .ask(crate::graph::manager::GetGraph)
    .await
    .map_err(map_send_err_inf)?;
ensure_knowledge_type(
    &graph,
    lo,
    &self.args.lo_slug,
    KnowledgeType::LearningOutcome,
    TOOL,
)?;
```

in:

* `LoReachTool`
* `LoCoverageTool`
* `LoAlignmentTool`
* `LoAssessmentsViewTool`
* `LoMissingCriteriaViewTool`
* `LoAnchorsViewTool`

**Local helper in `graph_tools/common.rs`:**

```rust
pub(crate) async fn load_lo_with_graph(
    graph_ref: &ActorRef<crate::graph::manager::GraphManager>,
    lo_slug: &str,
    tool: &'static str,
) -> Result<(crate::graph::CurriculumGraph, crate::graph::NodeId), ToolExecutionError> {
    let lo = resolve_slug(graph_ref, lo_slug.to_string(), tool).await?;
    let graph = graph_ref
        .ask(crate::graph::manager::GetGraph)
        .await
        .map_err(map_send_err_inf)?;
    ensure_knowledge_type(
        &graph,
        lo,
        lo_slug,
        KnowledgeType::LearningOutcome,
        tool,
    )?;
    Ok((graph, lo))
}
```

Then in each tool:

```rust
let (graph, lo) = load_lo_with_graph(&self.graph, &self.args.lo_slug, LO_ALIGNMENT).await?;
// ... then do reachability/coverage/etc.
```

Benefits:

* Removes repeated code blocks and makes the LO-focused tools read like “do LO stuff” rather than “boilerplate slug resolution + type checks”.
* Matches your overall style where repeated patterns get factored (like `parse_graph_command`, `paginate`).

You can analogously add helpers later if you find yourself repeating “resolve assessment + ensure AssessmentItem” or similar patterns (e.g., in `ExtraneousTool`).

---

### 3.5 Make `RedundantRequiresTool` look like the others

**File:** `graph_tools/redundant_requires.rs`

Right now:

```rust
struct RedundantRequiresTool {
    args:  RedundantRequiresArgs,
    state: CallState,
}

let edges: Vec<(String, String)> = self
    .state
    .graph
    .ask(RedundantRequires { prune: self.args.prune })
    .await
    .map_err(|e| map_send_err(e, REDUNDANT_REQUIRES))?;
```

This is the one tool that:

* Stores full `CallState` instead of just the `ActorRef<GraphManager>`.
* Doesn’t need metrics, model, or conversation_id.

**Local change:**

1. Change struct to:

```rust
struct RedundantRequiresTool {
    args:  RedundantRequiresArgs,
    graph: ActorRef<crate::graph::manager::GraphManager>,
}
```

2. Change `parse_redundant_requires` to pass `state.graph.clone()` instead of `state.clone()`.

3. Update `execute` to use `self.graph` instead of `self.state.graph`.

It’s a tiny refactor but it makes this tool consistent with all the others in `analysis.rs` that only store what they need (an `ActorRef<GraphManager>`).

You *could* also use `paginate` if you wanted, but because this tool has different `limit` semantics (default 200, max 500, vs `paginate`’s [1,200]) I’d keep the custom limit logic and only normalise the stored state.

---

### 3.6 Avoid unnecessary `payload.clone()` in `with_byte_hint` calls

This is small but shows polish.

Many tools do:

```rust
Ok(ToolOutput::with_byte_hint(payload.clone(), payload_size_bytes(&payload)))
```

You can rewrite this pattern to avoid clones:

```rust
let approx_bytes = payload_size_bytes(&payload);
Ok(ToolOutput::with_byte_hint(payload, approx_bytes))
```

Applies to various `*_view` and analysis tools (`DAGCheckTool`, `FirstPrinciplesViewTool`, many of the LO tools, etc.).

It’s micro, but it’s the kind of micro you *did* care about in the earlier architecture.

---

### 3.7 Clean up test duplication

**Files:** `tests/analysis_props.rs`, `tests/analysis_unit.rs`

Both define:

* A `mk_kn` function with nearly identical logic.
* Patterns like:

```rust
graph::IntroductionScope::InCourse
SourceRef { path: "dummy", ... }
```

**Local change:**

1. Create `tests/common.rs` or `tests/test_helpers/mod.rs` with:

   * `fn mk_kn(title: &str, kt: KnowledgeType) -> KnowledgeNode`
   * Maybe small wrappers like `fn add_concept(svc: &mut GraphService, slug: &str) -> NodeId`.

2. `pub use` them into the test modules and delete the duplicated helper code.

No behavior change; just smaller test files and less repeated setup noise.

---

## 4. Where code has bloated / unnecessary complexity, and how to trim

Here’s the “big picture” of bloated areas and the minimal surgery to fix them:

1. **`bon::Builder` derives on arg structs**

   * **Symptom:** Extra derives and `#[builder(with = ...)]` noise that isn’t actually used during parsing.
   * **Cost:** More cognitive load and compile time for zero runtime benefit.
   * **Fix:** Remove the derives (and the imports) where unused **or** commit to using them and centralize parsing via builder. I’d strongly lean toward removing them for the graph tools.

2. **Repetitive preview-handshake code**

   * **Symptom:** Four tools with nearly identical preview logic.
   * **Cost:** Harder to change the preview policy later; more code to read; diverges from your “framework-ish” feel.
   * **Fix:** Introduce a small `finalize_summary_tool` helper and call it from those tools.

3. **Manual validation of string flags (`direction`)**

   * **Symptom:** `Graph_neighbors` alone uses string options where everything else uses enums.
   * **Cost:** Inconsistent tool schema and more manual validation code.
   * **Fix:** Replace with a small enum `NeighborDirectionArg` + simple mapping.

4. **Boilerplate slug/type resolution for LO tools**

   * **Symptom:** Six tools re-implement the same 3-step pattern.
   * **Cost:** More code, harder to update if LO-type requirements change.
   * **Fix:** Add `load_lo_with_graph` helper to `graph_tools/common.rs`.

5. **Minor state and duplication quirks**

   * `RedundantRequiresTool` storing `CallState` instead of just `ActorRef`.
   * Tests defining their own `mk_kn` helpers instead of sharing.

---

### TL;DR

* **Semantically**, the graph stack is in good shape: the invariants are solid and well-tested, and the LLM surface is curated and typed in a way that fits your architecture.
* The “vibe-coded” feel mostly comes from:

  * Extra unused abstractions (`Builder`),
  * Some copy-pasted patterns (preview, LO resolution),
  * A couple of stringly-typed or slightly-off tools.

If you apply the local changes above:

* The graph tools will look and feel almost identical in *style* to your non-graph tools.
* The total code size shrinks a bit (especially from killing unused `Builder` machinery and boilerplate).
* Future work on alignment / gap analyses becomes easier because the patterns are factored the same way you factor everything else in the system.

If you want, next step I can zoom into a single area (e.g., the LO alignment family of tools) and sketch what the refactored code would look like in detail, keeping all behavior identical but applying the patterns above.

