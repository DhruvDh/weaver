# Two-Phase Autonomous Graph Construction - Implementation Plan

**Target**: Demo-ready implementation for tomorrow
**Approach**: Minimal changes to existing FileReader actor with mode-based behavior
**Estimated Time**: 4-6 hours of focused development
**Model**: gpt-oss-120b (FileReader + delegate_tasks contexts)

---

## Executive Summary

Implement a two-phase autonomous curriculum graph construction system:

1. **Phase 1 (Harvesting)**: Multiple parallel actors extract nodes from textbook chapters
2. **Deduplication Barrier**: Clean up duplicate nodes before wiring
3. **Phase 2 (Weaving)**: Multiple parallel actors connect nodes with edges

**Why Two-Phase Wins**: Solves the "missing target" problem (can't add edge until both nodes exist), minimal implementation complexity, leverages existing DeduplicationAgent.

**Key constraints baked into this plan**:
- Use existing tools correctly: assessments/LOs are created via `graph_insert_knowledge` with `knowledge_type=assessment_item` or `learning_outcome`; TeachingSteps stay in Harvester scope; Weavers get `graph_neighbors` and other read-only tools but no node mutations.
- Tags are standardized: every node gets `source:<chapter_path>` (workspace-relative for uniqueness,
  e.g., `source:unit1/intro.ptx`); Weavers discover nodes via `graph_list_nodes_by_tag` +
  `graph_search_nodes` instead of relying on slug hints.

**Engineering choices (flexible)**:
- Step ordering, concurrency, and exact prompts are suggested defaults; adapt per chapter/model/runtime.
- Numerical targets (node/edge counts, runtimes, costs) are demo heuristics, not invariants; tune based on observed output quality and token budgets.
- Platform assumption: exclusive GPU VM; goal is to saturate available compute with high concurrency (up to ~128 in-flight requests), not to minimize token/s costs.
- Parallelism is structural: multiple focused harvesters per chapter, then focused weavers; `skip_dedup_on_insert=true` in Phase 1, followed by a dedup barrier and validation before Phase 2.
- Context capture: Phase 2 receives chapter-scoped node lists (by tags or version delta) to avoid blind edge creation.
- App wiring: orchestrator needs `dedup` handle; add `glob = "0.3"` if using globbed chapter selection.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      PHASE 1: HARVESTING                    │
├─────────────────────────────────────────────────────────────┤
│  Parallel Actors (same FileReader, different modes):       │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ FactualHarvester │  │ConceptualHarvest │  ... (4-6x)    │
│  │ Mode: Harvester  │  │ Mode: Harvester  │                │
│  │ Type: Factual    │  │ Type: Conceptual │                │
│  └──────────────────┘  └──────────────────┘                │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      ▼                                      │
│           [GraphManager accumulates nodes]                 │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DEDUPLICATION BARRIER                          │
│  DeduplicationAgent.ask(RunDeduplication {                  │
│    auto_merge_threshold: 0.95,                              │
│    dry_run: false                                           │
│  })                                                         │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      PHASE 2: WEAVING                       │
├─────────────────────────────────────────────────────────────┤
│  Parallel Actors:                                           │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  RequiresWeaver  │  │ SupportsWeaver   │  ... (3-5x)    │
│  │  Mode: Weaver    │  │  Mode: Weaver    │                │
│  │  Type: Requires  │  │  Type: Supports  │                │
│  └──────────────────┘  └──────────────────┘                │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      ▼                                      │
│           [GraphManager with complete graph]               │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Add AgentMode to FileReader (30 mins)

**File**: `src/file_reader.rs`

**Changes**:

```rust
// Add after line 25 (after imports, before SYSTEM_PROMPT_TEMPLATE)

/// Operational mode for FileReader actors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    /// Phase 1: Extract nodes only (no edge creation)
    Harvester,
    /// Phase 2: Connect nodes only (no node creation)
    Weaver,
    /// Traditional mode: unrestricted tool access (default for interactive use)
    Interactive,
}

impl Default for AgentMode {
    fn default() -> Self {
        Self::Interactive
    }
}

/// Optional specialization for harvester actors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarvesterFocus {
    Factual,
    Conceptual,
    Procedural,
    Metacognitive,
    All,
}

/// Optional specialization for weaver actors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeaverFocus {
    Requires,
    Supports,
    Assesses,
    All,
}
```

**Modify FileReader struct** (line 48):

```rust
#[derive(Actor)]
pub struct FileReader {
    gateway:            ActorRef<LLMGateway>,
    model:              Arc<String>,
    root:               Arc<PathBuf>,
    metrics:            Arc<GatewayMetrics>,
    graph:              ActorRef<crate::graph::manager::GraphManager>,
    dedup:              ActorRef<DeduplicationAgent>,
    analysis_cache:     Arc<AnalysisCache>,
    rerun:              Option<ActorRef<crate::rerun_sink::RerunSink>>,
    depth:              usize,
    max_subdelegations: usize,
    actor_name:         Arc<String>,
    conversation_id:    Arc<String>,
    mode:               AgentMode,  // <-- ADD THIS
}
```

**Update ReaderDeps** (line 64):

```rust
#[derive(Clone)]
struct ReaderDeps {
    gateway:        ActorRef<LLMGateway>,
    model:          Arc<String>,
    root:           Arc<PathBuf>,
    metrics:        Arc<GatewayMetrics>,
    graph:          ActorRef<crate::graph::manager::GraphManager>,
    dedup:          ActorRef<DeduplicationAgent>,
    analysis_cache: Arc<AnalysisCache>,
    rerun:          Option<ActorRef<crate::rerun_sink::RerunSink>>,
    mode:           AgentMode,  // <-- ADD THIS
}
```

### Step 2: Mode-Specific System Prompts (45 mins)

**Replace** `SYSTEM_PROMPT_TEMPLATE` constant (line 27) with a function:

```rust
// Remove the const SYSTEM_PROMPT_TEMPLATE at line 27-34

fn system_prompt_for_mode(mode: AgentMode) -> &'static str {
    match mode {
        AgentMode::Harvester => HARVESTER_SYSTEM_PROMPT,
        AgentMode::Weaver => WEAVER_SYSTEM_PROMPT,
        AgentMode::Interactive => INTERACTIVE_SYSTEM_PROMPT,
    }
}

const HARVESTER_SYSTEM_PROMPT: &'static str = r#"You are Weaver's Knowledge Harvester, assigned to extract a COMPLETE, EXHAUSTIVE set of nodes from the UNCC CS2 PreTeXt project.

YOUR MISSION: Extract EVERY node from the assigned chapter. Do NOT create edges yet. COMPLETENESS IS CRITICAL.

EXTRACTION REQUIREMENTS:
You must extract 100% of the curriculum content:

1. **Knowledge Nodes** (use `graph_insert_knowledge`):
   - EVERY concept mentioned (conceptual)
   - EVERY definition, syntax rule, fact (factual)
   - EVERY algorithm, procedure, how-to (procedural)
   - EVERY metacognitive strategy (when/why to use X)
   - Atomic granularity: one concept per node

2. **Assessment Items** (use `graph_insert_knowledge` with `knowledge_type="assessment_item"`):
   - EVERY exercise, practice problem, checkpoint
   - EVERY worked example (these are assessment items too)
   - Include the full question text and any solution sketches

3. **Learning Outcomes** (use `graph_insert_knowledge` with `knowledge_type="learning_outcome"`):
   - EVERY stated learning objective
   - Decompose broad LOs into atomic rubric criteria
   - Each criterion should be measurable

4. **TeachingSteps** (use `graph_insert_teaching_step`):
   - EVERY section, subsection, paragraph with pedagogical intent
   - Mark method (exposition, example, exercise, definition)
   - Capture the narrative flow completely

QUALITY STANDARDS:
- **Slugs**: Normalized `{kind}.{snake_case_name}` (e.g., c.loop_invariant, p.insertion_sort, lo.write_test_cases)
- **Statements**: Complete, standalone text (someone reading just the node should understand it)
- **Source refs**: Exact file:line or section IDs
- **Tags for context**: Always add `source:<chapter_path>` for this chapter (workspace-relative,
  e.g., `source:unit1/intro.ptx`) and dependency hints as `req:slug`, `sup:slug`, or `ref:slug`
  (no other prefixes)
- **Coverage**: If you finish and have <50 knowledge nodes from a typical chapter, you missed content. Go back and extract more.

FORBIDDEN TOOLS (Phase 2 only):
- graph_add_requires / supports / assesses / precedes / anchors

VERIFICATION:
Before finishing, ask yourself:
- Did I extract every bold term, code example, exercise?
- Did I capture the discourse structure (intro → concept → example → practice)?
- Are my node statements clear enough for Phase 2 weavers to connect them?

Work methodically. Use `delegate_tasks` to parallelize sections if the chapter is large. Completeness over speed."#;

const WEAVER_SYSTEM_PROMPT: &'static str = r#"You are Weaver's Curriculum Architect, assigned to create a COMPLETE edge network for the UNCC CS2 PreTeXt project graph.

YOUR MISSION: Connect EVERY node with ALL appropriate edges. Do NOT create new nodes. COMPLETENESS IS CRITICAL.

EDGE CREATION REQUIREMENTS:
You must create 100% of the curriculum relationships:

1. **Requires Edges** (use `graph_add_requires`):
   - EVERY knowledge prerequisite relationship
   - Mark strength: necessary (hard requirement) vs helpful (soft prerequisite)
   - MUST maintain DAG property - use `graph_dag_check` if uncertain
   - Rationale must cite source text location
   - Expect 2-5 requires edges per knowledge node on average

2. **Supports Edges** (use `graph_add_supports`):
   - EVERY worked example → concept it demonstrates
   - EVERY analogy, counterexample, misconception fix
   - Mark support_kind: worked_example, analogy, counterexample, misconception_fix, strategy_hint
   - Mark case_tag: typical, edge_case, error_case
   - Include coverage_tags for constraint labels
   - Expect 1-3 support edges per conceptual/procedural node

3. **Assesses Edges** (use `graph_add_assesses`):
   - EVERY assessment item → LO it measures
   - Mark scope: target (full coverage) or enabling (partial)
   - List observation_features (what evidence you'd see)
   - Every LO should have ≥2 target assessments

4. **Discourse Edges** (use `graph_add_precedes`, `graph_add_anchors`):
   - EVERY TeachingStep sequence (narrative flow)
   - EVERY anchor from TeachingStep to Knowledge/LO/Assessment
   - Mark impact: introduce, use, refine, motivate, target
   - Reconstruct the pedagogical narrative completely

STRATEGY:
1. Start with `graph_first_principles` to find entry-point nodes
2. Use `graph_neighbors` to explore the local neighborhood
3. Read node `tags` for `req:`, `sup:`, or `ref:` hints from harvesters (and the
   `source:<chapter_path>` tag to scope work)
4. Use `source_refs` to read ONLY relevant passages (don't re-read full chapter)
5. Work systematically: requires → supports → assesses → discourse
6. Use `graph_lo_alignment_summary` and `graph_gap_summary` to find missing edges

FORBIDDEN TOOLS (Phase 1 only):
- graph_insert_* / graph_update_* / graph_rename_node / graph_remove_node

QUALITY CHECKS:
- **DAG property**: All requires edges must preserve acyclicity
- **Coverage**: Every node should have ≥1 edge (incoming or outgoing)
- **LO reachability**: Every LO must be reachable from first-principles via requires edges
- **Assessment coverage**: Every LO needs ≥2 target assessments

VERIFICATION:
Before finishing, run:
- `graph_gap_summary` - should show minimal gaps
- `graph_lo_alignment_summary` - all LOs should be reachable and covered
- `graph_dag_check` - should confirm DAG validity

Work methodically. If a chapter produced 80 nodes, you should produce 200-400 edges. Completeness over speed."#;

const INTERACTIVE_SYSTEM_PROMPT: &'static str = r#"You are Weaver's file-reading assistant assigned to explore the UNCC CS2 PreTeXt project.
Always stay within the provided workspace root and rely on the available tools to inspect files.
Call `list_directory` whenever you need to confirm the current structure instead of inferring it from file names.
Never assume content from file names alone—use `read_file_full` or `read_file_range` to inspect source material before describing or citing it.
When tasks can be partitioned, prefer launching delegated tasks in parallel. The `delegate_tasks` tool accepts a `tasks` array; wrap a single instruction in an array when needed. The runtime executes up to 8 tasks concurrently.
Large-result tools return a preview header first. Examine the reported size and token estimates, refine your arguments (e.g., smaller line ranges or narrower regex scopes), or delegate a summarisation task before opting into full payloads. Only set `fetch_body` to true when you are confident the resulting content fits within the conversation budget.
When creating knowledge nodes, generate slugs as `{kind}.{snake_case_name}` with lowercase prefixes: f, c, p, m, lo, a. Normalize names with underscores (e.g., `C.LoopBasics` -> `c.loop_basics`, `LO.Write Docstring` -> `lo.write_docstring`). Prefer unambiguous, descriptive names.
Only answer after gathering the necessary context via tool calls, and reference the specific files you actually examined."#;
```

**Update** `system_prompt()` method (line 155):

```rust
fn system_prompt(&self) -> String {
    let base = system_prompt_for_mode(self.mode);
    format!("{base}\n\nWorkspace root: {root}", root = self.root.display())
}
```

### Step 3: Tool Filtering by Mode (30 mins)

**Add new function** after `tool_identifiers()` (after line 165):

```rust
pub fn tool_identifiers_for_mode(mode: AgentMode) -> Result<Vec<&'static str>> {
    let all_tools = llm::all_tools()?;

    match mode {
        AgentMode::Interactive => {
            // All tools available
            Ok(all_tools.iter().map(|meta| meta.id).collect())
        }
        AgentMode::Harvester => {
            // Explicit allowlist: node creation/update + read-only helpers (no edges, no destructive ops)
            let allowed = [
                "graph_insert_knowledge",
                "graph_update_knowledge",
                "graph_insert_teaching_step",
                "graph_update_teaching_step",
                "graph_list_nodes_by_tag",
                "graph_list_nodes_by_kind",
                "graph_list_tags",
                "graph_get_node",
                "graph_neighbors",
                "graph_first_principles",
                "list_directory",
                "read_file_full",
                "read_file_range",
                "search_text",
            ];
            Ok(all_tools
                .iter()
                .filter(|meta| allowed.contains(&meta.id))
                .map(|meta| meta.id)
                .collect())
        }
        AgentMode::Weaver => {
            // Explicit allowlist: edge creation + inspection (no node creation/update/destructive ops)
            let allowed = [
                "graph_add_requires",
                "graph_add_supports",
                "graph_add_assesses",
                "graph_add_precedes",
                "graph_add_anchors",
                "graph_get_node",
                "graph_list_nodes_by_tag",
                "graph_list_nodes_by_kind",
                "graph_list_tags",
                "graph_search_nodes",
                "graph_neighbors",
                "graph_first_principles",
                "graph_dag_check",
                "graph_lo_alignment_summary",
                "graph_gap_summary",
                "list_directory",
                "read_file_full",
                "read_file_range",
                "search_text",
            ];
            Ok(all_tools
                .iter()
                .filter(|meta| allowed.contains(&meta.id))
                .map(|meta| meta.id)
                .collect())
        }
    }
}
```

### Step 3.5: Discovery Tools for Weavers (30 mins)

**Problem**: Weavers lack a way to enumerate orphan nodes or resolve fuzzy names; tags like `req:slug` can go stale after dedup.

**Add two read-only tools (graph_tools)**:
- `graph_list_nodes_by_tag(tag: String, limit: Option<usize>)` — returns slugs + titles for nodes matching a tag (use `source:<chapter_path>` to scope the working set).
- `graph_search_nodes(query: String, limit: usize)` — fuzzy search over titles/statements (substring/Jaro-Winkler) to resolve edge endpoints from text.
- (Optional) `graph_list_tags()` — enumerate available tags when chapters are numerous.
- (Optional) `graph_list_nodes_by_kind(kind: KnowledgeType, limit: Option<usize>)` — quick filtering when inventories get large.
  - Implementation note: start with case-insensitive substring on slug/title/statement; add Jaro-Winkler fallback. Keep it lightweight—no semantic/embedding search needed for demo.

**Prompt change for Weavers**:
1) Call `graph_list_nodes_by_tag(source:<chapter_path>)` to get inventory.
2) For each node, use its `source_refs` to read spans, derive relationships.
3) Resolve targets via `graph_search_nodes` (not slug tags), then create edges.

**Update allowlists**:
- Weaver mode: include `graph_list_nodes_by_tag`, `graph_search_nodes`.
- Harvester mode: optionally include `graph_list_nodes_by_tag` for self-checks; keep mutation-only tools as before.

### Step 4: Update Constructors (20 mins)

**Modify** `from_env` (line 78):

```rust
pub fn from_env(
    root: impl AsRef<Path>,
    gateway: ActorRef<LLMGateway>,
    metrics: Arc<GatewayMetrics>,
    graph: ActorRef<crate::graph::manager::GraphManager>,
    dedup: ActorRef<DeduplicationAgent>,
    rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
) -> Result<Self> {
    Self::from_env_with_mode(
        root,
        gateway,
        metrics,
        graph,
        dedup,
        rerun,
        AgentMode::Interactive,  // Default to interactive for backward compatibility
    )
}

/// Build a new FileReader with a specific mode (Harvester/Weaver/Interactive)
pub fn from_env_with_mode(
    root: impl AsRef<Path>,
    gateway: ActorRef<LLMGateway>,
    metrics: Arc<GatewayMetrics>,
    graph: ActorRef<crate::graph::manager::GraphManager>,
    dedup: ActorRef<DeduplicationAgent>,
    rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
    mode: AgentMode,
) -> Result<Self> {
    Self::from_env_with_limit_and_mode(
        root,
        gateway,
        metrics,
        graph,
        dedup,
        rerun,
        DEFAULT_MAX_SUBDELEGATIONS,
        mode,
    )
}
```

**Modify** `from_env_with_limit` (line 98):

```rust
pub fn from_env_with_limit(
    root: impl AsRef<Path>,
    gateway: ActorRef<LLMGateway>,
    metrics: Arc<GatewayMetrics>,
    graph: ActorRef<crate::graph::manager::GraphManager>,
    dedup: ActorRef<DeduplicationAgent>,
    rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
    max_subdelegations: usize,
) -> Result<Self> {
    Self::from_env_with_limit_and_mode(
        root,
        gateway,
        metrics,
        graph,
        dedup,
        rerun,
        max_subdelegations,
        AgentMode::Interactive,
    )
}

/// Build with custom delegation limit AND mode
pub fn from_env_with_limit_and_mode(
    root: impl AsRef<Path>,
    gateway: ActorRef<LLMGateway>,
    metrics: Arc<GatewayMetrics>,
    graph: ActorRef<crate::graph::manager::GraphManager>,
    dedup: ActorRef<DeduplicationAgent>,
    rerun: Option<ActorRef<crate::rerun_sink::RerunSink>>,
    max_subdelegations: usize,
    mode: AgentMode,
) -> Result<Self> {
    let model = Arc::new(
        env::var("OPENAI_MODEL")
            .map_err(|_| anyhow!("OPENAI_MODEL environment variable must be set"))?,
    );

    let root =
        Arc::new(root.as_ref().canonicalize().with_context(|| {
            format!("failed to canonicalize root {}", root.as_ref().display())
        })?);

    let analysis_cache = Arc::new(AnalysisCache::new());

    let deps = ReaderDeps {
        gateway,
        model,
        root,
        metrics,
        graph,
        dedup,
        analysis_cache,
        rerun,
        mode,  // <-- Pass mode through
    };
    Ok(Self::new(deps, 0, max_subdelegations))
}
```

**Update** `new()` (line 132):

```rust
fn new(deps: ReaderDeps, depth: usize, max_subdelegations: usize) -> Self {
    let mode_suffix = match deps.mode {
        AgentMode::Harvester => "/Harvester",
        AgentMode::Weaver => "/Weaver",
        AgentMode::Interactive => "",
    };
    let actor_name = if depth == 0 {
        format!("FileReader/Root{}", mode_suffix)
    } else {
        format!("FileReader/Delegate{}{}", depth, mode_suffix)
    };
    let conversation_id = make_conversation_id(&actor_name);
    Self {
        gateway: deps.gateway,
        model: deps.model,
        root: deps.root,
        metrics: deps.metrics,
        graph: deps.graph,
        dedup: deps.dedup,
        analysis_cache: deps.analysis_cache,
        rerun: deps.rerun,
        depth,
        max_subdelegations,
        actor_name: Arc::new(actor_name),
        conversation_id: Arc::new(conversation_id),
        mode: deps.mode,  // <-- Store mode
    }
}
```

### Step 5: Update Message Handler to Use Filtered Tools (15 mins)

**Modify** `FileReaderQuery` handler (line 253):

```rust
async fn handle(
    &mut self,
    FileReaderQuery { prompt }: FileReaderQuery,
    ctx: &mut Context<Self, Self::Reply>,
) -> Self::Reply {
    let tool_host = ctx.actor_ref().clone();
    let gateway = self.gateway.clone();
    let system_prompt = self.system_prompt();
    let model = self.model.clone();
    let actor_name = (*self.actor_name).clone();
    let conversation_id = (*self.conversation_id).clone();
    let rerun = self.rerun.clone();
    let mode = self.mode;  // <-- Capture mode

    ctx.spawn(async move {
        let system_msg: ChatCompletionRequestMessage =
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system_prompt)
                .build()?
                .into();
        let user_msg: ChatCompletionRequestMessage =
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()?
                .into();
        let request = ChatCompletionRequest {
            model,
            messages: vec![system_msg, user_msg],
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            tool_ids: Self::tool_identifiers_for_mode(mode)?,  // <-- Use mode-filtered tools
            max_iterations: MAX_TOOL_ITERATIONS,
            tool_host,
            actor_name,
            conversation_id,
            rerun,
        };

        let reply = gateway.ask(request).await?;
        Ok(reply)
    })
}
```

### Step 6: Create Orchestration Function in app.rs (60 mins)

**Add** to `src/app.rs` after the `run_app` function (after line 949):

```rust
/// Two-phase autonomous graph construction orchestrator
pub async fn run_two_phase_construction(
    cli: Cli,
    chapters: Vec<PathBuf>,
) -> Result<()> {
    use crate::file_reader::{AgentMode, FileReader, FileReaderQuery};
    use futures::future::join_all;

    // Initialize the app with skip_demo flag
    let mut modified_cli = cli.clone();
    modified_cli.skip_demo = true;
    modified_cli.skip_dedup_on_insert = true; // run Phase 1 fast; dedup at barrier

    let runtime_opts = RuntimeOptions {
        gateway_mode: GatewayMode::Real,
        min_autosave_secs: 5,
        trigger_initial_save: true,
        trigger_shutdown_save: true,
        on_started: Some(Arc::new(move |handles| {
            Box::pin(async move {
                let graph = handles.graph;
                let gateway = handles.gateway;
                let rerun = handles.rerun;
                let dedup_agent = handles.dedup;
                let metrics = gateway
                    .ask(GetGatewayMetrics)
                    .await
                    .unwrap_or_else(|_| Arc::new(GatewayMetrics::default()));

                info!("Starting Phase 1: Knowledge Harvesting");

                // ===== PHASE 1: HARVESTING =====
                // (Optional) warm-up ping here to pre-load LLM

                let harvester_tasks: Vec<_> = chapters
                    .iter()
                    .map(|chapter_path| {
                        let chapter = chapter_path.clone();
                        let graph_ref = graph.clone();
                        let gateway_ref = gateway.clone();
                        let metrics_ref = Arc::clone(&metrics);
                        let dedup_ref = dedup_agent.clone();
                        let rerun_ref = rerun.clone();

                        async move {
                            // For structural parallelism, duplicate this block per focus (e.g., Knowledge, Assessments, LOs, Discourse)
                            let actor = FileReader::from_env_with_mode(
                                modified_cli.workspace.clone(),
                                gateway_ref,
                                metrics_ref,
                                graph_ref.clone(),
                                dedup_ref,
                                rerun_ref,
                                AgentMode::Harvester,
                            )?;

                            let reader = FileReader::spawn(actor);

                            let prompt = format!(
                                "COMPLETE EXTRACTION from {}\n\n\
                                Your goal: Extract EVERY node from this chapter - aim for 80-150 total nodes.\n\n\
                                PHASE 1: KNOWLEDGE (50-100 nodes expected)\n\
                                - Extract EVERY concept (one per node, atomic granularity)\n\
                                - Extract EVERY definition, fact, syntax rule\n\
                                - Extract EVERY algorithm, procedure, how-to\n\
                                - Extract EVERY metacognitive strategy\n\
                                - Add dependency tags using req:/sup:/ref: prefixes and a chapter tag `source:{}`\n\n\
                                PHASE 2: ASSESSMENTS (10-30 nodes expected)\n\
                                - Extract EVERY exercise, practice problem (use knowledge_type=assessment_item)\n\
                                - Extract EVERY worked example as an assessment item\n\
                                - Include full question text\n\n\
                                PHASE 3: LEARNING OUTCOMES (5-15 nodes expected)\n\
                                - Extract EVERY stated learning objective (use knowledge_type=learning_outcome)\n\
                                - Decompose into atomic rubric criteria\n\n\
                                PHASE 4: DISCOURSE (15-30 nodes expected)\n\
                                - Extract EVERY section/subsection as a TeachingStep\n\
                                - Mark method: exposition/example/exercise/definition\n\
                                - Capture complete narrative flow\n\n\
                                Work methodically. Use delegate_tasks to parallelize sections. \
                                Before finishing, verify you have ≥80 total nodes. If not, you missed content.",
                                chapter.display(),
                                chapter.display()
                            );

                            reader.ask(FileReaderQuery { prompt }).await
                        }
                    })
                    .collect();

                // If rate limits bite, wrap join_all with buffer_unordered
                let phase1_results = join_all(harvester_tasks).await;
                info!(chapters_processed = phase1_results.len(), "Phase 1 complete");

                // ===== DEDUPLICATION BARRIER =====
                info!("Running deduplication...");
                let dedup_report = dedup_agent
                    .ask(RunDeduplication {
                        auto_merge_threshold: modified_cli.dedup_auto_merge_threshold,
                        dry_run: false,
                    })
                    .await?;
                info!(
                    clusters = dedup_report.clusters_analyzed,
                    auto_merged = dedup_report.auto_merged.len(),
                    pending_review = dedup_report.pending_review.len(),
                    "Deduplication complete"
                );

                info!("Running post-dedup validation...");
                graph.ask(AuditInvariants).await?;

                // ===== PHASE 2: WEAVING =====
                info!("Starting Phase 2: Edge Weaving");

                // Gather chapter-scoped node lists (via `source:<chapter_path>` tags or graph_version
                // delta) and inject into prompts

                let weaver_tasks: Vec<_> = chapters
                    .iter()
                    .map(|chapter_path| {
                        let chapter = chapter_path.clone();
                        let graph_ref = graph.clone();
                        let gateway_ref = gateway.clone();
                        let metrics_ref = Arc::clone(&metrics);
                        let rerun_ref = rerun.clone();

                        async move {
                            // For structural parallelism, duplicate per focus (e.g., Requires/Supports vs Discourse/Anchors)
                            let actor = FileReader::from_env_with_mode(
                                modified_cli.workspace.clone(),
                                gateway_ref,
                                metrics_ref,
                                graph_ref.clone(),
                                rerun_ref,
                                AgentMode::Weaver,
                            )?;

                            let reader = FileReader::spawn(actor);

                            let prompt = format!(
                                "COMPLETE EDGE NETWORK for {}\n\n\
                                Your goal: Connect EVERY node - aim for 200-400 total edges.\n\n\
                                STRATEGY:\n\
                                1. Start with graph_first_principles to find entry nodes\n\
                                2. Use graph_neighbors to explore local neighborhoods\n\
                                3. Read node tags for req:/sup:/ref: hints (and filter to this chapter via source:{} tag)\n\
                                4. Use source_refs to verify relationships (don't re-read full chapter)\n\n\
                                PHASE 1: REQUIRES (100-200 edges expected)\n\
                                - Create EVERY prerequisite relationship\n\
                                - Mark strength: necessary vs helpful\n\
                                - Verify DAG property with graph_dag_check\n\
                                - Expect 2-5 requires per knowledge node on average\n\n\
                                PHASE 2: SUPPORTS (50-150 edges expected)\n\
                                - Link EVERY worked example to concepts it demonstrates\n\
                                - Link EVERY analogy, counterexample, misconception fix\n\
                                - Mark support_kind and case_tag appropriately\n\
                                - Expect 1-3 supports per conceptual/procedural node\n\n\
                                PHASE 3: ASSESSES (20-50 edges expected)\n\
                                - Link EVERY assessment item to LOs it measures\n\
                                - Mark scope: target (full) or enabling (partial)\n\
                                - Every LO needs ≥2 target assessments\n\n\
                                PHASE 4: DISCOURSE (30-100 edges expected)\n\
                                - Create precedes edges for narrative flow\n\
                                - Create anchors from TeachingSteps to Knowledge/LO/Assessment\n\
                                - Mark impact: introduce/use/refine/motivate/target\n\n\
                                VERIFICATION before finishing:\n\
                                - Run graph_gap_summary (should show minimal gaps)\n\
                                - Run graph_lo_alignment_summary (all LOs reachable and covered)\n\
                                - Run graph_dag_check (confirm DAG validity)\n\
                                - Verify edge count ≥200 (if <200 with 80 nodes, you missed edges)",
                                chapter.display(),
                                chapter.display()
                            );

                            reader.ask(FileReaderQuery { prompt }).await
                        }
                    })
                    .collect();

                let phase2_results = try_join_all(weaver_tasks).await?;
                info!(chapters_processed = phase2_results.len(), "Phase 2 complete");

                // ===== FINAL VALIDATION =====
                info!("Running final validation...");
                graph_ref.ask(AuditInvariants).await?;

                info!("Two-phase construction complete!");
                Ok(())
            })
        })),
    };

    run_app(modified_cli, runtime_opts).await
}
```

**Note**: The above is a sketch. You'll need to:
1. Pass `dedup_agent` through `AppHandles`
2. Handle the chapter list properly
3. Add proper error handling

### Step 7: Add CLI Subcommand (30 mins)

**Modify** `src/app.rs` CLI parser (line 428):

```rust
// Add new CLI options for two-phase mode
let two_phase_mode = long("two-phase")
    .help("Enable two-phase autonomous construction mode")
    .switch();

let chapters_pattern = long("chapters")
    .help("Glob pattern for chapter files (e.g., 'source/sec-*.ptx')")
    .argument::<String>("pattern")
    .optional();
```

**Update** `Cli` struct (line 352):

```rust
#[derive(Clone, Debug)]
pub struct Cli {
    pub rerun_mode:                  RerunMode,
    pub rerun_file:                  PathBuf,
    pub workspace:                   PathBuf,
    pub graph_snapshot_path:         PathBuf,
    pub graph_autosave_secs:         u64,
    pub graph_course_commit:         Option<String>,
    pub graph_strict_quality:        bool,
    pub graph_prune_requires_s:      Option<u64>,
    pub skip_demo:                   bool,
    pub graph_validation_timeout_ms: u64,
    pub dedup_interval_secs:         u64,
    pub dedup_auto_merge_threshold:  f64,
    pub skip_dedup_on_insert:        bool,
    pub two_phase_mode:              bool,           // <-- ADD
    pub chapters_pattern:            Option<String>, // <-- ADD
}
```

### Step 8: Update main.rs Entry Point (15 mins)

**File**: `src/main.rs` (create if doesn't exist, or find existing)

```rust
use anyhow::Result;
use weaver::app::{cli, run_app, run_two_phase_construction, RuntimeOptions};
use bpaf::OptionParser;
use std::path::PathBuf;
use glob::glob;

#[tokio::main]
async fn main() -> Result<()> {
    let cli_args = cli().run();

    if cli_args.two_phase_mode {
        // Two-phase autonomous mode
        let pattern = cli_args.chapters_pattern.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--chapters pattern required in --two-phase mode"))?;

        let chapter_paths: Vec<PathBuf> = glob(pattern)?
            .filter_map(Result::ok)
            .collect();

        if chapter_paths.is_empty() {
            anyhow::bail!("No chapters found matching pattern: {}", pattern);
        }

        println!("Two-phase mode: processing {} chapters", chapter_paths.len());
        run_two_phase_construction(cli_args, chapter_paths).await
    } else {
        // Traditional interactive mode
        run_app(cli_args, RuntimeOptions::default()).await
    }
}
```

---

## Testing Strategy

### Pre-Demo Testing (1 hour)

**Test 1: Single Chapter Harvesting**
```bash
cargo build --release

# Test Phase 1 only with one chapter
OPENAI_MODEL=gpt-oss-120b \
OPENAI_API_KEY=<key> \
cargo run --release -- \
  --two-phase \
  --chapters "uncc_cs2-pretext-project/source/sec-intro-basics.ptx" \
  --skip-dedup-on-insert \
  --dedup-interval-secs 0
```

**Expected**:
- Harvester actors extract nodes from chapter
- No edges created yet
- Dedup runs after Phase 1
- Terminal shows "Phase 1 complete" message

**Test 2: Two Chapters End-to-End**
```bash
OPENAI_MODEL=gpt-oss-120b \
cargo run --release -- \
  --two-phase \
  --chapters "uncc_cs2-pretext-project/source/sec-intro-*.ptx" \
  --dedup-auto-merge-threshold 0.95
```

**Expected**:
- Phase 1 creates nodes from both chapters
- Dedup merges duplicates
- Phase 2 creates edges between chapters
- Final validation passes
- Graph snapshot saved

**Test 3: Validation**
```bash
# Inspect the graph
cat graph_snapshot.json | jq '.graph.nodes | length'  # Node count
cat graph_snapshot.json | jq '.graph.edges | length'  # Edge count
cat graph_snapshot.json | jq '.graph.edges[] | select(.kind.Requires) | length'  # Requires edges
```

### Demo Day Testing (30 mins before demo)

**Demo Script** (1 chapter, FULL extraction, ~60 min baseline runtime; expect lower with high concurrency on GPU host):
```bash
#!/bin/bash
set -e

echo "=== Weaver Two-Phase Autonomous Construction Demo ==="
echo "=== Target: COMPLETE curriculum graph from Chapter 1 ==="
echo ""

# Clean start
rm -f graph_snapshot.json
rm -rf graph_snapshot.state

# Identify shortest substantive chapter for demo
DEMO_CHAPTER="uncc_cs2-pretext-project/source/sec-intro-basics.ptx"

# Run full two-phase extraction
OPENAI_MODEL=gpt-oss-120b \
OPENAI_API_KEY=$OPENAI_API_KEY \
cargo run --release -- \
  uncc_cs2-pretext-project \
  --two-phase \
  --chapters "$DEMO_CHAPTER" \
  --dedup-auto-merge-threshold 0.95 \
  --graph-strict-quality \
  --rerun-mode both \
  --rerun-file demo.rrd \
  --graph-validation-timeout-ms 5000

echo ""
echo "=== Construction Complete ==="
echo "=== Validating Completeness ==="

# Check node counts
NODES=$(cat graph_snapshot.json | jq '.graph.nodes | length')
KNOWLEDGE=$(cat graph_snapshot.json | jq '[.graph.nodes[] | select(.kind.Knowledge)] | length')
ASSESSMENTS=$(cat graph_snapshot.json | jq '[.graph.nodes[] | select(.kind.AssessmentItem)] | length')
LOS=$(cat graph_snapshot.json | jq '[.graph.nodes[] | select(.kind.LearningOutcome)] | length')
TEACHING=$(cat graph_snapshot.json | jq '[.graph.nodes[] | select(.kind.TeachingStep)] | length')

# Check edge counts
EDGES=$(cat graph_snapshot.json | jq '.graph.edges | length')
REQUIRES=$(cat graph_snapshot.json | jq '[.graph.edges[] | select(.kind.Requires)] | length')
SUPPORTS=$(cat graph_snapshot.json | jq '[.graph.edges[] | select(.kind.Supports)] | length')
ASSESSES=$(cat graph_snapshot.json | jq '[.graph.edges[] | select(.kind.Assesses)] | length')
PRECEDES=$(cat graph_snapshot.json | jq '[.graph.edges[] | select(.kind.Precedes)] | length')
ANCHORS=$(cat graph_snapshot.json | jq '[.graph.edges[] | select(.kind.Anchors)] | length')

cat <<EOF
=== GRAPH STATISTICS ===
Nodes Total:        $NODES
  - Knowledge:      $KNOWLEDGE
  - Assessments:    $ASSESSMENTS
  - Learning Outcomes: $LOS
  - TeachingSteps:  $TEACHING

Edges Total:        $EDGES
  - Requires:       $REQUIRES
  - Supports:       $SUPPORTS
  - Assesses:       $ASSESSES
  - Precedes:       $PRECEDES
  - Anchors:        $ANCHORS

=== COMPLETENESS CHECKS ===
EOF

# Completeness heuristics
if [ "$KNOWLEDGE" -lt 30 ]; then
  echo "⚠️  WARNING: Only $KNOWLEDGE knowledge nodes - expected 50-100 for a full chapter"
else
  echo "✅ Knowledge node count looks complete ($KNOWLEDGE nodes)"
fi

if [ "$EDGES" -lt 100 ]; then
  echo "⚠️  WARNING: Only $EDGES edges - expected 200-500 for a complete graph"
else
  echo "✅ Edge count looks complete ($EDGES edges)"
fi

EDGE_NODE_RATIO=$(echo "scale=2; $EDGES / $NODES" | bc)
echo "Edge/Node Ratio: $EDGE_NODE_RATIO (expect 2.5-4.0 for complete graph)"

echo ""
echo "=== VALIDATION ==="
echo "Open demo.rrd in Rerun viewer to inspect:"
echo "  - Timeline of node/edge creation"
echo "  - Deduplication merge operations"
echo "  - Validation audit results"
echo ""
echo "Next steps:"
echo "  1. Manually inspect 5-10 random nodes for quality"
echo "  2. Check DAG visualization for sensible prerequisite flow"
echo "  3. Verify assessment-LO linkage makes pedagogical sense"
```

---

## Implementation Checklist

### Core Changes
- [ ] Add `AgentMode` enum and focus types to `file_reader.rs`
- [ ] Add mode-specific system prompts (Harvester/Weaver/Interactive)
- [ ] Implement `tool_identifiers_for_mode()` with filtering logic
- [ ] Update FileReader struct to include `mode` field
- [ ] Update ReaderDeps to include `mode`
- [ ] Add `from_env_with_mode()` constructor
- [ ] Modify `new()` to include mode in actor_name
- [ ] Update FileReaderQuery handler to use filtered tools
- [ ] Update delegate batch context to propagate mode

### Orchestration
- [ ] Add `dedup_agent` to AppHandles struct
- [ ] Create `run_two_phase_construction()` function in app.rs
- [ ] Add CLI flags: `--two-phase`, `--chapters`
- [ ] Update Cli struct with new fields
- [ ] Update CLI parser in `cli()` function
- [ ] Modify main.rs to branch on `two_phase_mode`

### Testing
- [ ] Test single-chapter harvesting
- [ ] Test two-chapter end-to-end
- [ ] Validate graph structure (nodes/edges counts)
- [ ] Test dedup merge behavior
- [ ] Verify tool restrictions (Harvester can't add edges, Weaver can't insert nodes)
- [ ] Check Rerun timeline visualization

### Demo Prep
- [ ] Write demo script (see Demo Script section)
- [ ] Identify shortest substantive chapter (check line counts in source/)
- [ ] Test with gpt-oss-120b (completeness over cost)
- [ ] Refine prompts based on initial test (boost coverage if needed)
- [ ] Prepare Rerun visualization walkthrough
- [ ] Create presentation showing before/after (empty graph → full graph)

---

## Troubleshooting Guide

### Issue: "Tool not found" errors
**Cause**: Tool filtering not applied correctly
**Fix**: Check `tool_identifiers_for_mode()` logic, ensure mode propagates through spawned actors

### Issue: Harvester creating edges
**Cause**: Tool whitelist allows edge tools
**Fix**: Verify `filter()` logic excludes `graph_add_*` edge tools in Harvester mode

### Issue: Weaver creating nodes
**Cause**: Tool whitelist allows insert tools
**Fix**: Ensure `graph_insert_knowledge` explicitly excluded in Weaver mode

### Issue: "Missing target node" in Phase 2
**Cause**: Dedup removed a node that Phase 2 tried to reference
**Fix**: Check dedup logs, verify `auto_merge_threshold` isn't too aggressive, consider using node slugs from fresh `graph_neighbors` call

### Issue: Slow execution
**Cause**: Sequential processing, large chapters
**Fix**: Confirm actors run in parallel (check `try_join_all`), raise concurrency to saturate GPU-hosted model (target up to ~128 in-flight), or use smaller chapters for demo

---

## Performance Estimates (Demo Day)

**Demo goal**: Produce a full curriculum graph for a chapter; adjust scope if quality/cost tradeoffs demand it. Assumes the baseline timings below come from moderate concurrency; with an exclusive GPU VM target high parallelism (e.g., ~128 in-flight) to saturate the model and cut wall-clock.

### Single Chapter (Full Extraction)

| Phase | Scope | Expected Time (baseline, moderate concurrency) | Token Usage (gpt-oss-120b) | Notes |
|-------|-------|----------------------------------------------|----------------------------|-------|
| Phase 1A: Knowledge Nodes | All concepts/facts/procedures | 10-15 min | 100-150K tokens | Wall-clock drops with higher parallelism; focus on saturation, not TPS |
| Phase 1B: Assessments | All exercises/questions | 5-8 min | 50-80K tokens | " |
| Phase 1C: Learning Outcomes | All LOs + rubric criteria | 3-5 min | 30-50K tokens | " |
| Phase 1D: TeachingSteps | Complete discourse layer | 8-12 min | 80-120K tokens | " |
| Dedup Barrier | Full graph scan | 15-30s | 0 tokens | " |
| Phase 2A: Requires Edges | All prerequisites | 8-12 min | 80-120K tokens | " |
| Phase 2B: Supports Edges | All scaffolds/examples | 5-8 min | 50-80K tokens | " |
| Phase 2C: Assesses Edges | Assessment-LO links | 3-5 min | 30-50K tokens | " |
| Phase 2D: Discourse Edges | Precedes/Anchors | 5-8 min | 50-80K tokens | " |
| Final Validation | DAG check, coverage | 30-60s | 0 tokens | " |
| **TOTAL (1 chapter)** | **Complete graph** | **~50-75 min (baseline)** | **~470-730K tokens** | With ~128 in-flight, expect materially lower wall-clock; cost optimization is secondary |

Token and cost numbers are placeholders; prioritize saturating the GPU-hosted model. Recalibrate once actual throughput/cost on gpt-oss-120b is known.

### Expected Output (Single Chapter)
- **Nodes**: 80-150 (50-100 knowledge, 10-30 assessments, 5-15 LOs, 15-30 TeachingSteps)
- **Edges**: 200-500+ (100-200 requires, 50-150 supports, 20-50 assesses, 30-100 discourse)
- **Coverage**: 100% of chapter content captured
- **Validation**: All invariants pass (DAG, reachability, coverage, purity)

**Scaling to 12 chapters**: ~10-15 hours runtime; cost depends on gpt-oss-120b (prev ~$30-55)

---

## Future Enhancements (Post-Demo)

1. **Specialized Actors**: Instead of generic Harvester, spawn `FactualHarvester`, `ConceptualHarvester`, etc. with focused prompts
2. **Incremental Dedup**: Run dedup after each chapter instead of end-of-phase
3. **Edge Conflict UI**: Integrate LLM-assisted resolution for conflicting edge types
4. **Parallelism Tuning**: Adjust concurrency based on rate limits
5. **Layered Multi-Pass**: Implement the 5-layer approach for more granular control
6. **Resume from Checkpoint**: Save phase progress, resume if interrupted
7. **Validation Budget**: Cap auto-merges per chapter to prevent cascading failures

---

## Success Criteria for Demo (targets; tune per chapter/model)

✅ **COMPLETENESS**:
- [ ] Phase 1 extracts ~50+ knowledge nodes from single chapter (adjust for chapter size)
- [ ] Exercises/examples captured as assessment items (~10+)
- [ ] Learning objectives captured (~5+ LOs)
- [ ] Discourse structure populated (~15+ TeachingSteps)
- [ ] Phase 2 produces ~200+ edges total (requires/supports/assesses/discourse mix)

✅ **QUALITY**:
- [ ] Final validation passes invariants
- [ ] Requires layer remains acyclic
- [ ] LOs reachable from first-principles
- [ ] Each LO has ≥2 target assessments (or justified exception)
- [ ] Edge/node ratio shows good connectivity (~2.5+); investigate outliers
- [ ] Dedup merges obvious duplicates; no regressions

✅ **DEMONSTRABLE**:
- [ ] Logs show Phase 1 → Dedup → Phase 2 flow
- [ ] Rerun visualization timeline is readable
- [ ] Walkthrough examples: complex concept with prerequisites; worked example support; assessment-LO link; discourse sequence
- [ ] Graph JSON spot-checks look sane

✅ **ROBUST**:
- [ ] No crashes/unhandled errors
- [ ] Runtime acceptable for chosen chapter (baseline 50-75 min; improve via high concurrency on GPU host)
- [ ] Handles hallucinated/missing slugs gracefully
- [ ] Gateway tolerates target concurrency (e.g., ~128 in-flight) without rate-limit churn; monitor cost separately

---

## Timeline (Demo Tomorrow)

**Today (6 hours)**:
- 09:00-10:30 (1.5h): Steps 1-3 (Add mode enum, prompts, tool filtering)
- 10:30-11:30 (1h): Steps 4-5 (Update constructors, message handler)
- 11:30-12:30 (1h): Step 6 (Orchestration function)
- **Lunch Break**
- 13:30-14:00 (0.5h): Step 7 (CLI subcommand)
- 14:00-14:15 (0.25h): Step 8 (main.rs update)
- 14:15-15:45 (1.5h): Testing (single chapter, validate completeness)
- 15:45-16:30 (0.75h): Demo script, prompts refinement, polish

**Demo Day**:
- Morning: Start full run (60-75 min) on demo chapter while preparing presentation
- 30 min before: Validate output completeness (node/edge counts)
- Demo: Show final graph + explain two-phase approach (15-20 min total)

---

## Open Questions / Decisions Needed

1. **Chapter selection for demo**: Which single chapter has good mix of content but is shortest? (Check PreTeXt source for line counts)
2. **Model choice**: gpt-oss-120b (current target); identify fallback if quality or cost requires it
3. **Dedup threshold**: 0.95 (RECOMMENDED - conservative, fewer false positives) or 0.92 (aggressive)?
4. **Sub-phase execution**: Sequential (safer) or parallel (faster but harder to debug)?
5. **Prompt iteration**: Should actors loop until target counts met, or single-pass only?
6. **Validation strictness**: Use `--graph-strict-quality` (promotes warnings to errors) or allow warnings?

---

## Contact / Support

For questions during implementation:
- Check white_paper.md for invariant definitions
- See AGENTS.md for coding conventions
- Review existing tests in tests/dedup_auto_merge.rs

Good luck with the demo! 🚀
