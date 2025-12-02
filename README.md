# Weaver

> **An Actor-Based Autonomous Curriculum Graph Construction System**

Weaver uses a novel **two-phase architecture** with specialized LLM agents to autonomously extract knowledge from textbook content and construct a comprehensive curriculum graph with validated learning relationships.

---

## 🎯 The Problem We Solve

### The Missing Target Problem

When building knowledge graphs with LLMs, a fundamental race condition exists:

```mermaid
flowchart LR
    subgraph "❌ Single-Pass Approach"
        A[LLM reads Chapter 1] --> B[Creates Node: 'Variables']
        C[LLM reads Chapter 2] --> D[Creates Edge: 'Loops' → 'Variables']
        D --> E[💥 ERROR: 'Loops' doesn't exist yet!]
    end
```

**The edge is created before its source node exists.** Traditional solutions involve complex retry logic and eventual consistency—fragile in practice.

### Our Solution: Two-Phase Architecture

```mermaid
flowchart TB
    subgraph Phase1["🌾 Phase 1: Harvest (Nodes Only)"]
        H1[Chapter 1 Harvester] --> N1[Variables, Types, ...]
        H2[Chapter 2 Harvester] --> N2[Loops, Conditionals, ...]
        H3[Chapter 3 Harvester] --> N3[Functions, Scope, ...]
    end
    
    Phase1 --> Barrier
    
    subgraph Barrier["🔒 Deduplication Barrier"]
        D1[Cluster Similar Nodes]
        D2[Merge Duplicates]
        D3[Validate & Persist]
        D1 --> D2 --> D3
    end
    
    Barrier --> Phase2
    
    subgraph Phase2["🕸️ Phase 2: Weave (Edges Only)"]
        W1[Requires Weaver] --> E1[Prerequisites]
        W2[Supports Weaver] --> E2[Examples & Scaffolds]
        W3[Assesses Weaver] --> E3[Assessment Alignments]
    end
    
    style Phase1 fill:#e8f5e9
    style Barrier fill:#fff3e0
    style Phase2 fill:#e3f2fd
```

**All nodes exist before any edges are created.** The deduplication barrier ensures a clean, deduplicated graph state.

---

## 🏗️ System Architecture

### Actor Model Overview

Weaver is built on **kameo**, a Rust actor framework. Every major component is an isolated actor with message-based communication:

```mermaid
flowchart TB
    subgraph Supervision["Supervision Hierarchy"]
        Sup[TwoPhaseSupervisor]
        Sup --> Orch1[TwoPhaseOrchestrator<br/>Chapter 1]
        Sup --> Orch2[TwoPhaseOrchestrator<br/>Chapter 2]
        Sup --> Orch3[TwoPhaseOrchestrator<br/>Chapter 3]
        Sup --> OrcN[...]
    end
    
    subgraph "Shared Services (Singleton Actors)"
        GW[LLMGateway<br/>━━━━━━━━━<br/>Rate limiting<br/>Retry logic<br/>Token tracking]
        GM[GraphManager<br/>━━━━━━━━━<br/>Validation<br/>Persistence<br/>Version control]
        DD[DeduplicationAgent<br/>━━━━━━━━━<br/>Similarity clustering<br/>Auto-merge]
        RR[RerunSink<br/>━━━━━━━━━<br/>Telemetry<br/>Visualization]
    end
    
    Orch1 -.-> GW
    Orch1 -.-> GM
    Orch2 -.-> GW
    Orch2 -.-> GM
    Orch3 -.-> GW
    Orch3 -.-> GM
    
    Sup --> DW[DedupPersistWorker]
    DW --> DD
    DW --> GM
    
    style Sup fill:#ffcdd2
    style GW fill:#bbdefb
    style GM fill:#c8e6c9
    style DD fill:#fff9c4
```

### Why Actors?

| Benefit | How We Use It |
|---------|---------------|
| **Isolation** | Each chapter has independent conversation state |
| **Concurrency** | Process N chapters in parallel safely |
| **Fault Tolerance** | Failed chapters don't crash the system |
| **Backpressure** | LLMGateway semaphore prevents API overload |
| **Cancellation** | Graceful shutdown with CancellationTokens |

---

## 🤖 LLM Gateway Architecture

The `LLMGateway` is the central coordinator for all LLM interactions:

```mermaid
sequenceDiagram
    participant R as Reader Actor
    participant GW as LLMGateway
    participant S as Semaphore<br/>(128 permits)
    participant API as OpenAI API
    participant TH as ToolHost

    R->>GW: ChatCompletionRequest
    GW->>S: acquire permit
    
    Note over S: Rate limiting:<br/>max 128 concurrent
    
    S-->>GW: permit granted
    GW->>API: POST /chat/completions

    alt API Error (429, 500, etc.)
        API-->>GW: Error
        GW->>GW: Jittered backoff<br/>(250ms → ~2s cap with current retry limit)
        GW->>API: Retry (up to 4 retries, 5 attempts total)
    end

    API-->>GW: Response with tool_calls
    
    loop For each tool call
        GW->>TH: ExecuteTool
        TH->>TH: Validate tool allowed for mode
        TH-->>GW: ToolOutput
        GW->>GW: Append to messages
    end
    
    GW->>API: Continue conversation
    API-->>GW: Final response
    GW-->>R: Result
    GW->>S: release permit
```

### Key Gateway Features

```rust
// Configurable constants (src/constants.rs)
LLM_MAX_CONCURRENT_REQUESTS: 128  // Semaphore permits
LLM_MAX_RETRIES: 5                // Per-request retry limit
REQUEST_TIMEOUT_SECS: 300         // 5-minute timeout
MAX_TOOL_ITERATIONS: 60           // Max tool loops per conversation
```

---

## 🔧 Tool System & Mode Enforcement

### The ToolHost Pattern

Each Reader actor wraps itself as a `ToolHost` that enforces mode-specific tool access:

```mermaid
flowchart TB
    subgraph "Reader&lt;HarvesterSpec&gt;"
        TH1[ToolHost::Harvester]
        TH1 --> Tools1["✅ graph_insert_knowledge<br/>✅ graph_update_knowledge<br/>✅ graph_insert_teaching_step<br/>✅ delegate_tasks<br/>✅ read_file_range<br/>✅ search_text<br/>❌ graph_add_requires<br/>❌ graph_add_supports"]
    end
    
    subgraph "Reader&lt;WeaverSpec&gt;"
        TH2[ToolHost::Weaver]
        TH2 --> Tools2["❌ graph_insert_knowledge<br/>❌ graph_update_knowledge<br/>✅ graph_add_requires<br/>✅ graph_add_supports<br/>✅ graph_add_assesses<br/>✅ graph_dag_check<br/>✅ graph_gap_summary<br/>✅ delegate_tasks"]
    end
    
    subgraph "LLM Gateway"
        GW[run_conversation<br/>exposes mode-scoped tool_ids]
        GW --> TH1
        GW --> TH2
        note over GW: Disallowed tools are not advertised; there is no per-call runtime deny list.
    end

    style Tools1 fill:#e8f5e9
    style Tools2 fill:#e3f2fd
```

### Compile-Time + Tool Surface

```rust
// Type-level mode specification (src/file_reader.rs)
pub trait ModeSpec: Send + Sync + 'static {
    const MODE: AgentMode;
    fn wrap_tool_host(actor: ActorRef<Reader<Self>>) -> ToolHost;
}

pub struct HarvesterSpec;  // Can only create nodes
pub struct WeaverSpec;     // Can only create edges

// Runtime enforcement
fn tool_identifiers_for_mode(mode: AgentMode) -> Vec<&'static str> {
    match mode {
        AgentMode::Harvester => vec![
            "graph_insert_knowledge",
            "graph_insert_teaching_step",
            // ... NO edge tools
        ],
        AgentMode::Weaver => vec![
            "graph_add_requires",
            "graph_add_supports",
            // ... NO node tools
        ],
    }
}
```

Gateway enforcement is by construction: only the mode's `tool_ids` are published to the LLM; there
is no per-call runtime deny list inside `LLMGateway::handle_tool_calls`.

---

## 🌾 Niche-Based Parallel Processing

Each phase uses **specialized niches** that run concurrently:

```mermaid
flowchart TB
    subgraph Harvest["Phase 1: Harvest Niches (Parallel)"]
        direction TB
        HN1["🔤 Factual/Conceptual<br/>Terms, definitions, principles"]
        HN2["📋 Procedural/Examples<br/>Procedures, worked examples"]
        HN3["📝 Assessments<br/>Quiz items, practice problems"]
        HN4["🎓 Teaching Steps<br/>Lessons, explanations"]
        HN5["🎨 Supports/Illustrations<br/>Analogies, diagrams, scaffolds"]
        HN6["🧠 Metacognitive<br/>Strategies, reflection cues"]
    end
    
    subgraph Weave["Phase 2: Weave Niches (Parallel)"]
        direction TB
        WN1["⬅️ Requires<br/>Prerequisites (DAG)"]
        WN2["🔗 Supports<br/>Fadeable scaffolds"]
        WN3["🎯 Assesses<br/>Assessment → LO alignment"]
        WN4["📖 Teaching Steps<br/>Precedes + Anchors edges"]
        WN5["🔍 Coverage Gap<br/>Find & fix missing edges"]
        WN6["✅ Cleanup/QA<br/>Validate DAG, fadeability"]
    end
    
    Harvest --> |"Dedup Barrier"| Weave
    
    style Harvest fill:#e8f5e9
    style Weave fill:#e3f2fd
```

### Tagging System for Coordination

```mermaid
flowchart LR
    subgraph "Harvester Creates Node"
        Node["P.bubble_sort<br/>━━━━━━━━━━━<br/>tags: [<br/>  source:Sorting/01_bubble.ptx<br/>  spec:procedural_examples<br/>  req:C.comparison_operator<br/>  sup:P.swap_function<br/>]"]
    end
    
    subgraph "Weaver Reads Hints"
        W1["Requires Weaver sees req:C.comparison_operator"]
        W2["Supports Weaver sees sup:P.swap_function"]
    end
    
    Node --> W1
    Node --> W2
    
    W1 --> E1["graph_add_requires(<br/>  from: 'P.bubble_sort',<br/>  to: 'C.comparison_operator'<br/>)"]
    W2 --> E2["graph_add_supports(<br/>  from: 'P.swap_function',<br/>  to: 'P.bubble_sort'<br/>)"]
```

---

## 🔄 Supervisor Lifecycle

### ChildHandle Pattern

The `TwoPhaseSupervisor` manages chapters via `ChildHandle` structs created by a `ChildLauncher`:

```mermaid
flowchart TB
    subgraph Supervisor["TwoPhaseSupervisor"]
        Start["StartTwoPhaseSupervision<br/>{chapters, cli_opts, ...}"]
        Start --> Spawn["Spawn up to max_concurrent_chapters"]
    end
    
    Spawn --> CL["ChildLauncher(chapter)"]
    
    subgraph CH["ChildHandle"]
        Join["join: JoinHandle&lt;()&gt;<br/>━━━━━━━━━━━━━━━<br/>Spawned tokio task"]
        Stop["stop: Box&lt;dyn FnOnce(...)&gt;<br/>━━━━━━━━━━━━━━━<br/>Cancellation callback"]
        HRx["harvest_rx: oneshot::Receiver<br/>━━━━━━━━━━━━━━━<br/>Receives HarvestReport"]
        WGate["weave_gate: oneshot::Sender<br/>━━━━━━━━━━━━━━━<br/>Signals 'start weave'"]
        Orch["orchestrator: ActorRef<br/>━━━━━━━━━━━━━━━<br/>Optional actor reference"]
    end
    
    CL --> CH
    
    style Supervisor fill:#ffcdd2
    style CH fill:#e3f2fd
```

### Phase Coordination via Oneshot Channels

```mermaid
sequenceDiagram
    participant Child as Chapter Child
    participant Sup as TwoPhaseSupervisor
    participant Dedup as DedupPersistWorker
    
    Note over Child: HARVEST PHASE<br/>Run 6 niches in parallel<br/>Create nodes only
    
    Child->>Sup: harvest_tx.send(HarvestReport)
    Note right of Sup: Collects from<br/>ALL chapters
    
    Note over Sup: Wait for all chapters<br/>to report harvest done
    
    Sup->>Dedup: RunDedupPersist
    Dedup-->>Sup: Ok(())
    Note over Sup: DEDUP BARRIER<br/>Cluster + merge + persist
    
    Sup->>Child: weave_gate.send(())
    Note over Child: weave_rx.await unblocks
    
    Note over Child: WEAVE PHASE<br/>Run 6 niches in parallel<br/>Create edges only
    
    Child->>Sup: supervisor.tell(ChildFinished)
    Note right of Child: { chapter, result: Ok(summary) }
```

### Cancellation Flow

```mermaid
sequenceDiagram
    participant Ext as External/Timeout
    participant Sup as TwoPhaseSupervisor
    participant CH as ChildHandle
    participant Token as CancellationToken
    participant Task as Child Task
    
    Ext->>Sup: StopAll { reason }
    
    loop For each inflight child
        Sup->>CH: child.stop(reason)
        CH->>Token: token.cancel()
    end
    
    Note over Task: tokio::select! {<br/>  work => ...<br/>  token.cancelled() => abort<br/>}
    
    Token-->>Task: wakes cancelled branch
    Task->>Sup: ChildFinished { result: Err("stopped") }
```

---

## ⏱️ Timeout & Cancellation Architecture

### Graceful Degradation

```mermaid
stateDiagram-v2
    [*] --> Harvesting: StartTwoPhase
    
    Harvesting --> GraceHarvest: harvest_timeout expires
    GraceHarvest --> WaitingForWeave: grace_period (3min) expires
    Harvesting --> WaitingForWeave: All niches complete
    
    WaitingForWeave --> Weaving: Dedup barrier complete
    
    Weaving --> GraceWeave: weave_timeout expires
    GraceWeave --> Completed: grace_period (3min) expires
    Weaving --> Completed: All niches complete
    
    GraceHarvest --> Stopped: StopTwoPhase
    GraceWeave --> Stopped: StopTwoPhase
    Harvesting --> Stopped: StopTwoPhase
    Weaving --> Stopped: StopTwoPhase
    
    Completed --> [*]
    Stopped --> [*]
    
    note right of GraceHarvest: CancellationToken.cancel()<br/>Active LLM calls abort
    note right of GraceWeave: In-flight work gracefully stops
```

### Cancellation Token Propagation

```mermaid
flowchart TB
    Orch[TwoPhaseOrchestrator] --> |"shares"| Token[CancellationToken]
    
    Token --> R1[Reader 1]
    Token --> R2[Reader 2]
    Token --> R3[Reader 3]
    
    R1 --> |"passes to"| GW[LLMGateway]
    R2 --> |"passes to"| GW
    R3 --> |"passes to"| GW
    
    GW --> |"select! with"| API[API call]
    GW --> |"select! with"| CancelBranch["token.cancelled() ?"]
    
    subgraph GraceTimeout["On Grace Timeout"]
        Orch --> |"token.cancel()"| Token
        Token --> |"wakes all select!"| R1
        Token --> |"wakes all select!"| R2
        Token --> |"wakes all select!"| R3
    end
    
    style Token fill:#ffcdd2
```

---

## 📊 Graph Data Model

### Node Types

```mermaid
classDiagram
    class NodePayload {
        +Uuid logical_id
        +String slug
        +NodeKind kind
        +Vec~String~ tags
    }

    class NodeKind {
        <<enum>>
        Knowledge(KnowledgeNode)
        TeachingStep(TeachingStepNode)
    }

    class KnowledgeNode {
        +String title
        +String statement
        +KnowledgeType knowledge_type
        +Vec~SourceRef~ source_refs
        +f32 confidence
        +Vec~String~ rubric_criteria
        +Vec~String~ construct_irrelevant_demands
        +Option~GrainLevel~ grain_level
        +Option~IntrinsicLoad~ intrinsic_load
        +IntroductionScope introduction_scope
    }

    class TeachingStepNode {
        +String title
        +String statement
        +TeachingPurpose purpose
        +Vec~String~ method_tags
        +String episode
        +Vec~SourceRef~ source_refs
        +Option~String~ rationale
    }

    class KnowledgeType {
        <<enum>>
        factual
        conceptual
        procedural
        metacognitive
        learning_outcome
        assessment_item
    }

    NodePayload --> NodeKind
    NodeKind --> KnowledgeNode
    NodeKind --> TeachingStepNode
    KnowledgeNode --> KnowledgeType
```

### Edge Types

```mermaid
flowchart LR
    subgraph "requires (DAG)"
        R1[P.bubble_sort] --> |"strength + rationale + evidence_refs"| R2[C.comparison]
    end

    subgraph "supports (fadeable)"
        S1[P.swap_example] --> |"support_kind + intended_effect<br/>case_tag + coverage_tags<br/>evidence_refs required"| S2[P.bubble_sort]
    end

    subgraph "assesses"
        A1[A.sorting_quiz] --> |"evidence_link {claim, observation_features, scope}"| A2[LO.apply_sorting]
    end

    subgraph "anchors"
        AN1[TS.explain_bubble] --> |"impact: introduce/use/refine/motivate/target"| AN2[P.bubble_sort]
    end

    subgraph "precedes"
        P1[TS.setup_problem] --> |"episode matches both steps"| P2[TS.explain_bubble]
    end
```

---

## 🚀 Running the System

### Two-Phase Mode (Autonomous)

```bash
# Set environment
export OPENAI_MODEL="your-model-name"
export OPENAI_API_BASE="http://your-endpoint:1234/v1"

# Demo run (matches run.sh: 13-way parallel, ~20/31 min timeouts)
./run.sh

# Manual run with defaults (4-way parallel, 1h/phase)
cargo run --release -- uncc_cs2-pretext-project \
  --max-concurrent-chapters 4 \
  --harvest-timeout-hours 1.0 \
  --weave-timeout-hours 1.0
```

### Monitoring

```bash
# Watch logs in real-time
tail -f logs/weaver.log | rg "WARN|ERROR|harvesting|weaving"

# Check for stuck processes
tail -f logs/weaver.log | rg "grace|cancel|timeout"
```

### Analyst Mode (Post-Processing)

After harvesting and weaving, use **Analyst Mode** for interactive analysis (read-only by default):

```bash
cargo run --release -- uncc_cs2-pretext-project --interactive
# alias: cargo run --release -- uncc_cs2-pretext-project --analyst
# opt-in to writable tools: cargo run --release -- uncc_cs2-pretext-project --interactive --interactive-writable
```

The Analyst has **read-only access** to powerful analysis tools:

```mermaid
flowchart TB
    subgraph Analyst["🔬 Analyst Mode"]
        direction TB
        
        subgraph Structure["Structural Analysis"]
            S1["graph_first_principles()"]
            S2["graph_dag_check()"]
            S3["graph_keystone()"]
        end
        
        subgraph LO["Learning Outcome Analysis"]
            L1["graph_lo_alignment_summary()"]
            L2["graph_lo_coverage()"]
            L3["graph_lo_missing_criteria_view()"]
        end
        
        subgraph Gaps["Gap Analysis"]
            G1["graph_gap_summary()"]
            G2["graph_example_gaps_view()"]
            G3["graph_fadeability_view()"]
            G4["graph_practice_gaps_view()"]
        end
        
        subgraph Discourse["Discourse Analysis"]
            D1["graph_borrow_ahead()"]
            D2["graph_discourse_orphans()"]
        end
    end
    
    style Analyst fill:#f3e5f5
```

**Example Analysis Questions:**

| Question | Tool to Use |
|----------|-------------|
| "What concepts are foundational?" | `graph_first_principles_summary()` |
| "Are there any cycles?" | `graph_dag_check()` |
| "Which LOs lack assessments?" | `graph_lo_missing_criteria_view()` |
| "What examples are missing?" | `graph_example_gaps_view()` |
| "What are the keystone concepts?" | `graph_keystone()` |
| "Are concepts used before taught?" | `graph_borrow_ahead()` |

---

## 🔬 Novel Contributions

### 1. Two-Phase Graph Construction
Separating node creation from edge creation eliminates the "missing target" problem that plagues single-pass approaches.

### 2. Mode-Specific Tool Enforcement
Compile-time generics + runtime filtering provide **defense in depth** against LLM "hallucinating" tool calls.

### 3. Niche-Based Parallelism
Specialized agents (factual, procedural, assessments, etc.) enable efficient parallel extraction without conflicts.

### 4. Actor-Based LLM Orchestration
Using the actor model for LLM coordination provides:
- Natural rate limiting via semaphores
- Conversation isolation
- Graceful cancellation
- Fault tolerance

### 5. Deduplication Barrier
A synchronization point that ensures graph consistency before edge creation, preventing duplicate edges and missed connections.

### 6. Read-Only Analyst Mode
Launch with `--interactive` (or `--analyst`) to open the TUI in read-only mode; add
`--interactive-writable` only when you explicitly want the legacy writable FileReader tools.
A dedicated analysis agent with **26 inspection tools** for validating curriculum structure:
- Structural analysis (DAG check, keystones, first principles)
- Learning outcome alignment (coverage, reachability, criteria gaps)
- Quality gaps (examples, fadeability, practice assessments)
- Discourse coherence (borrow-ahead, orphaned steps)

---

## 📁 Key Source Files

| File | Purpose |
|------|---------|
| `src/main.rs` | Entry point, CLI parsing |
| `src/app.rs` | Actor bootstrap, runtime options |
| `src/app/two_phase/orchestrator.rs` | Per-chapter state machine |
| `src/app/two_phase/supervisor.rs` | Multi-chapter coordination |
| `src/file_reader.rs` | Reader actor, mode specs, tool filtering |
| `src/llm_gateway.rs` | LLM coordination, retries, rate limiting |
| `src/graph/service.rs` | Graph mutations, validation |
| `src/graph/manager.rs` | Graph actor, persistence |
| `src/tools/llm/graph_tools/` | Tool implementations |

---

## 📚 Further Reading

- [`AGENTS.md`](./AGENTS.md) - Detailed agent documentation
- [`docs/white_paper.md`](./docs/white_paper.md) - Domain model specification
- [`docs/graph_tools.md`](./docs/graph_tools.md) - Tool reference

---

*Built with Rust 🦀 + kameo actors + OpenAI-compatible LLMs*
