Project: weaver — MVP Implementation Brief

Intent (first principles)

We need a coherent learning network built from LLM‑generated placeholder content, and we need to demo it live as the graph grows. Creativity happens at the edge (LLM or fallback proposals), but only one component may mutate state and every decision must emit an Event so the visualization stays in sync. The invariants that keep the system trustworthy:
- Nodes stay uniform, small, and typed (so edges remain meaningful).
- prerequisite_for edges must form a DAG; supports edges do not constrain rank.
- Visualization derives layout from the current prerequisite DAG on every draw; no mutable level field lives on the node.
- Everything is auditable (events, summaries, and a DOT export).

This MVP proves: generators → structured JSON → GraphAdder → Event stream → validated, watchable graph.

⸻

Hard scope (lock for MVP)
- Node types: concept, learning_outcome
- Granularity: "sentence" only (one complete sentence per node)
- Level metadata: nodes carry level ∈ {0,1,2,3}; used for validation, not layout
- Edge types: prerequisite_for (acyclic), supports (no DAG constraint); every edge proposal includes a required one-sentence rationale
- Single writer: exactly one GraphAdder owns all mutations and emits Event values after every decision
- Counts (demo defaults): ~25 concepts + ~5 LOs; ~40 edges mixed
- Visualization first: Viz subscribes to events so the graph is watchable before any LLM integration; fallback generators ship first
- No new dependencies beyond the provided Cargo.toml
(Use petgraph; do not add clap—parse args with std::env.)

Environment:
- OPENAI_API_KEY required for LLM calls.
- Optional: OPENAI_BASE_URL, OPENAI_MODEL (fallback to a sane default).

⸻

Granularity (teach-by-example guardrails)
- Chosen atom: “Single-sentence claim,” inspired by the granularity ladder exploration. Every node must be a single complete sentence that conveys one idea while allowing natural phrasing.
- Why: keeps the graph coherent (edge semantics stay sharp), easy for the judge to validate mechanically, and demo-friendly.
- Examples we vet against:
	- Concept: “A method’s signature communicates its name, parameter types, and return type.”
	- Concept: “Write tests before implementation to capture expected behavior.”
	- Learning outcome: “I can apply the design recipe to implement a method from a written specification.”
- Future ladder: Micro-explanations, concept paragraphs, and mini-lessons remain on the roadmap; storing `granularity` as `Sentence` today preserves room to extend later without breaking the schema.

⸻

Repository layout to create

src/
  main.rs               // CLI entrypoint; wires actors; drives MVP run flow
  model.rs              // Node/Edge core types + proposals + (de)serde + validation
  graph.rs              // GraphStore (petgraph), ID↔Index maps, DOT export
  adder.rs              // GraphAdder actor: the only mutator + invariants
  node_synth.rs         // NodeGenerator actor: LLM call + fallback generator
  edge_synth.rs         // EdgeGenerator actor: LLM call + heuristic fallback
  llm.rs                // async-openai wrapper (JSON-only responses)
  summary.rs            // stats & summaries (counts, DAG check, degree)
  viz.rs                // Visualization actor: subscribes to GraphAdder events, recomputes ranks from prerequisite DAG on every draw; supports edges are ignored for rank

⸻

Data model (implement exactly)

// src/model.rs
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum NodeKind { Concept, LearningOutcome }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Granularity { Sentence }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub id: uuid::Uuid,
    pub kind: NodeKind,
    pub granularity: Granularity,
    pub level: u8,                   // 0..=3 curricular metadata
    pub text: String,           // concise educational statement
    pub tags: Option<Vec<String>>,  // optional topical labels
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Relation { PrerequisiteFor, Supports }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edge {
    pub from: petgraph::graph::NodeIndex,
    pub to: petgraph::graph::NodeIndex,
    pub relation: Relation,
    pub rationale: String,         // one-sentence justification (required)
}

// Proposals coming from generators (LLM or fallback)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeProposal {
    pub kind: NodeKind,
    pub granularity: Granularity, // always Sentence in MVP
    pub level: u8,
    pub text: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeProposal {
    pub relation: Relation,
    // Reference accepted nodes by UUID assigned by the GraphAdder.
    pub from_id: uuid::Uuid,
    pub to_id: uuid::Uuid,
    pub rationale: String,
}

// Decision result for both nodes and edges
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Decision {
    pub accepted: bool,
    pub reason: Option<String>,
    pub assigned_id: Option<uuid::Uuid>, // for nodes
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Event {
    NodeAccepted { id: uuid::Uuid, kind: NodeKind, text: String },
    NodeRejected { proposal: NodeProposal, reason: String },
    EdgeAccepted { from: uuid::Uuid, to: uuid::Uuid, relation: Relation },
    EdgeRejected { proposal: EdgeProposal, reason: String },
}

Validation rules (code, not AI):
- NodeProposal → Node (granularity must be Sentence; text must be trimmed, non-empty, and a single complete sentence)
- Deduplicate by normalized text (lowercase, trimmed, single‑space)
- For LearningOutcome, require prefix: "I can " or "Students can "
- Level must be within {0,1,2,3}; optional tags accepted
- EdgeProposal → Edge
- from_id and to_id resolve to existing nodes
- No self‑loops
- prerequisite_for: endpoints must be Concept→(Concept|LearningOutcome); require from.level ≤ to.level; adding must not create a cycle in the prerequisite_for subgraph (check before commit)
- rationale is required; store the trimmed string for every edge
- GraphAdder emits Event::* immediately after each accept/reject so observers stay in sync

Relations and constraints (MVP)

| Relation           | Allowed endpoints                    | Additional checks                                  | In DAG? |
|--------------------|--------------------------------------|----------------------------------------------------|---------|
| prerequisite_for   | Concept → Concept or LearningOutcome | from.level ≤ to.level; must remain acyclic          | Yes     |
| supports           | Any → Any                            | forbid self-loops                                   | No      |

⸻

GraphStore & GraphAdder

GraphStore (src/graph.rs)
- Wrap petgraph::Graph<Node, Edge, petgraph::Directed>
- Maintain:
- HashMap<String /*normalized text*/, NodeIndex>
- HashMap<uuid::Uuid, NodeIndex>
- Functions (exact signatures may vary, keep semantics):
- fn add_node(&mut self, n: Node) -> NodeIndex
- fn add_edge(&mut self, e: Edge) -> petgraph::graph::EdgeIndex
- fn find_by_text(&self, norm: &str) -> Option<NodeIndex>
- fn would_cycle_prereq(&self, from: NodeIndex, to: NodeIndex) -> bool
- fn export_dot(&self) -> String

GraphAdder actor (src/adder.rs)
- Messages:
- AddNodes(pub Vec<NodeProposal>) -> Vec<Decision>
- AddEdges(pub Vec<EdgeProposal>) -> Vec<Decision>
- Inventory() -> Vec<(uuid::Uuid, NodeKind, u8 /*level*/, String /*text*/, Option<Vec<String>> /*tags*/)>
- Summary() -> crate::summary::Summary
- ExportDot() -> String
- Holds a tokio::sync::mpsc::Sender<Event> (or ActorRef<Viz>) to publish decisions
- Behavior:
- Validate → resolve → test DAG → commit or reject
- Emit Event::* synchronously after each decision (accepted and rejected)
- Emit tracing events in parallel so text logs stay readable

DAG check: Build a temporary subgraph of only prerequisite_for edges, tentatively add the new one, then use petgraph::algo::is_cyclic_directed (or a toposort) to decide.

⸻

Visualization (src/viz.rs)
- Start a rerun::RecordingStream once in `Viz::new`.
- Expose a `tokio::sync::mpsc::Sender<Event>` so GraphAdder can forward decisions; Viz runs on its own task consuming a Receiver<Event>.
- Maintain a lightweight mirror of accepted nodes/edges (HashMap<Uuid, NodeCache>) for layout calculations.
- On every NodeAccepted/EdgeAccepted: recompute ranks from the prerequisite_for subgraph via longest-path-from-sources (rank 0 with no inbound prereqs, otherwise 1 + max pred rank).
- Within each rank, keep insertion order deterministic for vertical placement; add minor jitter only if overlap occurs.
- Supports edges never influence rank; draw them as thin lines across ranks to emphasize cross-cutting support.
- Stored `level` metadata is curricular only; layout always derives from the live prerequisite DAG.
- Increment a monotonic `step` counter per event and call `rr.set_time_sequence("step", step)` before logging geometry.

⸻

LLM integration (src/llm.rs)
- Use async_openai chat/completions.
- Support:
- OPENAI_API_KEY (required)
- OPENAI_BASE_URL (optional)
- OPENAI_MODEL (optional; fallback to a small, cheap JSON‑friendly model)
- Always request JSON‑only response. If the server supports response_format: { "type": "json_object" }, use it; otherwise enforce “Return ONLY JSON” in the prompt.

NodeSynth system prompt (exact text):

You produce placeholder educational nodes for a learning network.
Rules:
- Emit pure JSON: { "nodes": [ NodeProposal, ... ] }
- Each node is a standalone statement that can be understood without citations.
- "kind" ∈ {"concept","learning_outcome"}.
- "granularity" is always "sentence".
- "level" must be an integer 0–3 where 0 is most foundational.
- Learning outcomes MUST start with "I can " or "Students can ".
- Optional "tags": include up to 3 topical labels if helpful.
- Text should be a single complete sentence with clear references (aim for clarity; natural phrasing is fine).
- Avoid duplicates; vary vocabulary.
Produce 25 concept nodes and 5 learning outcomes that a CS2 student should achieve.

EdgeSynth system prompt (exact text):

You propose placeholder edges among existing nodes.
Rules:
- Emit pure JSON: { "edges": [ EdgeProposal, ... ] }
- Edge kinds: "prerequisite_for", "supports".
- Use from_id and to_id copied exactly from the provided inventory of UUIDs.
- For "prerequisite_for", pick Concept → Concept/LO and ensure from.level ≤ to.level.
- Include a brief rationale string for every edge (required).
Target about 40 edges total with a balanced mix.


⸻

Fallback generators (no extra crates)

Node fallback (src/node_synth.rs):
- Deterministic combinator: choose from fixed subjects × verbs × objects lists to produce 25 concept sentences + 5 LO sentences (LOs prefixed with “I can …”).
- Assign deterministic levels 0–3 and optionally tag nodes so generators stay schema-compliant.
- Ensure each statement stands alone without relying on citations.

Edge fallback (src/edge_synth.rs):
- Heuristic:
- Build supports from concept → LO (each LO gets ≥3 supports).
- Build prerequisite_for by sorting concept texts lexicographically and connecting adjacent pairs; skip if DAG check fails.
- Attach deterministic one-sentence rationales to every proposed edge (e.g., template based on node texts).

⸻

CLI (src/main.rs)

Single entrypoint (parse std::env::args() manually):
- mvp run --topic "Design Recipe" --concepts 25 --los 5 --edges 40 [--use-llm true|false] [--export-dot path]

Behavior:
- Boot a kameo system, start Viz actor (spawns rerun::RecordingStream), then start GraphAdder with a clone of Viz’s sender.
- Run fallback NodeGenerator by default (LLM path is opt-in), forward NodeProposal[] to GraphAdder, print accept/reject summary.
- Get inventory from GraphAdder, run fallback EdgeGenerator to propose edges with rationales, print accept/reject summary.
- GraphAdder streams Event::* to Viz; rerun renders nodes left-to-right by derived rank and draws supports across ranks.
- Optionally export DOT when --export-dot is provided.
-  Finish by printing formatted summary (counts and DAG status).

Example output (target):

Nodes: 30 (concept=25, learning_outcome=5)
Edges: 41 (prerequisite_for=18, supports=23)
Prerequisite DAG: OK (no cycles)
Top LOs by inbound supports:
  LO-0003 (7), LO-0001 (5), LO-0005 (5)


⸻

Tests (must pass)
- test_add_node_validation: rejects empty or duplicate sentences; accepts a well-formed single sentence.
- test_add_edge_cycle_rejected: construct three nodes, propose edges forming a cycle; last proposal is rejected.
- test_edge_missing_rationale_rejected: edge proposal without rationale is rejected with a clear reason.
- test_export_dot_contains_labels: ensures node texts and relation labels appear.

⸻

Telemetry
- Use tracing for:
- node.accepted / node.rejected with reason
- edge.accepted / edge.rejected with reason
- summary payload printed at INFO

⸻

Definition of Done (DOD)
	1.	cargo run -- mvp run --use-llm false --topic "Design Recipe" --concepts 25 --los 5 --edges 40 accepts ≥ 20 of 30 nodes and ≥ 30 edges with zero prerequisite cycles.
	2.	The run emits visualization events and rerun displays node/edge activity during execution.
	3.	Visualization shows nodes appearing left-to-right by derived prerequisite rank, with supports edges clearly spanning across ranks.
	4.	The run prints counts and “Prerequisite DAG: OK”.
	5.	cargo run -- mvp run --use-llm false --export-dot graph.dot writes a non-empty DOT file with labeled nodes/edges.
	6.	All tests in cargo test pass.

⸻

Nice-to-have (only if all of the above is green)
- summary: compute topological order over prerequisite_for and print first 10 nodes as a “learning path”.
- Include rationale in DOT as an edge label for the first 10 edges.

⸻

Execute now
	1.	Create the files and modules exactly as specified.
	2.	Implement validation, GraphAdder invariants, and CLI subcommands.
	3.	Add unit tests described above.
	4.	Ensure the project builds and runs the 4 CLI flows end‑to‑end with fallback generators (no network).
	5.	Wire LLM paths behind --use-llm true, reading env vars as described.

If something is ambiguous, prefer the simplest implementation that preserves the invariants and the DOD.
