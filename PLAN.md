North Star (first principles)
	1.	Single writer, observable pipeline. GraphAdder remains the sole mutator and emits a precise Event stream after every decision so Viz and logs stay in lockstep.
	2.	Make it watchable from minute one. Prioritise live visualization; Rerun should reflect accepted nodes/edges immediately, even when generators are fake.
	3.	Derived structure, enforced invariants. Layout is recomputed from the prerequisite_for DAG each draw; supports edges never influence rank. Hard validation keeps the graph coherent while generators explore.

These guideposts keep the system demoable before any LLM integration and preserve your correctness guarantees.

⸻

Architecture at a glance

Crates/deps (minimal):
- petgraph for the directed graph and DAG checks.
- serde, serde_json, (optionally schemars, jsonschema) for typed JSON I/O + validation.
- uuid for IDs.
- thiserror for clean errors.
- tracing for logs; tokio runtime.
- async-openai (OpenAI‑compatible endpoint) for LLM calls.
- kameo for actors (your choice).
- rerun for live visualization (events + simple 2D layout).

Actors (message‑driven):
- NodeSynth (N copies): fallback deterministic generator ships first; LLM mode is optional.
- EdgeSynth (M copies): same pattern—heuristic fallback now, LLM later.
- GraphAdder (singleton): the only mutator; validates, inserts, and synchronously emits Event values for every accept/reject.
- Viz (singleton): consumes the Event stream, mirrors the graph read-only, recomputes ranks from prerequisite DAG, and records frames to Rerun.

Key invariant: Only GraphAdder can change the canonical petgraph; everyone else observes via events or read APIs.

⸻

Minimal schemas (structured JSON)

NodeProposal (from NodeSynth → GraphAdder)

{
  "kind": "concept|learning_outcome",
  "granularity": "sentence",
  "level": 0,
  "text": "One complete sentence that stands on its own.",
  "tags": ["tag-a","tag-b"]
}

EdgeProposal (from EdgeSynth → GraphAdder)

{
  "relation": "prerequisite_for|supports",
  "from_id": "UUID of existing node",
  "to_id": "UUID of existing node",
  "rationale": "Required brief justification."
}

Decision (GraphAdder → producers)

{
  "accepted": true,
  "assigned_id": "UUID or null for edges",
  "reason": null
}

Event (GraphAdder → Viz / observers)

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Event {
    NodeAccepted { id: uuid::Uuid, kind: NodeKind, text: String },
    NodeRejected { proposal: NodeProposal, reason: String },
    EdgeAccepted { from: uuid::Uuid, to: uuid::Uuid, relation: Relation },
    EdgeRejected { proposal: EdgeProposal, reason: String },
}

Network invariants enforced by GraphAdder
- Node kind ∈ {concept, learning_outcome}; granularity must be sentence.
- Node level ∈ {0,1,2,3}; learning outcomes must start with “I can ” or “Students can ”.
- Dedup: reject if text duplicates existing (normalized lowercase text).
- Rationale is required on every edge proposal; store the trimmed string.
- Relation domains and constraints:

| Relation           | Allowed endpoints                    | Extra checks                                  | In DAG? |
|--------------------|--------------------------------------|-----------------------------------------------|---------|
| prerequisite_for   | Concept → Concept or LearningOutcome | from.level ≤ to.level; must not introduce cycle| Yes     |
| supports           | Any → Any                            | forbid self-loops                              | No      |

⸻

Milestones with “done” checks

M0 — Compile & run
- Keep `main.rs` minimal: boot Tokio, init tracing, print args.
- `cargo run` succeeds, no functionality yet.

M1 — Core types + GraphAdder + Events (no LLM)
- Implement model.rs, graph.rs, adder.rs, events.rs (or model.rs) with Event enum.
- GraphAdder accepts nodes and edges, enforces validations, emits Event::* synchronously after every decision.
- Unit tests cover accept/reject paths (duplicate node, cycle edge, missing rationale).

M2 — Viz actor + Rerun hookup
- Add viz.rs; start rerun::RecordingStream once; provide an event receiver (mpsc sender is fine).
- Viz maintains a read-only mirror (local petgraph or lightweight caches), recomputes rank per draw from prerequisite_for DAG, and ignores supports for rank.
- Manual smoke test: feed a few NodeProposals + EdgeProposals; nodes appear left-to-right; edges draw; rejections log warnings.

M3 — Fallback generators (fake content first)
- node_synth.rs emits deterministic placeholder NodeProposals with level/tags metadata; edge_synth.rs emits EdgeProposals with required rationales.
- Pipeline run uses only fallback generators; demo shows acceptance/rejection counts while Viz animates growth.

M4 — CLI glue (mvp run …)
- Wire GraphAdder, Viz, fallback NodeSynth/EdgeSynth.
- Command `mvp run` streams events, prints summaries, optionally exports DOT.
- Definition of done: rerun UI plays live; supports edges visibly cross ranks; export works.

M5 — LLM path (optional once M4 is stable)
- Add `--use-llm true` to swap in async-openai; keep JSON-only contracts.
- GraphAdder logic remains unchanged; events continue to drive Viz.

⸻

Concrete prompts (drop‑in)

NodeSynth — “generate candidates”
- System:
“You generate placeholder nodes for a CS2 Design Recipe network. Each node is a single clear sentence. Avoid code. Do not repeat content. Do not include explanations. Output only JSON conforming to the provided schema.”
- User:
“Schema (JSON): { “nodes”: [ … NodeProposal … ] }
Produce {N} concept nodes and {K} learning outcomes.
Constraints:
- kind ∈ {concept, learning_outcome}
- granularity = “sentence”
- level ∈ {0,1,2,3} where 0 is most foundational
- text is a single complete sentence that reads clearly on its own.
- Learning outcomes must start with “I can ” or “Students can ”.
- Each sentence aims for clarity without imposing strict stylistic rules.
- Optional tags: include up to 3 topical labels if helpful.
Return only JSON.”

EdgeSynth — “propose edges” (LLM version)
- System:
“You propose pedagogically coherent edges between nodes. Use only the provided IDs. Output only JSON.”
- User:
“Given these nodes (id, kind, level, text)… Propose up to {E} edges.
Rules: prerequisite_for edges must connect Concept → Concept/LO, enforce from.level ≤ to.level, and preserve acyclicity. supports edges can be any→any except self-loops. Provide a one-sentence rationale for every edge. Return only JSON of EdgeProposal[].”

⸻

Granularity guardrails (MVP + roadmap)
- Atom enforced now: Single-sentence claim (one standalone sentence conveying a single idea). This comes from the design-recipe granularity ladder exploration and keeps judge logic deterministic while allowing natural phrasing.
- Validation cues for agents: confirm exactly one sentence, check LO prefix (“I can ” or “Students can ”), ensure level ∈ {0,1,2,3}, optionally accept tags, and flag truly ambiguous wording without rigid stylistic bans.
- Example set the team can reuse:
	- Concept — “A method’s signature communicates its name, parameter types, and return type.”
	- Concept — “Write tests before implementation to capture expected behavior.”
	- Learning outcome — “I can apply the design recipe to implement a method from a written specification.”
- Future ladder: keep `granularity` ready to expand (micro-explanation, concept paragraph, mini-lesson, scaffolding nodes). Plan an ADR once MVP is stable to widen the enum without breaking existing prompts.

⸻

GraphAdder validations (code, not model)
- Hard rules:
	- Node: granularity == Sentence; text trimmed, UTF-8, single sentence, deduped by normalized lowercase text, and should read clearly.
	- Learning outcomes must start with “I can ” or “Students can ”; levels must be within {0,1,2,3}; optional tags permitted.
- Prerequisite edges: ensure from/to exist, forbid self-loops, enforce Concept→(Concept|LO), require from.level ≤ to.level, and run a DAG check (clone prerequisite subgraph → add candidate → ensure acyclic).
- Supports edges: allowed Any→Any except self-loops; they never participate in the DAG check.
- Rationale string required for every edge; trim whitespace before storing.
- Emit Event::* synchronously after each decision so observers can replay the state faithfully.
- Layout ranks are always derived when rendering; stored `level` remains curricular metadata and does not drive Viz positioning.
- Soft rules (optional LLM‑Judge):
- “Is the sentence standalone, jargon‑light, and unambiguous?” → score 0–1.
- Reject below threshold and include the judge’s explanation in reason.

⸻

Minimal Rust wiring (sketch, no boilerplate)

// model.rs
pub struct Node { /* as above */ }
pub struct Edge { /* as above */ }

	pub struct NodeProposal { /* serde */ }
	pub struct EdgeProposal { /* serde */ }

// adder.rs
pub struct GraphAdder {
    graph: Graph<Node, Edge, Directed>,
    events: tokio::sync::mpsc::Sender<Event>, // emit after each decision
    /* indexes, maps */
}
impl GraphAdder {
	    pub fn add_nodes(&mut self, cands: Vec<NodeProposal>) -> Vec<Decision> { /* validate->insert */ }
	    pub fn add_edges(&mut self, cands: Vec<EdgeProposal>) -> Vec<Decision> { /* DAG + rationale checks */ }
    fn would_cycle(&self, from: NodeIndex, to: NodeIndex) -> bool { /* toposort or cycle check */ }
}

Hook actors around these functions; keep GraphAdder synchronous (owned by one actor) for simplicity; let NodeSynth/EdgeSynth be async (LLM calls).

⸻

Layout computation (Viz)
	1.	Build a prerequisite_only subgraph from accepted nodes/edges (local mirror is fine).
	2.	Run a toposort; assign rank[node] = 0 when no incoming prereqs, else 1 + max(rank[pred]).
	3.	For y-position, pack nodes within each rank by insertion order (stable Vec per rank) and add a tiny jitter if needed.
	4.	Supports edges never affect rank; just draw them across ranks.
	5.	Before logging each frame, increment a monotonic `step` and call `rr.set_time_sequence("step", step)`.

⸻

Demo script (what you’ll show)
	1.	Start the app; Rerun window opens immediately (Viz actor running).
	2.	Fallback NodeSynth streams deterministic candidates; console shows accept/reject with reasons while Viz plots them.
	3.	Fallback EdgeSynth proposes edges with rationales; accepted edges draw live; rejected ones warn in logs.
	4.	Show summary counts, export DOT, and point out left-to-right ranks derived from prerequisite DAG.
	5.	(Stretch) Print simple stats: node/edge counts by kind, average indegree/outdegree.

This aligns cleanly with your course deliverables: defined network (nodes/edges), a generative system design, and a working MVP with visualization and basic metrics.

⸻

Risks & mitigations (kept small)
- LLM emits invalid JSON. Mitigate with jsonschema check; on failure, ask once for “RETRY: emit valid JSON only”; otherwise discard batch.
- Duplicate spam. Keep a normalized text set; reject with reason “duplicate”.
- Cycles. Strict DAG check before every prerequisite commit; include short cycle witness in rejection reason.
- Layout stability. Derive ranks from prerequisite DAG each frame; keep per-rank ordering deterministic (e.g., insertion order) to avoid node jitter.
