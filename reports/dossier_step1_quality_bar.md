# Step 1 – Quality Bar Baseline (intentional stack, revised)

## Scope and sources

- Textual contract: `white_paper.md` (modeling rules, validations, evidence requirements).
- Intentional code: `src/main.rs`, `src/llm_gateway.rs`, `src/file_reader.rs`, `src/tools/llm/{mod.rs,list_directory.rs,read_file_full.rs,read_file_range.rs,search_text.rs,delegate_tasks.rs}`, `src/tools/filesystem.rs`, `src/tools/search.rs`, `src/constants.rs`, `src/rerun_sink.rs`.
- Excluded for now: `src/graph/*`, `src/analysis/*`, `src/tools/llm/graph_tools/*` (these are the vibe-coded targets).

## One-sentence bar

A contract-first, actorized system that treats the LLM as an expensive, bounded collaborator: schema-validated inputs, preview-first ergonomics, defensive async hygiene, explicit invariants, strong observability, and lean, typed interfaces that mirror the pedagogical model.

## Architectural portrait (what “intentional” looks like)

- **Actor isolation:** Every long-lived component is an actor with narrow messages (FileReader, LLMGateway, Scheduler, RerunSink, autosave/prune workers). State is owned, never shared; handlers are short and offload work with `ctx.spawn`.
- **LLM boundary discipline:** `LLMGateway` wraps OpenAI with semaphores, timeouts, jittered backoff, retry heuristics, and structured tool-call plumbing. Conversations have `max_iterations`, per-conversation token accounting, and a drop guard to reset prompt counters.
- **Preview + confirm:** File tools default to preview (cost header, token estimates, remaining budget, hints), require `fetch_body=true` to stream content, and compute safety margins from live metrics. Delegation is explicit and bounded.
- **Schema-first tools:** Arguments go through `schemars`/`bon::Builder` with `#[serde(deny_unknown_fields)]`; parse-time validation yields typed `ToolInputError`s. Duplicate tool IDs panic early. Workspace paths are canonicalized and sandboxed.
- **Observability:** Tracing fields on all key actions; metrics (latency, tokens, counts) live in `GatewayMetrics`; rerun sink receives scalar time series. Autosave/prune log duration and success/failure; tool calls log depth/bytes/scope.
- **Reproducibility:** CLI pins course commit and autosave path; persistent actors snapshot to disk; restores reapply runtime config and persist the updated settings.
- **Error taxonomy:** User vs internal errors at the tool boundary; `anyhow` with context for internal paths; no unchecked unwraps. Failures degrade gracefully (skip unreadable files, keep running when gateway persist fails, best-effort rerun).
- **Lean surfaces:** Few, orthogonal tools; small structs with clear names; imports grouped; lines kept short. Prompts are directive and concrete.

## Principles distilled into a checklist

1. **Actor ownership & non-blocking handlers**
   - State lives inside actors; mutation only via messages.
   - No heavy/blocking work inside handlers; use `spawn`/`spawn_blocking` + bounded parallelism (`buffered`, `Semaphore`).

2. **Cognitive economy (semantic macros, not assembly)**
   - Tools embody high-level intents, not primitive CRUD.
   - Default to preview; require explicit opt-in for large payloads.
   - Compute and expose cost/budget; provide actionable hints to narrow scope.

3. **Schema and validation discipline**
   - `#[serde(deny_unknown_fields)]`, typed enums/structs, builders that trim/validate.
   - Workspace paths canonicalized and confined; depth/iteration limits enforced.
   - Duplicate identifiers rejected at startup.

4. **Structured, typed feedback at boundaries**
   - User/LLM errors surfaced as structured codes (`missing_field`, `invalid_range`, `depth_exceeded`, `invalid_path`), not free text.
   - Internal faults remain `anyhow` but are logged with context.

5. **Observability & metrics**
   - Emit stable metric paths for latency, counts, token usage, successes/failures.
   - Best-effort, non-blocking telemetry (rerun sink) to avoid backpressure.
   - Trace key dimensions: depth, bytes, scope, attempt/retry, pruning counts.

6. **Explicit invariants and reproducibility**
   - Invariants are executable (cycle checks, budget checks, duplicate IDs) and fail fast.
   - Snapshots persist with version/commit; restores reapply config and re-persist.

7. **Human-facing affordances**
   - Prompts and previews tell the operator what to do next; CLI flags are explicit; hints are concrete ("use read_file_range", "re-run with fetch_body=true").

8. **Lean, modular code**
   - Small, purpose-built modules; no god objects; separate parsing, execution, persistence, and telemetry concerns.

## Evidence in the intentional code (why this bar is credible)

- **Actor hygiene in `main.rs`:** Autosave/prune run as separate actors scheduled by `Scheduler::SetInterval`. On restore, `GraphManager` re-applies runtime config then persists, ensuring persisted state reflects current flags. `LLMGateway` similarly restores, fetches metrics, and persists updates. Parents create required directories before spawning actors.
- **Gateway resilience:** `call_with_retry` wraps every OpenAI call with a timeout, bounded retries, jittered exponential backoff, and retry classification that skips caller-side errors. Each attempt logs rerun scalars (latency, backoff, errors) and records token usage. A semaphore caps concurrent requests at `LLM_MAX_CONCURRENT_REQUESTS`.
- **Structured tool surface:** Tool schemas are auto-derived; parse stages build validated structs and return typed `ToolInputError`s. Duplicate tool identifiers panic during tool table construction. File tools log depth, bytes, mode, scope, and include `byte_hint` for better budgeting.
- **Preview/budget mechanics:** Previews estimate tokens via learned estimators (per model) plus heuristic and safety margin. They compute remaining context from per-conversation prompt tokens and model context limit, then include actionable hints to narrow scope before fetching bodies.
- **Workspace confinement and path hygiene:** `resolve_workspace_path` canonicalizes, rejects escapes, and classifies the failure as user-facing. Paths are rendered relative in responses to avoid leaking host paths.
- **Observability plumbing:** `GatewayMetrics` stores per-conversation aggregates; `log_scalar` is best-effort and non-blocking. `RerunSink` throttles reconnects and prevents blocking the caller. Autosave/prune emit counts and durations to rerun, mirroring the expectation for any background graph maintenance.
- **Error taxonomy:** Tool boundaries distinguish user vs internal errors; internal errors are still logged with context. There are almost no unchecked `unwrap`s; failures degrade gracefully (skip unreadable files, continue after partial autosave failure).
- **Directive prompts:** The FileReader system prompt instructs the LLM to list directories, read files before summarizing, delegate in parallel, and respect preview budgets. CLI flags are concise and explicit.

## Detailed expectations to apply later (dimension by dimension)

- **Invariants:** Every white-paper rule should exist as executable code with typed results (e.g., `CycleError`, `CoverageGap { lo, missing }`, `FadeabilityIssue { assessment, supports }`, `GranularityWarning { node, kind }`, `EvidenceMissing { id }`). Mutations should trigger relevant validators; strict mode upgrades warnings to errors.
- **Telemetry:** Graph mutations and validations should emit timing, counts, and status to rerun under stable prefixes like `metrics/graph/validate/*` and `metrics/graph/mutate/*`, with tags for graph_version, course_commit, node/edge counts.
- **Tool UX:** Graph tools should mirror preview/body handshake, include byte/token hints, and surface narrowing suggestions (“filter by slug”, “limit edge kind”, “set fetch_body=true”). Bodies should carry byte hints so the gateway can price token usage.
- **Errors:** LLM/user errors should be structured codes (`cycle_detected`, `invalid_slug`, `unknown_node`, `denied_unknown_fields`, `strict_quality_violation`, `fadeability_blocked`) with concise remediation hints and involved IDs/paths.
- **Concurrency:** Expensive analyses (toposort, transitive reduction, reachability sweeps, fadeability checks) should run off the actor thread with bounded parallelism (rayon or buffered streams) and predictable resource limits.
- **Separation of concerns:** Keep persistence/versioning separate from validation and queries; GraphManager APIs should be semantic (e.g., `add_requires_edge` runs validation + persistence) rather than exposing raw petgraph handles.
- **Reproducibility:** Snapshots must include graph_version, schema_version, and course_commit. Load should verify version alignment or force migration; source refs must match the pinned commit and be immutable. No hand-editing snapshots.
- **Type-encoded pedagogy:** Use enums/newtypes for domain concepts (`SupportKind`, `AnchorImpact`, `AssessmentScope`, `GrainLevel`, `NodeId`) instead of raw strings. Provide defaults only where the white paper allows; otherwise require explicit caller input.
- **Safety defaults:** Strict quality mode should be easy to enable; default mode still reports all findings. Supports must remain fadeable; if fadeability cannot be proven, return a warning/error instead of accepting silently.

## What the white paper adds (must be reflected in code)

- **Pedagogical semantics:** Edge meanings, alignment predicate, fadeability, coverage, example minimums, granularity, and source-ref requirements are contractual, not advisory.
- **Evidence discipline:** Source refs are structured (path, line range, revision); assessments need observation features aligned to rubric criteria; supports need `intended_effect` and remain fadeable.
- **Validation set:** DAG acyclicity, LO reachability, coverage, purity, example variety, procedural practice, grain audits, keystone alerts, fadeability, source-ref guards, version pinning.
- **Minimal ontology:** Only the defined node/edge kinds; no ad-hoc extensions; tags over new types.
- **Borrow-ahead/discourse separation:** Teaching steps and anchors are distinct from prerequisite DAG; discourse is acyclic per episode.

## Gaps this bar exposes for the graph stack (hypotheses to verify in Step 2)

- **Semantic thinness:** Graph tools may expose petgraph-style primitives instead of semantic operations (violates cognitive economy).
- **Validator drift:** White-paper invariants may be partially implemented or only documented, not enforced.
- **Observability hole:** Graph mutations/validations likely lack rerun metrics and structured tracing.
- **Boundary laxity:** Inputs may skip `deny_unknown_fields`, path sandboxing, or structured errors.
- **Blocking work:** Heavy graph ops may run in actor handlers without bounding or offloading.
- **God objects:** Persistence, validation, and query logic may be intermingled.

## Concrete standards to apply to the graph side

- **Actor & concurrency:** GraphManager and friends must keep handlers short, offload heavy checks, and bound parallelism; no synchronous petgraph crunching inside the mailbox without guarding.
- **Tool design:** Graph tools should mirror preview/body handshake, carry token/byte hints, and use semantic verbs ("define_learning_outcome", "link_requires_with_rationale") instead of low-level edges.
- **Schema rigor:** All graph-facing payloads use `JsonSchema + deny_unknown_fields`; IDs validated; path inputs canonicalized within workspace; depth/iteration limits enforced for recursive ops.
- **Executable invariants:** Implement white-paper checks as functions with typed error variants; hook them into mutations and expose strict vs warning mode via `graph_strict_quality`.
- **Observability:** Add rerun metrics for validation passes, mutation latency, counts of rejected edges, coverage gaps, prune results; structured logs with graph_version, course_commit, edge/node counts.
- **Reproducible state:** Snapshots include graph version and course commit; migrations explicit; no hand edits to JSON snapshots.
- **User feedback:** Errors and previews carry actionable hints ("edge rejected: cycle via {path}", "missing rubric criteria: ..."), consistent with tool error shape.

## Tensions to respect (from the intentional stack)

- **Complexity for control:** The actor model and preview ergonomics add boilerplate; acceptable if they buy boundedness and observability. Graph fixes should not simplify away these controls.
- **Typed inputs vs `anyhow`:** Keep strict typing at the LLM/user edge; internal paths may stay on `anyhow` but must log context.
- **Boilerplate tolerance:** Some duplication (Message structs, builders) is fine; avoid “magical” shortcuts that bypass safety.

## Quality bar in plain language

Defensive-by-default, schema-driven, actor-oriented code that:

- rejects malformed inputs early and informatively,
- keeps expensive operations opt-in with clear cost previews,
- measures and logs what it does,
- encodes pedagogical semantics in types and validators,
- persists state reproducibly,
- and stays small, modular, and predictable.

## Next step

Use this bar to walk the graph stack (Step 2), confirm which invariants are missing or weak, and map violations to concrete local fixes.
