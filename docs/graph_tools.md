# Graph Tool Surface (LLM-facing)

This repo now distinguishes **views** (paged inspection) from **computations** (summaries). All graph tool outputs include `type` and `tool` fields for easy routing:

- `type: "graph_view"` — bounded lists with `limit`, `offset`, `has_more`.
- `type: "graph_analysis"` — summaries/analytics (often cost‑aware).
- `type: "graph_command"` — mutating operations.

Pagination defaults: `limit=50`, max `200`; `offset` defaults to `0`.

Cost/preview: summary tools accept `fetch_body` (default `false`). In preview mode they return a cost block; set `fetch_body=true` to receive the body.
Command previews also return a cost block with byte/token hints so the gateway can price mutations before execution.

## Tool quick reference

| id | kind | purpose | key params |
| --- | --- | --- | --- |
| `graph_neighbors` | view | Neighbors of a slug, filtered by edge kind/direction | `slug`, `edge_kind` (requires/supports/assesses/precedes/anchors), `direction` (incoming/outgoing/both), `limit`, `offset` |
| `graph_get_node` | view | Fetch a node with typed fields by slug | `slug` |
| `graph_list_nodes_by_tag` | view | Nodes that include a given tag (case-insensitive) | `tag`, `limit`, `offset` |
| `graph_list_nodes_by_kind` | view | Nodes filtered by knowledge_type, any_knowledge, or teaching_step | `selector`, `limit`, `offset` |
| `graph_list_tags` | view | Deduped list of all tags in the graph | — |
| `graph_search_nodes` | view | Fuzzy search across slug/title/statement | `query`, `limit` |
| `graph_first_principles` | view | Paged list of first-principle instructional nodes | `limit`, `offset` |
| `graph_first_principles_summary` | analysis | Counts of first principles by knowledge type | `fetch_body` |
| `graph_lo_reachability` | view | Reachability/coverage bundle for an LO (with anchors) | `lo_slug`, `limit`, `offset` |
| `graph_lo_coverage` | view | Coverage/criteria view for an LO (paged) | `lo_slug`, `limit`, `offset` |
| `graph_lo_alignment_summary` | analysis | Compact LO alignment summary (reachability, coverage, anchors) | `lo_slug`, `fetch_body` |
| `graph_lo_assessments_view` | view | Assessments for an LO with reachability flag | `lo_slug`, `reachable_only` (bool), `limit`, `offset` |
| `graph_lo_missing_criteria_view` | view | Missing rubric criteria for an LO | `lo_slug`, `limit`, `offset` |
| `graph_lo_anchors_view` | view | Anchoring teaching steps for an LO | `lo_slug`, `limit`, `offset` |
| `graph_gap_summary` | analysis | Counts of example gaps, fadeability issues, practice gaps | `fetch_body` |
| `graph_example_gaps_view` | view | Nodes failing example/variety rules | `limit`, `offset` |
| `graph_fadeability_view` | view | Assessments that fail fadeability | `limit`, `offset` |
| `graph_practice_gaps_view` | view | Procedural nodes lacking assessment practice | `limit`, `offset` |
| `graph_keystone` | analysis | Top keystone scores (capped at 20 entries) | — |
| `graph_dag_check` | analysis | Requires DAG boolean + topo length | — |
| `graph_extraneous` | analysis | Extraneous knowledge for assessment vs LO | `assessment_slug`, `lo_slug`, optional `intended_slugs` |
| `graph_redundant_requires` | analysis | List/prune redundant requires edges | `prune`, `limit`, `offset`, `apply` |
| `graph_assessment_gaps` | analysis | LOS w/o target assessments; orphan/unreachable assessments | `fetch_body` |
| `graph_discourse_orphans` | analysis | TeachingSteps lacking precedes links | `episode` (optional) |
| `graph_borrow_ahead` | analysis | Borrow-ahead uses within an episode | `episode` |
| `graph_analysis_cache_clear` | admin | Clear all cached graph analyses (ops) | — |
| `graph_edge_conflicts` | view | Edges carrying queued conflict payloads | `limit`, `offset` |

### Commands (mutations)
All command outputs now follow `{type: "graph_command", tool: <id>, status: "ok", ...}` and may echo key fields (e.g., slugs, path):

- `graph_insert_knowledge`, `graph_update_knowledge`
- `graph_insert_teaching_step`, `graph_update_teaching_step`
- `graph_add_requires`, `graph_add_supports`, `graph_add_assesses`, `graph_add_precedes`, `graph_add_anchors`
- `graph_rename_node`, `graph_remove_node`
- `graph_save_now`, `graph_load_snapshot`
- `graph_resolve_edge_conflict`

### Limits and safety notes
- Views are always bounded (`limit`/`offset`); use `has_more` to paginate.
- Summaries are compact; heavy tools provide previews unless `fetch_body=true`.
- Edge kind filters use the `EdgeKindFilter` enum—stringly values are rejected.
- Command preflight now resolves slugs with type guards: requires/supports expect knowledge nodes; assesses expects assessment_item -> learning_outcome; precedes/anchors ensure teaching_step sources (and anchors require knowledge/LO/assessment targets) before applying mutations.
- Strict mode: `--graph-strict-quality` promotes alignment/coverage/practice/example/discourse warnings to errors. Use it in CI; leave it off locally while staging edits.
- Supports: all supports must carry a `case_tag`; supports into high `intrinsic_load` targets must include `coverage_tags`.
- Purity & provenance: extraneous prerequisites and SourceRef revision mismatches are fatal regardless of strictness.
- Teaching steps: when unanchored, provide `rationale` (new optional field on insert/update teaching-step commands).
- Non-semantic algorithm tools (`graph_requires_pagerank`, cycles/bridges/articulation/shortest-path) are gated behind `WEAVER_DEBUG_GRAPH_ALGORITHMS=1` and stay out of the curated surface by default.

## Actor persistence
- GraphManager snapshots live under `<graph_snapshot_path>.state/graph_manager` (derived from the
  CLI `--graph-snapshot-path`) and are restored on startup before autosave.
- LLMGateway metrics persist under `<graph_snapshot_path>.state/llm_gateway` alongside the graph
  state.
- FileReader actors are intentionally ephemeral; conversation IDs use UUIDs so they remain unique
  across restarts, and no `.state/file_reader` snapshot is kept.
- Autosave ticks persist to the kameo state directories only; run `graph_save_now` when you need a
  legacy `graph_snapshot.json` dump.
- Analysis cache: see `docs/cache_coverage.md` for which graph tools share cached analyses.

## Cache guardrails
- When adding memoization (e.g., Foyer/Moka), key entries by `(graph_version, query_kind, args)` so
  graph mutations automatically invalidate cached results.
