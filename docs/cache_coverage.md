# Analysis cache coverage

The LLM graph tools reuse an in-process `AnalysisCache` keyed by `(graph_version, AnalysisKind)`.
Current coverage:

- `LoBundle { lo_slug }` → shared by `graph_lo_alignment_summary`, `graph_lo_reachability`,
  `graph_lo_assessments_view`, `graph_lo_coverage`, `graph_lo_missing_criteria_view`.
- `GapBundle` → shared by `graph_gap_summary`, `graph_example_gaps_view`,
  `graph_fadeability_view`, `graph_practice_gaps_view`.
- `Keystone` → used by `graph_keystone`.
- `AssessmentGaps` → used by `graph_assessment_gaps`.
- `DagCheck` → used by `graph_dag_check`.
- `BorrowAhead {episode}` → used by `graph_borrow_ahead`.
- `DiscourseOrphans {episode}` → used by `graph_discourse_orphans`.
- Requires-layer analyses:
  - `RequiresCycles` → `graph_requires_cycles` (stores top 500 SCCs, limits applied at read time)
  - `RequiresBridges` → `graph_requires_bridges`
  - `RequiresArticulation` → `graph_requires_articulation`
  - `RequiresFeedback` → `graph_requires_feedback_arcs`
  - `RequiresPagerank {damping_bits, iterations}` → `graph_requires_pagerank` (stores top 500)
  - `RequiresShortestPath {from, to}` → `graph_requires_shortest_path`

Notes:
- Bundles cache only the analysis payloads; previews still recompute cost/byte hints per request.
- Bundle keys include `graph_version`, so any mutation invalidates prior entries automatically.
- Use `GetGraphWithVersion` for tools that need a consistent `(graph, version)` pair.
