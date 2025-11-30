# Validation Matrix

This table maps white-paper rules to their enforcement level and the code paths that implement them. Levels show **default** behavior (strict flag off) and behavior when `--graph-strict-quality` is enabled (CI should set this flag).

| Rule | Level (default / strict) | Location |
| --- | --- | --- |
| Requires layer DAG | error / error | `src/graph/service.rs::validate_global_invariants` |
| SourceRef revision must equal `graph_course_commit` | error / error | `GraphService::validate_revision` + snapshot load |
| Non-empty statements on nodes | error / error | `GraphService::add_*_node` |
| Fadeability (supports cannot carry prerequisites) | error / error | `analysis::fadeability_issues` via `validate_global_invariants` |
| LO reachability (first principles → assessment → assesses target) | warn / error | `analysis::lo_reachability` in `validate_global_invariants` |
| Rubric coverage (observation_features cover rubric_criteria) | warn / error | `analysis::coverage_report` in `validate_global_invariants` |
| Rubric drift (criteria hash change) | warn / error | `validate_global_invariants` (hash cache) |
| Purity / extraneous prerequisites | error / error | `validate_global_invariants` + `analysis::extraneous_knowledge` |
| Procedural practice (requires→assessment→assesses target) | warn / error | `analysis::procedural_practice_gaps` |
| Example minimums / variety | warn / error | `analysis::example_gaps` |
| Supports `case_tag` required | error / error | `src/graph/specs.rs::SupportsSpec::validate` |
| High intrinsic_load targets require `coverage_tags` | error / error | `SupportsSpec::validate` + `validate_global_invariants` |
| Borrow-ahead severity ≥ CrossEpisode | error / error | `analysis::borrow_ahead` in `validate_global_invariants` |
| Missing introduce anchors for in-course knowledge | warn / error | `validate_global_invariants` |
| TeachingStep must have anchor or rationale | warn / error | `validate_global_invariants` |
| Discourse orphans (no precedes links) | warn / error | `analysis::discourse_orphans` |

Strict mode simply promotes warnings to errors; it does **not** downgrade any errors. For CI, run with `--graph-strict-quality` (or set the flag in configs) to make all warnings blocking.
