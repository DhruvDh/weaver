# Repository Guidelines
>
> Read `white_paper.md` first; it is the modeling contract.

## Project Structure & Module Organization

- `src/main.rs` wires CLI, scheduler, graph manager, LLM gateway, autosave, rerun.
- Modules: `graph/`, `analysis/`, `schema/`, `tools/llm/`, `constants.rs`, `llm_gateway.rs`.
- Tests: `tests/analysis_unit.rs`, `tests/analysis_props.rs` (all tests and test harnesses live under `tests/`). References: `docs/graph_tools.md`, `white_paper.md`, `graph_snapshot.json`, `uncc_cs2-pretext-project/`.
- Target toolchain: nightly (see `rust-toolchain.toml`).

## Build, Test, and Development Commands

- `cargo fmt` — format per `rustfmt.toml` (run every time).
- `cargo clippy --all-targets` — lint; fail on warnings (run every time).
- `cargo test` — unit + property tests; `PROPTEST_CASES=128` for sweeps (run every time).
- `cargo test --all --locked` — includes the CLI lifecycle integration (`cli_lifecycle`) that covers autosave clamping and `--skip-demo`.
- `cargo run -- <workspace-root>` — launch FileReader + graph pipeline (default `uncc_cs2-pretext-project/`).
- Config is CLI-only (no GRAPH_* env fallback). Key flags:
  - `--graph-snapshot-path <path>` (legacy JSON snapshot path; default `graph_snapshot.json`)
  - `--graph-autosave-secs <u64>` (default 300; validated >=1 and clamped to ≥5s)
  - `--graph-course-commit <hash>` (embedded in snapshots; snapshot commit wins by default, overrides warn on mismatch)
  - `-q|--graph-strict-quality` (promote warnings to errors)
  - `--graph-prune-requires-secs <u64>` (optional; disables when omitted)
  - `--rerun-mode grpc|file|both|none` (default grpc), `--rerun-file <path>` (default `weaver.rrd`)
  - `--skip-demo` (omit the startup FileReader run; useful for headless/CI)
  - positional `<workspace-root>` (default `uncc_cs2-pretext-project/`)
  - Env retained only for OpenAI (`OPENAI_MODEL` required, `OPENAI_API_BASE` optional).

## Modeling Basics (from the white paper)

- Edge layers: `requires` (acyclic), `supports` (scaffolds; no self-loops; fadeable), `assesses` (Assessment → LO with `evidence_link` + `scope`).
- Discourse: TeachingSteps use `precedes` (episode DAG) and `anchors` with `impact ∈ {introduce,use,refine,motivate,target}`.
- Alignment: each LO needs an assessment reachable from first principles via `requires*` and `assesses(..., scope=target)`.
- Example minimums: procedural need ≥2 worked examples (`typical` + `edge/error_case`) plus a practice assessment; conceptual need illustration + contrast; metacognitive need `strategy_hint` + reflection. Supports set `support_kind`, `case_tag`, `coverage_tags`, `intended_effect`.
- Source refs are mandatory on nodes and edges (path, line range, git revision); never store textbook text.
- Supports must stay fadeable: they cannot be the only path enabling an assessment; validators will reject if they carry prerequisite load.
- `GraphService` enforces invariants on every mutation (requires DAG, coverage, example minimums, practice links, fadeability). `GRAPH_STRICT_QUALITY=1` upgrades warnings to errors—use it locally to catch issues early.

## Coding Style & Naming

- Rust norms: `snake_case` items, `CamelCase` types, `SCREAMING_SNAKE_CASE` consts; keep lines ≤100 chars and grouped imports.
- Use `anyhow::Result` at the CLI boundary; prefer `thiserror` in libraries.
- Slugs for graph content: `{kind}.{short_name}` (see white paper §9).

## Runtime & Actors

- Actor-first (kameo); long-lived components should be `Actor`s so restarts are cheap and state stays isolated.
- Assume week-long uptime: no panics/`unwrap`/`expect`; propagate errors and let supervisors recover. Keep handlers idempotent for replays.
- Scheduler ticks: autosave every `--graph-autosave-secs`; optional redundant-requires pruning via `--graph-prune-requires-secs`. Persistence goes through `GraphManager`; tolerate partial writes and use bounded retries.
- `LLMGateway` is rate-limited by `LLM_MAX_CONCURRENT_REQUESTS`; it retries with backoff and logs token usage. Do not block within actor handlers.
- Rerun telemetry is optional; enable with `WEAVER_RERUN_MODE=grpc|file|both` and check `.rrd` artifacts before sharing.

## Analysis Cache (LLM graph tools)

- Treat `AnalysisCache` as ephemeral, in-process derived state (not an actor, not persisted). Keys are `(graph_version, AnalysisKind)`; always fetch `(graph, version)` together via `load_graph_with_version` to build keys.
- Use the cache helpers: `get_or_insert_with` for sync callers and `get_or_insert_with_async` for async. Both dedupe per key; no extra locks needed.
- Heavy work goes in `spawn_blocking` inside the compute closure so the async runtime and cache paths stay non-blocking. Never hold locks while doing the analysis itself.
- Cache only analysis payloads; previews still recompute byte/token hints. Graph mutations automatically bypass stale entries because `graph_version` changes.

## LLM Tools, Schemas, and Previews

- Tool args derive `serde::Deserialize`, `schemars::JsonSchema`, and `bon::Builder`; keep `#[serde(deny_unknown_fields)]` so unknown fields fail fast. Extend schemas via `schema_for_args::<T>()` to keep the OpenAI function specs in sync.
- File/graph tools are preview-first: default responses return a cost preview (`{type:"preview", mode:"preview"}`) with byte/token estimates; set `fetch_body=true` (or tool-specific flags) only when the payload is small enough. Respect context budget; previews include remaining-token estimates using `WEAVER_CONTEXT_LIMIT` (default 131_072 tokens).
- Provide `byte_hint` in `ToolOutput` for large payloads so the gateway can price token usage accurately.
- Path resolution is sandboxed: `resolve_workspace_path` rejects escapes outside the workspace. Always pass workspace-relative paths.
- Delegation: `delegate_tasks` fans out to child FileReaders, capped by `MAX_PARALLEL_DELEGATIONS` (8) and `DEFAULT_MAX_SUBDELEGATIONS` (6). Depth checks prevent runaway recursion.
- When adding new graph tools, expose them via `CURATED_TOOL_IDS` (Graph LLM surface) and document behavior/params in `docs/graph_tools.md`, including whether they preview or stream full bodies.

## Data Integrity & Validation

- Use `GraphManager` APIs for all mutations; they enforce duplicate-slug guards, endpoint rules (`requires` can’t originate from LOs/assessments, `supports` forbids self-loops, `assesses.claim` must equal target slug), and rerun validations on incident edges after updates.
- Evidence is required: `requires` needs `rationale` + `evidence_refs`; `supports` needs `evidence_refs`; `assesses` needs non-empty `observation_features`. Missing spans fail validation.
- Source refs must include path, start/end lines, and a git hash (7–40 hex). Mismatched revisions should be fixed by regenerating the snapshot for the new commit, not by editing spans.
- Renames: use the `graph_rename_node` tool or `GraphService::rename_node` so `assesses.claim` stays aligned with the target LO.
- Persistence: `graph_save_now` writes to a temp file then fsync/rename; don’t hand-edit `graph_snapshot.json`. `GRAPH_SNAPSHOT_PATH` and `GRAPH_COURSE_COMMIT` keep snapshots tied to the course revision.
- Context limits: `WEAVER_CONTEXT_LIMIT` (default 131_072 tokens) governs preview budgeting; supply `byte_hint` on large tool outputs to keep token estimates stable.

## Testing Guidelines

- Unit tests near touched code; property tests in `tests/` (`*_unit.rs`, `*_props.rs`).
- Add pass/fail cases for new invariants (cycle rejection, fadeable supports).
- Cap/seed proptests for CI stability; extend `analysis::*_gaps` tests when changing coverage logic.
- Validate new graph edits against `GraphService` invariants: requires DAG, example minimums, procedural practice, fadeability, coverage. Enable `GRAPH_STRICT_QUALITY=1` locally to surface warnings as errors.

## Commit & Pull Request Guidelines

- Commit messages: short, imperative, no trailing period.
- PRs: state intent, link issues, list commands run (`fmt`, `clippy`, `test`), and note new env vars or migrations; include a sample `cargo run` if behavior changes.
- When modifying graph tooling, update `CURATED_TOOL_IDS` in `src/tools/llm/graph_tools/mod.rs` to expose new tools to the assistant, and document them in `docs/graph_tools.md`.

## Security & Configuration

- Keep `OPENAI_*` secrets out of commits; load via env or `.env`. Avoid sharing `graph_snapshot.json` or rerun captures (`weaver.rrd`) without redaction.
- Snapshots are pinned to `GRAPH_COURSE_COMMIT`; regenerate rather than hand-edit when course sources change. Source refs must cite that commit hash.
