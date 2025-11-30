# PR Checklist (quality + tooling)

- [ ] Run `cargo fmt`.
- [ ] Run `cargo clippy --all-targets -- -D warnings`.
- [ ] Run `cargo test` (consider `PROPTEST_CASES=128` for sweeps).
- [ ] If graph changes: review validation matrix (`docs/validation_matrix.md`) and run with `--graph-strict-quality` to ensure alignment, coverage, purity, practice, discourse, and provenance checks pass.
- [ ] Check snapshots/backfill: use `cargo run --bin backfill_validation -- <snapshot>` to fill missing `case_tag`/`coverage_tags` and rationales; resolve any revision mismatches it reports.
