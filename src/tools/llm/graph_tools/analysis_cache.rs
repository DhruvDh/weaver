use std::sync::{Arc, OnceLock};

use dashmap::DashMap;
use serde_json::Value;
use tokio::sync::OnceCell;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum AnalysisKind {
    LoBundle {
        lo_slug: String,
    },
    GapBundle,
    Keystone,
    AssessmentGaps,
    DagCheck,
    BorrowAhead {
        episode: String,
    },
    DiscourseOrphans {
        episode: Option<String>,
    },
    RequiresCycles,
    RequiresBridges,
    RequiresArticulation,
    RequiresFeedback,
    RequiresPagerank {
        damping_bits: u64,
        iterations:   usize,
    },
    RequiresShortestPath {
        from: String,
        to:   String,
    },
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct AnalysisCacheKey {
    pub graph_version: u64,
    pub kind:          AnalysisKind,
}

#[derive(Debug, Clone)]
pub struct AnalysisCacheValue {
    pub payload: Value,
}

/// Simple in-process cache for graph analyses keyed by graph version and query
/// kind. This keeps repeated tool invocations from re-running expensive
/// analysis on an unchanged graph.
#[derive(Default)]
pub struct AnalysisCache {
    inner: DashMap<AnalysisCacheKey, Arc<CacheEntry>>,
}

#[derive(Default)]
struct CacheEntry {
    sync_value:  OnceLock<Arc<AnalysisCacheValue>>,
    async_value: OnceCell<Arc<AnalysisCacheValue>>,
}

impl AnalysisCache {
    pub fn new() -> Self {
        Self {
            inner: DashMap::new(),
        }
    }

    fn entry(&self, key: AnalysisCacheKey) -> Arc<CacheEntry> {
        self.inner
            .entry(key)
            .or_insert_with(|| Arc::new(CacheEntry::default()))
            .clone()
    }

    fn try_get(&self, key: &AnalysisCacheKey) -> Option<Arc<AnalysisCacheValue>> {
        self.inner.get(key).and_then(|entry| {
            entry
                .sync_value
                .get()
                .map(Arc::clone)
                .or_else(|| entry.async_value.get().map(Arc::clone))
        })
    }

    pub fn get_or_insert_with<F>(
        &self,
        key: AnalysisCacheKey,
        compute: F,
    ) -> Arc<AnalysisCacheValue>
    where
        F: FnOnce() -> Value,
    {
        if let Some(existing) = self.try_get(&key) {
            return existing;
        }

        let entry = self.entry(key);
        let value = Arc::new(AnalysisCacheValue { payload: compute() });

        // Populate both caches; whichever sets first wins.
        let _ = entry.sync_value.set(Arc::clone(&value));
        let _ = entry.async_value.set(Arc::clone(&value));

        entry.sync_value.get().map(Arc::clone).unwrap_or(value)
    }

    pub fn clear_for_version(&self, graph_version: u64) {
        self.inner
            .retain(|key, _| key.graph_version != graph_version);
    }

    pub fn clear_all(&self) {
        self.inner.clear();
    }

    /// Async helper to allow heavy computations off the main executor thread
    /// (e.g., via spawn_blocking) while still inserting into the cache exactly
    /// once.
    pub async fn get_or_insert_with_async<F, Fut>(
        &self,
        key: AnalysisCacheKey,
        compute: F,
    ) -> Arc<AnalysisCacheValue>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Value> + Send + 'static,
    {
        if let Some(existing) = self.try_get(&key) {
            return existing;
        }

        let entry = self.entry(key);

        // Fast-path in case a synchronous caller already populated the cache.
        if let Some(existing) = entry.sync_value.get() {
            return Arc::clone(existing);
        }

        let value = entry
            .async_value
            .get_or_init(|| async move {
                Arc::new(AnalysisCacheValue {
                    payload: compute().await,
                })
            })
            .await
            .clone();

        // Backfill the sync slot for future synchronous callers.
        let _ = entry.sync_value.set(Arc::clone(&value));

        value
    }
}

// Caching policy (keep in sync when adding tools):
// - LoBundle (graph_lo_alignment_summary, graph_lo_reachability,
//   graph_lo_assessments_view, graph_lo_coverage,
//   graph_lo_missing_criteria_view)
// - GapBundle (graph_gap_summary, graph_example_gaps_view,
//   graph_fadeability_view, graph_practice_gaps_view)
// - Keystone (graph_keystone)
// - AssessmentGaps (graph_assessment_gaps)
// - DagCheck (graph_dag_check)
// - BorrowAhead {episode} (graph_borrow_ahead)
// - DiscourseOrphans {episode} (graph_discourse_orphans)
// - RequiresCycles/Bridges/Articulation/Feedback/Pagerank/ShortestPath
//   (graph_requires_*)
