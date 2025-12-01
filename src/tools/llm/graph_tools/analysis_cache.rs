use std::sync::{Arc, OnceLock};

use dashmap::DashMap;
use serde_json::Value;
use tokio::sync::OnceCell;

const VERSION_BUDGET_PER_KIND: usize = 2;

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

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn entry(&self, key: &AnalysisCacheKey) -> Arc<CacheEntry> {
        self.inner
            .entry(key.clone())
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

        let entry = self.entry(&key);
        let value = entry
            .sync_value
            .get_or_init(|| Arc::new(AnalysisCacheValue { payload: compute() }))
            .clone();

        let _ = entry.async_value.set(Arc::clone(&value));

        self.evict_for_key(entry, &value, &key);
        value
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

        let entry = self.entry(&key);

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

        let cached = entry.sync_value.get().map(Arc::clone).unwrap_or(value);
        self.evict_for_key(entry, &cached, &key);
        cached
    }

    pub async fn get_or_try_insert_with_async<F, Fut, E>(
        &self,
        key: AnalysisCacheKey,
        compute: F,
    ) -> Result<Arc<AnalysisCacheValue>, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<serde_json::Value, E>> + Send + 'static,
    {
        if let Some(existing) = self.try_get(&key) {
            return Ok(existing);
        }

        let entry = self.entry(&key);

        if let Some(existing) = entry.sync_value.get() {
            return Ok(Arc::clone(existing));
        }

        let payload = compute().await?;
        let value = Arc::new(AnalysisCacheValue { payload });

        let _ = entry.sync_value.set(Arc::clone(&value));
        let _ = entry.async_value.set(Arc::clone(&value));

        let cached = entry.sync_value.get().map(Arc::clone).unwrap_or(value);
        self.evict_for_key(entry, &cached, &key);
        Ok(cached)
    }

    pub fn prune_for_version(&self, current_version: u64) -> usize {
        let mut evicted = 0usize;
        let window = VERSION_BUDGET_PER_KIND.saturating_sub(1) as u64;
        self.inner.retain(|key, _| {
            let keep = key.graph_version + window >= current_version;
            if !keep {
                evicted += 1;
            }
            keep
        });
        if evicted > 0 {
            tracing::debug!(
                target: "weaver.analysis_cache",
                current_version,
                evicted,
                size_after = self.inner.len(),
                "pruned analysis cache versions outside retention window"
            );
        }
        evicted
    }

    pub fn versions_for_kind(&self, kind: &AnalysisKind) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .inner
            .iter()
            .filter_map(|entry| {
                if &entry.key().kind == kind {
                    Some(entry.key().graph_version)
                } else {
                    None
                }
            })
            .collect();
        out.sort_unstable();
        out
    }

    fn evict_for_key(
        &self,
        entry: Arc<CacheEntry>,
        cached: &Arc<AnalysisCacheValue>,
        key: &AnalysisCacheKey,
    ) {
        // ensure backing entry is populated for any concurrent readers
        let _ = entry.sync_value.set(Arc::clone(cached));
        let _ = entry.async_value.set(Arc::clone(cached));

        let mut versions: Vec<u64> = self
            .inner
            .iter()
            .filter_map(|item| {
                if item.key().kind == key.kind {
                    Some(item.key().graph_version)
                } else {
                    None
                }
            })
            .collect();

        if versions.len() <= VERSION_BUDGET_PER_KIND {
            return;
        }

        versions.sort_unstable_by(|a, b| b.cmp(a)); // newest first
        let mut evicted = 0usize;
        for old in versions.into_iter().skip(VERSION_BUDGET_PER_KIND) {
            let drop_key = AnalysisCacheKey {
                graph_version: old,
                kind:          key.kind.clone(),
            };
            if self.inner.remove(&drop_key).is_some() {
                evicted += 1;
            }
        }

        if evicted > 0 {
            tracing::debug!(
                target: "weaver.analysis_cache",
                kind = ?key.kind,
                evicted,
                size_after = self.inner.len(),
                "evicted stale analysis cache versions for kind"
            );
        }
    }
}

/// Unified helper to prune to a specific graph version and fetch or compute an
/// analysis payload asynchronously. Returns only the cached payload to keep
/// call sites concise.
pub async fn with_cached_analysis<F, Fut>(
    cache: &AnalysisCache,
    key: AnalysisCacheKey,
    current_version: u64,
    compute: F,
) -> serde_json::Value
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = serde_json::Value> + Send + 'static,
{
    cache.prune_for_version(current_version);
    cache
        .get_or_insert_with_async(key, compute)
        .await
        .payload
        .clone()
}

/// Variant that propagates compute errors and only caches on success.
pub async fn with_cached_analysis_result<F, Fut, E>(
    cache: &AnalysisCache,
    key: AnalysisCacheKey,
    current_version: u64,
    compute: F,
) -> Result<serde_json::Value, E>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<serde_json::Value, E>> + Send + 'static,
{
    cache.prune_for_version(current_version);
    let value = cache.get_or_try_insert_with_async(key, compute).await?;
    Ok(value.payload.clone())
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
