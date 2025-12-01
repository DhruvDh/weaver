use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use futures::future::join_all;
use serde_json::json;
use weaver::tools::llm::analysis_cache::{AnalysisCache, AnalysisCacheKey, AnalysisKind};

#[test]
fn analysis_cache_reuses_entries_for_same_key() {
    let cache = AnalysisCache::new();
    let key = AnalysisCacheKey {
        graph_version: 1,
        kind:          AnalysisKind::GapBundle,
    };

    let mut calls = 0;

    let first = cache.get_or_insert_with(key.clone(), || {
        calls += 1;
        json!({ "value": 1 })
    });

    let second = cache.get_or_insert_with(key, || {
        calls += 1;
        json!({ "value": 2 })
    });

    assert_eq!(calls, 1);
    assert_eq!(first.payload, second.payload);
}

#[test]
fn analysis_cache_deduplicates_concurrent_sync_calls() {
    let cache = Arc::new(AnalysisCache::new());
    let key = AnalysisCacheKey {
        graph_version: 2,
        kind:          AnalysisKind::Keystone,
    };
    let calls = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..16 {
        let cache = Arc::clone(&cache);
        let key = key.clone();
        let calls = Arc::clone(&calls);
        handles.push(thread::spawn(move || {
            let _ = cache.get_or_insert_with(key, || {
                calls.fetch_add(1, Ordering::SeqCst);
                json!({ "value": 42 })
            });
        }));
    }

    for handle in handles {
        handle.join().expect("thread failed");
    }

    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn analysis_cache_deduplicates_concurrent_async_calls() {
    let cache = Arc::new(AnalysisCache::new());
    let key = AnalysisCacheKey {
        graph_version: 3,
        kind:          AnalysisKind::DagCheck,
    };
    let calls = Arc::new(AtomicUsize::new(0));

    let mut tasks = Vec::new();
    for _ in 0..16 {
        let cache = Arc::clone(&cache);
        let key = key.clone();
        let calls = Arc::clone(&calls);
        tasks.push(tokio::spawn(async move {
            let _ = cache
                .get_or_insert_with_async(key, || {
                    let calls = Arc::clone(&calls);
                    async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        json!({ "value": 99 })
                    }
                })
                .await;
        }));
    }

    join_all(tasks).await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn analysis_cache_bounds_versions_per_kind() {
    let cache = AnalysisCache::new();
    let kind = AnalysisKind::GapBundle;

    for version in 0..10 {
        let key = AnalysisCacheKey {
            graph_version: version,
            kind:          kind.clone(),
        };
        cache.get_or_insert_with(key, || json!({ "version": version }));
    }

    let versions = cache.versions_for_kind(&kind);
    assert!(
        versions.len() <= 2,
        "expected at most two cached versions per kind, got {versions:?}"
    );
    assert_eq!(versions, vec![8, 9]);
}

#[test]
fn analysis_cache_honors_ttl() {
    let cache = AnalysisCache::with_limits(usize::MAX, Duration::from_millis(5));
    let kind = AnalysisKind::GapBundle;
    let first = AnalysisCacheKey {
        graph_version: 1,
        kind:          kind.clone(),
    };
    cache.get_or_insert_with(first, || json!({ "version": 1 }));

    thread::sleep(Duration::from_millis(10));

    let second = AnalysisCacheKey {
        graph_version: 2,
        kind:          kind.clone(),
    };
    cache.get_or_insert_with(second, || json!({ "version": 2 }));

    let versions = cache.versions_for_kind(&kind);
    assert_eq!(versions, vec![2], "expected stale entries to be dropped after TTL expiry");
}

#[test]
fn analysis_cache_enforces_capacity() {
    let cache = AnalysisCache::with_limits(3, Duration::from_secs(60));
    let keys = vec![
        AnalysisCacheKey {
            graph_version: 1,
            kind:          AnalysisKind::GapBundle,
        },
        AnalysisCacheKey {
            graph_version: 1,
            kind:          AnalysisKind::Keystone,
        },
        AnalysisCacheKey {
            graph_version: 1,
            kind:          AnalysisKind::DagCheck,
        },
        AnalysisCacheKey {
            graph_version: 1,
            kind:          AnalysisKind::AssessmentGaps,
        },
    ];

    for (idx, key) in keys.into_iter().enumerate() {
        cache.get_or_insert_with(key, || json!({ "slot": idx }));
    }

    assert!(
        cache.len() <= 3,
        "capacity enforcement should evict least-recently-used entries"
    );
}
