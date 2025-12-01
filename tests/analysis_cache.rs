use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
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
