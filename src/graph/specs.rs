use crate::{
    analysis,
    graph::{model::*, service::GraphService},
    schema::types::{AssessmentScope, KnowledgeType},
};

/// Trait implemented per edge type to centralize validation and payload
/// construction. Implementations may use GraphService to enforce global
/// constraints (e.g., DAG guards).
pub trait EdgeSpec {
    type Attrs;
    const NAME: &'static str;

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        confidence: f32,
    ) -> Result<(), GraphError>;

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload;
}

pub struct RequiresSpec;
impl EdgeSpec for RequiresSpec {
    type Attrs = RequiresAttrs;
    const NAME: &'static str = "requires";

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        let (from_kt, to_kt) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => {
                (f.knowledge_type, t.knowledge_type)
            }
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        };

        crate::schema::validate::validate_requires(
            from_kt,
            to_kt,
            attrs.strength,
            &attrs.rationale,
            &attrs.evidence_refs,
        )
        .map_err(|e| GraphError::Schema(e.to_string()))?;

        if let Some(path) = svc.requires_path(to, from) {
            let cycle_slugs: Vec<String> = path
                .into_iter()
                .map(|id| svc.graph()[id].slug.clone())
                .collect();
            return Err(GraphError::RequiresCycle { cycle_slugs });
        }
        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload::new(EdgeKind::Requires(attrs), confidence)
    }
}

pub struct SupportsSpec;
impl EdgeSpec for SupportsSpec {
    type Attrs = SupportsAttrs;
    const NAME: &'static str = "supports";

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        if from == to {
            return Err(GraphError::Schema("supports self-loops are not allowed".to_string()));
        }
        let (from_kn, to_kn) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => (f, t),
            (NodeKind::Knowledge(f), NodeKind::TeachingStep(_)) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(&NodeKind::Knowledge(f.clone()))),
                    to:   Some(NodeKindPreview::TeachingStep),
                });
            }
            (NodeKind::TeachingStep(_), _) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::TeachingStep),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        };
        let from_kt = from_kn.knowledge_type;
        let to_kt = to_kn.knowledge_type;

        crate::schema::validate::validate_supports(
            from_kt,
            to_kt,
            attrs.support_kind,
            attrs.intended_effect,
            &attrs.evidence_refs,
        )
        .map_err(|e| GraphError::Schema(e.to_string()))?;

        if attrs.case_tag.is_none() {
            return Err(GraphError::Schema("supports.case_tag is required".to_string()));
        }

        if matches!(to_kn.intrinsic_load, Some(crate::graph::IntrinsicLoad::High))
            && attrs.coverage_tags.is_empty()
        {
            return Err(GraphError::Schema(
                "supports into high intrinsic_load targets must include coverage_tags".to_string(),
            ));
        }

        // Fadeability guard: adding this support must not create new
        // first-principle -> assessment reachability beyond the current graph.
        let fade_start = std::time::Instant::now();
        let fade_ctx = svc.fade_ctx();
        let carries_prereq =
            crate::analysis::support_would_break_fadeability(svc.graph(), &fade_ctx, from, to);
        let fade_elapsed_ms = fade_start.elapsed().as_secs_f64() * 1000.0;
        if fade_elapsed_ms > 10.0 {
            tracing::debug!(
                target: "weaver.graph.fadeability",
                elapsed_ms = fade_elapsed_ms,
                op = "supports_validate"
            );
        }
        if carries_prereq {
            return Err(GraphError::Schema(
                "support would carry prerequisite load (not fadeable)".to_string(),
            ));
        }

        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload::new(EdgeKind::Supports(attrs), confidence)
    }
}

pub struct AssessesSpec;
impl EdgeSpec for AssessesSpec {
    type Attrs = AssessesAttrs;
    const NAME: &'static str = "assesses";

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        let (from_kt, to_kt) = match (from_kind, to_kind) {
            (NodeKind::Knowledge(f), NodeKind::Knowledge(t)) => {
                (f.knowledge_type, t.knowledge_type)
            }
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        };

        crate::schema::validate::validate_assesses(from_kt, to_kt, &attrs.evidence_link)
            .map_err(|e| GraphError::Schema(e.to_string()))?;

        // evidence_link.claim must match target LO slug
        let target_slug = &svc.graph()[to].slug;
        if &attrs.evidence_link.claim != target_slug {
            return Err(GraphError::Schema(format!(
                "assesses.claim `{}` must equal target LO slug `{}`",
                attrs.evidence_link.claim, target_slug
            )));
        }

        if attrs.evidence_link.scope == AssessmentScope::Target {
            let intended = analysis::intended_knowledge_from_anchors(svc.graph(), to);
            let allows_extraneous = matches!(from_kind, NodeKind::Knowledge(k) if !k.construct_irrelevant_demands.is_empty());
            if intended.is_empty() && !allows_extraneous {
                return Err(GraphError::Schema(
                    "target assesses edges require intended knowledge anchors or an explicit \
                     construct_irrelevant_demands whitelist on the assessment"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload::new(EdgeKind::Assesses(attrs), confidence)
    }
}

pub struct PrecedesSpec;
impl EdgeSpec for PrecedesSpec {
    type Attrs = PrecedesAttrs;
    const NAME: &'static str = "precedes";

    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        match (from_kind, to_kind) {
            (NodeKind::TeachingStep(_), NodeKind::TeachingStep(_)) => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        }

        if let (NodeKind::TeachingStep(ts_from), NodeKind::TeachingStep(ts_to)) =
            (from_kind, to_kind)
            && (ts_from.episode != attrs.episode || ts_to.episode != attrs.episode)
        {
            return Err(GraphError::Schema(format!(
                "precedes episode `{}` must match both steps (`{}`, `{}`)",
                attrs.episode, ts_from.episode, ts_to.episode
            )));
        }

        if svc.has_precedes_path(to, from, &attrs.episode) {
            return Err(GraphError::Schema(
                "precedes edge would create a cycle in this episode".to_string(),
            ));
        }
        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload::new(EdgeKind::Precedes(attrs), confidence)
    }
}

pub struct AnchorsSpec;
impl EdgeSpec for AnchorsSpec {
    type Attrs = AnchorsAttrs;
    const NAME: &'static str = "anchors";

    /// Discourse anchors semantics:
    /// - Source must be TeachingStep.
    /// - impact = introduce/refine: target must be instructional knowledge (not
    ///   LO or assessment).
    /// - impact = target: target must be a LearningOutcome.
    /// - impact = use/motivate: general knowledge allowed; if target is an
    ///   assessment item, only `use` is permitted.
    fn validate(
        svc: &GraphService,
        from: NodeId,
        to: NodeId,
        attrs: &Self::Attrs,
        _confidence: f32,
    ) -> Result<(), GraphError> {
        let (from_kind, to_kind) = svc.node_kinds(Self::NAME, from, to)?;
        match from_kind {
            NodeKind::TeachingStep(_) => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::from_node(from_kind)),
                    to:   Some(NodeKindPreview::from_node(to_kind)),
                });
            }
        }

        match (&to_kind, attrs.impact) {
            (NodeKind::Knowledge(k), AnchorImpact::Introduce | AnchorImpact::Refine) => {
                if k.knowledge_type.is_learning_outcome() || k.knowledge_type.is_assessment_item() {
                    return Err(GraphError::Schema(
                        "introduce/refine anchors must target instructional knowledge".to_string(),
                    ));
                }
            }
            (NodeKind::Knowledge(k), AnchorImpact::Target) => {
                if k.knowledge_type != KnowledgeType::LearningOutcome {
                    return Err(GraphError::Schema(
                        "target anchors must point to learning_outcome nodes".to_string(),
                    ));
                }
            }
            (NodeKind::Knowledge(k), AnchorImpact::Use | AnchorImpact::Motivate) => {
                if k.knowledge_type.is_assessment_item() && attrs.impact != AnchorImpact::Use {
                    return Err(GraphError::Schema(
                        "anchors to assessment items must use impact=use".to_string(),
                    ));
                }
            }
            (NodeKind::TeachingStep(_), _) => {
                return Err(GraphError::InvalidEndpoints {
                    edge: Self::NAME,
                    from: Some(NodeKindPreview::TeachingStep),
                    to:   Some(NodeKindPreview::TeachingStep),
                });
            }
        }
        Ok(())
    }

    fn make_payload(attrs: Self::Attrs, confidence: f32) -> EdgePayload {
        EdgePayload::new(EdgeKind::Anchors(attrs), confidence)
    }
}

// Re-export spec types for callers that need them.
pub use AnchorsSpec as Anchors;
pub use AssessesSpec as Assesses;
pub use PrecedesSpec as Precedes;
pub use RequiresSpec as Requires;
pub use SupportsSpec as Supports;
