// ------------------------------
// Nodes
// ------------------------------

N::Knowledge {
  INDEX slug: String,

  title: String,
  statement: String,
  knowledge_type: String,
  source_refs: [String],
  confidence: F32 DEFAULT 0.7,
  rubric_criteria: [String],
  construct_irrelevant_demands: [String],
  grain_level: String DEFAULT "mid",
  intrinsic_load: String DEFAULT "medium",
  introduction_scope: String DEFAULT "in_course",
  tags: [String]
}

N::TeachingStep {
  INDEX slug: String,

  title: String,
  statement: String,
  episode: String,
  purpose: String,
  method_tags: [String],
  source_refs: [String],
  confidence: F32 DEFAULT 0.7
}

// ------------------------------
// Edges
// ------------------------------

E::Requires {
  From: Knowledge,
  To:   Knowledge,
  Properties: {
    strength: String,
    rationale: String,
    evidence_refs: [String],
    confidence: F32 DEFAULT 0.7
  }
}

E::Supports {
  From: Knowledge,
  To:   Knowledge,
  Properties: {
    support_kind: String,
    intended_effect: String,
    case_tag: String,
    coverage_tags: [String],
    evidence_refs: [String],
    confidence: F32 DEFAULT 0.7
  }
}

E::Assesses {
  From: Knowledge,
  To:   Knowledge,
  Properties: {
    claim: String,
    observation_features: [String],
    scope: String,
    confidence: F32 DEFAULT 0.7
  }
}

E::Anchors {
  From: TeachingStep,
  To:   Knowledge,
  Properties: {
    impact: String
  }
}

E::Precedes {
  From: TeachingStep,
  To:   TeachingStep,
  Properties: { }
}
