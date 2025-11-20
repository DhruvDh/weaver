// Lookup helpers -------------------------------------------------------------

QUERY GetKnowledgeBySlug(slug: String) =>
  node <- N<Knowledge>({ slug: slug })::RANGE(0, 1)
  RETURN node

QUERY GetTeachingStepBySlug(slug: String) =>
  step <- N<TeachingStep>({ slug: slug })::RANGE(0, 1)
  RETURN step

// Creation helpers -----------------------------------------------------------

QUERY CreateKnowledge(
  slug: String, title: String, statement: String,
  knowledge_type: String, source_refs: [String],
  confidence: F32, grain_level: String, intrinsic_load: String,
  introduction_scope: String, tags: [String],
  rubric_criteria: [String], construct_irrelevant_demands: [String]
) =>
  node <- AddN<Knowledge>({
    slug: slug, title: title, statement: statement,
    knowledge_type: knowledge_type, source_refs: source_refs,
    confidence: confidence, grain_level: grain_level,
    intrinsic_load: intrinsic_load, introduction_scope: introduction_scope,
    tags: tags,
    rubric_criteria: rubric_criteria,
    construct_irrelevant_demands: construct_irrelevant_demands
  })
  RETURN node

// Typed creators -----------------------------------------------------------

QUERY CreateFactual(
  slug: String, title: String, statement: String,
  source_refs: [String], confidence: F32
) =>
  node <- AddN<Knowledge>({
    slug: slug, title: title, statement: statement,
    knowledge_type: "factual",
    source_refs: source_refs,
    confidence: confidence,
    grain_level: "mid",
    intrinsic_load: "medium",
    introduction_scope: "in_course"
  })
  RETURN node

QUERY CreateConceptual(
  slug: String, title: String, statement: String,
  source_refs: [String], confidence: F32
) =>
  node <- AddN<Knowledge>({
    slug: slug, title: title, statement: statement,
    knowledge_type: "conceptual",
    source_refs: source_refs,
    confidence: confidence,
    grain_level: "mid",
    intrinsic_load: "medium",
    introduction_scope: "in_course"
  })
  RETURN node

QUERY CreateProcedural(
  slug: String, title: String, statement: String,
  source_refs: [String], confidence: F32, intrinsic_load: String
) =>
  node <- AddN<Knowledge>({
    slug: slug, title: title, statement: statement,
    knowledge_type: "procedural",
    source_refs: source_refs,
    confidence: confidence,
    grain_level: "mid",
    intrinsic_load: intrinsic_load,
    introduction_scope: "in_course"
  })
  RETURN node

QUERY CreateMetacognitive(
  slug: String, title: String, statement: String,
  source_refs: [String], confidence: F32
) =>
  node <- AddN<Knowledge>({
    slug: slug, title: title, statement: statement,
    knowledge_type: "metacognitive",
    source_refs: source_refs,
    confidence: confidence,
    grain_level: "mid",
    intrinsic_load: "medium",
    introduction_scope: "in_course"
  })
  RETURN node

QUERY CreateLearningOutcome(
  slug: String, title: String, statement: String,
  source_refs: [String], confidence: F32, rubric_criteria: [String]
) =>
  node <- AddN<Knowledge>({
    slug: slug, title: title, statement: statement,
    knowledge_type: "learning_outcome",
    source_refs: source_refs,
    confidence: confidence,
    grain_level: "mid",
    intrinsic_load: "medium",
    introduction_scope: "in_course",
    rubric_criteria: rubric_criteria
  })
  RETURN node

QUERY CreateAssessmentItem(
  slug: String, title: String, statement: String,
  source_refs: [String], confidence: F32, construct_irrelevant_demands: [String]
) =>
  node <- AddN<Knowledge>({
    slug: slug, title: title, statement: statement,
    knowledge_type: "assessment_item",
    source_refs: source_refs,
    confidence: confidence,
    grain_level: "mid",
    intrinsic_load: "medium",
    introduction_scope: "in_course",
    construct_irrelevant_demands: construct_irrelevant_demands
  })
  RETURN node
// Requires edge helpers ------------------------------------------------------

QUERY WouldCreateRequiresCycle(from_id: ID, to_id: ID) =>
  back_path <- N<Knowledge>(to_id)::ShortestPath<Requires>::To(from_id)
  cycle_count <- back_path::COUNT
  RETURN cycle_count

QUERY AddRequires(
  from_id: ID, to_id: ID,
  strength: String, rationale: String, evidence_refs: [String], confidence: F32
) =>
  edge <- AddE<Requires>({
    strength: strength,
    rationale: rationale,
    evidence_refs: evidence_refs,
    confidence: confidence
  })::From(from_id)::To(to_id)
  RETURN edge

// Supports edge helpers ------------------------------------------------------

QUERY AddSupports(
  from_id: ID, to_id: ID,
  support_kind: String, intended_effect: String,
  case_tag: String, coverage_tags: [String], evidence_refs: [String], confidence: F32
) =>
  edge <- AddE<Supports>({
    support_kind: support_kind,
    intended_effect: intended_effect,
    case_tag: case_tag,
    coverage_tags: coverage_tags,
    evidence_refs: evidence_refs,
    confidence: confidence
  })::From(from_id)::To(to_id)
  RETURN edge

// Precedes edge helpers ------------------------------------------------------

QUERY IsSameEpisode(from_step_id: ID, to_step_id: ID) =>
  from <- N<TeachingStep>(from_step_id)
  to   <- N<TeachingStep>(to_step_id)
  same_episode_count <- to::WHERE(_::{episode}::EQ(from::{episode}))::COUNT
  RETURN same_episode_count

QUERY WouldCreatePrecedesCycle(from_step_id: ID, to_step_id: ID) =>
  back <- N<TeachingStep>(to_step_id)::ShortestPath<Precedes>::To(from_step_id)
  cycle_count <- back::COUNT
  RETURN cycle_count

QUERY AddPrecedes(from_step_id: ID, to_step_id: ID) =>
  edge <- AddE<Precedes>()::From(from_step_id)::To(to_step_id)
  RETURN edge

// Other edge mutations -------------------------------------------------------

QUERY AddAssesses(
  from_assessment_id: ID, to_lo_id: ID,
  claim: String, observation_features: [String], scope: String, confidence: F32
) =>
  edge <- AddE<Assesses>({
    claim: claim,
    observation_features: observation_features,
    scope: scope,
    confidence: confidence
  })::From(from_assessment_id)::To(to_lo_id)
  RETURN edge

QUERY AddAnchors(step_id: ID, target_id: ID, impact: String) =>
  edge <- AddE<Anchors>({ impact: impact })::From(step_id)::To(target_id)
  RETURN edge

// Reporting / analytics ------------------------------------------------------

QUERY MissingExamplesByType() =>
  factual_missing <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("factual"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("worked_example"))
        ::COUNT::EQ(0)
    )
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("counterexample"))
        ::COUNT::EQ(0)
    )
    ::{ id, title, slug }

  conceptual_missing <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("conceptual"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("analogy"))
        ::COUNT::EQ(0)
    )
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("counterexample"))
        ::COUNT::EQ(0)
    )
    ::{ id, title, slug }

  procedural_missing_we <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("procedural"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("worked_example"))
        ::COUNT::LT(2)
    )::{ id, title, slug }

  metacognitive_missing <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("metacognitive"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("strategy_hint"))
        ::COUNT::EQ(0)
    )::{ id, title, slug }

  RETURN factual_missing, conceptual_missing, procedural_missing_we, metacognitive_missing

QUERY ProceduralVarietyGaps() =>
  missing_typical <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("procedural"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{case_tag}::EQ("typical"))
        ::COUNT::EQ(0)
    )::{ id, title, slug }

  missing_edge <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("procedural"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{case_tag}::EQ("edge"))
        ::COUNT::EQ(0)
    )
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{case_tag}::EQ("error_case"))
        ::COUNT::EQ(0)
    )::{ id, title, slug }

  RETURN missing_typical, missing_edge

QUERY ScaffoldingGaps() =>
  factual_gaps <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("factual"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("worked_example"))
        ::COUNT::EQ(0)
    )
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("counterexample"))
        ::COUNT::EQ(0)
    )::{ id, title, slug }

  conceptual_gaps <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("conceptual"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("analogy"))
        ::COUNT::EQ(0)
    )
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("counterexample"))
        ::COUNT::EQ(0)
    )::{ id, title, slug }

  procedural_gaps <- N<Knowledge>
    ::WHERE(_::{knowledge_type}::EQ("procedural"))
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{support_kind}::EQ("worked_example"))
        ::COUNT::LT(2)
    )
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{case_tag}::EQ("edge"))
        ::COUNT::EQ(0)
    )
    ::WHERE(
      _::InE<Supports>
        ::WHERE(_::{case_tag}::EQ("error_case"))
        ::COUNT::EQ(0)
    )::{ id, title, slug }

  RETURN factual_gaps, conceptual_gaps, procedural_gaps

QUERY KeystoneApprox() =>
  nodes <- N<Knowledge>
    ::{ id, title, slug, in_deg: _::In<Requires>::COUNT, out_deg: _::Out<Requires>::COUNT }
  RETURN nodes

QUERY BorrowAheadByEpisode(episode: String) =>
  uses <- N<TeachingStep>
    ::WHERE(_::{episode}::EQ(episode))
    ::WHERE(_::{purpose}::EQ("use"))
  RETURN uses::{ id, title, slug }

QUERY LOAssessments(lo_id: ID) =>
  lo <- N<Knowledge>(lo_id)
  lo_info <- lo::{ id, title, slug, rubric_criteria }
  assessed_by <- lo::In<Assesses>::{ id, title, slug, knowledge_type, construct_irrelevant_demands }
  RETURN lo_info, assessed_by
