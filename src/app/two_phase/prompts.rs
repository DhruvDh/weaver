use std::path::Path;

use super::types::{HarvestNiche, WeaveNiche};
use crate::file_reader::{HarvesterFocus, WeaverFocus};

const HARVEST_SHARED: &str = r#"## Harvest Specialist Instructions

You are a **harvest specialist** focused on ONE niche. Extract nodes comprehensively.
The system prompt above defines all node types and constraints—reference it.

### Workflow
1. `list_directory` → survey files in scope
2. For large files (>150 lines): `delegate_tasks` to split by section
3. Extract ALL relevant content; when uncertain, CREATE the node

### Mandatory Tags (on EVERY node)
- `source:<chapter_path>` — the chapter you're harvesting
- `spec:<niche>` — your specialization (given below)

### Relationship Hints (tags, NOT edges)
When you suspect relationships, add hint tags for weavers:
- `req:C.concept_slug` — probable prerequisite
- `sup:P.example_slug` — probable scaffold
- `ref:F.related_term` — related content

### Node Quality
- **statement**: 1-3 sentences, self-contained, precise terminology
- **source_refs**: REQUIRED — [{path, start_line, end_line}]
- **granularity**: One exam question per node. Split if too broad."#;

const WEAVE_SHARED: &str = r#"## Weave Specialist Instructions

You are a **weave specialist** focused on ONE edge type or validation task.
The system prompt above defines all edge types and constraints—reference it.

### Discovery (ALWAYS do this first)
Never guess slugs. Always verify nodes exist:
```
graph_list_nodes_by_tag(tag="source:<chapter>")  → all nodes in this chapter
graph_list_nodes_by_tag(tag="spec:<niche>")      → filter by type
graph_search_nodes(query="...")                   → find by title/statement
```

### Workflow
1. Discover nodes in scope
2. Plan 10-20 edges based on hint tags (req:*, sup:*, claim:*)
3. Create edges in batch
4. Validate: `graph_dag_check`, `graph_gap_summary`
5. Fix any issues
6. Repeat until complete

### Node Creation
You CANNOT create nodes—you only have edge tools. If a node is missing:
- Document it as a gap in your response
- Note which edge you would have created if the node existed

### Validation After Each Batch
- `graph_dag_check` — requires must stay acyclic
- `graph_gap_summary` — find example/fadeability/practice gaps
- `graph_lo_alignment_summary` — LO coverage"#;

pub fn harvest_prompt(
    _chapter_path: &Path,
    chapter_tag: &str,
    spec_tag: &str,
    niche: HarvestNiche,
) -> (&'static str, &'static str, String) {
    match niche {
        HarvestNiche::FactualConceptual => (
            "factual+conceptual",
            HarvesterFocus::All.directive(),
            format!(
                "{shared}

### Your Niche: Factual + Conceptual Knowledge

**Extract these knowledge_types:**

**factual** — Terms, definitions, symbols, notation
- Example: \"A method signature consists of name and parameter list\"
- Tag pattern: `F.term_name`

**conceptual** — Principles, models, categories, relationships
- Example: \"Preconditions divide responsibility between caller and callee\"
- Tag pattern: `C.principle_name`

**What to skip**: procedures, worked examples, assessments, teaching steps
(other specialists handle those)

**Tags**: {chapter_tag}, {spec_tag}
**Hint tags for relationships**: `req:C.prerequisite`, `ref:F.related_term`",
                shared = HARVEST_SHARED,
            ),
        ),
        HarvestNiche::ProceduralExamples => (
            "procedural/examples",
            HarvesterFocus::Procedural.directive(),
            format!(
                "{shared}

### Your Niche: Procedural Knowledge + Worked Examples

**Extract these:**

**procedural** (knowledge_type) — Algorithms, methods, step-by-step processes
- Example: \"Apply the design recipe: 1) stub, 2) write tests, 3) implement\"
- Tag pattern: `P.method_name`

**Examples** (also procedural knowledge_type) — Worked solutions demonstrating procedures
- Create as SEPARATE nodes from the procedure they demonstrate
- Add `case:typical`, `case:edge`, or `case:error` tags
- Add `sup:P.target_procedure` hint for weavers

**Example node**:
```
slug: P.docstring_typical_example
statement: \"Example showing Args/Returns/Raises for update_rating function\"
tags: [source:..., spec:procedural, case:typical, sup:P.python_docstring]
```

**Tags**: {chapter_tag}, {spec_tag}",
                shared = HARVEST_SHARED,
            ),
        ),
        HarvestNiche::Assessments => (
            "assessments",
            HarvesterFocus::All.directive(),
            format!(
                "{shared}

### Your Niche: Assessments + Learning Outcomes

**Extract these knowledge_types:**

**learning_outcome** — Measurable goals for the learner
- REQUIRED field: `rubric_criteria` — list of observable behaviors
- Example: \"Students can write a documented function with preconditions\"
- rubric_criteria: [\"identifies preconditions\", \"documents raises\", \"tests edge cases\"]
- Tag pattern: `LO.outcome_name`

**assessment_item** — Exercises, checkpoints, practice problems
- These are tasks that test whether students achieved LOs
- Tag pattern: `A.exercise_name`
- Add hint tags:
  - `claim:LO.target_outcome` — which LO this probably assesses
  - `obs:behavior_to_measure` — what observable behavior to check

**No edges** — weavers will connect assessments to LOs with `assesses` edges

**Tags**: {chapter_tag}, {spec_tag}",
                shared = HARVEST_SHARED
            ),
        ),
        HarvestNiche::TeachingSteps => (
            "teaching_steps",
            HarvesterFocus::All.directive(),
            format!(
                "{shared}

### Your Niche: TeachingSteps (Discourse Layer)

**Extract narrative moments from the text:**

**TeachingStep fields:**
- `purpose`: setup | idea | use | consolidate
- `episode`: section identifier (for precedes scoping)
- `method_tags`: [\"worked-example\"], [\"naive-first\"], [\"analogy\"], [\"guided-practice\"]

**Purpose meanings:**
- `setup` — Motivational framing, \"why this matters\"
- `idea` — Introduces a new concept/procedure
- `use` — Applies already-introduced knowledge
- `consolidate` — Summarizes, refines, or reinforces

**Hint tags for weavers:**
- `anchors:C.concept_slug` with suspected impact (introduce/use/refine/motivate)
- `precedes:TS.next_step_slug` for ordering

**Example node**:
```
slug: TS.contract_motivation
purpose: setup
episode: \"02_contracts\"
method_tags: [\"naive-first\"]
statement: \"Opens with a crashing function to motivate need for contracts\"
tags: [source:..., spec:teaching_steps, anchors:C.implicit_contract:motivate]
```

**Tags**: {chapter_tag}, {spec_tag}",
                shared = HARVEST_SHARED
            ),
        ),
        HarvestNiche::SupportsIllustrations => (
            "supports/illustrations",
            HarvesterFocus::All.directive(),
            format!(
                "{shared}

### Your Niche: Scaffolds and Illustrations

**Extract these as Knowledge nodes with special tags:**

**Analogies** — Comparisons that help learners understand
- Tag: `support:analogy`, `effect:reduce_load`
- Example: \"A contract is like a napkin agreement between functions\"

**Counterexamples** — Examples showing what NOT to do
- Tag: `support:counterexample`, `effect:contrast`
- Example: \"Missing Raises field causes silent failures\"

**Misconception fixes** — Addressing common mistakes
- Tag: `support:misconception_fix`, `effect:contrast`
- Example: \"Author bias: assuming callers know implementation details\"

**Strategy hints** — Tips for applying procedures
- Tag: `support:strategy_hint`, `effect:germane_load`
- Example: \"Start with stub + tests to control complexity\"

**Hint tags for weavers:**
- `sup:C.target_concept` or `sup:P.target_procedure` — what this scaffolds

**Tags**: {chapter_tag}, {spec_tag}",
                shared = HARVEST_SHARED
            ),
        ),
        HarvestNiche::Metacognitive => (
            "metacognitive",
            HarvesterFocus::Metacognitive.directive(),
            format!(
                "{shared}

### Your Niche: Metacognitive Knowledge

**Extract knowledge_type = metacognitive:**

These are strategies for self-regulation, planning, and monitoring learning.

**What to look for:**
- Debugging heuristics: \"When tests fail, check boundary conditions first\"
- Planning strategies: \"Sketch the design before coding\"
- Self-monitoring: \"Ask yourself: does this contract prevent the crash?\"
- Reflection prompts: \"What would have happened without the precondition?\"

**Context tags** to add:
- `context:debugging` — debugging/troubleshooting strategies
- `context:design` — design/planning strategies
- `context:testing` — testing strategies
- `context:reflection` — self-assessment prompts

**Example node**:
```
slug: M.test_first_strategy
knowledge_type: metacognitive
statement: \"Write a failing test before implementing to clarify requirements\"
tags: [source:..., spec:metacognitive, context:testing, context:design]
```

**Tags**: {chapter_tag}, {spec_tag}",
                shared = HARVEST_SHARED
            ),
        ),
    }
}

pub fn weave_prompt(
    _chapter_path: &Path,
    chapter_tag: &str,
    niche: WeaveNiche,
) -> (&'static str, &'static str, String) {
    match niche {
        WeaveNiche::Requires => (
            "requires",
            WeaverFocus::Requires.directive(),
            format!(
                "{shared}

### Your Niche: requires Edges (Prerequisite DAG)

**Your job**: Build the prerequisite dependency graph for {chapter_tag}.

**Edge structure**:
```
graph_add_requires(
    from_slug: \"C.prerequisite_concept\",    // Knowledge node
    to_slug: \"C.dependent_concept\",         // Knowledge or AssessmentItem
    strength: \"necessary\",                  // necessary | strong | helpful
    rationale: \"Understanding X is needed to apply Y because...\",
    apply: true
)
```

**Find candidates**:
1. `graph_list_nodes_by_tag(tag=\"{chapter_tag}\")` — all chapter nodes
2. Look for `req:slug` hint tags left by harvesters
3. `graph_neighbors(slug=\"...\")` to see existing connections

**Validation after each batch**:
```
graph_dag_check()  // MUST return is_dag: true
```

**If cycle detected**:
1. The edge will be rejected
2. Identify which edge creates the cycle
3. Consider: Is the dependency actually bidirectional? (Split the concept)
4. Consider: Is one direction weaker? (Remove the helpful edge, keep the necessary one)

**Strength guidelines**:
- `necessary`: Cannot proceed without this knowledge
- `strong`: Very difficult without, but technically possible
- `helpful`: Makes learning easier but not required",
                shared = WEAVE_SHARED,
            ),
        ),
        WeaveNiche::Supports => (
            "supports",
            WeaverFocus::Supports.directive(),
            format!(
                "{shared}

### Your Niche: supports Edges (Scaffolding)

**Your job**: Connect examples, analogies, and scaffolds to the concepts they support.

**Edge structure**:
```
graph_add_supports(
    from_slug: \"P.worked_example\",          // The scaffold/example
    to_slug: \"P.target_procedure\",          // What it supports
    support_kind: \"worked_example\",         // Type of scaffold
    intended_effect: \"reduce_extraneous_load\",
    case_tag: \"typical\",                    // For examples: typical | edge | error_case
    apply: true
)
```

**support_kind values**:
- `worked_example`: Step-by-step demonstration
- `analogy`: Comparison to familiar concept
- `counterexample`: Shows what NOT to do
- `misconception_fix`: Addresses common mistake
- `strategy_hint`: Tip for applying knowledge
- `rubric_note`: Clarification for LO

**intended_effect values**:
- `reduce_extraneous_load`: Makes learning easier by removing distractions
- `increase_germane_load`: Deepens understanding through connections
- `motivate`: Creates desire to learn
- `contrast`: Highlights differences/boundaries

**Find candidates**:
1. Look for `sup:slug`, `case:*`, `support:*`, `effect:*` hint tags
2. `graph_list_nodes_by_tag(tag=\"spec:procedural\")` for examples

**Fadeability rule**:
Supports edges CANNOT be the only path to an assessment. After adding supports,
check with `graph_fadeability_view()`. If violations found, either:
1. Add a requires edge to provide an alternate path, OR
2. The supports edge should actually be a requires edge",
                shared = WEAVE_SHARED
            ),
        ),
        WeaveNiche::Assesses => (
            "assesses",
            WeaverFocus::Assesses.directive(),
            format!(
                "{shared}

### Your Niche: assesses Edges (Assessment → LO Links)

**Your job**: Connect AssessmentItems to the LearningOutcomes they measure.

**Edge structure**:
```
graph_add_assesses(
    from_slug: \"A.docstring_exercise\",      // AssessmentItem
    to_slug: \"LO.write_documented_function\", // LearningOutcome
    scope: \"target\",                         // target | enabling
    observation_features: [\"identifies preconditions\", \"documents raises\"],
    apply: true
)
```

**scope values**:
- `target`: This assessment directly measures the LO (primary assessment)
- `enabling`: This assessment measures a sub-part or prerequisite

**observation_features**:
These MUST cover the LO's `rubric_criteria`. First check what criteria the LO has:
```
graph_get_node(slug=\"LO.target_outcome\")
// Look at rubric_criteria field
```

Then ensure your observation_features address each criterion.

**Find candidates**:
1. `graph_list_nodes_by_kind(selector=\"assessment_item\")` — all assessments
2. Look for `claim:LO.slug`, `obs:feature` hint tags from harvesters
3. `graph_list_nodes_by_kind(selector=\"learning_outcome\")` — all LOs

**Validation**:
```
graph_lo_alignment_summary(lo_slug=\"LO.xxx\", fetch_body=true)
```
Check that:
- Every LO has ≥1 assesses edge with scope=target
- Coverage shows no missing criteria",
                shared = WEAVE_SHARED
            ),
        ),
        WeaveNiche::TeachingSteps => (
            "teaching_steps",
            WeaverFocus::All.directive(),
            format!(
                "{shared}

### Your Niche: Discourse Wiring (precedes + anchors)

**Your job**: Connect TeachingSteps in sequence and anchor them to knowledge.

**precedes edges** (TeachingStep → TeachingStep):
```
graph_add_precedes(
    from_slug: \"TS.motivation\",    // Earlier step
    to_slug: \"TS.definition\",      // Later step
    episode: \"02_contracts\",       // Must match both steps' episode
    apply: true
)
```
- Must be acyclic within each episode
- Look for `precedes:slug` hint tags

**anchors edges** (TeachingStep → Knowledge/LO/Assessment):
```
graph_add_anchors(
    from_slug: \"TS.definition_step\",
    to_slug: \"C.contract_components\",
    impact: \"introduce\",             // introduce | use | refine | motivate | target
    apply: true
)
```

**impact values and when to use**:
- `introduce`: The step presents this concept for the first time
  - Use with `idea` purpose steps
- `use`: The step applies already-known knowledge
  - Use with `use` purpose steps
- `refine`: The step adds nuance or specialization
  - Use with `consolidate` purpose steps
- `motivate`: The step creates motivation to learn this
  - Use with `setup` purpose steps
- `target`: The step articulates expectations for an LO
  - Use when step describes rubric/grading

**Find candidates**:
1. `graph_list_nodes_by_tag(tag=\"spec:teaching_steps\")` — all TeachingSteps
2. Look for `anchors:slug:impact` hint tags
3. Filter by {chapter_tag}",
                shared = WEAVE_SHARED
            ),
        ),
        WeaveNiche::CoverageGap => (
            "coverage/gaps",
            WeaverFocus::All.directive(),
            format!(
                "{shared}

### Your Niche: Gap Analysis and Repair

**Your job**: Find and fix structural problems in the graph.

**Analysis tools**:
```
graph_dag_check()                → is_dag must be true
graph_gap_summary(fetch_body=true)       → overview of all gaps
graph_example_gaps_view()        → nodes missing required examples
graph_fadeability_view()         → supports that carry prerequisite load
graph_practice_gaps_view()       → procedural nodes without practice links
graph_lo_alignment_summary(lo_slug=\"...\") → per-LO coverage analysis
```

**Common gaps and fixes**:

1. **Example gaps** (procedural nodes need typical + edge examples)
   - Search for example nodes that harvesters may have created
   - Add supports edges to connect examples to procedures
   - If no example nodes exist, document as \"missing example nodes\"

2. **Fadeability violations** (supports carry prerequisite load)
   - Add a requires edge to provide an alternate path
   - This ensures supports can be \"faded\" without breaking reachability

3. **Practice gaps** (procedural nodes not reaching assessments)
   - Add requires edges: Procedure → Assessment
   - Verify assessment has assesses edge to an LO

4. **LO coverage gaps** (missing criteria coverage)
   - Check LO's rubric_criteria vs incoming assesses edges' observation_features
   - If assessment nodes exist, add assesses edges with proper observation_features
   - If no assessment nodes exist, document as \"missing assessment nodes\"

**Remember**: You can only create EDGES, not nodes. Document missing nodes as issues.

**Scope**: Focus on {chapter_tag}. Document cross-chapter gaps but don't fix them
(other chapters' weavers handle their own).",
                shared = WEAVE_SHARED
            ),
        ),
        WeaveNiche::CleanupQa => (
            "cleanup/qa",
            WeaverFocus::All.directive(),
            format!(
                "{shared}

### Your Niche: QA and Validation

**Your job**: Final validation and gap-fixing for {chapter_tag}.

**What you CAN do** (you have edge tools only):

1. **Fix fadeability violations**:
   - Run `graph_fadeability_view()` to find supports edges carrying prerequisite load
   - Fix by adding requires edges to provide alternate paths

2. **Fix practice gaps**:
   - Run `graph_practice_gaps_view()` to find procedural nodes without assessment paths
   - Add requires edges to connect procedures to assessments

3. **Verify LO alignment**:
   - For each LO, run `graph_lo_alignment_summary(lo_slug=\"LO.xxx\", fetch_body=true)`
   - Add missing assesses edges if assessments exist but aren't connected

4. **Verify DAG integrity**:
   - Run `graph_dag_check()` — must return is_dag: true

**Validation checklist**:
```
graph_dag_check()                  → is_dag: true
graph_gap_summary(fetch_body=true) → review all gaps
graph_fadeability_view()           → should be empty
graph_practice_gaps_view()         → should be empty
```

**What you CANNOT do** (node updates require harvester tools):
- Remove hint tags (req:*, sup:*, etc.)
- Set introduction_scope
- Modify source_refs

Document these as \"known issues\" in your final report.

**Final report** (generate at end):
- Validation status for each check (pass/fail)
- Edge counts added during this pass
- Known issues requiring manual cleanup
- Cross-chapter dependencies noted",
                shared = WEAVE_SHARED
            ),
        ),
    }
}
