# White Paper: First-Principles Learning Network

> **Purpose.** This work provides a defensible, auditable representation of both (a) **what must be learned** (the Knowledge layer) and (b) **how the text actually teaches it** (the Discourse layer). The Knowledge layer is a multiplex network over a **prerequisite DAG** (`requires`) with scaffolding (`supports`) and assessment evidence (`assesses`). The Discourse layer is a **local narrative graph** of authored **TeachingSteps** linked by `precedes` and grounded to specific knowledge and outcomes via `anchors`. This separation preserves the purity of prerequisites and assessments while making the pedagogical storyline explicit and analyzable.

## 1. Why a *directed* network, and why *acyclic*?

**First principles reasoning** is a *directional* activity: you start from primitives and build up. In a network, that means arrows go from enabling knowledge to the knowledge that depends on it. If you can lay out the nodes so all arrows point the same way (earlier → later), the graph is **acyclic**. Acyclic directed networks have nice properties: they admit a **topological order** (a valid teach/learn sequence), and their **adjacency matrix** can be permuted into a strictly upper‑triangular form—both handy for checking that our structure is coherent and cycle‑free.  

Keeping edges typed in **separate layers** (requires/supports/assesses) also fits the standard notion of a **multiplex/multilayer network**, where the same nodes are connected by different kinds of ties represented per layer. This gives us clean semantics and clean analytics.  

> **Scope of the DAG.** The acyclicity requirement applies **only to the `requires` layer**. Those prerequisite edges form the dependency backbone and must admit a topological order. The `supports` and `assesses` layers are analyzed separately—they can branch, cross, or even contain cycles without jeopardizing the prerequisite DAG. The Discourse layer’s `precedes` relation is also acyclic but scoped **per episode** (section/lesson). We do not require global linearity; a linear reading order is a derived view.
>
> **Takeaway.** We aim for a DAG so that “reasoning chains” are literally **paths** from first principles to outcomes, while edge *types* live in parallel layers that we can analyze or visualize separately.

## 2. The theoretical foundations (what each theory is, why it matters, how we use it)

### 2.1 Constructive Alignment (Biggs & Tang)

* **What it is.** A curriculum design principle: align **Intended Learning Outcomes (ILOs)**, **teaching/learning activities**, and **assessment**.
* **Why it exists.** To make sure students are practicing and being assessed on the very outcomes we claim to value, not on side skills.
* **How we use it.** In our network, we treat **first‑principle knowledge** as the set of knowledge nodes with no incoming `requires` edges. A Learning Outcome **L** is aligned iff there exists an assessment item **A** such that (a) at least one first‑principle node reaches **A** via a `requires*` path (possibly through intermediate knowledge nodes), and (b) **A → L** exists in `assesses` with `scope = target`. The `assesses` edge carries the evidentiary argument, so reachability and evidence are both explicit even when we ingest an existing course.

Reachability alone only shows that the learner could reasonably attempt the task. Evidence‑Centered Design demands that the **observations** collected from **A** genuinely support **L**. We capture that distinction by storing `observation_features` on every `assesses` edge and by validating that those features cover the LO’s rubric criteria (see §6).

### 2.2 Knowledge Space Theory (Doignon & Falmagne)

* **What it is.** A mathematical framework for **prerequisite** relations among items/skills, yielding *quasi‑orders* and feasible “knowledge states.”
* **Why it exists.** To model *which sets of skills are reachable* given prerequisites, and to support adaptive sequencing/diagnosis.
* **How we use it.** We treat **requires** edges as KST‑style surmise relations; the induced partial order is the spine of our DAG. Every `requires` edge—no matter whether its `strength` is `necessary`, `strong`, or `helpful`—participates in the acyclicity test. If the agent proposes a cycle, it violates feasibility and must be revised while preserving assessment items as sinks.

### 2.3 Revised Bloom’s Taxonomy — *Knowledge Dimension*

* **What it is.** An expansion of Bloom (Anderson & Krathwohl, 2001) adding **factual, conceptual, procedural, metacognitive** knowledge types.
* **Why it exists.** To distinguish *what kind* of knowledge a node embodies (a fact vs. a concept vs. a method vs. self‑regulation strategy).
* **How we use it.** This gives us a **defensible node ontology** (below), so we are not inventing ad‑hoc categories.

### 2.4 Evidence‑Centered Design (Mislevy & colleagues)

* **What it is.** A framework that ties **claims (competencies)** to **tasks** and **evidence rules**—assessment as an evidentiary argument.
* **Why it exists.** To ensure an assessment item *actually provides evidence* about the intended claim and nothing extraneous.
* **How we use it.** Our **assesses** edges mean: “this task elicits evidence for that LO/construct,” and we store the item’s evidence model metadata on the edge.

### 2.5 Cognitive Load / Worked Examples (Sweller et al.)

* **What it is.** Worked examples reduce extraneous load during early acquisition; they are especially powerful supports for procedures.
* **Why it exists.** Learners initially lack schemas; good examples scaffold without overloading working memory.
* **How we use it.** We represent worked examples, analogies, and counterexamples as **supports** edges into the target concept/procedure.

### 2.6 Discourse / Narrative Structure (minimal rhetorical grounding)

* **What it is.** A representation of the *authored flow*—the sequence of TeachingSteps (motivation, definition, example, practice, refinement) used to bring learners to the knowledge.
* **Why it matters.** A naïve attempt or a motivational vignette is pedagogically useful but not a logical prerequisite. Modeling discourse separately prevents polluting the prerequisite DAG while letting us analyze motivation coverage, example/practice patterns, and borrow‑ahead moments.
* **How we use it.** We add a single node kind (**TeachingStep**) and two relations: `precedes` (authored order within a section) and `anchors` (a step’s impact on specific knowledge or outcomes).

### 2.7 Granularity Policy ("Strict Mid-Grain")

We adopt a **Two-Speed Model** to separate logic from narrative:

* **Knowledge Layer (Logic):** Modeled at **Level B (Micro-Concept)**.
  * *The Litmus Test:* “Can I write a single exam question targeting *only* this specific idea?”
  * *Handling Micro-Details:* Syntax rules or trivia (e.g., “semicolons end lines”) must **not** be nodes. Treat them as attributes/constraints on a mid-grain node or as metadata on `supports` edges.
* **Discourse Layer (Narrative):** Modeled at **Level D (Episode/Mini-Lesson)**.
  * *The Litmus Test:* “Is this a distinct story, worked example, or exercise block in the text?”

**Micro-grain (fragments):** sub-facts or syntax trivia better represented as examples, rubric details, or **constraints within a mid-grain node’s statement**. Prefer to fold these into the parent node’s definition or `supports` metadata rather than stand-alone nodes.

**Audits:**

* **Over-bundled:** `Statement > 2 sentences` **and** `In-degree(requires) > 3` → Flag as Macro (needs splitting).
* **Fragment:** `Statement < 15 tokens` **and** no `assesses`/`supports` edges → Flag as Micro (fold into parent).

## 3. Our node ontology: **what** we model (and why)

Each node has a **knowledge type** from Revised Bloom’s Knowledge Dimension. That gives us theoretical grounding while staying simple enough for an MVP:

* **Factual** — terms, symbols, conventions, simple facts (e.g., “A method’s signature is name + parameter list in Java.”)
* **Conceptual** — categories, principles, models (e.g., “Preconditions vs. postconditions divide responsibility.”)
* **Procedural** — methods, algorithms, heuristics (e.g., “Apply the OOP design recipe to implement a method.”)
* **Metacognitive** — strategies for monitoring/planning (e.g., “Start with a stub + tests to control complexity.”)

Two additional “meta” nodes are used sparingly (both grounded in ECD/CLT use-cases):

* **Learning Outcome (LO)** — an assessable claim about what the learner can do. (Constructive Alignment anchor.)
* **Assessment Item** — a task that can elicit evidence for one or more LOs. (ECD anchor.)

**Discourse node (single new kind).**

* **TeachingStep** — an atomic authored step (paragraph, code block, figure, prompt) within an episode (section/lesson). It carries:
  * `purpose` ∈ {`setup`, `idea`, `use`, `consolidate`},
  * `method_tags` (open set; e.g., `naive-first`, `breakdown`, `analogy`, `worked-example`, `guided-practice`),
* `source_refs` spans and an `episode` identifier.
TeachingSteps do **not** change what is true; they document how the text leads the learner.

> **Why only these?** They cover the validated distinctions: *what knowledge is* (Bloom), *what progression requires* (KST), and *how we know we’ve achieved it* (CA + ECD). Everything else (e.g., examples, analogies, counterexamples, rubrics) appears via **supports** edges rather than proliferating node kinds—keeping the ontology lean and defensible.

> *Tagging, not types.* Use a generic `tags: [String]` attribute (see Appendix A) to label conceptual nodes as `"principle"` or `"misconception"`; do **not** mint new knowledge types for those concepts.

## 4. Edge types: the *meaning* of arrows

We reason about three **node sets** and three **edge sets**:

* **Knowledge nodes** — the factual/conceptual/procedural/metacognitive entries.
* **Learning outcomes (LOs).**
* **Assessment items.**

The edge layers connect these sets as follows:

* `requires`: prerequisite edges from knowledge → knowledge/assessment.
* `supports`: scaffolds from knowledge → knowledge/LO.
* `assesses`: evidence edges from assessment → LO.

All invariants are defined per layer: only `requires` must remain acyclic; `supports` forbids self-loops but may contain cycles; `assesses` must always point from an assessment item to an LO. With that framing we can express validations precisely and avoid leaking constraints across layers.

We use exactly three edge types, each with its own **layer** in a multiplex representation (same node set, different ties).

> **Only stored edge kinds.** The data model persists exactly three edge layers—`requires`, `supports`, `assesses`. All other pedagogical relationships must be expressed via Discourse `anchors` and/or `supports` metadata (see §4.2 and §4.x); no additional graph edge kinds are permitted.

1. **requires (A → B)**

   * **Semantics.** Mastery of *A* is a **prerequisite** for dependable mastery of *B*. This is the KST backbone.
   * **Constraints.** Must preserve a **DAG** (no cycles). Enforced by topological checks/Kahn‑style peeling; the adjacency can be permuted to strictly upper‑triangular if acyclic. `requires` edges may **originate only from knowledge nodes** and may terminate at knowledge or assessment nodes. Assessment items never emit `requires` edges, so they remain sinks in the prerequisite layer, and LOs never participate in `requires` at all.
   * **Typical spans.** factual→conceptual, conceptual→procedural, procedural→assessment_item (when the task depends on that knowledge). Metacognitive nodes may feed assessments **only** when the LO explicitly intends to measure strategy use; otherwise note such demands as `construct_irrelevant_demands`. We highlight `strength` to guide pedagogy (e.g., schedule necessary edges earlier), but the DAG invariant treats all strengths equally.
   > **Rationale provenance.** Cite either a course `source_refs` span or an internal justification note (e.g., `notes/dep-justifications.md#slug`) in every `requires.rationale`/`evidence_refs` pair so reviewers can audit the dependency claim.

2. **supports (S → B)**

   * **Semantics.** *S* improves learnability or robustness of *B* (worked examples, analogies, counterexamples, common misconceptions with repairs, strategy prompts). These edges represent **scaffolding**, not logical dependency.
   * **Constraints.** No self‑loops. Cycles among supports are permitted because scaffolds can reinforce one another, but supports should remain *fadeable*: removing them must not break any `requires` reachability. Supports into LOs are rare and document motivational framing (e.g., scenario videos) rather than prerequisites; justify them explicitly.

3. **assesses (Assessment → LO)**

   * **Semantics.** The assessment item is the task/process intended to elicit evidence for the LO’s claim (ECD). The edge stores the evidence link, not content text, and `evidence_link.claim` must equal the LO id at the target so the evidentiary argument is self-contained. `observation_features` enumerate the observable cues or scoring dimensions tied to the LO’s rubric.
   * **Constraints.** Items are *sinks* in the “first principles → assessment” portion of the graph (requires flow); they do not point to knowledge nodes. Every assessment must point via `assesses` to at least one LO so evaluators know what evidence it’s meant to produce. When a single assessment maps to multiple LOs, each `assesses` edge tags its own `scope` (`target` or `enabling`), and coverage analysis treats the scopes independently.

> **Why this orientation?** When designing from scratch you might think “first principles → LO → assessment.” For the evaluation network we ingest today, we encode “first principles → assessment → LO” so a single forward walk lets analysts see which knowledge supports each task and which outcomes that task evidences, while still keeping `requires` acyclic.

### 4.x Discourse logic (the narrative layer)

We add two relations over TeachingSteps plus cross‑layer anchoring:

1. **`precedes (Step A → Step B)`**
   * **Meaning.** Authored order **within an episode** (section/lesson).
   * **Constraint.** Acyclic per episode (local DAG). Branching and optional detours are allowed; a linear sequence is a derived view.

2. **`anchors (TeachingStep → KnowledgeOrLO){impact}`**
   * **Meaning.** This step interacts with a specific knowledge node or LO.
   * **Attributes.** `impact` ∈ {`introduce`, `use`, `refine`, `motivate`, `target`}:
     * `introduce`: the step presents a new concept/procedure,
     * `use`: the step applies already‑introduced knowledge (examples, practice, presenting an assessment item),
     * `refine`: the step specializes or improves a known idea (syntax sugar, patterns),
     * `motivate`: the step frames the need or goal for a target idea,
     * `target`: the step explicitly articulates performance expectations for an LO (rubric exposition).

## 5. What goes **on** edges and nodes (attributes, with justification)

### 5.1 Minimal node attributes (all nodes)

* **`title`** — stable, human‑readable identifier.
* **`knowledge_type`** ∈ {factual, conceptual, procedural, metacognitive, learning_outcome, assessment_item}. (Revised Bloom + CA/ECD anchors.)
* **`statement`** — a concise, self‑contained sentence or two (for LOs: performance statement).
* **`source_refs`** — *path + line range + commit hash* for the exact PreTeXt source (no excerpted text), e.g., `source/DesignRecipe/section.xml#L120–L180 @ 1a2b3c4d`. Stored as structured objects `{ path, start_line, end_line, revision }`, never as free strings. These references are immutable with respect to that commit; when the upstream repo changes you regenerate or migrate the graph for the new commit rather than “sliding” offsets.
* **`confidence`** — model’s self‑estimate (0–1) to flag uncertain extractions.
* **LO-only fields.** `rubric_criteria: [String]` stores the observable behaviors or scoring dimensions promised by the LO. These criteria are what `observation_features` must cover.
* **Assessment-only fields.** `construct_irrelevant_demands: [String]` documents any known skills the item happens to require but is *not* intended to measure (e.g., advanced prose, tricky notation). Recording them makes purity checks interpretable.
* **`introduction_scope`** ∈ {`in_course`, `prior`, `external`} — default `in_course`; used by the borrow-ahead guardrail in §6.
* **`tags`** — open set for meta labels such as `["principle"]`, `["misconception"]`, or `["keystone"]`; see §3.

### 5.2 Edge attributes by type

#### requires

* **`strength`** ∈ {necessary, strong, helpful}. *Necessary* edges approximate KST surmise relations; *helpful* approximates “facilitates but not required.”
* **`rationale`** — a short natural‑language justification (why *A* really precedes *B*).
* **`evidence_refs`** — optional parent pointers to course sources (again, file/line references only).

#### supports

* **`support_kind`** ∈ {worked_example, analogy, counterexample, misconception_fix, strategy_hint, rubric_note}. (CLT/ECD‑inspired scaffolds.)
* **`intended_effect`** ∈ {reduce_extraneous_load, increase_germane_load, motivate, contrast}. (So the system “knows” why the support exists; `reduce_load`/`germane_load` are accepted aliases on ingest.)

#### assesses

* **`evidence_link`** — ECD pointers: `{claim: LO_id, observation_features: [...], scope ∈ {target,enabling}}`. Claim must equal the LO id at the target of the edge.
* **`scope`** ∈ {target, enabling} to indicate whether an item targets the LO directly or only sub‑parts. Every LO must have at least one incoming `assesses` edge with `scope = target`.
  Observation features are scored behaviors or cues (“loop invariant holds,” “unit tests pass for negative inputs”) that trace directly to the LO’s rubric criteria; write them at the granularity you can actually observe or autograde.

> **Why file/line *references* but no text?** You said you don’t want to store textbook prose. These references are enough to justify nodes/edges and let humans audit decisions later.

### 5.3 Discourse attributes

**TeachingStep (node).**

* `purpose` ∈ {`setup`, `idea`, `use`, `consolidate`}
* `method_tags`: [String] (style markers; e.g., `naive-first`, `breakdown`, `analogy`, `worked-example`)
* `episode`: String (section/lesson label)
* `source_refs`: [SourceRef]
* *Anchoring guidance.* `idea` steps **must** `anchors(..., impact=introduce)` at least one Knowledge node. `use` steps **must** anchor with `impact=use`. `consolidate` steps that refine should include `impact=refine`. `setup` steps should anchor via `impact=motivate` where possible; otherwise allow zero anchors with a short rationale.

**Anchors (edge).**

* `impact` ∈ {`introduce`, `use`, `refine`, `motivate`, `target`}
* Anchors to **assessment items** use `impact=use`. The evidentiary mapping remains the Knowledge‑layer `assesses(Assessment → LO)`.
* *Constraint.* `introduce` and `refine` impacts must target **Knowledge** nodes; only `target` impacts may point directly to **Learning Outcomes** (typically via the assessment narrative).

> **Where examples live.** Examples are authored as TeachingSteps in the Discourse layer and captured structurally via `supports` edge metadata. There is no standalone “example” node type.

### 5.4 Example Metadata (on `supports` edges)

We treat examples not as standalone nodes but as **rich edges**. Every `supports` edge must carry:

* `support_kind ∈ {worked_example, analogy, counterexample, misconception_fix, strategy_hint, rubric_note}`
* `case_tag ∈ {typical, edge, error_case}` *(critical for variance analysis)*
* `coverage_tags: [String]` (e.g., `["negative_input", "empty_list"]` to track specific constraint coverage)
* `intended_effect ∈ {reduce_extraneous_load, increase_germane_load, motivate, contrast}` *(ingestion aliases: `reduce_load`, `germane_load`)*

`rubric_note` is used sparingly to attach clarifications or exemplar snippets to an LO; it complements, but does not replace, `anchors(..., target)`.

**Note on refutation.** We do **not** add a `refutes` edge type. To refute a misconception, model the misconception as a node and connect it to the principle via `supports` with `support_kind="misconception_fix"` and `intended_effect="contrast"`.

**Consistency rule.** For any single misconception, choose exactly one representation: either model it as a conceptual node with an incoming `misconception_fix` support, **or** encode only the fix as a `supports` edge into the target concept. Do not do both.

### 5.5 Example compliance table (author + validator contract)

| Artifact/edge | Required evidence | Where it lives |
|---------------|------------------|----------------|
| **Conceptual node** | ≥ 1 positive illustration TeachingStep **plus** ≥ 1 counterexample or misconception_fix support tagged `contrast` | `anchors(introduce/use)` + `supports` edges (`support_kind`, `case_tag`) |
| **Procedural node** | ≥ 2 worked examples (one `typical`, one `edge` or `error_case`) **and** ≥ 1 practice assessment that reaches a target LO | `supports` edges + `anchors` + `requires` path into an `assessment_item` with `assesses(scope=target)` |
| **Metacognitive node** | ≥ 1 `strategy_hint` support **and** ≥ 1 anchored reflection/transfer activity | `supports` edges + TeachingStep anchors |
| **`requires` edge** | 1–2 sentence `rationale` + `evidence_refs` | Edge attributes |
| **`supports` edge** | `support_kind`, `intended_effect`, `case_tag`, `coverage_tags` (when constraints exist), `evidence_refs` | Edge attributes |
| **Learning Outcome** | Exhaustive `rubric_criteria` list | Node attributes |
| **Assessment item** | `observation_features` that cover every LO criterion | `assesses` edge attributes |
| **TeachingStep** | ≥ 1 `anchors(...)` link or a short rationale explaining why it is unanchored | Node attributes / Anchors |

Sections §6 (“Example minimums”, “Variety check”) and §10.2 reference this table; validators enforce the requirements defined here.

## 6. How this becomes a *coherent* network (and how we check it)

1. **Requires DAG.** Run Kahn’s algorithm (or adjacency triangularization) over the subgraph induced by `requires` edges. Every edge participates regardless of strength, assessment items never emit `requires`, and any residual nodes after peeling indicate a violation. *Remediation order:* when breaking a cycle, prefer to relax/drop `helpful` edges before `strong`, and `strong` before `necessary`.

2. **First principles & schedule.** Knowledge nodes with indegree 0 in `requires` are treated automatically as *first principles*. A topological order over the DAG yields a teach/learn schedule; longest paths expose depth, and nodes that share the same rank highlight parallelizable topics.

3. **LO reachability predicate.** A Learning Outcome **L** is reachable iff ∃ assessment **A** such that (i) at least one first‑principle node reaches **A** via a `requires*` path (knowledge → … → assessment, never reversing edge direction), and (ii) `assesses(A, L)` exists with `scope = target`. This predicate replaces vague “reachable via assessment” language and is what alignment reports compute.

4. **Coverage check (Constructive Alignment).** For each LO, take the union of `observation_features` on all incoming `assesses` edges with `scope = target`. That union must cover the LO’s `rubric_criteria`; if some criteria are only partially covered, flag them for author review or add additional assessments.

5. **Rubric drift.** Any change to an LO’s `rubric_criteria` forces a fresh coverage computation. Missing coverage for a new criterion is an error; `observation_features` that reference removed criteria emit warnings until the edge metadata is updated.

6. **Purity check (ECD).** For each target assessment **A** of LO **L**, compute `Extraneous(A, L) = Requires*(A) \ Intended(L)`, where `Intended(L)` is the set of knowledge nodes that L explicitly references or depends on via rubric criteria. Large differences—or any presence of knowledge listed under `construct_irrelevant_demands`—signal possible construct‑irrelevant variance.

7. **Support sanity (fadeable test).** Temporarily remove all `supports` edges and recompute `requires*` reachability from first principles to assessments. Any difference is a **type error**: the implicated `supports` edge is carrying prerequisite load and must be recast as `requires` (or the target split).

8. **Minimal multiplex.** Staying with the three named layers keeps analytics interpretable while remaining faithful to multiplex‑network literature (see §4).

9. **Discourse continuity (per episode).** `precedes` is acyclic; every TeachingStep in the episode lies on a `precedes` path (no orphans).

10. **Anchoring integrity.** `idea` steps anchor with `impact=introduce`; `use` steps anchor with `impact=use`. `consolidate` steps that claim refinement must `impact=refine`. `setup` steps should `impact=motivate` or include a rationale if unanchored.

11. **Introduction scope + borrow-ahead severity.** Knowledge nodes can be flagged `introduction_scope ∈ {in_course, prior, external}`. Only `in_course` nodes must have at least one upstream `idea` step with `impact=introduce`. Borrow-ahead levels: Level 1 (use before introduce inside the same episode) = warning; Level 2 (across episodes) = error unless `introduction_scope ∈ {prior, external}`; Level 3 (across chapters) = error and requires explicit prior/external annotation plus a pointer.

12. **Pedagogical coverage (suggestion).** For every Knowledge node introduced in an episode, recommend ≥1 downstream `use` step anchored to the same node. Missing → **support gap** (advisory).

13. **Alignment bridge (unchanged).** Introduced Knowledge should be reachable to at least one assessment via `requires*` and that assessment must have `assesses(..., scope=target)` to some LO.

14. **Assessable Atom Test (AAT).** A knowledge node is valid only if (a) its `statement` is a single claim or single method (≤ 2 sentences) and (b) it participates in the network either as a prerequisite (≥ 1 distinct `requires` consumer) or via ≥ 2 distinct TeachingSteps anchored with `introduce/use/refine`. Failures flag for merge (violates (a)) or “fold into parent/example” (violates (b)).

15. **Procedural practice (Constructive Alignment bridge).** Every **procedural** node must be an ancestor of ≥ 1 assessment item in the `requires` DAG (i.e., `Procedure →requires→ Assessment`). That assessment must, in turn, `assesses(..., scope=target)` at least one LO so practice is auditable evidence, not merely advisory.

16. **Keystone analysis (critical path).** Compute betweenness centrality over the `requires` DAG (batch job) and use the `KeystoneApprox` query (Appendix B.7) in CI as a fast proxy (`score = in_reach * out_reach`). Nodes with high centrality/score are **Keystones**. Any Keystone with < 2 `worked_example` supports is flagged **high risk** because a brittle node would block large portions of the learning graph (and Keystones ideally serve ≤ 8 direct dependents before being split).

17. **Example minimums (see §5.5).** Use the compliance table to enforce per-type minima: conceptual nodes need both a positive illustration and a counterexample/misconception fix, procedural nodes need the worked-example pair plus a practice assessment, metacognitive nodes need a strategy hint + reflection, etc.

18. **Variety check.** The union of `case_tag`s over a node’s examples must include **`typical`**; for **procedural** nodes it must include **`typical` + `edge`** or `error_case`, and the audit ensures at least one non-typical sample exists.

19. **Drift guard.** If a procedural node’s assessment coverage exists but all examples are missing or only `typical`, raise a **quality warning** (examples haven’t kept pace with the assessment demand).

## 7. What success looks like (evaluating the network we’ve built)

* **No cycles in `requires`.** The topological peel completes; the adjacency permutes to strictly upper‑triangular. (Automatable invariant.)
* **Reachable LOs.** Every LO satisfies the predicate in §6: some assessment reachable from first principles via `requires*` also `assesses` the LO with `scope = target`.
* **Coverage & purity proven.** Target assessments’ observation features cover every LO rubric criterion, and `Extraneous(A, L)` is empty (or explained via `construct_irrelevant_demands`).
* **Rubric drift proof.** Whenever LO rubric criteria change, coverage is re-evaluated and blockers are surfaced immediately.
* **Purposeful scaffolding.** High intrinsic‑load procedures are supported by worked examples or analogies early; supports can be *faded* (removable without breaking reachability) as mastery increases.
* **Auditability.** Every node/edge can be traced to **source_refs** or **evidence_link**—file names, line ranges, and commit hashes, not copied text.
* **Reasoning paths exist.** For each LO, you can traverse a **path** that reads like a human‑intelligible reasoning chain from primitives to outcome—this is the “first principles” promise realized as graph structure.
* **Multiplex clarity.** Analysts can slice by layer (requires vs. supports vs. assesses) and by knowledge type (factual/conceptual/procedural/metacognitive) to answer “what to teach, what helps, how we’ll know.”
* **Discourse quality.** Motivation anchors precede `idea` steps, `setup→idea→use/refine` sequences stay acyclic per episode, and any borrow-ahead is annotated with the severity levels defined in §6.10.
* **Granularity sanity.** ≥ 90% of non‑macro nodes are tagged **mid** and pass the acceptance criteria; flagged macro/micro outliers are reviewed or re‑cast as bundles/examples.
* **Example completeness.** 100% of **knowledge nodes** meet the **example minimums** for their type; 100% of **procedural nodes** have **typical + edge** worked examples **and** at least one **practice assessment** that targets an LO.
* **Edge-case entropy.** No procedural node relies solely on `typical` examples; at least one `supports` edge per procedure carries `case_tag ∈ {edge, error_case}` so assessments are stress-tested.

## 8. A tiny walk‑through (illustrative, domain‑agnostic)

* **Factual → Conceptual → Procedural → Assessment → LO**

  1. *Factual:* “A method’s signature is name + parameter list.”
  2. *Conceptual:* “Preconditions/postconditions divide caller/callee responsibility.”
  3. *Procedural:* “Apply the design recipe to implement method `average` with pre/post.”
  4. *LO:* “Given a requirement, produce a correct, tested Java method by applying the OOP design recipe.”
  5. *Assessment Item:* A short, autograded task prompting the same.
     **Edges:** factual → conceptual (**requires**), conceptual → procedural (**requires**), worked example → procedural (**supports**), procedural → assessment (**requires**), assessment → LO (**assesses**).
     *LO rubric criteria:* {“identifies pre/postconditions”, “produces tests that cover edge cases”, “refactors to remove duplication”}.
     *Observation features on the `assesses` edge:* {“unit tests pass for boundary values”, “explicit contract documented”, “method decomposes behavior per recipe”}.
     CLT predicts the worked example early will help; ECD demands those observation features collectively cover the rubric criteria while avoiding construct‑irrelevant demands (e.g., essay writing).

**Discourse slice for the same topic (episode “Introducing loops”).**

1. `setup` — “Repeating code is brittle” → `anchors(motivate → Procedural: Looping)`
2. `idea` — “Indefinite loop (`loop` + `break` )” → `anchors(introduce → Procedural: Indefinite Loop)`
3. `use` — “Sum inputs until 0” → `anchors(use → Procedural: Indefinite Loop)`
4. `consolidate` — “`while`/`for` idioms as refinements” → `anchors(refine → Procedural: Loop Idioms)`
5. `use` — “Practice: sum even numbers in a range” → `anchors(use → Procedural: Loop Idioms)`

`precedes` edges follow 1→2→3→4→5 (branches allowed for separate `while` and `for` vignettes).

## 9. Practical notes for your stack

* **Graph engine (Rust + petgraph).** Represent the curriculum as a single directed `petgraph::Graph<NodePayload, EdgeKind, Directed>`; `EdgeKind` discriminates `requires`/`supports`/`assesses`. Use filtered views of that graph per layer when running algorithms: `algo::toposort` or `is_cyclic_directed` for the `requires` DAG; `has_path_connecting`/BFS for reachability and LO predicates; simple reachability counts for keystone approximations. Serialize nodes/edges plus the pinned repo commit hash with `serde` (JSON or similar) so snapshots are reproducible and auditable.
* **Visualization (Rerun).** Show three layers togglable: **requires** (backbone DAG), **supports** (scaffolds), **assesses** (assessment→LO links).
* **Validation hooks.** Add checks for (a) DAG property, (b) LO reachability predicate, (c) coverage (observation features vs. rubric criteria), (d) purity (Extraneous sets + construct-irrelevant demands), (e) orphan nodes, (f) item without claim, (g) supports that create self-loops.
* **Slug convention.** Use `{kind}.{short_name}` for every node/edge slug, where `kind ∈ {F,C,P,M,R,LO,A}` denotes factual, conceptual, procedural, metacognitive, principle/misconception tags, learning outcome, or assessment item (e.g., `C.contract_components`, `R.python_docstring`, `LO.clear_contract`, `A.update_rating_docstring`).

## 10. Authoring standards (checklists + example table pattern)

Authoring mid-grain nodes should feel procedural. Every node inherits a chapter-default `grain_level=mid`, authors fill a lightweight “example table” (columns: `support_kind`, `case_tag`, `coverage_tags`, `intended_effect`, `evidence_refs`) that maps 1:1 onto `supports` edges, and validations confirm the table made it into the graph. Treat that table as the canonical place to list examples before copying them into edges. *Ingestion rule:* each table row must have those columns populated; ingestion creates exactly one `supports` edge per row and fails fast if required columns are missing.

### 10.1 Mid-grain node checklist (use for 95% of nodes)

* **Statement:** ≤ 2 sentences, one claim or one method.
* **Scope:** not a grab-bag; if you are listing three different rules, split.
* **Edges:**
  * `requires`: only prerequisites you would actually **test** before using this idea.
  * `supports`: add examples with `support_kind` and `case_tag`.
  * `assesses`: (procedural) reachable to an assessment targeting an LO.
* **Discourse:** at least one `introduce` anchor and one `use/refine` anchor.

### 10.2 Example minimums (by knowledge_type)

* **Factual**
  * ≥ 1 example (`worked_example` or `counterexample`), preferably **typical**.
  * Optional quick check (MC/TF) as assessment.
* **Conceptual**
  * ≥ 1 `analogy` **or** `counterexample` (+ `contrast` intended_effect).
  * ≥ 1 `use`-anchored step.
  * Optional short assessment (identify/recognize).
* **Procedural**
  * ≥ 2 `worked_example` (`typical` + `edge`).
  * ≥ 1 practice **assessment** with observation features that map to the LO rubric.
  * Prefer a **counterexample** showing a common failure (`error_case`).
* **Metacognitive**
  * ≥ 1 `strategy_hint` and a reflection exercise (assessment item is fine).

*(If a node is tagged `intrinsic_load=high`, add one more example and at least one `coverage_tag` per major constraint.)*

**Example table template (copy/paste for authoring):**

```
| support_kind     | case_tag    | coverage_tags            | intended_effect              | step_ref (source_refs)                     | note/rationale                |
|------------------|------------|--------------------------|------------------------------|-------------------------------------------|-------------------------------|
| worked_example   | typical    | ["rating_range"]         | reduce_extraneous_load       | 02_contracts_purpose.ptx#docstring_step   | Happy-path docstring          |
| worked_example   | edge       | ["empty_title","None"]   | reduce_extraneous_load       | 02_contracts_purpose.ptx#edge_case_step   | Boundary & invalid inputs     |
| misconception_fix| error_case | ["author_bias"]          | contrast                     | 02_contracts_purpose.ptx#crash_vignette   | Assumption failure vignette   |
```

## 11. Worked application: Design Recipe chapters

Use §02 (“Method Signature & Purpose Statement”) to sanity-check the policy:

* **Grain.** Nodes such as `C.implicit_contract`, `C.contract_components`, `C.signature_vs_purpose`, `C.explicit_boundary`, `C.living_documentation`, `R.python_docstring` are all mid-grain; tag them `grain_level=mid`.
* **Examples.**
  * `C.implicit_contract` — supports: worked example (`error_case`) from the `None`/`10` crash; `case_tag=error_case`, `intended_effect=motivate` or `contrast`.
  * `C.contract_components` — supports: analogy (napkin → spec), counterexample (missing `Raises` field).
  * `R.python_docstring` — supports: worked example (`typical`) covering Args/Returns/Raises plus a worked example (`edge`) that documents constraints (rating in [1..5]).
  * `C.explicit_boundary` — supports: misconception fix (“author bias” anti-pattern) and optionally a strategy hint on where to enforce vs. document.
* **Practice assessments.** “Quick Practice: Documenting a Function” and the end-of-section contracts exercise become `assessment_item` nodes with `assesses(..., scope=target)` edges into `LO.write.docstring` and `LO.explicit.boundaries`.
* **Validation.** The new queries confirm that `R.python_docstring` owns both `typical` and `edge` worked examples and that each procedure routes to at least one practice assessment aligned to an LO.

## 12. Why these changes hit the two priorities

* **Granularity becomes operational.** Authors set an explicit grain tag, inherit sensible defaults, and rely on audits that push over-bundled nodes into macro bundles and fragments into supports metadata.
* **Examples move from advice to requirements.** Minimum counts, case-tag variety, and Constructive Alignment links make “examples for everything” enforceable, not aspirational.
* **The network stays auditable.** Supports edges now carry the data you already track in example tables (case tags, coverage tags, intended effects), so reviews and queries can prove sufficiency.
* **Authorship friction drops.** Checklists, example tables, and the Design Recipe walkthrough show exactly how to comply without guessing.

## 13. Quality heuristics to stage for later

* **Keystone detector.** Flag nodes that appear on many of the DAG’s longest paths and automatically recommend adding more example rows (worked or counterexample) before treating them as stable keystones.
* **Edge-case entropy.** When a procedure’s supports are all tagged `typical`, prompt the author to add at least one `edge` or `error_case` example so assessments do not drift beyond the documented scaffolds.

## Appendix A: Minimal schema sketch (language-agnostic)

```text
Node {
  id: UUID
  title: String
  knowledge_type: enum { factual, conceptual, procedural, metacognitive, learning_outcome, assessment_item }
  statement: String
  source_refs: [ { path: String, start_line: u32, end_line: u32, revision: String } ]
  confidence: f32                                   // 0..1
  rubric_criteria?: [String]                        // only for LOs
  construct_irrelevant_demands?: [String]           // only for assessment items
  grain_level?: enum { macro, mid, micro }          // default = mid
  intrinsic_load?: enum { low, medium, high }       // optional author signal
  introduction_scope?: enum { in_course, prior, external } // default = in_course
  tags?: [String]                                   // e.g., ["principle"], ["misconception"], ["keystone"]
}

Edge {
  id: UUID
  kind: enum { requires, supports, assesses }
  from: NodeId
  to: NodeId
  attrs:
    confidence?: f32                            // 0..1 self-estimate for this edge
    // requires
    strength?: enum { necessary, strong, helpful }
    rationale?: String
    evidence_refs?: [ { path: String, lines: String } ]

    // supports
    support_kind?: enum { worked_example, analogy, counterexample, misconception_fix, strategy_hint }
    intended_effect?: enum { reduce_extraneous_load, increase_germane_load, motivate, contrast }
    case_tag?: enum { typical, edge, error_case }
    coverage_tags?: [String] // e.g., ["rating_range", "empty_input"]

    // assesses (assessment -> LO)
    evidence_link?: {
      claim: NodeId,                 // must equal the `to` LO id
      observation_features: [String],
      scope: enum { target, enabling }
    }
}
```

This keeps the *theory* (why these kinds and edges exist) distinct from the *implementation* (how we store them). Acyclicity and multiplex layering are standard network notions, so the graph remains analyzable with familiar tools (adjacency matrices, topological sorts, layer‑wise analytics).

## Appendix B: Detailed schema and implementation guide

This appendix inlines the standalone `schema.md` guidance so the white paper now contains both the theoretical framing and one concrete Rust implementation path (petgraph-based) without locking you to a particular backend.

### B.1 Design goals (from the white paper)

* **Multiplex DAG.** Three edge layers—`requires`, `supports`, `assesses`—with arrows flowing from first principles into assessment tasks and finally into outcomes. `requires` must remain acyclic.
* **Auditability.** Every node/edge cites its provenance via `source_refs`/`evidence_refs` (path + line span + revision), never raw textbook excerpts.
* **Constructive alignment.** Every Learning Outcome (LO) satisfies the reachability predicate: some assessment reachable from a first principle via `requires*` assesses the LO with `scope = target`.
* **Evidence coverage & purity.** Target assessments’ observation features cover each LO’s rubric criteria, and extraneous required knowledge is either intended or documented as construct-irrelevant.
* **Purposeful scaffolding.** `supports` edges record the pedagogical device (`support_kind`) and intended cognitive effect, stay fadeable, and avoid self-loops.
* **Evidence-centered assessments.** `assesses` edges store the ECD evidence link (`claim`, `observation_features`, `scope`) plus observation-feature granularity guidance.

### B.2 Node types

| Node Type | Description | Required Fields | Notes |
|-----------|-------------|-----------------|-------|
| `factual` | Terms, symbols, single facts. | `title`, `statement`, `source_refs`, `knowledge_type=factual`, `confidence`. | Statements stay concise; provenance uses repo path + line span. |
| `conceptual` | Categories, principles, mental models. | Same as factual with `knowledge_type=conceptual`. | |
| `procedural` | Algorithms, heuristics, workflows. | Same fields, `knowledge_type=procedural`. | |
| `metacognitive` | Self-monitoring strategies, planning heuristics. | Same fields, `knowledge_type=metacognitive`. | |
| `learning_outcome` | Assessable claim. | `title`, `statement`, `source_refs`, `confidence`, `rubric_criteria`. | Knowledge type fixed to `learning_outcome`; rubric criteria align with observation features. |
| `assessment_item` | Task that elicits evidence. | `title`, `statement` (short prompt summary), `source_refs`, `confidence`, `construct_irrelevant_demands?`. | `knowledge_type=assessment_item`; note any unavoidable non-target demands. |

#### Common node schema

```text
Node {
  id: UUID
  title: String
  statement: String
  knowledge_type: enum { factual, conceptual, procedural, metacognitive, learning_outcome, assessment_item }
  source_refs: [ { path: String, start_line: u32, end_line: u32, revision: String } ]
  confidence: f32
  rubric_criteria?: [String]
  construct_irrelevant_demands?: [String]
  grain_level?: enum { macro, mid, micro }
  intrinsic_load?: enum { low, medium, high }
}
```

### B.3 Edge types

#### B.3.1 `requires` (Knowledge → Knowledge/Assessment)

| Field | Type | Description |
|-------|------|-------------|
| `from` | NodeId | prerequisite node (`knowledge_type ∈ {factual, conceptual, procedural, metacognitive}`). |
| `to` | NodeId | dependent knowledge node **or** assessment item (`knowledge_type ∈ {factual, conceptual, procedural, metacognitive, assessment_item}`). |
| `strength` | enum { `necessary`, `strong`, `helpful` } | Mirrors Knowledge Space Theory surmise strength. |
| `rationale` | NonEmptyString | Natural-language justification referencing the learning dependency. |
| `evidence_refs` | [SourceRef] | Optional extra citations. |
| `confidence` | f32 | Self-estimate (0..1) attached by the extractor to prioritize review. |

Constraints:

* Graph of `requires` edges must be acyclic. Enforce via `ShortestPath` pre-check or topological sort validation for **every** strength value.
* Do **not** allow `learning_outcome` or `assessment_item` as `from`.
* Assessment items are sinks for `requires` (no outgoing `requires` edges).

#### B.3.2 `supports` (Scaffold → Knowledge/LO)

| Field | Type | Description |
|-------|------|-------------|
| `from` | NodeId | Supporting knowledge node (`knowledge_type ∈ {factual, conceptual, procedural, metacognitive}`). |
| `to` | NodeId | Target knowledge node or LO (`knowledge_type ∈ {factual, conceptual, procedural, metacognitive, learning_outcome}`). |
| `support_kind` | enum { `worked_example`, `analogy`, `counterexample`, `misconception_fix`, `strategy_hint`, `rubric_note` } | Pedagogical device classification. |
| `intended_effect` | enum { `reduce_extraneous_load`, `increase_germane_load`, `motivate`, `contrast` } | Declares the cognitive role (aliases `reduce_load`/`germane_load` normalize on ingest). |
| `case_tag` | enum { `typical`, `edge`, `error_case` } | Encodes which scenario the example represents. |
| `coverage_tags` | [String] | Optional fine-grained constraint coverage (e.g., “negative_input”). |
| `evidence_refs` | [SourceRef] | Provenance for the support artifact. |
| `confidence` | f32 | Extractor confidence (0..1) for this scaffold. |

Constraints:

* No self-loops; cycles are allowed but should remain fadeable.
* Removing supports must not change `requires*` reachability; violations indicate the edge really belongs in `requires`.
* Supports into LOs should include rationale describing the framing or motivational strategy they provide.

#### B.3.3 `assesses` (Assessment Item → LO)

| Field | Type | Description |
|-------|------|-------------|
| `from` | NodeId | `assessment_item`. |
| `to` | NodeId | `learning_outcome`. |
| `evidence_link.claim` | NodeId | LO id (must equal the `to` node so the evidence payload is self-contained). |
| `evidence_link.observation_features` | [NonEmptyString] | Measurable cues evaluated for evidence (e.g., “loop invariant holds,” “test suite passes”). |
| `scope` | enum { `target`, `enabling` } | Distinguishes full LO coverage vs. partial probe. Every LO must have ≥ 1 incoming `assesses` edge with `scope = target`. |
| `confidence` | f32 | Extractor confidence (0..1) for the evidence link. |

Guidance:

* `observation_features` must correspond to LO rubric criteria.
* If an assessment targets multiple LOs, give each edge its own `scope` and observation features; scopes may differ (`target` for L1, `enabling` for L2).

### B.4 Derived/supporting structures

1. **SourceSpan meta nodes** (optional)  
   `N::SourceSpan { path, start_line, end_line, revision }` with edges `cites` from any node/edge to the span. Use when the same segment justifies multiple nodes and you need repo revision tracking.
2. **Vector nodes** (optional embedding layer)  
   * `V::KnowledgeEmbedding` storing `[F64]`.  
   * Edges `has_embedding` from any knowledge/LO node to its vector.  
   * Vectors are created from agent-generated summaries (no raw textbook text stored).  
   * Exclude embedding nodes from coverage/DAG audits.

### B.5 Validation and integrity checks

1. **Requires DAG.** For each proposed `requires` edge, use `has_path_connecting` (on the `requires`-only view) or a temporary insert + `toposort` to detect cycles; reject edges that would introduce a cycle or originate from non-knowledge nodes. Assessment items must remain sinks in this layer.
2. **First-principle tagging.** Automatically tag nodes with indegree 0 (in `requires`) as first principles; persist the tag so reports can anchor reasoning paths without manual bookkeeping.
3. **LO reachability predicate.** For every LO, confirm the existence of an assessment satisfying the predicate from §6 (reachable from a first principle via `requires*` and `assesses` the LO with `scope = target`).
4. **Coverage audit.** For each LO, intersect its `rubric_criteria` with the union of `observation_features` across incoming target edges; raise errors for missing criteria and warnings when coverage depends only on enabling edges. Any rubric change automatically re-runs this audit (rubric drift guard).
5. **Purity audit.** For every target assessment‐LO pair, compute `Extraneous(A, L)` and compare it against the LO’s intended knowledge plus the assessment’s declared `construct_irrelevant_demands`. Non-empty differences require explanation or task redesign.
6. **Supports sanity.** Reject self-loops, track clusters that are not fadeable (i.e., removing them changes reachability), and ensure each edge carries an `intended_effect` aligned with CLT.
7. **Evidence presence.** Require at least one `source_refs` entry and `evidence_refs` (when applicable) on every node/edge; missing references surface in nightly audits.
8. **SourceRef guards.** Enforce structured spans: `path` non-empty, `start_line >= 1`, `end_line >= start_line`, `revision` present and matches `[0-9a-f]{7,40}`. Reject inserts that violate these or that lack `revision` equal to the graph’s pinned commit.

9. **Example minimums.**
   * Enforce the §6 “Example minimums” item (backed by the §5.5 table) by checking each knowledge type’s minimum count and `case_tag` variety.

10. **Procedural practice.**
   * Enforce the §6 “Procedural practice” rule: each procedural node must be on a path to ≥ 1 assessment with `assesses(..., scope=target)` to some LO.

11. **Variety check.**
    * Enforce the §6 “Variety check” so every procedural node has both `typical` and `edge` (or `error_case`) examples recorded via `supports`.

12. **Assessable Atom Test.**
    * Enforce the §6 AAT: statements must be ≤ 2 sentences and every node must either serve as a prerequisite (≥ 1 distinct `requires` consumer) or have ≥ 2 anchored TeachingSteps; otherwise flag for merge/fold.

13. **Keystone analysis.**
    * Compute betweenness centrality over the `requires` DAG; any node above the Keystone threshold must own ≥ 2 `worked_example` supports or trigger a **high-risk** alert.

14. **Granularity audits.**
    * Over‑bundling detector: `len(statement) > 2 sentences` AND `in_degree(requires) ≥ 4` → suggest `grain_level=macro` or split.
    * Fragment detector: `len(statement) < 15 tokens` AND no `assesses` AND no `supports` → suggest fold into an example (`grain_level=micro`).
    * **Complexity warning:** if `in_degree(requires) ≥ 4` **and** `intrinsic_load != high`, warn the author to bump the load tag or split the node; heavy prerequisite fans imply higher cognitive load.

15. **Fading readiness.**
    * For nodes with `intrinsic_load=high`, require ≥ 2 supports; warn if the set is not fadeable (removing supports changes `requires*` reachability).

### B.6 Implementation notes: Rust + petgraph core

We implement the multiplex graph in-process with `petgraph` and serialize snapshots with `serde`.

* **Core types (sketch)**

  ```rust
  use petgraph::{graph::Graph, Directed};

  pub type CurriculumGraph = Graph<NodePayload, EdgeKind, Directed>;

  #[derive(Clone, Debug)]
  pub struct NodePayload {
      pub id: uuid::Uuid,
      pub title: String,
      pub statement: String,
      pub knowledge_type: KnowledgeType,
      pub source_refs: Vec<SourceRef>,
      pub confidence: f32,
      pub rubric_criteria: Vec<String>,
      pub construct_irrelevant_demands: Vec<String>,
      pub grain_level: Option<GrainLevel>,
      pub intrinsic_load: Option<IntrinsicLoad>,
      pub introduction_scope: IntroductionScope,
      pub tags: Vec<String>,
  }

  #[derive(Clone, Debug)]
  pub enum EdgeKind {
      Requires(RequiresAttrs),
      Supports(SupportsAttrs),
      Assesses(AssessesAttrs),
  }
  ```

  *Represent LOs either as a dedicated node variant or as `knowledge_type="learning_outcome"`; assessments remain sinks for `requires`.*

* **Layered algorithms.** Build filtered views per edge kind:
  * `requires` view → `petgraph::algo::toposort` / `is_cyclic_directed` to enforce DAG; `has_path_connecting` for fast “would this edge close a cycle?” checks before insert.
  * `assesses` + `requires` → reachability for LO predicate: first-principle → … → assessment → LO.
  * `supports` view → fadeability checks: removing supports must not change `requires*` reachability.
  * Keystone approx = `in_reach(n) * out_reach(n)` via repeated BFS/DFS on the `requires` view.

* **Validation hooks.** Reuse the Rust validators in `schema::validate` for spans, enums, and edge constraints before mutating the graph. Prefer “fail fast” on ingest.

* **Persistence.** Serialize `CurriculumGraph` nodes/edges plus the pinned source commit hash (and optional remote URL) to JSON/CBOR. Treat each commit as an immutable snapshot; regenerate on repo changes rather than editing in place.

### B.7 Example graph operations (Rust-oriented recipes)

1. **insert_factual**
   *Signature*: `fn insert_factual(g: &mut CurriculumGraph, payload: NodePayload) -> NodeIndex`  
   Validates `knowledge_type=factual`, `source_refs`, then inserts node.

2. **add_requires_if_acyclic**
   *Signature*: `fn add_requires_if_acyclic(g: &mut CurriculumGraph, from: NodeIndex, to: NodeIndex, attrs: RequiresAttrs) -> Result<EdgeIndex, CycleError>`  
   Steps: validate attrs → check `has_path_connecting(requires_view(g), to, from, None)` → if true, return cycle error; else insert `EdgeKind::Requires`.

3. **lo_alignment**
   *Signature*: `fn lo_alignment(g: &CurriculumGraph, lo: NodeIndex) -> AlignmentReport`  
   For each assessment with `assesses(lo, scope=target)`, report whether some first‑principle node reaches that assessment via `requires*`; include missing-coverage rubric criteria.

4. **borrow_ahead**
   *Signature*: `fn borrow_ahead(g: &CurriculumGraph, episode: &str) -> Vec<BorrowAheadWarning>`  
   Find `TeachingStep` nodes in the episode whose `anchors(impact=use)` targets a knowledge node that lacks any prior `introduce` anchor in the same or earlier episode (configurable scope); grade severity per §6.

5. **add_precedes_if_acyclic (discourse layer)**
   *Signature*: `fn add_precedes_if_acyclic(g: &mut CurriculumGraph, from: NodeIndex, to: NodeIndex, episode: &str)`  
   Restrict to `TeachingStep` nodes of that episode; reject if `has_path_connecting` already links `to → from` within that episode’s `precedes` edges.

6. **missing_examples_report**
   *Signature*: `fn missing_examples(g: &CurriculumGraph) -> Vec<NodeIndex>`  
   Return knowledge nodes whose incoming `supports` lack required `support_kind`/`case_tag` variety per §6 table.

   ```hx
   QUERY MissingExamplesByType() =>
     K <- N<Knowledge>()
     F <- K::WHERE(_::{knowledge_type}::EQ("factual"))::Out<Supports>::COUNT()
     C <- K::WHERE(_::{knowledge_type}::EQ("conceptual"))::Out<Supports>::COUNT()
     P <- K::WHERE(_::{knowledge_type}::EQ("procedural"))::Out<Supports>::WHERE(_::{support_kind}::EQ("worked_example"))::COUNT()
     M <- K::WHERE(_::{knowledge_type}::EQ("metacognitive"))::Out<Supports>::WHERE(_::{support_kind}::EQ("strategy_hint"))::COUNT()
     RETURN {
       factual_missing: K::WHERE(_::{knowledge_type}::EQ("factual"))::WHERE(F::EQ(0))::{id,title},
       conceptual_missing: K::WHERE(_::{knowledge_type}::EQ("conceptual"))::WHERE(C::EQ(0))::{id,title},
       procedural_missing_we: K::WHERE(_::{knowledge_type}::EQ("procedural"))::WHERE(P::< 2)::{id,title},
       metacognitive_missing: K::WHERE(_::{knowledge_type}::EQ("metacognitive"))::WHERE(M::EQ(0))::{id,title}
     }
   ```

9. **Procedural nodes lacking practice assessments**

   ```hx
   QUERY ProceduralPracticeGaps() =>
     P <- N<Knowledge>()::WHERE(_::{knowledge_type}::EQ("procedural"))
     A <- P::Out<Requires*>()::To(N<AssessmentItem>())
     L <- A::Out<Assesses>()::To(N<LearningOutcome>())::WHERE(_::{scope}::EQ("target"))
     RETURN P::WHERE(L::{count}::EQ(0))::{ id, title }
   ```

10. **Variety check: ensure typical + edge examples for procedures**

   ```hx
   QUERY ProceduralVarietyGaps() =>
     P <- N<Knowledge>()::WHERE(_::{knowledge_type}::EQ("procedural"))
     Etyp <- P::Out<Supports>()::WHERE(_::{case_tag}::EQ("typical"))::COUNT()
     Eedge <- P::Out<Supports>()::WHERE(_::{case_tag}::EQ("edge"))::COUNT()
     RETURN P::WHERE(Etyp::EQ(0) OR Eedge::EQ(0))::{ id, title }
   ```

11. **Minimum viable scaffolding**

   ```hx
   QUERY ScaffoldingGaps() =>
     K <- N<Knowledge>()
     factual_gaps <- K
       ::WHERE(_::{knowledge_type}::EQ("factual"))
       ::WHERE(
         Out<Supports>()
           ::WHERE(_::{support_kind}::IN(["worked_example","counterexample"]))
           ::COUNT()::<1
       )::{id,title}
     conceptual_gaps <- K
       ::WHERE(_::{knowledge_type}::EQ("conceptual"))
       ::WHERE(
         Out<Supports>()
           ::WHERE(_::{support_kind}::IN(["analogy","counterexample"]))
           ::COUNT()::<1
       )::{id,title}
     procedural_gaps <- K
       ::WHERE(_::{knowledge_type}::EQ("procedural"))
       ::WHERE(
         Out<Supports>()::WHERE(_::{support_kind}::EQ("worked_example"))::COUNT()::<2
         OR
         Out<Supports>()::WHERE(_::{case_tag}::IN(["edge","error_case"]))::COUNT()::EQ(0)
       )::{id,title}
     RETURN {
       factual_gaps: factual_gaps,
       conceptual_gaps: conceptual_gaps,
       procedural_gaps: procedural_gaps
     }
   ```

12. **Keystone approximation (cheap deterministic score)**

   ```hx
   QUERY KeystoneApprox() =>
     K <- N<Knowledge>()
     in_reach <- K::In<Requires*>()::COUNT()
     out_reach <- K::Out<Requires*>()::COUNT()
     score <- in_reach * out_reach
     RETURN K::{ id, title, score }::ORDER_DESC(score)
   ```

### B.8 Versioning & drift

`source_refs` spans are only trustworthy for the precise repo commit that produced them. Rather than trying to “slide” line numbers after upstream edits, pin every graph to a commit hash and treat updates as fresh ingestions or explicit migrations.

#### Workflow

1. Capture `git rev-parse HEAD` (and optionally the remote URL) before ingestion; stamp those values into `graph_metadata` and into every `source_refs.revision`.
2. Ingest nodes/edges using that pinned checkout. The resulting database row set is immutable with respect to the commit hash.
3. When the course repository changes, rerun the entire extraction for the new commit (or run a purpose-built migrator). Do not attempt to offset line numbers by hand or via naive diffs; accuracy depends on re-reading the updated sources.

#### Enforcement hooks

* Schema validation rejects any `source_refs` entry lacking a revision or whose revision does not match the graph’s recorded commit.
* Ingestion code fails fast if `git status` is dirty or if the requested target commit differs from what is already stored, forcing collaborators to acknowledge they are replacing the graph.
* Observability jobs can emit warnings whenever a downstream query references a graph whose commit no longer matches the latest upstream default branch, prompting a fresh run.

### B.9 Next steps

1. **Finalize the petgraph data model** using the shapes above (single `CurriculumGraph` with `EdgeKind` discriminants; optional separate LO node type vs. `knowledge_type="learning_outcome"`).
2. **Codify enums** in Rust (`KnowledgeType`, `SupportKind`, `Strength`, `AssessmentScope`) and expose typed builders/validators for each edge kind.
3. **Author validation routines** (reachability report, coverage audit, purity/construct-irrelevant scan, cycle detector) and run them in CI on every regenerated snapshot.
4. **Backfill initial data** for the Design Recipe units to test end-to-end: insert nodes, add edges, run `toposort`/reachability, inspect LO coverage; serialize the snapshot with its commit hash.

Once these steps are complete, the implementation will reflect the white paper’s theoretical commitments while remaining practical for a Rust + petgraph stack.

## Appendix C: Discourse layer (MDC) schema

```text
// New node kind
TeachingStep {
  id: UUID
  title: String
  statement: String                  // concise summary of the authored step
  purpose: enum { setup, idea, use, consolidate }
  method_tags: [String]              // e.g., "naive-first", "breakdown", "analogy", "worked-example"
  episode: String                    // section/lesson anchor
  source_refs: [SourceRef]           // { path, start_line, end_line, revision }
}

// New relations
precedes {
  from: TeachingStepId
  to: TeachingStepId
  // constraint: acyclic per episode
}

anchors {
  from: TeachingStepId
  to: KnowledgeOrLOId                // includes assessment_item and learning_outcome
  impact: enum { introduce, use, refine, motivate, target }
}
```

## Appendix D: Derived narrative views (optional)

You may expose views for collaborators who like “motivation/evolution” edges **without storing them**:

* **Derived Motivation:** If S (setup) precedes I (idea) and `anchors(S → K, motivate)` and `anchors(I → K, introduce)`, render a virtual edge **K —motivates→ K** in a pedagogy view.
* **Derived Evolution:** If R (consolidate) `anchors(refine → K₂)` and an earlier step introduced K₁ in the same topic family, render a virtual edge **K₁ —evolves→ K₂**.

These are visualization projections; the database remains MDC‑clean.
