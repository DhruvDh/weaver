# White Paper: First-Principles Learning Network

> **Purpose.** We want a defensible, auditable way to represent how learners build understanding “from first principles” all the way to assessed learning outcomes. We do this by encoding the learning process as a **directed acyclic graph (DAG)** in which each edge encodes *why* one thing should precede or support another. The result doubles as an instruction plan, a quality checklist, and a scaffold the agents can extend.


## 1. Why a *directed* network, and why *acyclic*?

**First principles reasoning** is a *directional* activity: you start from primitives and build up. In a network, that means arrows go from enabling knowledge to the knowledge that depends on it. If you can lay out the nodes so all arrows point the same way (earlier → later), the graph is **acyclic**. Acyclic directed networks have nice properties: they admit a **topological order** (a valid teach/learn sequence), and their **adjacency matrix** can be permuted into a strictly upper‑triangular form—both handy for checking that our structure is coherent and cycle‑free.  

Keeping edges typed in **separate layers** (requires/supports/assesses) also fits the standard notion of a **multiplex/multilayer network**, where the same nodes are connected by different kinds of ties represented per layer. This gives us clean semantics and clean analytics.  

> **Takeaway.** We aim for a DAG so that “reasoning chains” are literally **paths** from first principles to outcomes, while edge *types* live in parallel layers that we can analyze or visualize separately. 


## 2. The theoretical foundations (what each theory is, why it matters, how we use it)

### 2.1 Constructive Alignment (Biggs & Tang)

* **What it is.** A curriculum design principle: align **Intended Learning Outcomes (ILOs)**, **teaching/learning activities**, and **assessment**.
* **Why it exists.** To make sure students are practicing and being assessed on the very outcomes we claim to value, not on side skills.
* **How we use it.** In our network, **LO nodes** must (a) be **required by** the relevant first‑principle nodes and (b) **assess** to concrete assessment items designed to elicit evidence for those LOs. The edges enforce the “alignment” by construction.

### 2.2 Knowledge Space Theory (Doignon & Falmagne)

* **What it is.** A mathematical framework for **prerequisite** relations among items/skills, yielding *quasi‑orders* and feasible “knowledge states.”
* **Why it exists.** To model *which sets of skills are reachable* given prerequisites, and to support adaptive sequencing/diagnosis.
* **How we use it.** We treat **requires** edges as KST‑style surmise relations; the induced partial order is the spine of our DAG. If the agent proposes a cycle, it violates feasibility and must be revised.

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


## 3. Our node ontology: **what** we model (and why)

Each node has a **knowledge type** from Revised Bloom’s Knowledge Dimension. That gives us theoretical grounding while staying simple enough for an MVP:

* **Factual** — terms, symbols, conventions, simple facts (e.g., “A method’s signature is name + parameter list in Java.”)
* **Conceptual** — categories, principles, models (e.g., “Preconditions vs. postconditions divide responsibility.”)
* **Procedural** — methods, algorithms, heuristics (e.g., “Apply the OOP design recipe to implement a method.”)
* **Metacognitive** — strategies for monitoring/planning (e.g., “Start with a stub + tests to control complexity.”)

Two additional “meta” nodes are used sparingly (both grounded in ECD/CLT use-cases):

* **Learning Outcome (LO)** — an assessable claim about what the learner can do. (Constructive Alignment anchor.)
* **Assessment Item** — a task that can elicit evidence for one or more LOs. (ECD anchor.)

> **Why only these?** They cover the validated distinctions: *what knowledge is* (Bloom), *what progression requires* (KST), and *how we know we’ve achieved it* (CA + ECD). Everything else (e.g., examples, analogies, counterexamples, rubrics) appears via **supports** edges rather than proliferating node kinds—keeping the ontology lean and defensible.


## 4. Edge types: the *meaning* of arrows

We use exactly three edge types, each with its own **layer** in a multiplex representation (same node set, different ties). 

1. **requires (A → B)**

   * **Semantics.** Mastery of *A* is a **prerequisite** for dependable mastery of *B*. This is the KST backbone.
   * **Constraints.** Must preserve a **DAG** (no cycles). Enforced by topological checks/Kahn‑style peeling; the adjacency can be permuted to strictly upper‑triangular if acyclic. 
   * **Typical spans.** factual→conceptual, conceptual→procedural, factual→procedural, and any → LO (when the LO aggregates prior knowledge).

2. **supports (S → B)**

   * **Semantics.** *S* improves learnability or robustness of *B* (worked examples, analogies, counterexamples, common misconceptions with repairs, strategy prompts). Not strictly necessary, but pedagogically **scaffolding**.
   * **Constraints.** Should not create cycles that would imply “you can’t understand S without B” unless S is *also* required—in which case it belongs in **requires**.

3. **assesses (LO → Item)**

   * **Semantics.** The **LO** is the **claim**, the **item** is the **task** intended to elicit evidence for that claim (ECD). The edge stores the evidence link, not content text.
   * **Constraints.** Items are *sinks* in the “first principles → LO → assessment” flow; they do not point to knowledge nodes (no teaching by test).

> **Why this orientation?** The overall flow is “first principles → composed understanding → **learning outcome** → **assessment**.” That preserves acyclicity and matches constructive alignment (LOs precede assessment design).


## 5. What goes **on** edges and nodes (attributes, with justification)

### 5.1 Minimal node attributes (all nodes)

* **`title`** — stable, human‑readable identifier.
* **`knowledge_type`** ∈ {factual, conceptual, procedural, metacognitive, LO, assessment_item}. (Revised Bloom + CA/ECD anchors.)
* **`statement`** — a concise, self‑contained sentence or two (for LOs: performance statement).
* **`source_ref`** — *paths and ranges* to your PreTeXt sources (no excerpted text), e.g., `source/DesignRecipe/section.xml#L120–L180`.
* **`confidence`** — model’s self‑estimate (0–1) to flag uncertain extractions.

### 5.2 Edge attributes by type

**requires**

* **`strength`** ∈ {necessary, strong, helpful}. *Necessary* edges approximate KST surmise relations; *helpful* approximates “facilitates but not required.”
* **`rationale`** — a short natural‑language justification (why *A* really precedes *B*).
* **`evidence_ref`** — optional parent pointers to course sources (again, file/line references only).

**supports**

* **`support_kind`** ∈ {worked_example, analogy, counterexample, misconception_fix, strategy_hint, rubric_note}. (CLT/ECD‑inspired scaffolds.)
* **`intended_effect`** ∈ {reduce_extraneous_load, increase_germane_load, motivate, contrast}. (So the system “knows” why the support exists.)

**assesses**

* **`evidence_link`** — ECD pointers: `{claim: LO_id, observation_features: [...], scoring_rule: ...}`.
* **`scope`** ∈ {target, enabling} to indicate whether an item targets the LO directly or only sub‑parts.

> **Why file/line *references* but no text?** You said you don’t want to store textbook prose. These references are enough to justify nodes/edges and let humans audit decisions later.


## 6. How this becomes a *coherent* network (and how we check it)

1. **Acyclicity by design.** The main layer (requires) must pass an acyclicity test (peel off nodes with no outgoing edges until none remain). If we stall with nodes remaining, a cycle exists and must be resolved. 

2. **Topological schedule = learning path.** The topological order yields a teach/learn sequence. Longest paths reveal *depth*; branching reveals *parallelizable* topics. (Adjacency‑matrix checks/triangularity are a quick sanity test.)  

3. **Alignment coverage.** Every **LO** must (a) be reachable from first‑principle nodes via requires, and (b) have at least one **assesses** edge to an item whose evidence model matches the LO. (Constructive Alignment + ECD.)

4. **Support is purposeful.** Each **supports** edge specifies its *intended effect* on load or motivation; supports should cluster around high‑intrinsic‑load procedures (worked examples first, then faded).

5. **Minimal multiplex.** Keep just three layers (requires / supports / assesses). This gives interpretability without edge‑type explosion, yet is true to the standard concept of multiplex networks. 


## 7. What success looks like (evaluating the network we’ve built)

* **No cycles in `requires`.** The topological peel completes; the adjacency permutes to strictly upper‑triangular. (Automatable invariant.) 
* **Coverage & connectedness.** Every LO has at least one incoming requires path from first‑principle nodes **and** at least one outgoing assesses edge to an item. (Constructive alignment achieved.)
* **Purposeful scaffolding.** High intrinsic‑load procedures are supported by worked examples or analogies early; supports can be *faded* (removable without breaking reachability) as mastery increases.
* **Auditability.** Every node/edge can be traced to **source_ref** or **evidence_link**—file names and line ranges, not copied text (fits your Helix‑DB intent).
* **Reasoning paths exist.** For each LO, you can traverse a **path** that reads like a human‑intelligible reasoning chain from primitives to outcome—this is the “first principles” promise realized as graph structure.
* **Multiplex clarity.** Analysts can slice by layer (requires vs. supports vs. assesses) and by knowledge type (factual/conceptual/procedural/metacognitive) to answer “what to teach, what helps, how we’ll know.”


## 8. A tiny walk‑through (illustrative, domain‑agnostic)

* **Factual → Conceptual → Procedural → LO → Assessment**

  1. *Factual:* “A method’s signature is name + parameter list.”
  2. *Conceptual:* “Preconditions/postconditions divide caller/callee responsibility.”
  3. *Procedural:* “Apply the design recipe to implement method `average` with pre/post.”
  4. *LO:* “Given a requirement, produce a correct, tested Java method by applying the OOP design recipe.”
  5. *Assessment Item:* A short, autograded task prompting the same.
     **Edges:** factual → conceptual (**requires**), conceptual → procedural (**requires**), worked example → procedural (**supports**), LO → item (**assesses**).
     CLT predicts the worked example early will help; ECD demands the item’s scoring/evidence rules map to the LO.


## 9. Practical notes for your stack

* **Helix‑DB.** Store nodes/edges plus `source_ref`/`evidence_link`. Avoid storing textbook text; keep only file paths and line ranges.
* **Visualization (Rerun).** Show three layers togglable: **requires** (backbone DAG), **supports** (scaffolds), **assesses** (LO→item links).
* **Validation hooks.** Add checks for (a) DAG property, (b) LO coverage, (c) orphan nodes, (d) item without claim, (e) supports that create cycles.


## Appendix (for implementers): minimal schema sketch (language‑agnostic)

```text
Node {
  id: UUID
  title: String
  knowledge_type: enum { factual, conceptual, procedural, metacognitive, LO, assessment_item }
  statement: String
  source_ref: [ { path: String, lines: String } ]   // e.g., "source/DesignRecipe/section.xml#L120–L180"
  confidence: f32                                   // 0..1
}

Edge {
  id: UUID
  kind: enum { requires, supports, assesses }
  from: NodeId
  to: NodeId
  attrs:
    // requires
    strength?: enum { necessary, strong, helpful }
    rationale?: String
    evidence_ref?: [ { path: String, lines: String } ]

    // supports
    support_kind?: enum { worked_example, analogy, counterexample, misconception_fix, strategy_hint, rubric_note }
    intended_effect?: enum { reduce_extraneous_load, increase_germane_load, motivate, contrast }

    // assesses
    evidence_link?: {
      claim: NodeId,                 // LO id (redundant with 'to' but explicit for ECD)
      observation_features: [String],
      scoring_rule: String
    }
}
```

This keeps the *theory* (why these kinds and edges exist) distinct from the *implementation* (how we store them). Acyclicity and multiplex layering are standard network notions, so the graph remains analyzable with familiar tools (adjacency matrices, topological sorts, layer‑wise analytics).
