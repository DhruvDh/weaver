# Slide 1: Network Science as a Lens

- **Core Idea:** Network science is not just a bag of algorithms; it is a perspective.
- **The Promise:** By modeling entities and relationships, we can discover simple, insightful truths hidden within complex systems.
- **Examples:**
  - Centrality reveals influencers in social graphs.
  - Clustering reveals functional modules in biological protein networks.
  - Path analysis reveals bottlenecks in logistics.

# Slide 2: The Constructivist Reality

- **The Catch:** Networks are rarely "found in nature" ready-made.
- **The Challenge:** We don't walk into the woods and trip over an adjacency matrix.
- **The "Lens" Design:** The network *is* the model. We have to choose:
  - What counts as a node?
  - What counts as an edge?
- **The Consequence:** These choices are not neutral. They determine strictly what truths can be uncovered.
  - *Bad design:* GIGO (Garbage In, Garbage Out).
  - *Good design:* Emergent properties appear.

# Slide 3: Network Design as a Creative Problem

- **The Thesis:** Uncovering truth depends primarily on the *design of the lens*.
- **The Art:** It is an act of ontology engineering.
  - Too coarse? You miss the dynamics.
  - Too fine? You drown in noise.
- **The Goal:** To craft a representation where the topological properties of the graph map meaningfully to the semantic properties of the domain.

# Slide 4: Case Study: The "Curriculum" Network

- **The Domain:** Educational Textbooks (Computer Science).
- **The Raw Material:** Unstructured, linear prose. Chapter 1 follows Chapter 2.
- **The Design Question:** How do we turn a linear book into a network that reveals *pedagogical* truth?
- **The Naive Approach:** 
  - Nodes = Paragraphs.
  - Edges = "Follows".
  - *Result:* A line. Zero insight.

# Slide 5: Designing the "Weaver" Lens

- **Our Creative Choice:** The "Dual-Layer" Ontology.
- **Layer 1: The Knowledge Graph (The Logic)**
  - *Nodes:* Concepts, Facts, Procedures (not paragraphs).
  - *Edges:* `requires` (Prerequisite DAG), `supports` (Scaffolding).
  - *Insight:* Reveals the "Deep Structure" of the subject matter.
- **Layer 2: The Discourse Graph (The Narrative)**
  - *Nodes:* Teaching Steps (Narrative moments).
  - *Edges:* `precedes` (Time), `anchors` (Pedagogical Intent).
  - *Insight:* Reveals the author's "Strategy" for teaching that structure.

# Slide 6: Why This Specific Design Matters

- **The "Truths" This Lens Uncovers:**
  - **Keystones:** Nodes with high betweenness centrality in the `requires` layer are the "load-bearing" concepts of the course.
  - **Orphans:** Concepts introduced in the Narrative (Discourse) but disconnected in the Logic (Knowledge).
  - **Alignment:** Does the structure of the Assessment graph map to the structure of the Learning Outcome graph?
- **Conclusion:** By designing a specific, opinionated lens (Weaver), we transform a flat text into a queryable structure where network science tools (centrality, reachability, topological sort) yield educational insights.

# Slide 7: The Engineering Challenge (Realizing the Design)

- **The Obstacle:** LLMs are probabilistic, not logical.
- **The "Missing Target" Problem:**
  - If you ask an agent to build the network in one pass, it tries to connect *Concept A* to *Concept B* before it has read the chapter containing *Concept B*.
  - *Result:* Hallucinated nodes, broken edges, and a "noisy lens."
- **The Implication:** A bad engineering process destroys the beautiful ontological design.

# Slide 8: The Architectural Solution: Two-Phase Construction

- **Phase 1: Harvest (The Nodes)**
  - *Goal:* Identify the entities.
  - *Constraint:* Agents can **only** create nodes. No connections allowed.
  - *Result:* A "Bag of Concepts" – the raw material.
- **The Barrier:** Deduplication and Persistence. (Cleaning the lens).
- **Phase 2: Weave (The Edges)**
  - *Goal:* Draw the connections.
  - *Constraint:* Agents can **only** connect existing nodes.
  - *Result:* A valid, consistent graph structure.

# Slide 9: The Topology of the Resulting Network

- **Structure:** A **Multiplex Network**.
  - Same set of nodes, different layers of edges.
- **Layer 1: `requires` (The Prerequisite Backbone)**
  - *Topology:* **Directed Acyclic Graph (DAG)**.
  - *Property:* Enables Topological Sort (a valid teaching sequence).
  - *Analysis:* Longest path analysis = Course depth.
- **Layer 2: `supports` (The Scaffolding)**
  - *Topology:* Cyclic, dense.
  - *Property:* Redundancy.
  - *Analysis:* "Fadeability" (Can we remove these edges and still have a path? If yes, it's a scaffold. If no, it's a dependency).

# Slide 10: Putting the Lens to Use (Analyst Mode)

- **The Pivot:** Now that we have a high-fidelity network, we can ask questions the text couldn't answer.
- **Metric: Betweenness Centrality**
  - *Question:* "Which concepts are the bottlenecks?"
  - *Weaver Tool:* `graph_keystone()`
- **Metric: Reachability**
  - *Question:* "Do our assessments actually cover the stated learning outcomes?"
  - *Weaver Tool:* `graph_lo_alignment_summary()`
- **Metric: Structural Gaps**
  - *Question:* "Are we teaching concepts we never use?"
  - *Weaver Tool:* `graph_discourse_orphans()`

# Slide 11: Summary

- **Network Science** is the method of inquiry.
- **Ontology Design** is the creative act of defining the lens.
- **Weaver** is the engineering system that polishes that lens.
- **The Result:** We turn the "art" of curriculum design into the "science" of network analysis.
