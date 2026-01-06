# Qyra

**Qyra** is a library designed to solve complex combinatorial optimization problems by bridging the gap between the predictive intuition of **Deep Learning** and the mathematical rigor of **System-Level Programming**.

While traditional solvers struggle with scale and neural networks struggle with strict constraints, Qyra combines both: it uses GNNs to "guide" a high-performance backend towards the optimal solution.

---

## The Philosophy: "AI-Guided Exactness"

In complex systems, whether financial markets, cloud clusters, or biological networks, the best solution is often hidden in the relationships (topology) between entities, not just their individual values.

Qyra solves optimization problems in three distinct phases:

1.  **Translation:** Converts abstract business domains into a unified mathematical language (QUBO - Quadratic Unconstrained Binary Optimization).
2.  **Guidance:** Uses Graph Neural Networks (GNNs) to analyze the topology of the data, predicting which nodes are most likely to be part of the solution (Active Learning).
3.  **Execution:** Passes the mathematical matrix and the GNN's "intuition" to a optimized solver, which navigates the search space to find the global minimum energy state.

## Solvable Problem Classes

Qyra is built around "Translators": specialized logic units that frame your problem for the solver.

### 1. Affinity & Clustering Problems
*For scenarios where items must be grouped based on hidden relationships.*

* **The Logic:** The solver looks for items that move together or share latent features. It maximizes the "magnetic attraction" between correlated entities while applying regularization penalties to prevent trivial solutions (like selecting everything).
* **How it works:** It transforms correlation matrices and topological proximity into negative energy states (rewards), forcing the solver to "clump" related items together.

### 2. Conflict & Bin Packing Problems
*For scenarios involving limited resources, distinct constraints, or mutual exclusivity.*

* **The Logic:** The solver operates under a "penalty regime." It assigns a high energy cost (punishment) to forbidden pairs or over-capacity buckets. The goal is to fit the maximum number of high-value items into the solution without triggering a constraint violation.
* **How it works:** It utilizes adjacency matrices to map out "danger zones." If the GNN predicts a future conflict (e.g., two servers likely to crash if paired), it preemptively creates a high-penalty barrier between them.

---

## Architecture Overview

* **Interface Layer:** Handles data ingestion and creates the agnostic "Decoder Ring," ensuring that the final output is returned as domain objects, not just raw numbers.
* **Topological Layer:** Processes the graph structure to generate probability weights (Priors).
* **Solver Layer:** A compiled binary engine that performs the heavy lifting of combinatorial search, operating outside the Python Global Interpreter Lock (GIL) for maximum speed.

---

## License

**Proprietary Software.**
Copyright © 2026 Guilherme Amaro. All rights reserved.