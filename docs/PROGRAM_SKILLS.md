# PROGRAM_SKILLS.md — builderbrain

## 0. purpose

Program skills are discrete latent tokens (z) mapped to small adapters (LoRA/hypernets). They represent reusable **primitives** that the builder rail can compose into larger behaviors. This doc defines their lifecycle: **birth → reuse → merge → distill → retire**.

---

## 1. representation

* **token:** discrete `z ∈ {1..K}` sampled via ST-Gumbel.
* **adapter:** small parameterized block (LoRA/MLP/hypernet) attached to builder rail.
* **embedding:** each program has an embedding vector used in Compose and Graph heads.

Typical adapter size: 0.1–1% of base model params.

---

## 2. birth

### 2.1 triggers

* high residual build loss (ℓ_build) despite reuse.
* persistent graph mismatch (ℓ_g2g high).
* novelty requirement in curriculum.

### 2.2 mechanics

* allocate new program id `z_new`.
* initialize adapter from random or from nearest cluster centroid.
* log birth event with metadata (trigger, time, parent skills).

### 2.3 constraints

* birth rate capped (≤1 per N steps).
* penalized by ℓ_Δθ (param growth cost).

---

## 3. reuse

### 3.1 objective

* encourage model to call existing programs often; reduce entropy of z distribution.

### 3.2 metrics

* skill frequency histogram.
* entropy H(z).
* reuse loss ℓ_reuse.

### 3.3 desired pattern

* long-tail distribution with reuse of core skills; occasional exploration of rarer skills.

---

## 4. merge

### 4.1 motivation

* prevent adapter sprawl.
* combine redundant or highly similar programs.

### 4.2 criteria

* cosine similarity of embeddings > τ.
* functional overlap measured by ablation (performance unaffected when one removed).

### 4.3 mechanics

* average parameters (or distill into one).
* update graphs to redirect edges to merged skill.
* retire old id; keep mapping for logs.

---

## 5. distill

### 5.1 purpose

* compress multiple programs into smaller shared adapter.
* improve efficiency and generalization.

### 5.2 method

* train teacher model with full skill set.
* distill into student with reduced K.
* align program embeddings via Procrustes/CCA.

---

## 6. retire

* programs unused for long horizon (frequency < ε) flagged.
* distill knowledge if needed, then deactivate.
* keep id in registry for reproducibility; mark `status=retired`.

---

## 7. clustering & libraries

### 7.1 clustering

* periodically cluster program embeddings (k-means, spectral).
* identify redundant or related skills.
* drive merge/distill decisions.

### 7.2 libraries

* maintain domain-specific skill libraries (API, UI, robotics).
* reuse across runs by freezing/adapting embeddings.

---

## 8. dashboards

* skill frequency histograms.
* entropy of z distribution.
* adapter growth curves.
* similarity matrices of program embeddings.
* active vs retired counts.
* merge/distill events log.

---

## 9. failure smells

* **mode collapse:** single skill dominates; entropy ↓ to near 0.
* **sprawl:** uncontrolled growth; many births, few merges.
* **dead adapters:** large fraction unused.
* **identity drift:** skill ids change semantics between runs.
* **merge errors:** merged skills degrade performance.

---

## 10. ethos

skills are **lego bricks**, not coral reefs. keep them small, reusable, and prunable. birth rarely, reuse aggressively, merge often, distill periodically, retire gracefully.
