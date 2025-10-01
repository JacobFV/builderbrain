# LOSS_BANK.md — builderbrain

## 0. purpose

Loss bank catalogs all training objectives in builderbrain. For each loss we specify:

* **formal definition**
* **units / scale / expected range**
* **normalization methods**
* **role in lagrangian**
* **diagnostic metrics**
* **common failure smells**
* **interactions with other losses**

This doc is the canonical reference for configuring, interpreting, and debugging training runs.

---

## 1. task loss (ℓ_task)

* **def:**

  * LM mode: cross-entropy per token.
  * Policy mode: negative log-likelihood of actions.
  * RL mode: advantage-weighted log-prob, or TD-λ value regression.
* **units:** nats/token or per-step utility.
* **range:** [0, ∞), but usually ~1–5 nats/token on text.
* **normalization:** none.
* **role:** primary objective, anchor for all constraints.
* **diagnostics:** perplexity, average return, reward curves.
* **failure smells:** stagnation at high loss; divergence after online RL updates.
* **interactions:** All auxiliary losses must not overwhelm ℓ_task; dual λ should stabilize so ℓ_task continues to decrease.

---

## 2. grammar prior loss (ℓ_cfs)

* **def:**
  [ ℓ_{cfs} = \mathbb E_t [ \max(0, -\log P_{CFG}(x_{1:t}) - τ) ] ]
* **units:** nats (log-prob).
* **range:** [0, ∞). For valid samples, near 0.
* **normalization:** rank normalization (robust) or winsorized z-score.
* **role:** constraint with target c_cfs.
* **diagnostics:** violation rate (% tokens rejected by grammar), λ_cfs trajectory.
* **failure smells:** violation rate ↑ while LM loss ↓ (wireheading), mask collapse (beam emptied).
* **interactions:** Conflicts with ℓ_task in creative channels; PCGrad needed to resolve.

---

## 3. graph-to-graph loss (ℓ_g2g)

* **def:** kernel surrogate of graph edit distance:
  [ ℓ_{g2g} = 1 - \frac{k(\hat{G}, G^*)}{\sqrt{k(\hat{G},\hat{G}) k(G^*, G^*)}} ]
* **units:** dimensionless [0,1].
* **range:** 0=perfect similarity, 1=max dissimilarity.
* **normalization:** direct or rank.
* **role:** constraint with target c_g2g.
* **diagnostics:** GED distribution, coverage ratio (fraction of traces with valid DAGs).
* **failure smells:** graphs look plausible but unused in execution; GED plateau despite training.
* **interactions:** With ℓ_build (consistency) and ℓ_reuse (skill allocation).

---

## 4. buildability loss (ℓ_build)

* **def:**
  [ ℓ_{build} = | Proj(h^C_t) - Compose({Prog(z_{t-i})}_{i=1..m}) |_2^2 ]
* **units:** squared L2 norm.
* **range:** ≥0, expected O(0.1–1.0).
* **normalization:** winsorized.
* **role:** encourages hidden states to be derivable from invoked skills.
* **diagnostics:** correlation between hidden states and compositional reconstructions.
* **failure smells:** collapse to trivial zeros; overfitting to Compose architecture.
* **interactions:** Synergistic with ℓ_g2g; conflicts with ℓ_task if model prefers shortcut encodings.

---

## 5. reuse + parameter efficiency (ℓ_reuse, ℓ_Δθ)

* **def:**

  * Reuse: ( ℓ_{reuse} = -\frac{1}{|V|}\sum_v \log \pi(z_v) )
  * Param L1: ( ℓ_{Δθ} = ||Δθ^C||_1 )
* **units:** nats (reuse), weight sum (param L1).
* **range:** ≥0.
* **normalization:** rank (reuse), winsorized (param).
* **role:** keep skill library small; penalize uncontrolled growth.
* **diagnostics:** entropy of skill distribution, adapter growth curves.
* **failure smells:** mode collapse on a single skill; adapter explosion.
* **interactions:** With ℓ_build (constructibility) and ℓ_task (performance); dual λ ensures balance.

---

## 6. KL budget (ℓ_kl)

* **def:**
  [ ℓ_{kl} = D_{KL}(π || π_0) ]
* **units:** nats.
* **range:** [0, ∞). Target β ~0.01–0.1 nats/step.
* **normalization:** none.
* **role:** rational inattention constraint; limits divergence from prior.
* **diagnostics:** KL per step, cumulative divergence.
* **failure smells:** persistent overshoot → dual instability; undershoot → model frozen.
* **interactions:** Competes with ℓ_task in RL; stabilizes with reverse-KL priors.

---

## 7. calibration loss (ℓ_cal)

* **def:** Expected calibration error (ECE):
  [ ECE = \sum_b \frac{|B_b|}{N} |acc(B_b) - conf(B_b)| ]
* **units:** probability.
* **range:** [0,1]. Targets c_cal ≤0.05.
* **normalization:** rank.
* **role:** constraint for reliable uncertainty.
* **diagnostics:** reliability diagrams, Brier score.
* **failure smells:** good calibration on training but collapse under shift.
* **interactions:** Supports EVSI and deferral heads.

---

## 8. Lyapunov safety (V_s)

* **def:** surrogate risk energy ≥0.
* **constraint:** ΔV_s ≤ 0 (hard, non-negotiable).
* **units:** dimensionless energy.
* **range:** [0, ∞). Absolute values domain-dependent.
* **normalization:** none.
* **role:** promotion gate.
* **diagnostics:** ΔV_s distribution on shadow eval.
* **failure smells:** Vs ↑ without clear driver; safety regressions.
* **interactions:** Overrides all other losses; hard veto.

---

## 9. EVSI objective (ℓ_evsi)

* **def:**
  [ ℓ_{evsi} = - (V_{tool} - V_{no}) ]
* **units:** utility units (task-dependent).
* **range:** can be negative or positive.
* **normalization:** winsorized.
* **role:** auxiliary for gating tool/robot calls.
* **diagnostics:** EVSI margin (mean >0 when calling).
* **failure smells:** tool always chosen (helplessness); never chosen (hubris).
* **interactions:** Needs calibration; interacts with ℓ_cal and ℓ_kl.

---

## 10. task-specific auxiliary losses

### 10.1 reconstruction losses (ℓ_auto)

* **def:** L2 or BCE reconstruct of features, graphs, or tokens.
* **units:** L2 norm or nats.
* **range:** ≥0.
* **normalization:** rank.
* **diagnostics:** recon error distribution.
* **failure smells:** trivial copy; mode collapse.

### 10.2 entropy regularization (ℓ_ent)

* **def:** negative entropy of policy.
* **units:** nats.
* **range:** ≥0.
* **normalization:** none.
* **diagnostics:** entropy curves; exploration vs exploitation.
* **failure smells:** entropy collapse; uncontrolled entropy growth.

### 10.3 novelty penalty (ℓ_nov)

* **def:** penalty if info gain > τ.
* **units:** nats.
* **range:** ≥0.
* **normalization:** winsorized.
* **diagnostics:** info gain traces.
* **failure smells:** agent over-seeking novelty; ignoring rewards.

---

## 11. normalization summary

* **rank normalization:** default for discrete or heavy-tailed (ℓ_cfs, ℓ_reuse, ℓ_cal).
* **winsorized z-score:** default for continuous bounded/unbounded (ℓ_build, ℓ_evsi).
* **direct:** ℓ_task, ℓ_kl, V_s.

---

## 12. dashboards & monitoring

For every run, log:

* raw + normalized losses.
* dual λ trajectories.
* violation rates (CFS, GED).
* skill distribution entropy, adapter growth.
* calibration reliability curves.
* ΔV_s shadow eval.
* EVSI margins.
* gradient cosine heatmaps.
* constraint satisfaction rates.

---

## 13. failure pattern catalog

* **wireheading:** ℓ_cfs or ℓ_g2g minimized without real structural improvement. Symptom: loss ↓ but violation rate unchanged.
* **sprawl:** ℓ_reuse ignored, adapters proliferate. Symptom: λ_reuse ↑ steadily, but growth continues.
* **safety mirage:** task loss improves, but ΔV_s ↑. Must rollback.
* **overconstraint:** grammar too strict, no valid decodes. Symptom: mask collapse.
* **underconstraint:** grammar too loose, invalid traces slip. Symptom: false positives in golden tests.
* **instability:** duals oscillate without convergence. Symptom: constraints violated erratically.
