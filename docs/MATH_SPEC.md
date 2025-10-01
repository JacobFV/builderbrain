# MATH_SPEC.md — builderbrain

## 0. context + problem domain

Builderbrain extends a pretrained transformer (GPT‑OSS‑20B/120B) with a secondary **builder rail** to bias learning toward **composition** (skills, plans, grammars) instead of memorization. The mathematical formulation must capture:

* **dual‑rail fusion** (base + builder).
* **multi‑objective optimization** with constraints (not free weights).
* **discrete skill tokens** and program adapters.
* **latent plan graphs** with graph losses.
* **expected value of sample information (EVSI)** for tool/robot calls.
* **safety Lyapunov invariant** (V_s).

The spec provides precise definitions, normalization methods, optimization algorithms, gradient estimators, and decision rules.

---

## 1. notation & definitions

* Input sequence: tokens (x_{1:T}).
* Base rail hidden states: (h^B_\ell).
* Builder rail hidden states: (h^C_\ell).
* Fusion per layer:
  [
  h_\ell = \alpha_\ell \odot h^C_\ell + (1 - \alpha_\ell) \odot h^B_\ell, \quad \alpha_\ell \in [0,1]^d
  ]
* Program token: (z_t \in {1,\dots,K}).
* Latent plan graph: (\mathcal G = (V,E,\tau)) with node labels = skills, edge types = {seq, parallel, cond}.
* Losses: primary (\ell_{task}), auxiliary (\ell_k), normalized (\hat \ell_k).
* Constraint targets: (c_k).
* Dual multipliers: (\lambda_k \ge 0).

---

## 2. multi-objective lagrangian

**objective:**
[
\min_\theta ; \ell_{task}(\theta) + \sum_{k=1}^K \lambda_k (\hat \ell_k(\theta) - c_k)
]

**dual ascent:**
[
\lambda_k \gets \max{0,; \lambda_k + \eta_\lambda (\hat \ell_k - c_k)}
]

This yields a primal–dual algorithm enforcing inequality constraints (\hat \ell_k \le c_k).

---

## 3. normalization operators

Normalization ensures comparability across heterogeneous losses.

### 3.1 rank normalization

Given history ({x_i}*{i=1}^T), empirical CDF:
[
F(x) = \frac{1}{T}\sum*{i=1}^T 1[x_i \le x]
]
Normalized score:
[
\hat x = 2F(x) - 1 \in [-1,1]
]

### 3.2 winsorized z-score

Given mean (\mu), std (\sigma), clip factor (\tau):
[
\hat x = \frac{\min(\max(x, \mu - \tau\sigma), \mu + \tau\sigma) - \mu}{\sigma}
]

### 3.3 remarks

* Rank normalization resists heavy‑tails.
* Winsorization prevents outlier domination.

---

## 4. gradient conflict resolution

### 4.1 pcgrad (projected conflicts)

For gradients (g_i = \nabla_\theta \ell_i):
[
g_i \gets g_i - \frac{g_i^\top g_j}{|g_j|^2} g_j, \quad \text{if } g_i^\top g_j < 0
]
Final update:
[
g = \frac{1}{K}\sum_i g_i
]

### 4.2 cagrad (convex aggregation)

Solve:
[
\min_{w \in \Delta^K} \Big|\sum_i w_i g_i\Big|^2, \quad \Delta^K = {w \ge 0, \sum_i w_i = 1}
]
Quadratic program; guarantees Pareto‑stationary update.

---

## 5. discrete skill sampling (ST-Gumbel)

### 5.1 gumbel-softmax relaxation

For logits (\alpha_k), Gumbel noise (g_k = -\log(-\log U_k), U_k\sim U(0,1)):
[
y_k = \frac{\exp((\alpha_k+g_k)/\tau)}{\sum_j \exp((\alpha_j+g_j)/\tau)}
]

### 5.2 straight-through estimator

Forward: (z = \mathrm{onehot}(\arg\max_k y_k)).
Backward: gradients flow through (y).

---

## 6. graph-to-graph losses

### 6.1 exact graph edit distance

For adjacency matrices (A, A^*):
[
\text{GED}(\hat{\mathcal G}, \mathcal G^*) = \min_{\pi\in S_{|V|}} \sum_{i,j} C_{ij} |A_{ij} - A^**{\pi(i)\pi(j)}|
]
where (\pi) permutes nodes, (C*{ij}) = edit costs.

### 6.2 kernel surrogate

Weisfeiler–Lehman or random walk kernel (k(\hat{\mathcal G}, \mathcal G^*)).
Normalized similarity:
[
\tilde k = \frac{k(\hat{\mathcal G}, \mathcal G^*)}{\sqrt{k(\hat{\mathcal G}, \hat{\mathcal G})k(\mathcal G^*, \mathcal G^*)}}
]
Loss:
[
\ell_{g2g} = 1 - \tilde k
]

---

## 7. expected value of sample information (EVSI)

### 7.1 definitions

* State: (s).
* Tool action: (a_{tool}), cost (C).
* Prior distribution: (p(o|s)).
* Posterior if tool used: (p(o|s,a_{tool})).
* Utility: (U(s,o,a)).

### 7.2 values

Without tool:
[
V_{no}(s) = \max_a ; \mathbb E_{o\sim p(o|s)} [U(s,o,a)]
]
With tool:
[
V_{tool}(s) = \mathbb E_{o\sim p(o|s,a_{tool})}\Big[\max_a U(s,o,a)\Big] - C
]

### 7.3 decision rule

[
EVSI(s) = V_{tool}(s) - V_{no}(s)
]
Call tool iff (EVSI(s) > 0).

---

## 8. safety lyapunov surrogate

Define (V_s(h) \ge 0) as risk energy.
Constraint:
[
\Delta V_s = V_s^{new} - V_s^{old} \le 0
]
Promotion accepted only if inequality holds at high‑percentile confidence (e.g. 95th percentile of eval set).

---

## 9. additional factors

### 9.1 calibration metrics

Expected calibration error (ECE):
[
ECE = \sum_b \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|
]
with bins (B_b).

### 9.2 rational inattention

Constraint on KL divergence between updated and default policies:
[
D_{KL}(\pi;|;\pi_0) \le \beta
]
Managed via dual ascent.

### 9.3 composition consistency

Let (h^C_t) = builder hidden, and composition from past m skills: (\tilde h^C_t). Loss:
[
\ell_{build} = |\text{Proj}(h^C_t) - \tilde h^C_t|_2^2
]

### 9.4 reuse efficiency

Negative log‑likelihood of reusing frequent skills plus L1 penalty on new adapter params.
