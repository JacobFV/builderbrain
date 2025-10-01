# DUAL_OPTIMIZER.md — builderbrain

## 0. purpose

Dual optimization governs multi-objective training without “free weights.” Each auxiliary objective is a **constraint** with a target `c_k`. Lagrange multipliers `λ_k ≥ 0` are updated online to keep normalized losses near their targets while the **primary task loss** is minimized.

---

## 1. math recap (one screen)

Primal objective with constraints:
[
\min_\theta ; \ell_{task}(\theta) + \sum_{k=1}^K \lambda_k (\hat\ell_k(\theta) - c_k), \quad \lambda_k \ge 0
]
Dual ascent:
[
\lambda_k \gets \max{0,; \lambda_k + \eta_\lambda (\hat\ell_k - c_k)}
]
`\hat\ell_k = N(\ell_k)` is the normalized loss (rank or winsorized z-score; see **MATH_SPEC.md**).

Gradient conflict resolution (PCGrad/CaGrad) should be applied on per-loss VJPs before summation.

---

## 2. python api

### 2.1 module layout

```
bb_losses/dual/
  ├─ dual_engine.py        # core class
  ├─ normalizers.py        # RankNormalizer, WinsorNormalizer
  ├─ targets.py            # schedulers, adaptive targets
  ├─ pcgrad.py             # projected gradient utilities
  └─ dashboards.py         # logging helpers
```

### 2.2 core class

```python
from typing import Dict, Callable
import torch

class DualEngine:
    def __init__(self, losses_cfg, eta_lambda=1e-2, pcgrad=True):
        self.cfg = losses_cfg  # dict[k]: {target, normalizer, weight=1.0, hard=False}
        self.lmb = {k: torch.tensor(0.0) for k in losses_cfg}
        self.norms = {k: build_normalizer(v["normalizer"]) for k,v in losses_cfg.items()}
        self.eta = eta_lambda
        self.pcgrad = pcgrad

    def normalize(self, raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.norms[k](raw[k].detach()) for k in raw}

    def lagrangian(self, L_task: torch.Tensor, raw_losses: Dict[str, torch.Tensor]):
        hat = self.normalize(raw_losses)
        L = L_task.clone()
        for k, v in hat.items():
            L = L + self.lmb[k].to(L) * (v - self.cfg[k]["target"]) * self.cfg[k].get("weight", 1.0)
        return L, hat

    @torch.no_grad()
    def step_duals(self, hat: Dict[str, torch.Tensor]):
        for k, v in hat.items():
            delta = v - self.cfg[k]["target"]
            self.lmb[k] = torch.clamp(self.lmb[k] + self.eta * delta, min=0.0)

    def state_dict(self):
        return {"lambda": {k: float(v) for k,v in self.lmb.items()},
                "norm": {k: self.norms[k].state_dict() for k in self.norms}}

    def load_state_dict(self, sd):
        for k,v in sd["lambda"].items():
            self.lmb[k] = torch.tensor(v)
        for k,v in sd["norm"].items():
            self.norms[k].load_state_dict(v)
```

### 2.3 normalizers

```python
class RankNormalizer:
    def __init__(self, window=50000): ...
    def __call__(self, x): ...          # returns in [-1,1]
    def state_dict(self): ...
    def load_state_dict(self, sd): ...

class WinsorNormalizer:
    def __init__(self, tau=3.0): ...
    def __call__(self, x): ...          # clipped z-score
```

### 2.4 usage snippet

```python
dual = DualEngine({
  "cfs":   {"target": 0.0,  "normalizer": "rank"},
  "g2g":   {"target": 0.2,  "normalizer": "rank"},
  "build": {"target": 0.0,  "normalizer": "winsor"},
  "reuse": {"target": 0.5,  "normalizer": "rank"},
  "kl":    {"target": 0.05, "normalizer": "none"}
}, eta_lambda=5e-3)

for batch in loader:
    L_task, aux_raw = model(batch)           # aux_raw: dict[str, tensor]
    L, aux_hat = dual.lagrangian(L_task, aux_raw)
    grads = pcgrad(L, per_loss_grads=aux_raw)  # optional
    opt.step(grads)
    dual.step_duals(aux_hat)
    log(dual.lmb, aux_hat)
```

---

## 3. targets: how to set `c_k`

### 3.1 fixed targets (simple)

Use domain knowledge. Examples:

* Grammar energy (strict channels): `c_cfs = 0`.
* Graph similarity: `c_g2g ≤ 0.2` (high similarity).
* KL budget: `c_kl = β` (e.g., 0.05 nats/step).
* Calibration: `c_cal ≤ 0.05`.

### 3.2 percentile targets (robust under drift)

Set `c_k` to a percentile of a rolling baseline distribution:

* `c_k = P90(\hat\ell_k^{baseline})` → keep current run no worse than the 90th percentile of baseline.

### 3.3 curriculum/annealing

* Start loose (`c_k` high), tighten gradually according to eval success.
* Use **sigmoid** schedule: `c_k(t) = c_min + (c_max - c_min)·σ(−γ(t−t0))`.

### 3.4 adaptive via outer loop

* Periodically re-optimize `c_k` on long-horizon validation (Bayesian optimization or PBT) while keeping dual updates online.

---

## 4. λ stability tricks (make it behave)

1. **clipped updates:** `λ_k ← clip(λ_k + η·Δ, 0, λ_max)`; typical `λ_max∈[10,100]`.
2. **ema on measurements:** `Δ = EMA(\hat\ell_k − c_k; α=0.1)` reduces jitter.
3. **per-constraint learning rate:** set `η_λ` per `k` (fast for easy constraints, slow for noisy ones).
4. **integral + leak control:** `λ_k ← (1−ρ)λ_k + η·Δ` with small leak `ρ∈[1e−4,1e−3]` to avoid long-term windup.
5. **deadzone around target:** if `|\hat\ell_k − c_k| < ε`, skip update.
6. **anti-covariance audit:** pause updates if covariance(`λ_k`, ease-metric) > τ (wireheading smell).
7. **coupled budgets:** tie families of constraints (e.g., sum of grammar λ across domains capped) to prevent one exploding.
8. **freeze-on-spike:** if a constraint suddenly spikes (e.g., parser outage), freeze `λ_k` to avoid runaway penalties.

---

## 5. gradient plumbing (PCGrad/CaGrad)

* Always compute **per-loss VJPs** to allow PCGrad/CaGrad.
* Prefer PCGrad for simplicity; switch to CaGrad when K is small and you can afford QP.
* Monitor **gradient cosine matrices** before and after surgery; store top singular value as a redundancy score.

---

## 6. dashboards (what to stare at)

* **Dual traces:** `λ_k` time series, with clip bands.
* **Constraint satisfaction:** `\hat\ell_k − c_k` violin plots; % within deadzone.
* **Normalizer health:** histograms of raw vs normalized losses; drift detectors.
* **Gradient cosines:** heatmap pre/post PCGrad; top-1 singular value.
* **Task vs constraints:** scatter of `Δℓ_task` vs `Δ\hat\ell_k` to spot tradeoffs.
* **Wireheading indicators:** covariance(`λ_k`, loss ease metrics); sudden denominator shifts.
* **Release gate:** snapshot of all `λ`, satisfaction rates, and ΔV_s before/after candidate.

---

## 7. recipes (by domain)

### 7.1 API/JSON agent

* `c_cfs = 0` (hard mask at decode), `η_λ(cfs)=1e−2`.
* Tight `c_g2g` for plan shape if mapping requests → multi-call flows.
* KL budget small (`β≈0.02`).
* Expect quick convergence; monitor mask collapses.

### 7.2 Phone/Video call agent

* Semi-strict grammar (tags) → `c_cfs` > 0 with slack.
* Strong calibration constraint (`c_cal ≤ 0.03`) to support deferrals.
* EVSI auxiliary on tool lookups; dual-couple with latency budget.

### 7.3 Robots/Factory

* `c_cfs = 0` for DSL; `c_g2g ≤ 0.1` (high similarity to reference).
* Penalize param growth (ℓ_Δθ) aggressively.
* Tie safety metrics to pre-promotion dashboard; do not relax.

---

## 8. failure modes → fixes

* **Oscillating λ:** lower `η_λ`, add EMA on Δ, widen deadzone ε.
* **Constraint never satisfied:** verify normalizer; raise `c_k` (curriculum) then tighten.
* **Task loss starved:** cap `λ_max`, or lower weight for that constraint.
* **Wireheading:** randomize normalization windows; freeze `λ` temporarily; inspect covariance dashboard.
* **Conflicting constraints:** enable CaGrad; relax the least critical constraint; stagger updates (asynchronous λ).

---

## 9. hydra config (example)

```yaml
losses:
  cfs:   {target: 0.0,  normalizer: rank,   weight: 1.0}
  g2g:   {target: 0.2,  normalizer: rank,   weight: 1.0}
  build: {target: 0.0,  normalizer: winsor, weight: 0.5}
  reuse: {target: 0.5,  normalizer: rank,   weight: 0.5}
  kl:    {target: 0.05, normalizer: none,   weight: 1.0}

dual:
  eta_lambda: 5e-3
  lambda_max: 50.0
  deadzone: 0.02
  ema_alpha: 0.1
  leak: 1e-4
  pcgrad: true
```

---

## 10. testing & CI

* **unit tests:** λ update monotonicity; deadzone behavior; clip bounds.
* **property tests:** invariance to affine scaling of raw losses (after normalization).
* **goldens:** replay a known training segment; assert λ traces within tolerance bands.

---

## 11. ops notes

* Persist `λ` and normalizer state with checkpoints.
* On resume after long pause, warm-up with smaller `η_λ` for N steps.
* Per-constraint alerting thresholds (e.g., λ hitting `λ_max`, satisfaction < 80%).
