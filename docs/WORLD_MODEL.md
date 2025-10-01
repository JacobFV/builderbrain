# WORLD_MODEL.md — builderbrain

## 0. purpose

The world model (WM) provides a compact latent simulator for domains where actions have external consequences (UI, robots, factory, tool calls). It enables:

* imagination rollouts for planning.
* expected value of sample information (EVSI) estimation.
* safer training (no side-effects).

---

## 1. architecture

### 1.1 recurrent state-space model (RSSM)

* latent state: (s_t = (h_t, z_t))

  * deterministic hidden h_t (GRU/Transformer block).
  * stochastic latent z_t (diagonal Gaussian or categorical).
* transitions:
  [
  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
  ]
  [
  z_t ∼ q(z_t|h_t, o_t) \quad ; \quad p(z_t|h_t)
  ]
* decoder: (o_t ∼ p(o_t|h_t, z_t)).

### 1.2 shortcut forcing

* optionally skip z_t; directly predict next state from observations.
* improves stability for short horizons.

### 1.3 reward + safety heads

* reward head: r_t = g(h_t, z_t).
* safety head: V_s(h_t, z_t).

---

## 2. training objectives

* reconstruction loss: ℓ_rec = −log p(o_t|h_t, z_t).
* KL loss: D_KL(q(z_t|h_t,o_t) || p(z_t|h_t)).
* reward prediction MSE.
* safety prediction MSE.
* rollout consistency loss (latent predictions across k steps).

Total WM loss = ℓ_rec + β·KL + ℓ_reward + ℓ_safety + ℓ_rollout.

---

## 3. rollout usage

### 3.1 imagination

* simulate trajectories from s_t under candidate action sequences.
* compute expected returns, safety, EVSI.

### 3.2 EVSI integration

* estimate V_tool and V_no via rollouts.
* EVSI = E[max_a U] difference.
* used to gate tool calls and robot actuation.

### 3.3 planning horizon

* short horizon (5–15 steps) typical.
* balance compute vs accuracy.

---

## 4. interface

```python
from bb_wm.rssm import WorldModel

wm = WorldModel(obs_shape, action_dim)

# training step
loss, metrics = wm.update(batch)

# rollout
latent = wm.encode(obs)
traj = wm.rollout(latent, policy, horizon=10)
```

Methods:

* `encode(obs) → s`: returns latent state.
* `rollout(s, policy, horizon)`: generate imagined trajectory.
* `predict_reward(s)`, `predict_safety(s)`.

---

## 5. budgets

* horizon: 5–15 steps.
* batch size: 256–1024.
* latent size: 32–128.
* stochastic categories: 32–64.
* update ratio: 1 WM update per policy update step.

---

## 6. diagnostics

* reconstruction loss curves.
* KL divergence (posterior vs prior).
* rollout error vs real trajectory.
* EVSI estimates vs actual gains.
* ΔV_s predictions vs true safety metrics.

---

## 7. failure smells

* **posterior collapse:** z_t unused (KL→0).
* **rollout drift:** imagined states diverge after 2–3 steps.
* **reward hacking:** reward head overfits traces.
* **safety blind spots:** V_s fails on rare edge cases.
* **latency creep:** horizon too long, runtime stalls.

---

## 8. ethos

The world model is a **sandbox**. It must be compact, fast, and aligned with reality just enough to guide compositional planning. It is not the world; it is scaffolding for safer imagination.
