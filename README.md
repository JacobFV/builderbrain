# builderbrain

> build, don’t just search.
> a dual-rail extension to pretrained transformers that learns reusable skills, executable plans, and safe compositional reasoning across domains.

## INTERNAL README

**status:** pre-alpha, unstable, expect breakage.
**audience:** internal MLEs. not for public release.

---

## 0. guiding principle

transformers memorize. we need **composition**.
→ bolt a **builder rail** onto a frozen gpt-oss-20b/120b, and force it to:

* represent **skills** as discrete programs,
* emit **plan graphs** that are executable + auditable,
* obey **grammars**,
* stay inside **safety rails**.

no vibes, no magic. just bias the system toward *buildability*.

---

## 1. architecture (mental model)

```
           base rail (frozen oss 20b/120b)
      ────────────────────────────────
      h^B_ℓ ──────────────────────▶

           builder rail (learned comp)
      ────────────────────────────────
      h^C_ℓ = CompBlock(h^C_ℓ, h^B_ℓ)
         + program adapters z_t
         + latent call DAG G_t

           gating
      ────────────────────────────────
      h_ℓ = α_ℓ ⊙ h^C_ℓ + (1−α_ℓ) ⊙ h^B_ℓ
```

* **base rail (B):** oss transformer. no online updates except rare LoRA merges.
* **builder rail (C):** ssm/gru/attn hybrids, cross-attn into B.
* **program library:** K discrete skills (ST-gumbel). adapters = LoRA/hypernets.
* **call DAG:** latent plan graph; later validated/executed.
* **gates α:** per-layer, capped by global budget (\bar α). grows only when metrics improve.

---

## 2. objectives

* **primary:** lm loss / decision loss.
* **constraints (dual variables, not free weights):**

  * `cfs_energy`: grammar prior (cfg/peg).
  * `g2g_loss`: graph recon.
  * `build_loss`: hidden ≈ composition of past skills.
  * `reuse_loss`: encourage reusing skill tokens; penalize new param bloat.
  * `kl_budget`: keep policy divergence bounded.
  * `calibration_loss`: prequential ece.
* **safety (hard):** lyapunov (V_s) must not increase post-update/promotion.

### lagrangian

[
L = L_{\text{task}} + \sum_k \lambda_k (\hat \ell_k - c_k), \quad
\lambda_k \gets [\lambda_k + η(\hat \ell_k - c_k)]_+
]

* use **rank/winsor normalization** per loss.
* resolve gradient conflict via **pcgrad/cagrad**.

---

## 3. priors

* **grammars:** strict channels (api/json, robot dsl) → hard masks at decode. softer for chat/social.
* **graphs:** each domain has plan schema (yaml) → codegen into checker.
* **composition efficiency:** reuse > bloat. if adapters sprawl, you did it wrong.
* **invariance:** same workflow, different skins → must hold.

---

## 4. data

* **instrumented traces:** tool calls, ui macros, phone flows, robot dsl, social pipelines.
* **graph schemas:** derive from orchestration logs.
* **grammars:** one per domain; keep minimal + precise.
* **synthetic domain shifts:** skins, layouts, policy variants.

note: no massive re-pretrain. lean on oss weights; builder rail soaks structure.

---

## 5. training phases

1. **stage0 boot:**

   * load oss base, freeze.
   * insert builder rail + gates (α≈0.05).
   * init program adapters (K≈32).
   * wire parsers, schemas.

2. **stage1 offline:**

   * optimize task + constraints.
   * watch duals, cfs violation rate, ged, grad cosines.
   * slowly lift (\bar α) if compositional metrics improve.

3. **stage2 planner (optional):**

   * small wm (rssm w/ shortcut forcing).
   * policy/value in wm.
   * tool/robot gating via evsi.

4. **stage3 runtime:**

   * decode w/ grammar masks.
   * plan check every dag.
   * fallback to base rail on violation.

5. **stage4 continual:**

   * adapters-only online learning.
   * promotion requires shadow eval + (V_s) gate + kl trust region.
   * auto-rollback on fail.

---

## 6. evaluation must-pass

* **skill stacking:** new tasks mastered w/ sublinear param growth.
* **ablations:** removing skills kills compositional perf.
* **syntax:** zero invalid api/json; < threshold elsewhere.
* **graph fidelity:** ged low under domain shift.
* **takeover sanity:** as (\bar α↑), compositional win-rate ↑, safety flat.
* **ops:** refund errors, pii leaks, robot near-misses ↓.

---

## 7. repo structure (enforced)

```
bb_core/    # pure math/protocols, no torch
bb_nn/      # torch impls
bb_train/   # training loops, data, eval
bb_runtime/ # serving, decode, plan exec
bb_safety/  # Vs, shadow eval, promotion
bb_domains/ # plugins per domain (api_json, robots, phone, social)
bb_infra/   # logging, config, ci
tests/      # unit, property, golden, e2e
```

* import rules enforced (importlinter).
* grammars + schemas live in `bb_domains/*`. codegen into parsers + checkers.
* **do not** import training code into runtime.

---

## 8. safety invariants

* builder rail only online-trained modules.
* plans must pass checker before execution.
* promotion requires battery pass, (V_s) non-increase, kl bound.
* logs = immutable artifacts (model hash + grammar/schema hash + config).

---

## 9. what will break (known pain)

* **wireheading attempts:** builder may try to game cfs/reuse losses. duals keep it bounded, but audit covariance.
* **fake graphs:** model can output pretty but unused dags. fix = counterfactual execution: reward only if chosen plan > alt.
* **adapter sprawl:** l1 + cap birthrate, plus merge/distill.
* **grammar bottleneck:** if over-tight, creativity tanks. soften energy; keep hard mask only where correctness matters.
* **compute:** dags + pcgrad = overhead. profile and prune.

---

## 10. rules of engagement

* all new domains go in `bb_domains/*`. add `grammar.peg`, `plan_schema.yaml`, `tool_adapters.py`.
* always write golden tests for grammar + plan schemas.
* never bypass planchecker for side-effects.
* promotion must run through `bb_safety.promote.gate`. no shortcuts.
* logs must include: duals, grad cosines, cfs rate, ged, evsi, Vs delta.

---

## 11. near-term milestones

* [ ] m0: dual-rail skeleton, gates, program library, dummy grammars.
* [ ] m1: offline train on synthetic domain; reduce cfs+g2g+build losses.
* [ ] m2: syntax-aware decode + planchecker.
* [ ] m3: adapters-only online learning + promotion gate stub.
* [ ] m4: wm + evsi tool gating.
* [ ] m5: real traces (api/json + phone flows) → valid plans.

---

## 12. ethos

this is not “product polish.” this is **research infra for composition bias**.
expect half the code to be thrown away. measure everything. falsify aggressively.
remember: **the base rail is memory. the builder rail must become construction.**
