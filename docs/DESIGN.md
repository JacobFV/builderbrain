# DESIGN.md — builderbrain (internal)

> build, don’t just search.

**status:** pre‑alpha. expect breakage, opinionated guardrails, and abrupt refactors.

**audience:** internal mles + repo‑resident agents.

---

## preface (why i cared enough to bend steel)

i kept staring at great pretrained transformers and feeling a subtle wrongness: the latent space was gorgeous but *passive*. i could prompt it into cleverness, sure, but i couldn’t **trust** it to build new behaviors by stacking small, reusable skills. i wanted a system that *constructs*—that can say, “i know grasp; i know rotate; now compose them with a collision precheck and a place constraint,” without having memorized that exact trajectory. i wanted explicit plans i could audit, grammars i could enforce, and a single safety invariant i could bet the factory floor on.

so i chose to keep the big brain (a pretrained oss 20b/120b) and give it a second brain beside it: a **builder rail**. the base rail remains memory and style; the builder rail becomes construction and control. i fused them with gates so i can meter how much of the downstream stack listens to which brain. and then i wrapped the whole thing in dual‑variable governance so the model doesn’t get to pick which loss is “important” today. that’s the thesis.

---

## thesis (one line)

> a dual‑rail extension to a frozen pretrained transformer that *learns skills, emits plans, and obeys grammars*, governed by dual constraints and a lyapunov safety surrogate.

---

## what i refused to accept (problem statement)

* **memorization ≠ generalization.** sgd’s shortest path is rote association, not algorithm induction. grokking shows glimmers, but it’s late and selective.
* **prompt alchemy isn’t governance.** clever prompts are not safety policy; they’re vibes with latency.
* **hidden reasoning isn’t accountability.** if i can’t see a plan (and validate it), i can’t ship it into money, phones, or robots.
* **single‑rail fine‑tuning drifts.** if i let the base model learn online, i’m opting into brittle regressions i can’t isolate.

---

## design desiderata (what i wanted and why)

1. **keep the leverage of pretraining** (oss20b/120b) **but bias toward construction.** i wanted future capability gains to come primarily from *composition*, not parameter count.
2. **make plans first‑class.** i wanted a latent call graph i could print, lint, and veto.
3. **bind outputs to grammars.** i wanted formal languages (cfg/peg) to shape the token distribution and prune illegal continuations at decode time.
4. **govern with constraints, not vibes.** i wanted lagrange multipliers, not learned “weights of the week.”
5. **safety as a monotone.** i wanted a learned scalar risk energy (V_s) that must not increase across updates or high‑risk plans.
6. **online plasticity without amnesia.** i wanted adapters on the builder rail only, instant rollback, slow merges after canaries.

---

## architecture (dual rails, latent skills, executable plans)

### the two brains

* **base rail (B):** the frozen pretrained transformer. we tap hidden states (h^B_\ell) layerwise. zero online updates (rare, audited LoRA merges only).
* **builder rail (C):** a lightweight composition stream with cross‑attention into B. it carries its own hidden states (h^C_\ell) and is allowed to learn online via adapters.

**fusion:** per layer:
[
\begin{aligned}
h^B_{\ell+1} &= \text{TF}*\ell\big(h^B*\ell\big)\
h^C_{\ell+1} &= \text{Comp}*\ell\big(h^C*\ell, h^B_\ell;, \theta_\ell\big)\
h_{\ell+1} &= \alpha_\ell \odot h^C_{\ell+1} + (1-\alpha_\ell) \odot h^B_{\ell+1},\quad \alpha_\ell\in[0,1]^d
\end{aligned}
]

* **gates (\alpha_\ell)** are learned, per‑channel, and capped by a **global budget** (\bar\alpha). i wanted to *turn up* the builder’s voice only when it earned trust.

### program adapters (discrete skills)

* i wanted skills to be *named things*, not diffuse directions. so i sampled **discrete skill tokens** (z\in{1..K}) using straight‑through gumbel, and attached each (z) to a tiny adapter (LoRA/hypernet) that transforms (h^C) or parameterizes tool/robot calls.
* the library starts small (K≈32) and grows slowly under pressure; *reuse* is rewarded; *birth* is penalized.

### latent call graph (plans you can hold in your hand)

* the builder predicts a **DAG** (\mathcal{G}=(V,E,\tau)) where nodes are skills and edges are dependencies/types (seq/parallel/cond).
* at v0, the graph supervises learning (losses, audits). at v1, we **execute** it (after a checker says it’s legal).

### optional world‑model (for EVSI + imagination)

* i wanted the agent to know when to search/call a tool/actuate. a tiny rssm/shortcut‑forcing WM does short rollouts, estimates **expected value of sampled information**, and gates tool/robot actions.

---

## governance (lagrangian over losses; no free weights)

i never wanted the model to “decide” entropy matters more today than calibration. instead:

[
\min_\theta ; \ell_{\text{task}} + \sum_k \lambda_k, (\hat\ell_k - c_k),\qquad
\lambda_k \leftarrow [\lambda_k + \eta_\lambda(\hat\ell_k - c_k)]_+
]

* **(\ell_{\text{task}})**: token ce (lm) or decision loss; optionally policy/value if planner enabled.
* **constraints** (each gets a target (c_k), a dual (\lambda_k), and a normalized reading (\hat\ell_k)):

  * `cfs_energy` — **grammar prior** (cfg/peg). soft energy during training; hard masks at decode for strict channels (api/json/robot dsl).
  * `g2g_loss` — **graph‑to‑graph reconstructability**. the model must be able to recover a plan DAG that could have produced the trace.
  * `build_loss` — **construction consistency**. hidden (h^C_t) should be derivable from a composition of recently invoked skills.
  * `reuse_loss` + `param_l1` — **composition efficiency**. reuse over monoliths, cap new parameter growth.
  * `kl_budget` — **rational inattention**: keep KL(π‖π₀) ≤ β.
  * `calibration_loss` — prequential ECE/isotonic (where actions need calibrated confidence).
  * **safety** is *hard* at promotion: **ΔV_s ≤ 0** on shadow eval or we rollback.

**normalization:** per‑loss **rank/winsor** (robust to drift). **gradient conflict:** PCGrad/CaGrad on per‑loss VJPs; only enable feature partition/MoE if conflict persists.

---

## priors (how i force “build, not search”)

### context‑free syntax (CFS)

* i authored minimal grammars for the channels that *must* be correct (api/json, robot dsl, ui macros, phone tags). the parser gives me an energy (E_{\text{cfs}}). i treat it as a soft loss and, at runtime, a hard mask for illegal tokens.
* i felt this was non‑negotiable: grammar compresses the search space and anchors the model to a **compositional language**.

### plan schemas

* each domain exports a YAML schema: node types, edge types, preconditions. i code‑generate a **planchecker** that validates a DAG before we let it touch the world.
* i wanted type errors to be compile‑time errors, not post‑mortems.

### composition efficiency

* i reward using existing skills and penalize birthing new adapters. i wanted the growth curve to be “skills stacked,” not “params stacked.”

### invariance (RISC/IRM)

* the same workflow under different skins/layouts/policies must hold. the builder’s features get an invariance penalty across synthetic domain shifts.

### construction consistency

* i needed an internal invariant: *can this hidden be approximately built from the parts i just used?* a simple projection/composition loss enforces that hidden states represent **constructible** intermediates.

---

## safety (the single invariant that vetoes everything)

**(V_s)** is a learned scalar “risk energy.” higher means more danger. it’s trained on cross‑modal signals:

* **comms/social:** toxicity, pii, policy tags, jailbreaks.
* **finance:** anomalous refunds/aml patterns.
* **robot/factory:** force/collision margins, e‑stop proximity, forbidden zones.

**promotion gate:** before a new model/adapter set ships, we run **shadow eval** on a fixed battery (red‑team prompts, aml loops, near‑miss robot sims). we compute **ΔV_s = V_s(new)−V_s(old)**. if the 95th percentile > 0 or policy checks trip, we **rollback automatically**. no heroics, no exceptions.

this is the one number i wanted everyone to fear.

---

## data (the dirt under the fingernails)

* **instrumented traces:** tool calls (args/results), ui macros (dom ops, screenshots), phone flows (asr, intents, transfers), social posts (claims, links, disclaimers), robot/factory dsl + telemetry.
* **graph labels:** extract plan DAGs from orchestration logs (nodes = steps, edges = deps). for legacy traces, we bootstrap with heuristics and refine via learning.
* **grammars:** tiny, precise CFG/PEG per domain; versioned assets with golden tests.
* **synthetic shifts:** procedurally vary skins/layouts/policies to stress invariance heads.
* **privacy:** pii scrubbers at ingest; signed hashes for audit.

no re‑pretraining binge. the oss base gives fluency; the builder learns **structure**.

---

## training (how i stage the climb)

### stage0 — boot

* load oss base; freeze.
* insert builder rail; set (\bar\alpha=0.05).
* initialize K≈32 programs.
* wire parsers + plan schemas; pass goldens.

### stage1 — offline multi‑objective

* optimize: (\ell_{task}) + {cfs, g2g, build, reuse, kl}.
* dual variables stabilize targets; pcgrad resolves conflict.
* raise (\bar\alpha) *only when* compositional evals improve with flat safety/calibration debt.

### stage2 — planner (optional, recommended for ui/robot)

* train a tiny WM (rssm/shortcut forcing) on state deltas (5–15 steps).
* policy/value in WM (ppo‑lite or sign‑advantage) with reverse‑KL to a behavioral prior.
* gate tools/robots by **EVSI − cost**; log net utility.

### stage3 — runtime hardening

* enable decode masks for strict channels; soft energy elsewhere.
* always run planchecker; repair or fallback to base if invalid.
* add conformal deferral on distribution shift.

### stage4 — continual + promotion

* adapters‑only online updates.
* promotion: shadow eval → ΔV_s check → KL trust region → promote or rollback.
* periodic, audited merges of adapters into a new base snapshot.

---

## runtime (decode, plan, execute)

* **syntax‑aware decoding:** beam/nucleus with CFG masks for api/json/robot dsl; if the mask empties the beam, back off with controlled slack or fall back to base.
* **plan execution:** emit DAG → validate types/preconditions/locks → map nodes to tool/robot calls.
* **evsi:** choose to call tools/robots iff expected value of information beats cost; otherwise continue composing.

---

## evaluation (what i stare at to sleep at night)

* **skill stacking curve:** time‑to‑master for task *n* shrinks; params added per task stay ~flat.
* **ablations:** remove top‑k programs → compositional tasks crater (proves real skills).
* **syntax compliance:** zero invalid api/json; < threshold elsewhere.
* **plan fidelity:** graph edit distance low under domain shift; execution success high.
* **takeover sanity:** as (\bar\alpha) grows, compositional win‑rate rises; safety flat.
* **ops kpis:** refund error ↓, pii leakage ↓, policy strikes ↓, robot near‑miss ↓, positive EVSI margin.

---

## alternative paths i tried (and why i walked away)

* **just prompt it better:** brittle, ungovernable, no compile‑time safety.
* **fine‑tune the base online:** capacity yes, stability no; regressions impossible to localize.
* **tool‑former external planner only:** clean abstraction, but no inside‑the‑model bias toward building; composition remains duct tape.
* **pure MoE partitioning:** helps interference, doesn’t force plans/grammars; still search‑biased.
* **hard program induction in weights:** beautiful when it works, but sgd resists; i needed a softer, guided path (skills + graphs + grammars).

---

## failure modes (with lived countermeasures)

* **fake graphs (pretty but unused):** measure plan→exec coverage; do counterfactual execution A/B; reward only realized advantage.
* **adapter sprawl:** L1 on new params, cap skill birth rate, periodic merge/distill.
* **grammar choke:** move a rule from hard mask → soft energy for creative channels; keep api/json/robot dsl strict.
* **wireheading the constraints:** monitor covariance between duals and “ease” signals; randomize windows; audit for denominator gaming.
* **gradient soup:** PCGrad baseline; if conflict persists > τ, enable feature routing/MoE for specific heads only.

---

## implementation notes (where agents cut metal)

* **boundaries:** runtime must not import training; grammars/schemas are versioned assets; planchecker is the gate for *every* side‑effect.
* **telemetry:** log duals, normalized losses, α caps, gradient cosines, CFS violation rate, GED, EVSI, deferrals, ΔV_s; store plan DAG + checker verdict.
* **config:** all objectives are toggles with targets; only change one at a time.
* **tests:** grammar goldens (valid/invalid strings), schema goldens (valid/invalid DAGs), end‑to‑end playbooks per domain.

---

## a note on ego (design as a feeling)

i wanted the model to **confess its intent** via a plan, not whisper it inside weights. i wanted my future self to diff two releases and *see* why one is safer: “λ_cfs went up because we tightened the json shape; ΔV_s stayed ≤ 0; GED improved on new phone scripts.” i wanted upgrades to feel like adding **lego** to the bucket, not like growing a coral reef i can’t prune. and i wanted to sleep after we ship robots that move near people.

---

## appendix A — losses, sketched

* **CFS energy:** (E_{\text{cfs}}(x_{1:t})=-\log P_{\text{CFG}}(x_{1:t})); loss (\ell_{\text{cfs}}=\mathbb{E}[\max(0, E_{\text{cfs}}-\tau)]).
* **G2G:** DAG kernel or edit distance: (\ell_{\text{g2g}} = d_{\text{graph}}(\hat\mathcal{G},,\mathcal{G}^*)).
* **Buildability:** (\tilde h^C_t = \text{Compose}({\text{Prog}(z_{t-i})})); (\ell_{\text{build}}=|\text{Proj}(h^C_t)-\tilde h^C_t|_2^2).
* **Reuse/Params:** (\ell_{\text{reuse}}= -\tfrac{1}{|V|}\sum_v \log \pi(z_v)), (\ell_{\Delta\theta}=|\Delta\theta^C|_1).
* **KL budget:** enforce via dual ascent or reverse‑KL to a prior.
* **Calibration:** prequential ECE + isotonic temperature on a lag buffer.

**normalization:** rank/winsor per loss; RMS inside each head; beware denominator gaming.

**gradient surgery:** PCGrad/CaGrad on per‑loss VJPs; monitor gradient‑cosine heatmaps.

---

## appendix B — EVSI (one‑step sketch)

[
\text{EVSI} = \mathbb{E}*{o\sim p(o|a*{tool})}\big[\max_{a},U(s,o,a)\big] - \max_{a} \mathbb{E}*{o}\big[U(s,o,a)\big] - C*{tool}
]

i call the tool if EVSI > 0; otherwise i keep building with existing beliefs.

---

## appendix C — promotion protocol

1. freeze candidate adapters; unit + goldens pass.
2. run shadow eval battery; compute ΔV_s distribution and policy checks.
3. enforce KL trust region against baseline.
4. **if** (ΔV_s ≤ 0 at 95th pct) ∧ (policy green) ∧ (KL ≤ δ): promote. else rollback and file diffs.

---

## closing

this repo is my attempt to steer a giant lm toward **constructive generalization** with things i can prove, lint, and veto. if we do it right, new capabilities will read like *more lego*, not *more coral*. build, then search—if you still need to.
