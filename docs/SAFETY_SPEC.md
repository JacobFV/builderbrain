# SAFETY_SPEC.md — builderbrain

## 0. purpose

Safety is the hard constraint of builderbrain. All other objectives are soft. This doc defines:

* the surrogate risk energy **V_s**.
* domains + features for safety heads.
* promotion protocol.
* shadow eval design.
* rollback and audit rules.

---

## 1. V_s (risk energy)

### 1.1 definition

* scalar ≥0, predicted from hidden state (h^B, h^C, z, graph features).
* learned head `V_s(h)` trained on labeled risk events.

### 1.2 constraint

[
ΔV_s = V_s^{new} - V_s^{old} ≤ 0
]
Promotion only if inequality holds at high confidence (95th percentile across eval set).

### 1.3 training signals

* **comms/social:** toxicity, PII, policy violations, jailbreak triggers.
* **finance:** anomalous refunds, AML patterns, compliance breaches.
* **robot/factory:** force/torque spikes, collision proximity, e-stop triggers.

---

## 2. features for V_s

* fused hidden (h^B, h^C).
* program token sequence z_t.
* plan graph features (nodes, edges, resource locks).
* decoder outputs (tokens, logits).
* external telemetry (robot sensors, financial logs).

---

## 3. promotion protocol

### 3.1 steps

1. freeze candidate adapters.
2. run shadow eval battery.
3. compute ΔV_s distribution vs baseline.
4. enforce KL trust region.
5. gate decision: promote or rollback.

### 3.2 criteria

* ΔV_s ≤ 0 at 95th percentile.
* policy checks passed.
* KL divergence ≤ δ.

### 3.3 rollback

* automatic on any violation.
* artifact logs preserved.

---

## 4. shadow eval battery

### 4.1 comms/social

* red-team prompts (toxicity, PII, policy edge).
* long-jailbreak attempts.

### 4.2 finance

* AML synthetic traces.
* refund loops.
* mislabelled compliance docs.

### 4.3 robots/factory

* near-miss trajectories.
* forbidden zones.
* randomized perturbations.

### 4.4 metrics

* violation counts.
* ΔV_s distributions.
* constraint satisfaction rates.

---

## 5. runtime safety checks

* plan checker: enforce schema invariants.
* conformal deferral under shift.
* EVSI gating (tools/robots).
* sandbox execution for high-risk plans.

---

## 6. logging + audit

* log V_s predictions per action.
* log ΔV_s vs baseline per promotion.
* immutable artifacts: model hash, grammar/schema version, config.
* audit trail of rejected promotions.

---

## 7. dashboards

* V_s time series.
* ΔV_s histograms.
* domain-specific risk metrics.
* promotion gate outcomes.
* rollback events.

---

## 8. failure smells

* **V_s drift:** predictions collapse to constant.
* **false negatives:** risky behavior passes shadow eval.
* **false positives:** safe upgrades blocked by spurious ΔV_s.
* **overfit battery:** passing eval but failing in wild.

---

## 9. ethos

Safety is **non-negotiable**. All other objectives can bend; V_s cannot. Promotion is gated, rollback is automatic, logs are immutable. If in doubt, defer.
