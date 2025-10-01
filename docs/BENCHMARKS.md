# BENCHMARKS.md — builderbrain

## 0. purpose

Benchmarks define the canonical evaluation tasks for BuilderBrain across domains, establish acceptance thresholds for deployment, and track skill-stacking improvements over time. These benchmarks ensure the system meets production requirements and demonstrates compositional generalization.

---

## 1. evaluation framework

### 1.1 benchmark categories

**compositional tasks:**
* Multi-step reasoning requiring skill composition
* Cross-domain knowledge transfer
* Novel task adaptation from existing skills

**grammar compliance:**
* Syntax correctness in structured outputs
* Format preservation across domains
* Constraint satisfaction under variation

**plan execution:**
* DAG validation and precondition checking
* Execution success rates and error recovery
* Resource utilization and timing accuracy

**safety and reliability:**
* Risk energy calibration and violation detection
* Fallback mechanism effectiveness
* Constraint satisfaction under adversarial conditions

### 1.2 evaluation metrics

**accuracy metrics:**
* Task completion rate (end-to-end success)
* Constraint satisfaction rate (grammar/plan compliance)
* Error rate by type and severity
* User satisfaction scores

**efficiency metrics:**
* Response time (P50, P95, P99)
* Resource utilization (CPU, GPU, memory)
* Token efficiency (output quality per token)
* Plan execution efficiency (nodes/second)

**robustness metrics:**
* Performance degradation under load
* Constraint violation recovery rates
* Adversarial input handling
* Domain shift adaptation

---

## 2. domain-specific benchmarks

### 2.1 API/JSON agent benchmarks

**json formatting:**
* Valid JSON generation rate: >99%
* Schema compliance: >98%
* Nested structure accuracy: >95%

**api call composition:**
* Multi-call sequence accuracy: >90%
* Parameter binding correctness: >95%
* Error handling and recovery: >85%

**test cases:**
```json
{
  "name": "complex_api_sequence",
  "description": "Generate nested API calls with error handling",
  "input": "Create a user profile and send welcome email",
  "expected_structure": {
    "steps": [
      {"action": "validate_input", "params": {...}},
      {"action": "create_user", "params": {...}, "on_error": "retry"},
      {"action": "send_email", "params": {...}, "depends_on": "create_user"}
    ]
  },
  "acceptance_threshold": 0.9
}
```

### 2.2 robot manipulation benchmarks

**trajectory planning:**
* Collision-free path generation: >95%
* Precondition satisfaction: >98%
* Execution time accuracy: <10% deviation

**skill composition:**
* Multi-step manipulation success: >85%
* Tool selection accuracy: >90%
* Constraint propagation: >80%

**test cases:**
```yaml
name: pick_place_sequence
description: Pick object and place in target location
preconditions:
  - gripper.open: true
  - object.visible: true
  - target_location.clear: true
steps:
  - action: move_to_object
    params: {object_id: "red_cube"}
  - action: grasp
    params: {object_id: "red_cube", force: 5.0}
  - action: move_to_target
    params: {target_pose: {x: 0.5, y: 0.2, z: 0.1}}
  - action: place
    params: {release_force: 2.0}
acceptance_threshold: 0.85
```

### 2.3 phone/video agent benchmarks

**call flow accuracy:**
* Intent recognition: >90%
* State transition correctness: >95%
* Format compliance: >98%

**conversation management:**
* Context preservation: >85%
* Escalation accuracy: >80%
* Multi-turn coherence: >75%

**test cases:**
```json
{
  "name": "customer_service_flow",
  "description": "Handle customer inquiry with proper escalation",
  "input": "My order hasn't arrived yet",
  "expected_flow": [
    {"intent": "order_inquiry", "confidence": 0.9},
    {"action": "check_order_status", "params": {...}},
    {"condition": "order_delayed", "then": "escalate_to_agent"},
    {"response": "polite_escalation_message"}
  ],
  "acceptance_threshold": 0.8
}
```

### 2.4 chat/social agent benchmarks

**response quality:**
* Relevance to query: >85%
* Grammar correctness: >95%
* Appropriateness: >90%

**creativity and coherence:**
* Novel response generation: >70%
* Context awareness: >80%
* Multi-turn consistency: >75%

**test cases:**
```json
{
  "name": "creative_storytelling",
  "description": "Generate coherent story continuation",
  "input": "Once upon a time in a magical forest...",
  "expected": {
    "coherence": 0.8,
    "creativity": 0.7,
    "grammar": 0.95,
    "length_appropriateness": 0.85
  },
  "acceptance_threshold": 0.75
}
```

---

## 3. skill-stacking curves

### 3.1 skill acquisition metrics

**time-to-mastery:**
* First skill acquisition: <10 training examples
* Skill composition: <50 examples for 2-skill combinations
* Complex planning: <100 examples for 3+ skill sequences

**generalization metrics:**
* Zero-shot adaptation: >60% success on novel combinations
* Few-shot learning: <10 examples for new skill variants
* Cross-domain transfer: >70% retention across domains

### 3.2 performance scaling

**skill count vs performance:**
* 1 skill: baseline performance
* 4 skills: 2x performance improvement
* 8 skills: 3x performance improvement
* 16 skills: 4x performance improvement

**training efficiency:**
* Parameter efficiency: <10% parameter increase per skill
* Training time scaling: sublinear with skill count
* Inference time: <20% increase per additional skill

### 3.3 compositional benchmarks

**composition depth:**
* 1-step: single skill execution
* 2-step: sequential skill composition
* 3-step: conditional skill branching
* 4+ step: complex multi-branch planning

**composition breadth:**
* Single domain: intra-domain composition
* Multi-domain: cross-domain skill reuse
* Meta-composition: composition of compositions

---

## 4. acceptance thresholds

### 4.1 deployment gates

**production deployment:**
* Grammar compliance: >99% on validation set
* Plan execution success: >90% on test cases
* Constraint satisfaction: >95% on adversarial inputs
* Safety violations: <0.1% false negative rate

**model updates:**
* Performance regression: <5% on benchmark suite
* Constraint degradation: <2% on validation metrics
* Safety improvements: >10% reduction in risk energy

### 4.2 continuous monitoring

**daily thresholds:**
* Error rate: <1% on production traffic
* Response time: P95 < 2 seconds
* Constraint violations: <0.5% of requests

**weekly thresholds:**
* Performance drift: <2% degradation
* New constraint violations: <10 total incidents
* User satisfaction: >4.2/5 average rating

---

## 5. benchmark datasets

### 5.1 synthetic benchmarks

**grammar compliance dataset:**
* 10,000 synthetically generated examples per grammar
* Balanced positive/negative examples
* Domain-specific edge cases
* Adversarial constraint violations

**plan execution dataset:**
* 5,000 DAG structures of varying complexity
* Precondition satisfaction scenarios
* Resource constraint variations
* Error injection and recovery cases

### 5.2 real-world benchmarks

**api interaction corpus:**
* 50,000 real API call sequences
* Multi-step transaction patterns
* Error handling scenarios
* Schema evolution cases

**robot task corpus:**
* 10,000 manipulation trajectories
* Multi-object interactions
* Constraint satisfaction scenarios
* Safety violation cases

---

## 6. evaluation protocols

### 6.1 automated evaluation

**syntax validation:**
* JSON schema validation
* Grammar parsing success rates
* Format compliance checking

**semantic evaluation:**
* Intent recognition accuracy
* Plan semantic correctness
* Constraint satisfaction verification

**performance evaluation:**
* Execution time measurement
* Resource utilization tracking
* Throughput and latency analysis

### 6.2 human evaluation

**quality assessment:**
* Response relevance and helpfulness
* Grammar and style quality
* Plan execution effectiveness

**safety evaluation:**
* Risk energy calibration
* Constraint violation detection
* Fallback mechanism assessment

**usability evaluation:**
* Interface intuitiveness
* Error message clarity
* Recovery procedure effectiveness

---

## 7. skill-stacking validation

### 7.1 incremental skill addition

**validation protocol:**
1. Train baseline model on single skill
2. Add second skill and measure composition improvement
3. Continue adding skills, tracking generalization
4. Validate cross-domain skill transfer

**success criteria:**
* Each new skill improves performance by >10%
* Composition efficiency >80% of theoretical maximum
* Cross-domain transfer >60% effectiveness

### 7.2 ablation studies

**skill importance:**
* Remove each skill individually
* Measure performance degradation
* Identify critical vs optional skills

**composition analysis:**
* Compare skill composition vs monolithic approaches
* Measure parameter efficiency gains
* Validate generalization improvements

---

## 8. performance tracking

### 8.1 dashboard metrics

**real-time tracking:**
* Benchmark scores by category
* Performance trends over time
* Constraint satisfaction rates
* Error pattern analysis

**historical comparison:**
* Performance improvements across versions
* Skill-stacking curve evolution
* Constraint effectiveness trends

### 8.2 regression detection

**automated monitoring:**
* Performance degradation alerts
* Constraint violation spikes
* Quality metric declines

**investigation protocols:**
* Automated rollback on regression
* Human review for complex issues
* Root cause analysis and fixes

---

## 9. future benchmark development

**planned enhancements:**
* Multi-domain composition benchmarks
* Long-horizon planning evaluation
* Adversarial robustness testing
* Real-time adaptation benchmarks

**research directions:**
* Automated benchmark generation
* Self-improving evaluation metrics
* Cross-system benchmark standardization
* Ethical AI evaluation frameworks
