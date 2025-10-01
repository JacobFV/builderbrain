# RUNTIME_PLAYBOOK.md — builderbrain

## 0. purpose

Runtime playbook defines how BuilderBrain executes in production: decode modes, plan repair, fallbacks, deferral logic, timeouts, and retry policies. This ensures reliable operation across domains (chat, API, robots, phone, social).

---

## 1. decode modes

### 1.1 syntax-aware decoding

**hard masking (strict domains):**
* API/JSON: 100% compliance required
* Robot DSL: safety-critical, no violations allowed
* Phone callflows: format must be preserved

**soft energy (flexible domains):**
* Chat/social: creativity encouraged, minor violations acceptable
* UI macros: structural correctness with content flexibility

**implementation:**
```python
# Strict mode
mask = GrammarMask(grammar, tokenizer, strict=True)
logits = mask(logits, prefix)

# Flexible mode
mask = GrammarMask(grammar, tokenizer, strict=False)
logits = logits - energy_scale * grammar_energy(tokens)
```

### 1.2 beam search with constraints

**constrained beam search:**
* Maintains beam diversity while respecting grammar
* Prunes invalid continuations at each step
* Fallback to unconstrained search if beam empties

**parameters:**
* `num_beams`: 4-8 for most domains
* `diversity_penalty`: 1.0 for exploration, 0.0 for consistency
* `grammar_weight`: 2.0 for strict, 0.5 for flexible

### 1.3 temperature annealing

**adaptive temperature:**
* Start high (1.2) for exploration
* Anneal to low (0.7) for exploitation
* Domain-specific schedules

---

## 2. plan execution flows

### 2.1 validation pipeline

**pre-execution checks:**
1. Parse DAG proposal from latent graph head
2. Verify node types against schema
3. Check edge relationships and cycles
4. Validate preconditions against current state
5. Estimate resource requirements

**rejection handling:**
* Invalid plans → fallback to base rail template
* Resource conflicts → defer or queue
* Precondition failures → wait or alternative path

### 2.2 execution patterns

**sequential execution:**
```python
for node_id in topological_order:
    result = await execute_node(node_id, state)
    if not result.success:
        handle_failure(node_id, result.error)
        break  # Stop on critical failures
```

**parallel execution:**
```python
# Resource-aware parallel execution
concurrent_nodes = identify_parallel_groups(plan_dag)
for group in concurrent_nodes:
    await execute_parallel(group, resource_locks)
```

**conditional execution:**
```python
if evaluate_condition(node.precondition, current_state):
    await execute_node(node, state)
```

### 2.3 plan repair flows

**automatic repair:**
* Missing parameters → infer from context
* Invalid edges → remove and reconnect
* Resource conflicts → reschedule or substitute

**repair strategies:**
1. **parameter completion:** use base rail to fill missing args
2. **edge repair:** remove invalid edges, add necessary dependencies
3. **node substitution:** replace with functionally equivalent nodes
4. **plan simplification:** remove optional branches if core fails

---

## 3. fallback mechanisms

### 3.1 graceful degradation

**fallback hierarchy:**
1. **Constrained generation** (preferred)
2. **Base rail template** (fallback)
3. **Human intervention** (last resort)
4. **Error response** (fail safe)

**domain-specific fallbacks:**
* **API calls:** return error schema with explanation
* **Robot actions:** safe default poses and stop commands
* **Phone transfers:** polite hold message and escalation
* **Chat responses:** "I need more information" template

### 3.2 circuit breakers

**failure thresholds:**
* Consecutive constraint violations > 5 → switch to fallback
* Grammar energy > τ for 3+ steps → constrained mode
* Plan validation failures > 80% → human review required

**recovery procedures:**
* Reset dual variables after fallback period
* Gradually re-enable constraints
* Log failure patterns for analysis

---

## 4. deferral logic

### 4.1 when to defer

**deferral triggers:**
* High uncertainty (confidence < 0.7)
* Ambiguous user intent
* Insufficient context for safe decision
* Resource constraints (memory, compute, time)
* Domain expertise gaps

**deferral responses:**
* **Clarification questions:** "What type of X are you looking for?"
* **Information requests:** "Please provide more details about Y"
* **Alternative suggestions:** "Would you like me to Z instead?"
* **Human escalation:** "I'll connect you with a specialist"

### 4.2 deferral calibration

**confidence thresholds by domain:**
* Chat/social: 0.6 (forgiving)
* API calls: 0.8 (strict)
* Robot actions: 0.9 (safety critical)
* Phone transfers: 0.7 (moderate)

**calibration updates:**
* Track deferral accuracy (did user find answer helpful?)
* Adjust thresholds based on feedback
* Domain-specific learning from deferral patterns

---

## 5. timeout management

### 5.1 execution timeouts

**per-operation limits:**
* Token generation: 30s max
* Plan validation: 5s max
* Node execution: domain-specific (1-10s)
* Full conversation: 5min max

**timeout handling:**
* Graceful truncation with partial results
* Context summarization for long conversations
* Resource cleanup on timeout
* User notification of truncation

### 5.2 retry policies

**retry strategies:**
* **Exponential backoff:** 1s, 2s, 4s, 8s max
* **Circuit breaker:** 3 failures → cooldown period
* **Partial retry:** retry failed components only
* **Alternative paths:** try different execution strategies

**retry conditions:**
* Transient failures (network, resource contention)
* Non-deterministic errors (timing, race conditions)
* Recoverable state corruption

---

## 6. error handling

### 6.1 error classification

**error types:**
* **Grammar violations:** invalid syntax/structure
* **Plan failures:** execution errors, precondition failures
* **Resource errors:** memory, compute, bandwidth limits
* **Safety violations:** risk energy above threshold
* **Integration errors:** external service failures

**severity levels:**
* **Critical:** safety violations, system failures
* **High:** constraint violations, execution failures
* **Medium:** resource limits, performance issues
* **Low:** minor formatting issues, style violations

### 6.2 error recovery

**recovery actions by type:**
* **Grammar errors:** fallback to unconstrained generation
* **Plan errors:** repair or human escalation
* **Resource errors:** queue, defer, or simplify
* **Safety errors:** immediate rollback and alert
* **Integration errors:** retry with backoff or alternative

**monitoring and alerting:**
* Error rate dashboards
* Failure pattern analysis
* Automated incident response
* Human oversight for critical errors

---

## 7. performance optimization

### 7.1 caching strategies

**cache types:**
* Grammar parsing results (token validity)
* Plan validation outcomes
* Model hidden states (for repeated queries)
* External API responses

**cache policies:**
* LRU eviction for memory management
* TTL for staleness prevention
* Domain-specific cache sizes

### 7.2 batching and parallelism

**execution optimization:**
* Batch similar operations across requests
* Parallel execution of independent plan nodes
* Pre-compute expensive grammar operations
* Async I/O for external service calls

**resource management:**
* Memory pooling for large models
* GPU utilization optimization
* Network connection pooling
* Rate limiting for external APIs

---

## 8. monitoring and observability

### 8.1 key metrics

**performance metrics:**
* Response latency (P50, P95, P99)
* Throughput (requests/second)
* Error rates by type and domain
* Resource utilization (CPU, GPU, memory)

**quality metrics:**
* Grammar compliance rates
* Plan execution success rates
* Constraint satisfaction scores
* User satisfaction scores

### 8.2 operational dashboards

**real-time monitoring:**
* Error rate trends
* Resource utilization graphs
* Constraint violation alerts
* Performance degradation detection

**historical analysis:**
* Failure pattern identification
* Performance regression detection
* Constraint effectiveness tracking
* User behavior insights

---

## 9. domain-specific playbooks

### 9.1 API/JSON agent

**decode mode:** strict hard masking
**plan validation:** schema compliance required
**fallback:** error response with explanation
**deferral:** request clarification for ambiguous endpoints

### 9.2 robot/factory agent

**decode mode:** strict hard masking
**plan validation:** safety preconditions mandatory
**fallback:** safe default poses and stop commands
**deferral:** human oversight for complex trajectories

### 9.3 phone/video agent

**decode mode:** semi-strict with content flexibility
**plan validation:** format preservation required
**fallback:** standard response templates
**deferral:** escalation to human operators

### 9.4 chat/social agent

**decode mode:** soft energy with creativity
**plan validation:** content appropriateness
**fallback:** generic helpful responses
**deferral:** topic clarification questions

---

## 10. operational procedures

### 10.1 deployment checklist

**pre-deployment:**
* All constraints calibrated and tested
* Grammar parsers validated against golden tests
* Plan schemas verified for completeness
* Fallback mechanisms tested
* Performance benchmarks met

**post-deployment:**
* Monitor error rates and constraint violations
* Validate grammar compliance in production
* Track plan execution success rates
* User feedback collection and analysis

### 10.2 incident response

**incident levels:**
* **P0:** safety violations, system failures
* **P1:** constraint violations > threshold
* **P2:** performance degradation
* **P3:** minor formatting issues

**response procedures:**
* Automated rollback for P0/P1
* Human investigation for P1/P2
* Code fixes for P2/P3
* Documentation updates for all levels

---

## 11. future enhancements

**planned improvements:**
* Advanced plan repair with learned repair policies
* Dynamic constraint adaptation based on context
* Multi-domain plan composition
* Advanced deferral strategies with reinforcement learning
* Real-time constraint calibration from user feedback

**research directions:**
* Grammar induction from execution traces
* Plan generalization across domains
* Safety constraint learning from demonstrations
* Multi-objective optimization improvements
