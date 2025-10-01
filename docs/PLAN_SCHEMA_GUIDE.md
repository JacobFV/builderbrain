# PLAN_SCHEMA_GUIDE.md — builderbrain

## 0. purpose

Builderbrain emits latent **plan DAGs** (directed acyclic graphs) to represent compositional actions. Plan schemas define the valid shape of these DAGs:

* **node types** = skills, primitives, or domain actions.
* **edge types** = dependencies (seq, parallel, conditional).
* **preconditions** = invariants that must hold before execution.
* **checker API** = runtime validator that enforces schemas.

Schemas allow us to:

* audit plans before execution.
* map latent DAGs → executable tool/robot workflows.
* detect invalid or unsafe plans early.

---

## 1. schema structure

Each domain provides a YAML schema:

```
bb_domains/<domain>/plan_schema.yaml
```

### 1.1 top-level keys

* `nodes`: list of allowed node types.
* `edges`: allowed edge types and semantics.
* `preconditions`: per-node logical constraints.

### 1.2 node definition

```yaml
nodes:
  - id: pick
    type: grasp
    params: [pose, gripper_ok]
  - id: orient
    type: rotate
    params: [angle<=180]
  - id: place
    type: place
    params: [pose_clear]
```

* **id:** canonical symbolic name.
* **type:** semantic category.
* **params:** required inputs / invariants.

### 1.3 edge definition

```yaml
edges:
  - from: pick
    to: orient
    type: seq
  - from: orient
    to: place
    type: seq
```

* **type:** `seq` (must complete before), `par` (parallelizable), `cond` (conditional branch).

### 1.4 preconditions

```yaml
preconditions:
  pick: gripper.state == "open" and force < 2.0
  orient: collision_free == true
  place: bin.free_space > 10cm
```

* boolean expressions over telemetry or domain-specific state.

---

## 2. DAG semantics

* **seq:** edge (u,v) means u must finish before v starts.
* **par:** u and v may run concurrently, but resource locks must not conflict.
* **cond:** v executes iff predicate holds at runtime.
* **acyclic:** DAG must remain cycle-free.
* **root:** at least one entry node with no incoming edges.
* **sink:** at least one terminal node with no outgoing edges.

---

## 3. checker API

### 3.1 validation pipeline

1. Parse DAG proposal.
2. Verify node types ∈ schema.
3. Verify edge types valid + no cycles.
4. Check preconditions syntactically.
5. Simulate resource locks (domain-specific).

### 3.2 python interface

```python
from bb_runtime.plancheck import PlanChecker

checker = PlanChecker("bb_domains/robots/plan_schema.yaml")
result = checker.validate(plan)
if result.valid:
    execute(plan)
else:
    handle_invalid(result.errors)
```

### 3.3 result object

```python
{
  "valid": False,
  "errors": [
    {"node": "orient", "error": "param angle > 180"},
    {"edge": ("pick","orient"), "error": "cycle detected"}
  ]
}
```

---

## 4. mapping DAG → execution

### 4.1 node mapping

* Each node id maps to a **tool adapter** (API call, function, or robot primitive).
* Tool adapters live in `bb_domains/<domain>/adapters.py`.

### 4.2 parameter binding

* Extract params from node.
* Bind to adapter args.
* Resolve runtime values via environment state.

### 4.3 execution order

* Topological sort respecting `seq`.
* For `par`, schedule concurrently with locks.
* For `cond`, evaluate predicate at branch time.

### 4.4 example: robot

```
plan:
  nodes: [pick, orient, place]
  edges: [(pick,orient,seq), (orient,place,seq)]
execution:
  grasp(pose)
  rotate(angle)
  place(pose)
```

---

## 5. golden tests

### 5.1 valid examples

* canonical minimal plan.
* edge-case plan with parallel branches.

### 5.2 invalid examples

* cycles.
* unknown node type.
* missing precondition.

### 5.3 location

```
bb_domains/<domain>/tests/plans_valid.yaml
bb_domains/<domain>/tests/plans_invalid.yaml
```

---

## 6. failure smells

* **cycle detection triggered frequently** → schema too permissive.
* **false negatives (valid plans rejected)** → schema incomplete.
* **runtime errors despite valid check** → preconditions underspecified.
* **adapter mismatch** → node id drifted from tool mapping.

---

## 7. ethos

schemas are **contracts** between latent plans and executable systems. write them precise, keep them minimal, and test them brutally. if a plan can’t pass the checker, it must never touch the world.
