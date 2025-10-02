---
layout: post
title: "Plan Execution: From Plans to Actions in BuilderBrain"
date: 2024-10-04
categories: ai ml robotics planning
excerpt: "How BuilderBrain converts generated plans into executable actions with validation, error handling, and real-world integration."
---

## The Gap Between Plans and Actions

Most AI systems generate text or high-level descriptions, but BuilderBrain goes further: it generates **executable plans** that can be validated and executed in the real world.

The challenge: bridging the gap between abstract plans and concrete actions.

## What is Plan Execution?

BuilderBrain generates **Directed Acyclic Graphs (DAGs)** representing executable workflows:

```
Input: "Pick up the red cube and place it on the blue platform"

Generated Plan:
├── validate_gripper_open
├── move_to_object(red_cube)
├── grasp(red_cube, force=5.0)
├── move_to_target(blue_platform)
└── place(red_cube, release_force=2.0)

With dependencies: grasp → move_to_target → place
```

Each node is an **executable action** with parameters and preconditions.

## Plan Structure

### Nodes: Executable Actions
```python
@dataclass
class PlanNode:
    id: str                    # Unique identifier
    action_type: str          # "grasp", "move", "place", etc.
    parameters: Dict[str, Any] # Action parameters
    preconditions: List[str]  # Required state conditions
    resource_cost: Dict[str, float]  # CPU, memory, etc.
```

### Edges: Dependencies
```python
@dataclass
class PlanEdge:
    from_node: str
    to_node: str
    edge_type: str  # "seq", "par", "cond"
    condition: Optional[str]  # For conditional edges
```

## Plan Validation

Before execution, plans are validated against domain schemas:

```python
def validate_plan(plan_dag: Dict) -> ValidationResult:
    # Check node types against schema
    for node in plan_dag['nodes']:
        if node['type'] not in schema.allowed_types:
            return ValidationError(f"Unknown action: {node['type']}")

    # Check edge relationships
    for edge in plan_dag['edges']:
        if not schema.allows_edge(edge['from'], edge['to'], edge['type']):
            return ValidationError(f"Invalid edge: {edge}")

    # Check preconditions
    for node in plan_dag['nodes']:
        if not check_preconditions(node['preconditions'], current_state):
            return ValidationError(f"Precondition failed: {node}")

    return ValidationSuccess()
```

## Execution Engine

### Sequential Execution
```python
async def execute_sequential(nodes: List[PlanNode], state: Dict) -> ExecutionResult:
    for node in nodes:
        # Execute node
        result = await execute_node(node, state)

        # Update state
        state.update(result.state_changes)

        # Handle errors
        if not result.success:
            return ExecutionFailure(node.id, result.error)

    return ExecutionSuccess()
```

### Parallel Execution
```python
async def execute_parallel(node_groups: List[List[PlanNode]], state: Dict) -> ExecutionResult:
    # Execute independent groups in parallel
    for group in node_groups:
        tasks = [execute_node(node, state) for node in group]
        results = await asyncio.gather(*tasks)

        # Check for failures
        for node, result in zip(group, results):
            if not result.success:
                return ExecutionFailure(node.id, result.error)

    return ExecutionSuccess()
```

### Conditional Execution
```python
async def execute_conditional(node: PlanNode, condition: str, state: Dict) -> Optional[ExecutionResult]:
    if evaluate_condition(condition, state):
        return await execute_node(node, state)
    return None
```

## Error Handling and Recovery

### Retry Logic
```python
async def execute_with_retry(node: PlanNode, max_retries: int = 3) -> ExecutionResult:
    for attempt in range(max_retries):
        result = await execute_node(node)

        if result.success:
            return result

        # Exponential backoff
        await asyncio.sleep(2 ** attempt)

    return ExecutionFailure(f"Failed after {max_retries} attempts")
```

### Fallback Strategies
```python
async def execute_with_fallback(node: PlanNode) -> ExecutionResult:
    # Try primary execution
    result = await execute_node(node)

    if not result.success:
        # Try alternative approach
        fallback_node = create_fallback_node(node)
        result = await execute_node(fallback_node)

    return result
```

## Real-World Integration

### Tool Adapters
Each plan node maps to a **tool adapter** that knows how to execute the action:

```python
class ToolAdapter:
    async def execute(self, parameters: Dict, state: Dict) -> ExecutionResult:
        # Implementation specific to tool
        pass

# Example adapters
grasp_adapter = GraspAdapter(robot_interface)
move_adapter = MoveAdapter(robot_interface)
api_adapter = APIAdapter(http_client)
```

### State Management
```python
class ExecutionState:
    def __init__(self):
        self.robot_state = {}      # Joint positions, gripper state
        self.environment = {}      # Object positions, obstacles
        self.api_tokens = {}       # Authentication tokens
        self.resource_usage = {}   # CPU, memory, network usage

    def update(self, changes: Dict):
        # Deep merge state updates
        self._merge_recursive(self.robot_state, changes.get('robot', {}))
        self._merge_recursive(self.environment, changes.get('environment', {}))
```

## Monitoring and Observability

### Execution Metrics
```python
@dataclass
class ExecutionMetrics:
    total_time: float
    nodes_executed: int
    nodes_failed: int
    resource_usage: Dict[str, float]
    error_patterns: List[str]
```

### Real-time Monitoring
```python
class ExecutionMonitor:
    async def monitor_execution(self, execution_id: str):
        while execution_is_running(execution_id):
            metrics = collect_execution_metrics(execution_id)
            send_metrics_to_dashboard(metrics)

            if metrics.error_rate > threshold:
                trigger_error_handling(execution_id)
```

## Code Example: Robot Manipulation

```python
# Generate plan for robot task
plan_text = model.generate("Pick up red cube and place on blue platform")
plan_dag = parse_plan_from_text(plan_text)

# Validate plan
validation = plan_checker.validate_plan(plan_dag)
if not validation.valid:
    # Repair or reject invalid plan
    repaired_plan = repair_plan(plan_dag, validation.errors)
    plan_dag = repaired_plan

# Execute plan
executor = PlanExecutor(tool_adapters)
result = await executor.execute_plan(plan_dag, current_state)

# Handle result
if result.success:
    print(f"Task completed in {result.execution_time}s")
else:
    print(f"Task failed: {result.error}")
    # Trigger human intervention or retry
```

## Benefits of Plan Execution

1. **Reliability**: Plans are validated before execution
2. **Safety**: Precondition checking prevents dangerous actions
3. **Efficiency**: Parallel execution of independent actions
4. **Debugging**: Clear execution logs and error reporting
5. **Recovery**: Automatic retry and fallback mechanisms

## Challenges and Solutions

**Challenge**: Plans can become very complex
**Solution**: Hierarchical planning with abstraction levels

**Challenge**: Real-world state is uncertain
**Solution**: Robust precondition checking with sensor feedback

**Challenge**: Execution failures require recovery
**Solution**: Comprehensive error handling with multiple fallback strategies

## Next Steps

In the next post, we'll explore [safety invariants](/ai/ml/safety/ethics/2024/10/05/safety-invariants/) - how BuilderBrain prevents harmful behavior during execution.

---

*Plan execution transforms BuilderBrain from a text generator into an action-taking system. It's the bridge between abstract reasoning and real-world impact.*
