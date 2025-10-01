"""
Plan executor for executing validated plan DAGs.

Maps plan nodes to actual tool/robot calls and manages execution flow.
"""

from typing import Dict, List, Any, Optional, Callable
import asyncio
import time


class PlanExecutor:
    """
    Executes validated plan DAGs by mapping nodes to tool calls.

    Handles sequential, parallel, and conditional execution patterns.
    """

    def __init__(self, tool_adapters: Dict[str, Callable]):
        self.tool_adapters = tool_adapters

    async def execute_plan(self, plan_dag: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete plan DAG.

        Args:
            plan_dag: Validated plan with nodes and edges
            context: Execution context (state, resources, etc.)

        Returns:
            Execution results and final state
        """
        nodes = plan_dag.get('nodes', [])
        edges = plan_dag.get('edges', [])

        # Build execution graph
        execution_order = self._compute_execution_order(nodes, edges)

        # Execute in topological order
        results = {}
        current_state = context.copy()

        for node_id in execution_order:
            node = next(n for n in nodes if n['id'] == node_id)

            # Execute node
            result = await self._execute_node(node, current_state)
            results[node_id] = result

            # Update state
            current_state.update(result.get('state_updates', {}))

        return {
            'results': results,
            'final_state': current_state,
            'execution_time': time.time() - context.get('start_time', time.time())
        }

    def _compute_execution_order(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
        """Compute topological execution order."""
        # Simple topological sort
        # In practice would handle parallel execution

        in_degree = {node['id']: 0 for node in nodes}
        adj_list = {node['id']: [] for node in nodes}

        for edge in edges:
            if edge.get('type') == 'seq':
                adj_list[edge['from']].append(edge['to'])
                in_degree[edge['to']] += 1

        # Topological sort (simplified)
        order = []
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]

        while queue:
            node_id = queue.pop(0)
            order.append(node_id)

            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    async def _execute_node(self, node: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single plan node."""
        node_id = node['id']
        node_type = node.get('type', '')

        if node_type not in self.tool_adapters:
            return {'error': f'No adapter for node type: {node_type}'}

        # Get adapter function
        adapter_fn = self.tool_adapters[node_type]

        # Prepare parameters
        params = node.get('params', {})
        params.update({'state': state})

        try:
            # Execute tool call
            result = await adapter_fn(**params)

            return {
                'success': True,
                'result': result,
                'execution_time': time.time() - state.get('node_start_time', time.time())
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - state.get('node_start_time', time.time())
            }

    def estimate_execution_cost(self, plan_dag: Dict[str, Any]) -> Dict[str, float]:
        """Estimate computational cost of plan execution."""
        nodes = plan_dag.get('nodes', [])

        total_cost = {
            'compute': 0.0,
            'memory': 0.0,
            'network': 0.0,
            'time': 0.0
        }

        for node in nodes:
            node_type = node.get('type', '')

            # Default cost estimates by type
            cost_estimates = {
                'grasp': {'compute': 0.8, 'memory': 0.2, 'network': 0.1, 'time': 2.0},
                'rotate': {'compute': 0.6, 'memory': 0.1, 'network': 0.0, 'time': 1.5},
                'place': {'compute': 0.7, 'memory': 0.2, 'network': 0.1, 'time': 1.0},
                'move': {'compute': 0.9, 'memory': 0.3, 'network': 0.2, 'time': 3.0}
            }

            costs = cost_estimates.get(node_type, {'compute': 0.5, 'memory': 0.1, 'network': 0.1, 'time': 1.0})

            for resource, cost in costs.items():
                total_cost[resource] += cost

        return total_cost


class MockToolAdapters:
    """Mock tool adapters for demonstration."""

    @staticmethod
    async def grasp_adapter(object_id: str, pose: Dict[str, float], state: Dict[str, Any], **kwargs):
        """Mock grasp operation."""
        await asyncio.sleep(0.1)  # Simulate execution time
        return {
            'object_grasped': object_id,
            'gripper_force': 5.0,
            'success': True
        }

    @staticmethod
    async def rotate_adapter(target_orientation: Dict[str, float], max_force: float, state: Dict[str, Any], **kwargs):
        """Mock rotate operation."""
        await asyncio.sleep(0.05)
        return {
            'final_orientation': target_orientation,
            'force_applied': min(max_force, 3.0),
            'success': True
        }

    @staticmethod
    async def place_adapter(target_pose: Dict[str, float], release_force: float, state: Dict[str, Any], **kwargs):
        """Mock place operation."""
        await asyncio.sleep(0.03)
        return {
            'object_placed': True,
            'final_position': target_pose,
            'success': True
        }
