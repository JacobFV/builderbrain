"""
Plan checker for validating and executing plan DAGs.

Ensures plans conform to domain schemas before execution.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import yaml
from dataclasses import dataclass
from .plan_schemas import PlanSchema, NodeDefinition, EdgeDefinition, Precondition


@dataclass
class ValidationResult:
    """Result of plan validation."""
    valid: bool
    errors: List[str]
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PlanChecker:
    """
    Validates plan DAGs against domain schemas.

    Checks node types, edge relationships, preconditions, and resource constraints.
    """

    def __init__(self, schema_path: str):
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)

        self.schema = PlanSchema.from_dict(schema_data)
        self._build_validation_cache()

    def _build_validation_cache(self):
        """Build internal caches for efficient validation."""
        self.node_types = {node.id for node in self.schema.nodes}
        self.edge_types = {edge.type for edge in self.schema.edges}

        # Build adjacency constraints
        self.outgoing_edges = {node.id: [] for node in self.schema.nodes}
        self.incoming_edges = {node.id: [] for node in self.schema.nodes}

        for edge in self.schema.edges:
            self.outgoing_edges[edge.from_node].append(edge)
            self.incoming_edges[edge.to_node].append(edge)

    def validate_plan(self, plan_dag: Dict[str, Any]) -> ValidationResult:
        """
        Validate a complete plan DAG.

        Args:
            plan_dag: Plan representation with nodes, edges, and metadata

        Returns:
            ValidationResult with validity and any errors/warnings
        """
        errors = []
        warnings = []

        # Extract plan components
        nodes = plan_dag.get('nodes', [])
        edges = plan_dag.get('edges', [])
        metadata = plan_dag.get('metadata', {})

        # 1. Validate nodes
        node_errors = self._validate_nodes(nodes)
        errors.extend(node_errors)

        # 2. Validate edges
        edge_errors = self._validate_edges(edges, nodes)
        errors.extend(edge_errors)

        # 3. Check DAG structure (no cycles)
        cycle_errors = self._check_acyclic(edges, nodes)
        errors.extend(cycle_errors)

        # 4. Validate preconditions
        precondition_errors = self._validate_preconditions(nodes, metadata.get('state', {}))
        errors.extend(precondition_errors)

        # 5. Check resource constraints
        resource_warnings = self._check_resource_constraints(nodes, edges)
        warnings.extend(resource_warnings)

        # 6. Validate node parameter compatibility
        param_errors = self._validate_node_parameters(nodes)
        errors.extend(param_errors)

        valid = len(errors) == 0

        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings
        )

    def _validate_nodes(self, nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate that all nodes conform to schema."""
        errors = []

        for node in nodes:
            node_id = node.get('id')
            node_type = node.get('type')

            if node_id not in self.node_types:
                errors.append(f"Unknown node type: {node_type}")

            # Check required parameters
            if node_id in [n.id for n in self.schema.nodes]:
                schema_node = next(n for n in self.schema.nodes if n.id == node_id)
                for param in schema_node.required_params:
                    if param not in node.get('params', {}):
                        errors.append(f"Missing required parameter '{param}' for node '{node_id}'")

        return errors

    def _validate_edges(self, edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate that all edges conform to schema."""
        errors = []

        node_ids = {node['id'] for node in nodes}

        for edge in edges:
            from_node = edge.get('from')
            to_node = edge.get('to')
            edge_type = edge.get('type')

            # Check nodes exist
            if from_node not in node_ids:
                errors.append(f"Edge source node '{from_node}' not found")
                continue

            if to_node not in node_ids:
                errors.append(f"Edge target node '{to_node}' not found")
                continue

            # Check edge type is allowed
            if edge_type not in self.edge_types:
                errors.append(f"Unknown edge type: {edge_type}")

            # Check if this specific edge is allowed by schema
            allowed_edge = self._is_edge_allowed(from_node, to_node, edge_type)
            if not allowed_edge:
                errors.append(f"Edge {from_node} -> {to_node} ({edge_type}) not allowed by schema")

        return errors

    def _is_edge_allowed(self, from_node: str, to_node: str, edge_type: str) -> bool:
        """Check if an edge is allowed by the schema."""
        # Check if there's a matching edge definition
        for edge in self.schema.edges:
            if (edge.from_node == from_node and
                edge.to_node == to_node and
                edge.type == edge_type):
                return True

        return False

    def _check_acyclic(self, edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[str]:
        """Check that the graph is acyclic."""
        errors = []

        # Build adjacency list
        adj_list = {node['id']: [] for node in nodes}
        for edge in edges:
            adj_list[edge['from']].append(edge['to'])

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in nodes:
            node_id = node['id']
            if node_id not in visited:
                if has_cycle(node_id):
                    errors.append(f"Cycle detected in plan graph involving node '{node_id}'")

        return errors

    def _validate_preconditions(self, nodes: List[Dict[str, Any]], state: Dict[str, Any]) -> List[str]:
        """Validate that preconditions are satisfied."""
        errors = []

        for node in nodes:
            node_id = node['id']

            # Find matching schema node
            schema_node = next((n for n in self.schema.nodes if n.id == node_id), None)
            if schema_node is None:
                continue

            # Check each precondition
            for precondition in schema_node.preconditions:
                if not self._evaluate_precondition(precondition, state):
                    errors.append(f"Precondition failed for node '{node_id}': {precondition.expression}")

        return errors

    def _evaluate_precondition(self, precondition: Precondition, state: Dict[str, Any]) -> bool:
        """Evaluate a precondition expression against current state."""
        # Simple evaluation - in practice would use a proper expression evaluator
        try:
            # Replace state variables in expression
            expr = precondition.expression
            for var_name, var_value in state.items():
                expr = expr.replace(var_name, str(var_value))

            # Simple evaluation (would need a proper evaluator for complex expressions)
            return eval(expr)
        except Exception:
            return False  # Fail safe

    def _check_resource_constraints(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
        """Check resource constraints and concurrency limits."""
        warnings = []

        # Simple resource checking - in practice would be more sophisticated
        concurrent_nodes = set()

        for edge in edges:
            if edge.get('type') == 'par':  # Parallel edges
                concurrent_nodes.add(edge['from'])
                concurrent_nodes.add(edge['to'])

        if len(concurrent_nodes) > 10:  # Arbitrary threshold
            warnings.append(f"High concurrency detected: {len(concurrent_nodes)} parallel nodes")

        return warnings

    def _validate_node_parameters(self, nodes: List[Dict[str, Any]]) -> List[str]:
        """Validate node parameter types and ranges."""
        errors = []

        for node in nodes:
            node_id = node['id']
            params = node.get('params', {})

            # Find schema node
            schema_node = next((n for n in self.schema.nodes if n.id == node_id), None)
            if schema_node is None:
                continue

            # Check parameter constraints
            for param_name, param_value in params.items():
                constraint = getattr(schema_node, f'param_{param_name}_constraint', None)
                if constraint:
                    if not self._check_parameter_constraint(param_value, constraint):
                        errors.append(f"Parameter '{param_name}' for node '{node_id}' violates constraint: {constraint}")

        return errors

    def _check_parameter_constraint(self, value: Any, constraint: str) -> bool:
        """Check if parameter value satisfies constraint."""
        # Simple constraint checking - in practice would be more sophisticated
        try:
            if '<=' in constraint:
                _, bound = constraint.split('<=')
                return value <= float(bound)
            elif '>=' in constraint:
                _, bound = constraint.split('>=')
                return value >= float(bound)
            elif '<' in constraint:
                _, bound = constraint.split('<')
                return value < float(bound)
            elif '>' in constraint:
                _, bound = constraint.split('>')
                return value > float(bound)
            else:
                return True  # No constraint or unknown format
        except Exception:
            return True  # Fail safe

    def estimate_execution_time(self, plan_dag: Dict[str, Any]) -> float:
        """Estimate total execution time for the plan."""
        nodes = plan_dag.get('nodes', [])

        # Simple estimation based on node types
        total_time = 0.0
        for node in nodes:
            node_type = node.get('type', '')
            # Default time estimates by type
            time_estimates = {
                'grasp': 2.0,
                'rotate': 1.5,
                'place': 1.0,
                'move': 3.0,
                'sense': 0.5
            }
            total_time += time_estimates.get(node_type, 1.0)

        return total_time

    def get_resource_requirements(self, plan_dag: Dict[str, Any]) -> Dict[str, float]:
        """Get resource requirements for plan execution."""
        nodes = plan_dag.get('nodes', [])

        resources = {
            'cpu': 0.0,
            'memory': 0.0,
            'network': 0.0,
            'compute': 0.0
        }

        for node in nodes:
            node_type = node.get('type', '')

            # Resource estimates by type
            resource_estimates = {
                'grasp': {'cpu': 0.8, 'memory': 0.2, 'network': 0.1, 'compute': 0.5},
                'rotate': {'cpu': 0.6, 'memory': 0.1, 'network': 0.0, 'compute': 0.3},
                'place': {'cpu': 0.7, 'memory': 0.2, 'network': 0.1, 'compute': 0.4},
                'move': {'cpu': 0.9, 'memory': 0.3, 'network': 0.2, 'compute': 0.8},
                'sense': {'cpu': 0.4, 'memory': 0.5, 'network': 0.8, 'compute': 0.2}
            }

            estimates = resource_estimates.get(node_type, {'cpu': 0.5, 'memory': 0.1, 'network': 0.1, 'compute': 0.3})

            for resource, amount in estimates.items():
                resources[resource] += amount

        return resources
