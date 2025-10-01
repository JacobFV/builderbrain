"""
Plan schema definitions for different domains.

Defines node types, edge types, preconditions, and constraints for plan validation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml


@dataclass
class Precondition:
    """Represents a precondition for a node or edge."""
    expression: str  # Boolean expression as string
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Precondition':
        return cls(
            expression=data['expression'],
            description=data.get('description', '')
        )


@dataclass
class NodeDefinition:
    """Defines a node type in the plan schema."""
    id: str
    type: str
    description: str = ""
    required_params: List[str] = None
    optional_params: List[str] = None
    preconditions: List[Precondition] = None
    resource_cost: Dict[str, float] = None

    def __post_init__(self):
        if self.required_params is None:
            self.required_params = []
        if self.optional_params is None:
            self.optional_params = []
        if self.preconditions is None:
            self.preconditions = []
        if self.resource_cost is None:
            self.resource_cost = {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeDefinition':
        return cls(
            id=data['id'],
            type=data['type'],
            description=data.get('description', ''),
            required_params=data.get('required_params', []),
            optional_params=data.get('optional_params', []),
            preconditions=[Precondition.from_dict(p) for p in data.get('preconditions', [])],
            resource_cost=data.get('resource_cost', {})
        )


@dataclass
class EdgeDefinition:
    """Defines an edge type in the plan schema."""
    from_node: str
    to_node: str
    type: str  # 'seq', 'par', 'cond'
    precondition: Optional[Precondition] = None
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeDefinition':
        return cls(
            from_node=data['from'],
            to_node=data['to'],
            type=data['type'],
            precondition=Precondition.from_dict(data['precondition']) if data.get('precondition') else None,
            description=data.get('description', '')
        )


@dataclass
class PlanSchema:
    """Complete plan schema for a domain."""
    name: str
    version: str
    description: str
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]
    global_preconditions: List[Precondition] = None
    resource_limits: Dict[str, float] = None

    def __post_init__(self):
        if self.global_preconditions is None:
            self.global_preconditions = []
        if self.resource_limits is None:
            self.resource_limits = {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanSchema':
        return cls(
            name=data['name'],
            version=data['version'],
            description=data.get('description', ''),
            nodes=[NodeDefinition.from_dict(n) for n in data['nodes']],
            edges=[EdgeDefinition.from_dict(e) for e in data['edges']],
            global_preconditions=[Precondition.from_dict(p) for p in data.get('global_preconditions', [])],
            resource_limits=data.get('resource_limits', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export schema to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'nodes': [
                {
                    'id': node.id,
                    'type': node.type,
                    'description': node.description,
                    'required_params': node.required_params,
                    'optional_params': node.optional_params,
                    'preconditions': [
                        {'expression': p.expression, 'description': p.description}
                        for p in node.preconditions
                    ],
                    'resource_cost': node.resource_cost
                }
                for node in self.nodes
            ],
            'edges': [
                {
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'type': edge.type,
                    'precondition': {
                        'expression': edge.precondition.expression,
                        'description': edge.precondition.description
                    } if edge.precondition else None,
                    'description': edge.description
                }
                for edge in self.edges
            ],
            'global_preconditions': [
                {'expression': p.expression, 'description': p.description}
                for p in self.global_preconditions
            ],
            'resource_limits': self.resource_limits
        }

    def save_to_file(self, filepath: str):
        """Save schema to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'PlanSchema':
        """Load schema from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class RobotPlanSchema(PlanSchema):
    """Specialized schema for robot manipulation tasks."""

    def __init__(self):
        super().__init__(
            name="robot_manipulation",
            version="1.0",
            description="Schema for robot manipulation plans",
            nodes=[
                NodeDefinition(
                    id="pick",
                    type="grasp",
                    description="Pick up an object",
                    required_params=["object_id", "pose"],
                    preconditions=[
                        Precondition("gripper.state == 'open'", "Gripper must be open"),
                        Precondition("object.position.z > 0", "Object must be above surface")
                    ],
                    resource_cost={"compute": 0.8, "time": 2.0}
                ),
                NodeDefinition(
                    id="orient",
                    type="rotate",
                    description="Rotate object to target orientation",
                    required_params=["target_orientation", "max_force"],
                    preconditions=[
                        Precondition("gripper.state == 'closed'", "Must be holding object"),
                        Precondition("target_orientation.w != 0", "Valid orientation")
                    ],
                    resource_cost={"compute": 0.6, "time": 1.5}
                ),
                NodeDefinition(
                    id="place",
                    type="place",
                    description="Place object at target location",
                    required_params=["target_pose", "release_force"],
                    preconditions=[
                        Precondition("gripper.state == 'closed'", "Must be holding object"),
                        Precondition("target_pose.z > 0", "Valid placement position")
                    ],
                    resource_cost={"compute": 0.7, "time": 1.0}
                ),
                NodeDefinition(
                    id="move",
                    type="move",
                    description="Move robot to target pose",
                    required_params=["target_pose", "velocity"],
                    preconditions=[
                        Precondition("robot.status == 'ready'", "Robot must be ready"),
                        Precondition("target_pose.reachable", "Target must be reachable")
                    ],
                    resource_cost={"compute": 0.9, "time": 3.0}
                )
            ],
            edges=[
                EdgeDefinition("pick", "orient", "seq", description="Pick before orient"),
                EdgeDefinition("orient", "place", "seq", description="Orient before place"),
                EdgeDefinition("move", "pick", "seq", description="Move to object before picking")
            ],
            resource_limits={
                "max_concurrent_actions": 3,
                "max_compute_load": 0.95,
                "max_time_per_plan": 30.0
            }
        )


class APIPlanSchema(PlanSchema):
    """Specialized schema for API call sequences."""

    def __init__(self):
        super().__init__(
            name="api_calls",
            version="1.0",
            description="Schema for API call sequences",
            nodes=[
                NodeDefinition(
                    id="authenticate",
                    type="auth",
                    description="Authenticate with service",
                    required_params=["api_key", "service"],
                    preconditions=[
                        Precondition("network.connected", "Network must be available"),
                        Precondition("credentials.valid", "Valid credentials required")
                    ],
                    resource_cost={"network": 0.1, "time": 0.5}
                ),
                NodeDefinition(
                    id="query",
                    type="request",
                    description="Make API request",
                    required_params=["endpoint", "method", "data"],
                    preconditions=[
                        Precondition("auth_token != null", "Must be authenticated"),
                        Precondition("endpoint.rate_limit_ok", "Rate limit not exceeded")
                    ],
                    resource_cost={"network": 0.5, "time": 1.0}
                ),
                NodeDefinition(
                    id="parse_response",
                    type="parse",
                    description="Parse API response",
                    required_params=["response_format"],
                    preconditions=[
                        Precondition("response.status == 200", "Successful response required")
                    ],
                    resource_cost={"compute": 0.3, "time": 0.3}
                )
            ],
            edges=[
                EdgeDefinition("authenticate", "query", "seq"),
                EdgeDefinition("query", "parse_response", "seq")
            ],
            resource_limits={
                "max_concurrent_requests": 5,
                "max_network_load": 0.8,
                "max_requests_per_minute": 100
            }
        )
