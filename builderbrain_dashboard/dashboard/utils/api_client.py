"""
API client for communicating with BuilderBrain backend services.

Handles inference requests, model interactions, and real-time updates.
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime


class APIClient:
    """Client for BuilderBrain API services."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = 30  # seconds

    def health_check(self) -> bool:
        """Check if the API service is healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and configuration."""
        try:
            response = self.session.get(
                f"{self.base_url}/model/status",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def run_inference(
        self,
        prompt: str,
        model_scale: str = "small",
        grammar_strict: bool = True,
        max_tokens: int = 100
    ) -> Dict[str, Any]:
        """Run inference with the specified model."""
        try:
            payload = {
                "prompt": prompt,
                "model_scale": model_scale,
                "grammar_strict": grammar_strict,
                "max_tokens": max_tokens
            }

            response = self.session.post(
                f"{self.base_url}/inference/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_grammar_constraints(self) -> Dict[str, Any]:
        """Get available grammar constraints."""
        try:
            response = self.session.get(
                f"{self.base_url}/grammar/constraints",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def validate_plan(self, plan_dag: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a plan DAG against current schema."""
        try:
            response = self.session.post(
                f"{self.base_url}/plans/validate",
                json=plan_dag,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics from active trainer."""
        try:
            response = self.session.get(
                f"{self.base_url}/training/metrics",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_constraint_metrics(self) -> Dict[str, Any]:
        """Get constraint satisfaction metrics."""
        try:
            response = self.session.get(
                f"{self.base_url}/constraints/metrics",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            response = self.session.get(
                f"{self.base_url}/system/metrics",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_model_scales(self) -> List[str]:
        """Get available model scales."""
        try:
            response = self.session.get(
                f"{self.base_url}/models/scales",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("scales", [])
        except Exception as e:
            return ["tiny", "small", "production"]

    def set_model_scale(self, scale: str) -> Dict[str, Any]:
        """Set the active model scale."""
        try:
            response = self.session.post(
                f"{self.base_url}/models/scale",
                json={"scale": scale},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_grammar_preview(self, sample_text: str, grammar_type: str = "json") -> Dict[str, Any]:
        """Preview how text would be constrained by grammar."""
        try:
            response = self.session.post(
                f"{self.base_url}/grammar/preview",
                json={"text": sample_text, "grammar_type": grammar_type},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_plan_execution_preview(self, plan_dag: Dict[str, Any]) -> Dict[str, Any]:
        """Preview plan execution without actually running it."""
        try:
            response = self.session.post(
                f"{self.base_url}/plans/preview",
                json=plan_dag,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def export_model(self, scale: str, format: str = "hf") -> Dict[str, Any]:
        """Export model in specified format."""
        try:
            response = self.session.post(
                f"{self.base_url}/models/export",
                json={"scale": scale, "format": format},
                timeout=self.timeout * 2  # Longer timeout for export
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """Check status of model export."""
        try:
            response = self.session.get(
                f"{self.base_url}/exports/{export_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


class MockAPIClient(APIClient):
    """Mock API client for development when backend is not available."""

    def __init__(self):
        super().__init__()
        self.mock_responses = {
            "health_check": True,
            "model_status": {
                "model_scale": "small",
                "status": "ready",
                "grammar_enabled": True,
                "plan_validation_enabled": True,
                "last_training": "2024-01-15T10:30:00Z"
            },
            "grammar_constraints": {
                "available_grammars": ["json", "api", "robot_dsl", "phone_flow"],
                "strict_modes": ["json", "api", "robot_dsl"],
                "flexible_modes": ["phone_flow"]
            },
            "model_scales": ["tiny", "small", "production"],
            "training_metrics": {
                "current_step": 1500,
                "total_loss": 2.34,
                "task_loss": 2.12,
                "constraint_losses": {
                    "grammar": 0.15,
                    "graph2graph": 0.08,
                    "reuse": 0.03
                },
                "dual_variables": {
                    "grammar": 1.2,
                    "graph2graph": 0.8,
                    "reuse": 0.5
                }
            },
            "constraint_metrics": {
                "grammar_compliance_rate": 0.95,
                "plan_execution_success_rate": 0.88,
                "constraint_violation_rate": 0.02,
                "safety_energy": 0.05
            }
        }

    def health_check(self) -> bool:
        return self.mock_responses["health_check"]

    def get_model_status(self) -> Dict[str, Any]:
        return self.mock_responses["model_status"]

    def run_inference(self, prompt: str, **kwargs) -> Dict[str, Any]:
        time.sleep(0.1)  # Simulate processing time
        return {
            "prompt": prompt,
            "response": f"Mock response to: {prompt[:50]}...",
            "model_scale": kwargs.get("model_scale", "small"),
            "grammar_strict": kwargs.get("grammar_strict", True),
            "tokens_generated": len(prompt.split()) + 20,
            "processing_time": 0.1,
            "grammar_violations": 0 if kwargs.get("grammar_strict") else 2
        }

    def get_grammar_constraints(self) -> Dict[str, Any]:
        return self.mock_responses["grammar_constraints"]

    def validate_plan(self, plan_dag: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(0.05)  # Simulate validation time
        return {
            "valid": True,
            "validation_time": 0.05,
            "errors": [],
            "warnings": ["Consider adding more preconditions for safety"]
        }

    def get_training_metrics(self) -> Dict[str, Any]:
        return self.mock_responses["training_metrics"]

    def get_constraint_metrics(self) -> Dict[str, Any]:
        return self.mock_responses["constraint_metrics"]

    def get_model_scales(self) -> List[str]:
        return self.mock_responses["model_scales"]

    def get_grammar_preview(self, sample_text: str, **kwargs) -> Dict[str, Any]:
        return {
            "original_text": sample_text,
            "constrained_text": sample_text,  # Mock constraint
            "violations": [],
            "suggestions": ["Consider using proper JSON formatting"]
        }

    def get_plan_execution_preview(self, plan_dag: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "estimated_execution_time": 2.5,
            "resource_requirements": {"cpu": 0.3, "memory": 0.2},
            "risk_assessment": "low",
            "optimization_suggestions": ["Consider parallelizing independent steps"]
        }

    def export_model(self, scale: str, **kwargs) -> Dict[str, Any]:
        return {
            "export_id": f"export_{scale}_{int(time.time())}",
            "status": "completed",
            "download_url": f"/mock/download/{scale}",
            "file_size": "1.2GB"
        }
