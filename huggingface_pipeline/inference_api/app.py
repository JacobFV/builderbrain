"""
FastAPI inference server for BuilderBrain.

Provides REST API endpoints for model inference, grammar validation,
and real-time monitoring data.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import sys
import os

# Add parent directory to path for BuilderBrain imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Pydantic models for request/response
class InferenceRequest(BaseModel):
    prompt: str
    model_scale: str = "small"
    grammar_strict: bool = True
    max_tokens: int = 100

class GrammarPreviewRequest(BaseModel):
    text: str
    grammar_type: str = "json"

class PlanValidationRequest(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class ModelExportRequest(BaseModel):
    scale: str
    format: str = "hf"

class ModelScaleRequest(BaseModel):
    scale: str

# Global state for mock responses (in production, this would connect to actual BuilderBrain)
class MockBuilderBrainState:
    def __init__(self):
        self.current_scale = "small"
        self.grammar_enabled = True
        self.plan_validation_enabled = True
        self.training_active = False
        self.current_step = 1500
        self.total_loss = 2.34

        # Mock training history
        self.training_history = {
            'total_loss': [5.0, 4.5, 3.8, 3.2, 2.8, 2.5, 2.3, 2.34, 2.32, 2.31],
            'task_loss': [4.8, 4.2, 3.5, 2.9, 2.5, 2.2, 2.0, 2.1, 2.05, 2.03],
            'constraint_losses': {
                'grammar': [0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.035],
                'graph2graph': [0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02],
                'reuse': [0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.006]
            },
            'dual_variables': {
                'grammar': [1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.75, 0.7, 0.65],
                'graph2graph': [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35],
                'reuse': [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
            }
        }

# Initialize state
brain_state = MockBuilderBrainState()

# Create FastAPI app
app = FastAPI(
    title="BuilderBrain Inference API",
    description="REST API for BuilderBrain model inference and monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model/status")
async def get_model_status():
    """Get current model status."""
    return {
        "model_scale": brain_state.current_scale,
        "status": "ready",
        "grammar_enabled": brain_state.grammar_enabled,
        "plan_validation_enabled": brain_state.plan_validation_enabled,
        "last_training": "2024-01-15T10:30:00Z"
    }

@app.post("/inference/generate")
async def run_inference(request: InferenceRequest):
    """Run inference with the specified model."""
    # Simulate processing time
    await asyncio.sleep(0.1)

    # Mock response generation
    prompt_words = len(request.prompt.split())
    response_text = f"Mock response to: {request.prompt[:50]}..."

    if request.grammar_strict:
        response_text = '{"response": "Properly formatted JSON response"}'

    return {
        "prompt": request.prompt,
        "response": response_text,
        "model_scale": request.model_scale,
        "grammar_strict": request.grammar_strict,
        "tokens_generated": prompt_words + 20,
        "processing_time": 0.1,
        "grammar_violations": 0 if request.grammar_strict else 2,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/grammar/constraints")
async def get_grammar_constraints():
    """Get available grammar constraints."""
    return {
        "available_grammars": ["json", "api", "robot_dsl", "phone_flow"],
        "strict_modes": ["json", "api", "robot_dsl"],
        "flexible_modes": ["phone_flow"]
    }

@app.post("/grammar/preview")
async def get_grammar_preview(request: GrammarPreviewRequest):
    """Preview how text would be constrained by grammar."""
    await asyncio.sleep(0.02)  # Simulate processing

    return {
        "original_text": request.text,
        "constrained_text": request.text,  # Mock constraint
        "violations": [],
        "suggestions": ["Consider using proper JSON formatting"]
    }

@app.post("/plans/validate")
async def validate_plan(request: PlanValidationRequest):
    """Validate a plan DAG against current schema."""
    await asyncio.sleep(0.05)  # Simulate validation time

    return {
        "valid": True,
        "validation_time": 0.05,
        "errors": [],
        "warnings": ["Consider adding more preconditions for safety"]
    }

@app.post("/plans/preview")
async def get_plan_execution_preview(request: PlanValidationRequest):
    """Preview plan execution without actually running it."""
    await asyncio.sleep(0.03)

    return {
        "estimated_execution_time": 2.5,
        "resource_requirements": {"cpu": 0.3, "memory": 0.2},
        "risk_assessment": "low",
        "optimization_suggestions": ["Consider parallelizing independent steps"]
    }

@app.get("/training/metrics")
async def get_training_metrics():
    """Get current training metrics from active trainer."""
    return {
        "current_step": brain_state.current_step,
        "total_loss": brain_state.total_loss,
        "task_loss": brain_state.training_history['task_loss'][-1],
        "constraint_losses": {
            k: v[-1] for k, v in brain_state.training_history['constraint_losses'].items()
        },
        "dual_variables": {
            k: v[-1] for k, v in brain_state.training_history['dual_variables'].items()
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/constraints/metrics")
async def get_constraint_metrics():
    """Get constraint satisfaction metrics."""
    return {
        "grammar_compliance_rate": 0.95,
        "plan_execution_success_rate": 0.88,
        "constraint_violation_rate": 0.02,
        "safety_energy": 0.05,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system/metrics")
async def get_system_metrics():
    """Get system performance metrics."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
        "disk_percent": disk.percent,
        "disk_used_gb": disk.used / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
        "active_processes": len(psutil.pids()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/scales")
async def get_model_scales():
    """Get available model scales."""
    return {"scales": ["tiny", "small", "production"]}

@app.post("/models/scale")
async def set_model_scale(request: ModelScaleRequest):
    """Set the active model scale."""
    brain_state.current_scale = request.scale
    return {"status": "success", "scale": request.scale}

@app.post("/models/export")
async def export_model(request: ModelExportRequest):
    """Export model in specified format."""
    await asyncio.sleep(2.0)  # Simulate export time

    return {
        "export_id": f"export_{request.scale}_{int(time.time())}",
        "status": "completed",
        "download_url": f"/mock/download/{request.scale}",
        "file_size": "1.2GB"
    }

@app.get("/exports/{export_id}")
async def get_export_status(export_id: str):
    """Check status of model export."""
    return {
        "export_id": export_id,
        "status": "completed",
        "progress": 100,
        "download_url": f"/mock/download/{export_id}"
    }

# Background task for simulating training
async def simulate_training():
    """Background task to simulate ongoing training."""
    while True:
        if brain_state.training_active:
            # Update training metrics
            brain_state.current_step += 1

            # Simulate loss improvement
            improvement_factor = 0.999
            brain_state.total_loss *= improvement_factor

            # Update history
            if len(brain_state.training_history['total_loss']) >= 10:
                brain_state.training_history['total_loss'].pop(0)
                brain_state.training_history['task_loss'].pop(0)
                for k in brain_state.training_history['constraint_losses']:
                    brain_state.training_history['constraint_losses'][k].pop(0)
                for k in brain_state.training_history['dual_variables']:
                    brain_state.training_history['dual_variables'][k].pop(0)

            brain_state.training_history['total_loss'].append(brain_state.total_loss)
            brain_state.training_history['task_loss'].append(brain_state.total_loss * 0.85)

            for k in brain_state.training_history['constraint_losses']:
                brain_state.training_history['constraint_losses'][k].append(
                    brain_state.training_history['constraint_losses'][k][-1] * improvement_factor
                )
            for k in brain_state.training_history['dual_variables']:
                brain_state.training_history['dual_variables'][k].append(
                    brain_state.training_history['dual_variables'][k][-1] * 0.995
                )

        await asyncio.sleep(1.0)  # Update every second

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup."""
    asyncio.create_task(simulate_training())

@app.post("/training/start")
async def start_training():
    """Start training simulation."""
    brain_state.training_active = True
    return {"status": "training_started"}

@app.post("/training/stop")
async def stop_training():
    """Stop training simulation."""
    brain_state.training_active = False
    return {"status": "training_stopped"}

@app.get("/training/status")
async def get_training_status():
    """Get current training status."""
    return {
        "active": brain_state.training_active,
        "current_step": brain_state.current_step,
        "total_loss": brain_state.total_loss,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
