# BuilderBrain Hugging Face Integration

Complete integration pipeline for exporting BuilderBrain models to Hugging Face Hub and running inference APIs.

## 🚀 Overview

This integration provides:

- **Model Export**: Convert BuilderBrain models to HF-compatible format
- **Inference API**: FastAPI server for grammar-constrained generation
- **Training Integration**: HF Trainer compatibility with dual optimization
- **Production Deployment**: Docker and Kubernetes deployment manifests
- **Hugging Face Hub**: Upload models with comprehensive model cards

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Hugging Face  │    │  BuilderBrain   │    │   Production    │
│      Hub        │◄──►│     Models      │◄──►│   Deployment    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model Cards    │    │  Inference API  │    │  Docker/K8s     │
│  & Downloads    │    │  & Validation   │    │  Deployment     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
huggingface_pipeline/
├── model_export/
│   ├── export.py              # Model serialization utilities
│   ├── upload_to_hub.py       # HF Hub upload utilities
│   └── README.md              # Export documentation
├── inference_api/
│   ├── app.py                 # FastAPI inference server
│   └── requirements.txt       # API dependencies
├── training_integration/
│   ├── hf_trainer.py          # HF Trainer adaptation
│   └── dataset_loader.py      # HF Dataset integration
└── deployment/
    ├── Dockerfile             # Multi-stage container
    ├── docker-compose.yml     # Local deployment
    └── k8s/                   # Kubernetes manifests
```

## 🔧 Model Export

### Export to Hugging Face Format

```python
from huggingface_pipeline.model_export.export import ModelExporter

exporter = ModelExporter()
result = exporter.export_to_huggingface(
    model_path="builderbrain_final.ckpt",
    config_path="configs/small.yaml",
    scale="small"
)

print(f"Exported to: {result['export_path']}")
```

### Upload to Hugging Face Hub

```python
from huggingface_pipeline.model_export.upload_to_hub import HubUploader

uploader = HubUploader(token="your_hf_token")
result = uploader.upload_model(
    export_path="exports/builderbrain_small_1234567890",
    model_name="builderbrain",
    scale="small",
    description="BuilderBrain small scale model"
)

print(f"Uploaded to: {result['repo_url']}")
```

## 🚀 Inference API

### Start the API Server

```bash
# Using Python directly
python -m huggingface_pipeline.inference_api.app

# Using Docker
docker run -p 8000:8000 builderbrain/builderbrain-api:latest

# Using Docker Compose
docker-compose up builderbrain-api
```

### API Endpoints

#### Health & Status
- `GET /health` - Health check
- `GET /model/status` - Current model configuration
- `GET /system/metrics` - System performance metrics

#### Inference
- `POST /inference/generate` - Run grammar-constrained generation
- `GET /grammar/constraints` - Available grammar types
- `POST /grammar/preview` - Grammar validation preview

#### Plans & Validation
- `POST /plans/validate` - Validate plan DAG structure
- `POST /plans/preview` - Execution preview and cost estimation

#### Training & Monitoring
- `GET /training/metrics` - Current training state
- `GET /constraints/metrics` - Constraint satisfaction rates
- `POST /training/start` / `POST /training/stop` - Training control

#### Model Management
- `GET /models/scales` - Available model scales
- `POST /models/scale` - Switch active model
- `POST /models/export` - Export model to HF format

### Example Usage

```python
import requests

# Generate response with grammar constraints
response = requests.post("http://localhost:8000/inference/generate", json={
    "prompt": "Generate a JSON API call for user authentication",
    "model_scale": "small",
    "grammar_strict": True,
    "max_tokens": 150
})

result = response.json()
print(f"Response: {result['response']}")
print(f"Grammar violations: {result['grammar_violations']}")

# Validate a plan
plan = {
    "nodes": [{"id": "grasp", "type": "grasp"}],
    "edges": []
}

validation = requests.post("http://localhost:8000/plans/validate", json=plan)
print(f"Plan valid: {validation.json()['valid']}")
```

## 🎯 Training Integration

### HF Trainer Integration

```python
from huggingface_pipeline.training_integration.hf_trainer import (
    BuilderBrainTrainer,
    create_hf_training_args,
    load_builderbrain_model
)

# Load model
model = load_builderbrain_model("gpt2")

# Create training arguments
args = create_hf_training_args(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4
)

# Create trainer with dual optimization
trainer = BuilderBrainTrainer(
    model=model,
    args=args,
    # ... dataset and other arguments
)

# Train with dual constraints
trainer.train()
```

### WandB Integration

Training automatically logs to Weights & Biases:

```bash
export WANDB_API_KEY="your_wandb_key"
python -m huggingface_pipeline.training_integration.hf_trainer
```

## 🚀 Production Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Access services
echo "Dashboard: http://localhost:8501"
echo "API: http://localhost:8000/docs"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f huggingface_pipeline/deployment/k8s/

# Check status
kubectl get pods -l app=builderbrain-api
kubectl get services

# Access via ingress
echo "Dashboard: http://builderbrain.local"
```

### Environment Variables

```bash
# Required for HF Hub uploads
export HF_TOKEN="your_huggingface_token"

# Optional for experiment tracking
export WANDB_API_KEY="your_wandb_key"

# API configuration
export API_BASE_URL="http://localhost:8000"
```

## 🧪 Testing & Validation

### Run Tests

```bash
# API tests
python -m pytest huggingface_pipeline/tests/

# Integration tests
python -m pytest tests/integration/

# Load tests
python -m pytest tests/load/
```

### Demo Scripts

```bash
# Complete demo
python demo_huggingface.py

# Dashboard demo
python demo_dashboard.py

# Individual component demos
python -c "from huggingface_pipeline.model_export.export import ModelExporter; print('Export demo')"
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The API server exposes Prometheus metrics:

```bash
# View metrics
curl http://localhost:8000/metrics

# Prometheus dashboard
open http://localhost:9090
```

### Grafana Dashboards

Pre-configured dashboards for:

- **API Performance**: Response times, error rates, throughput
- **Model Metrics**: Training progress, constraint satisfaction
- **System Health**: CPU, memory, disk usage
- **Business KPIs**: Grammar compliance, plan success rates

### Logging

Structured logging with configurable levels:

```python
# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

# API server logs
tail -f /var/log/builderbrain/api.log

# Dashboard logs
tail -f /var/log/builderbrain/dashboard.log
```

## 🔧 Configuration

### Model Export Settings

```python
# In export.py
export_config = {
    "model_type": "builderbrain",
    "scale": "small",
    "max_sequence_length": 1024,
    "grammar_support": True,
    "plan_validation": True
}
```

### API Configuration

```python
# In inference_api/app.py
API_CONFIG = {
    "model_scales": ["tiny", "small", "production"],
    "max_batch_size": 32,
    "timeout_seconds": 30,
    "enable_caching": True,
    "cache_ttl": 300
}
```

### Training Configuration

```python
# In hf_trainer.py
TRAINING_CONFIG = {
    "dual_optimizer": {
        "eta_lambda": 1e-2,
        "lambda_max": 50.0,
        "use_pcgrad": True
    },
    "constraints": {
        "grammar": {"target": 0.0, "normalizer": "rank"},
        "graph2graph": {"target": 0.2, "normalizer": "rank"},
        "reuse": {"target": 0.5, "normalizer": "rank"}
    }
}
```

## 🔍 Troubleshooting

### Common Issues

**Model export fails:**
- Check if model files exist and are readable
- Verify configuration files are valid YAML
- Ensure sufficient disk space for export

**API server won't start:**
- Check port 8000 is available
- Verify Python dependencies are installed
- Check BuilderBrain modules are importable

**Grammar validation errors:**
- Verify grammar files are properly configured
- Check if selected grammar type is supported
- Review constraint configuration in model

**Training integration issues:**
- Ensure HF transformers is properly installed
- Check dataset format compatibility
- Verify dual optimizer configuration

**HF Hub upload fails:**
- Verify HF token is valid and has write permissions
- Check repository name doesn't already exist
- Ensure model files are properly formatted

### Performance Tuning

**API Performance:**
```bash
# Increase worker processes
uvicorn --workers 4 ...

# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Adjust batch sizes
API_CONFIG["max_batch_size"] = 64
```

**Training Performance:**
```bash
# Enable mixed precision
args.fp16 = True

# Increase batch size
args.per_device_train_batch_size = 8

# Enable gradient accumulation
args.gradient_accumulation_steps = 2
```

## 📚 API Reference

### Model Export

```python
from huggingface_pipeline.model_export.export import ModelExporter

exporter = ModelExporter(output_dir="./exports")
result = exporter.export_to_huggingface(
    model_path="path/to/model.ckpt",
    config_path="configs/small.yaml",
    scale="small"
)
```

### HF Hub Upload

```python
from huggingface_pipeline.model_export.upload_to_hub import HubUploader

uploader = HubUploader(token="hf_token")
result = uploader.upload_model(
    export_path="./exports/model_dir",
    model_name="my-builderbrain-model",
    scale="small"
)
```

### Inference Client

```python
import requests

client = requests.Session()

# Generate with constraints
response = client.post("http://localhost:8000/inference/generate", json={
    "prompt": "Create a structured API response",
    "model_scale": "small",
    "grammar_strict": True,
    "max_tokens": 200
})

result = response.json()
```

## 🤝 Contributing

### Adding New Features

1. **Model export formats:** Extend `export.py` for new formats
2. **API endpoints:** Add to `inference_api/app.py` with proper validation
3. **Training integrations:** Update `hf_trainer.py` for new frameworks
4. **Deployment targets:** Add new container/deployment configurations

### Code Standards

- Use type hints for all public functions
- Add comprehensive docstrings
- Follow PEP 8 style guidelines
- Include unit tests for new functionality
- Update documentation for API changes

### Testing Requirements

- Unit tests for individual components
- Integration tests for API endpoints
- End-to-end tests for complete workflows
- Performance benchmarks for optimization changes

## 📄 License

This integration is part of BuilderBrain and follows the same Apache 2.0 license.

## 🙏 Acknowledgments

Built on top of:
- [Hugging Face](https://huggingface.co/) for model sharing and transformers
- [FastAPI](https://fastapi.tiangolo.com/) for high-performance APIs
- [Docker](https://docker.com/) for containerization
- [Kubernetes](https://kubernetes.io/) for orchestration
- [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) for monitoring
- [BuilderBrain](https://github.com/JacobFV/builderbrain) for the core AI system

---

**BuilderBrain Hugging Face Integration** - Seamless model deployment and inference for compositional AI systems.
