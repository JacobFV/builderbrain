# BuilderBrain Dashboard

Real-time monitoring and visualization dashboard for BuilderBrain training and inference.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
uv add streamlit plotly fastapi uvicorn pandas psutil

# Run dashboard
python demo_dashboard.py
```

### Access URLs

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 📋 Features

### 📊 Real-time Monitoring

- **System Health**: CPU, memory, disk usage, active processes
- **Training Progress**: Loss curves, constraint evolution, dual variables
- **Model Performance**: Grammar compliance, plan execution success
- **Constraint Satisfaction**: Real-time violation monitoring

### 🧠 Interactive Testing

- **Grammar-Constrained Generation**: Test structured outputs
- **Plan Validation**: Interactive DAG validation and execution preview
- **Model Scale Selection**: Switch between tiny/small/production models
- **Export Functionality**: Download results and metrics

### ⚙️ Configuration Management

- **Model Parameters**: View and adjust model configurations
- **Constraint Settings**: Monitor and tune constraint targets
- **Training Controls**: Start/stop/pause training operations
- **System Diagnostics**: Health checks and performance monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     FastAPI     │    │  BuilderBrain   │
│   Dashboard     │◄──►│   Inference     │◄──►│    Models       │
│                 │    │     API         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Real-time      │    │  Grammar &      │    │  Training       │
│  Metrics        │    │  Plan Validation│    │  History        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
builderbrain_dashboard/
├── dashboard/
│   ├── app.py                 # Main Streamlit application
│   ├── components/            # Reusable UI components
│   ├── pages/                 # Dashboard pages
│   │   ├── overview.py        # Main overview page
│   │   ├── training.py        # Training monitoring
│   │   ├── inference.py       # Interactive testing
│   │   └── config.py          # Configuration management
│   └── utils/
│       ├── data_loader.py     # Data loading utilities
│       └── api_client.py      # API communication
```

## 🔧 API Endpoints

### Health & Status

- `GET /health` - Health check
- `GET /model/status` - Current model status
- `GET /system/metrics` - System performance metrics

### Inference

- `POST /inference/generate` - Run model inference
- `GET /grammar/constraints` - Available grammar types
- `POST /grammar/preview` - Grammar validation preview

### Plans & Validation

- `POST /plans/validate` - Validate plan DAG
- `POST /plans/preview` - Execution preview

### Training & Monitoring

- `GET /training/metrics` - Current training metrics
- `GET /constraints/metrics` - Constraint satisfaction
- `POST /training/start` - Start training simulation
- `POST /training/stop` - Stop training simulation

### Model Management

- `GET /models/scales` - Available model scales
- `POST /models/scale` - Set active model scale
- `POST /models/export` - Export model to HF format

## 🎨 Dashboard Pages

### Overview Page

**High-level KPIs and system status:**
- Training progress with loss curves
- System resource utilization
- Constraint compliance rates
- Recent activity feed
- Quick action buttons

### Training Page

**Detailed training monitoring:**
- Real-time loss evolution charts
- Constraint analysis and correlations
- Dual variable evolution tracking
- Training statistics and health metrics
- Export functionality for training data

### Inference Page

**Interactive model testing:**
- Grammar-constrained generation
- Plan validation and execution preview
- Model scale selection
- Grammar compliance testing
- Export results and validation reports

### Configuration Page

**System and model management:**
- Model parameter visualization
- Constraint target adjustment
- Training control interface
- System diagnostics and logs
- Model export configuration

## 🔄 Real-time Updates

The dashboard supports multiple update mechanisms:

### Auto-Refresh
- Configurable refresh intervals (1-60 seconds)
- Automatic page updates with new data
- Background polling for live metrics

### WebSocket Support
- Real-time streaming updates (planned)
- Live training progress updates
- Instant constraint violation alerts

### Manual Refresh
- Force refresh buttons on each page
- Export and download operations
- Configuration changes

## 📊 Metrics & Visualization

### Supported Chart Types

- **Time Series**: Loss curves, dual variables, system metrics
- **Bar Charts**: Constraint violations, resource usage
- **Heatmaps**: Constraint correlations, performance matrices
- **Progress Bars**: Training progress, system health
- **Scatter Plots**: Loss vs constraint relationships

### Key Metrics Tracked

- **Training**: Total loss, task loss, constraint losses, dual variables
- **System**: CPU, memory, disk usage, active processes
- **Quality**: Grammar compliance, plan execution success
- **Performance**: Response times, throughput, error rates

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_BASE_URL=http://localhost:8000

# Dashboard Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Hugging Face Integration
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

### Configuration Files

Dashboard reads configuration from:
- `configs/small.yaml` - Model and training configuration
- `training_history.json` - Training progress data
- Environment variables for runtime settings

## 🚀 Deployment

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f builderbrain-api

# Stop services
docker-compose down
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f huggingface_pipeline/deployment/k8s/

# Check status
kubectl get pods -l app=builderbrain-api

# Scale deployment
kubectl scale deployment builderbrain-api --replicas=5
```

### Manual Deployment

```bash
# Start API server
python -m huggingface_pipeline.inference_api.app

# Start dashboard (new terminal)
python -m streamlit run builderbrain_dashboard/dashboard/app.py
```

## 🧪 Testing

### Run Tests

```bash
# Dashboard tests
python -m pytest builderbrain_dashboard/tests/

# API tests
python -m pytest huggingface_pipeline/tests/

# Integration tests
python -m pytest tests/integration/
```

### Demo Scripts

```bash
# Dashboard demo
python demo_dashboard.py

# Hugging Face integration demo
python demo_huggingface.py
```

## 🔍 Troubleshooting

### Common Issues

**Dashboard not loading:**
- Check if API server is running on port 8000
- Verify Python dependencies are installed
- Check browser console for JavaScript errors

**API connection errors:**
- Ensure API server is running and accessible
- Check firewall and network configuration
- Verify CORS settings if accessing from different domain

**Training data not appearing:**
- Check if `training_history.json` exists
- Verify file permissions and paths
- Run training to generate data

**Grammar validation failing:**
- Check if grammar files are properly configured
- Verify model supports grammar constraints
- Check API server logs for detailed errors

### Performance Tuning

**Dashboard responsiveness:**
- Adjust auto-refresh interval (lower = more responsive)
- Reduce chart complexity for slower machines
- Enable caching for expensive operations

**API performance:**
- Adjust model batch sizes in configuration
- Enable GPU acceleration if available
- Monitor resource usage and scale accordingly

## 📚 API Reference

### Inference Request

```python
import requests

response = requests.post("http://localhost:8000/inference/generate", json={
    "prompt": "Generate a JSON API call",
    "model_scale": "small",
    "grammar_strict": True,
    "max_tokens": 100
})

result = response.json()
print(result["response"])
```

### Grammar Validation

```python
response = requests.post("http://localhost:8000/grammar/preview", json={
    "text": '{"name": "test"}',
    "grammar_type": "json"
})

preview = response.json()
print(f"Violations: {len(preview['violations'])}")
```

### Plan Validation

```python
response = requests.post("http://localhost:8000/plans/validate", json={
    "nodes": [{"id": "grasp", "type": "grasp"}],
    "edges": []
})

validation = response.json()
print(f"Valid: {validation['valid']}")
```

## 🤝 Contributing

### Development Setup

1. **Install dependencies:**
   ```bash
   uv add -e .
   ```

2. **Run tests:**
   ```bash
   python -m pytest
   ```

3. **Start development servers:**
   ```bash
   # Terminal 1: API server
   python -m huggingface_pipeline.inference_api.app

   # Terminal 2: Dashboard
   python -m streamlit run builderbrain_dashboard/dashboard/app.py
   ```

### Code Style

- Follow PEP 8 conventions
- Use type hints for all functions
- Add docstrings for public methods
- Keep functions focused and testable

### Adding New Features

1. **Dashboard features:** Add to appropriate page in `pages/`
2. **API endpoints:** Add to `inference_api/app.py`
3. **Model integration:** Update `training_integration/hf_trainer.py`
4. **Documentation:** Update this README and add examples

## 📄 License

This project is part of BuilderBrain and follows the same Apache 2.0 license.

## 🙏 Acknowledgments

Built on top of:
- [Streamlit](https://streamlit.io/) for the dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API backend
- [Plotly](https://plotly.com/) for interactive visualizations
- [Hugging Face](https://huggingface.co/) for model sharing
- [BuilderBrain](https://github.com/JacobFV/builderbrain) for the core AI system

---

**BuilderBrain Dashboard** - Real-time monitoring and control for compositional AI systems.
