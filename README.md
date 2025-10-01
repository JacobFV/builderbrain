# BuilderBrain: Dual-Rail Compositional AI System

> **Build, don't just search.**
> A dual-rail extension to pretrained transformers that learns reusable skills, executable plans, and safe compositional reasoning across domains.

## 🚀 Quick Start

> We are now live on GitHub! https://github.com/JacobFV/builderbrain.git

```bash
# Run the demo
uv run python main.py --mode demo

# Train the model
uv run python main.py --mode train

# Serve for inference
uv run python main.py --mode serve
```

## 📋 Overview

BuilderBrain extends large language models with a secondary "builder rail" that learns to compose discrete skills into executable plans. The system maintains the leverage of pretrained models while adding:

- **Compositional reasoning** via discrete program skills
- **Explicit planning** with DAG-based execution graphs
- **Grammar constraints** for structured outputs
- **Safety invariants** with automatic rollback
- **Multi-objective optimization** with dual constraints

## 🏗️ Architecture

### Dual-Rail Design
```
Base Rail (Frozen) ──────────────────▶ h^B_{ℓ+1}
                     │
Builder Rail ────────┼─▶ h^C_{ℓ+1}
                     │
Program Adapters ────┼─▶ z_t (discrete skills)
                     │
Fusion Gates ────────┼─▶ α_ℓ (gating values)
                     │
Latent Plans ────────┼─▶ 𝓖_t (DAG structures)
```

### Key Components

- **bb_core/**: Mathematical foundations and protocols
- **bb_nn/**: Dual-rail neural network architecture
- **bb_priors/**: Grammar parsers and token masking
- **bb_runtime/**: Plan validation and execution
- **bb_losses/**: Multi-objective constraint losses
- **bb_train/**: Training pipeline and orchestration
- **bb_safety/**: Safety invariants and promotion gates

## 🎯 Core Features

### 1. Grammar-Constrained Generation
```python
# JSON grammar enforcement
grammar = JSONGrammar()
mask = GrammarMask(grammar, tokenizer, strict=True)
constrained_logits = mask(logits, prefix)
```

### 2. Plan Validation & Execution
```python
# Validate plan DAG
checker = PlanChecker("robot_schema.yaml")
result = checker.validate_plan(plan_dag)

# Execute if valid
if result.valid:
    executor = PlanExecutor(tool_adapters)
    await executor.execute_plan(plan_dag, context)
```

### 3. Dual Constraint Optimization
```python
# Multi-objective training
dual_optimizer = DualOptimizer(constraint_configs)
total_loss, normalized_losses = dual_optimizer.compute_lagrangian(
    task_loss, constraint_losses
)
```

## 📊 Status

✅ **Fully Implemented & Documented:**
- **Core Architecture:** Dual-rail neural system with composition blocks
- **Grammar System:** CFG/PEG parsing with real tokenizer integration
- **Plan System:** DAG validation and execution with precondition checking
- **Training Pipeline:** Multi-objective optimization with real data
- **Runtime Operations:** Grammar-constrained generation and execution
- **Safety Framework:** Foundation for Lyapunov invariants and rollback
- **Documentation:** Complete operational and technical documentation
- **Scale Testing:** Works from MacBook Pro (GPT-2) to production (2.7B+)

🚧 **Future Enhancements:**
- Advanced model heads (graphs, calibration, safety prediction)
- World model for planning and EVSI computation
- Domain-specific plugins (API, robots, phone, social)
- Safety invariants with automatic promotion/rollback
- Comprehensive testing suite and benchmarks

## 🔧 Installation

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and run demo
uv run python main.py --mode demo

# Or create virtual environment with uv (optional)
uv venv
source .venv/bin/activate
uv sync  # Install all dependencies from pyproject.toml
```

## 📚 Documentation

### **Core Documentation** ✅
- **[DESIGN.md](docs/DESIGN.md)**: System rationale and tradeoffs
- **[MATH_SPEC.md](docs/MATH_SPEC.md)**: Mathematical derivations
- **[GRAMMAR_GUIDE.md](docs/GRAMMAR_GUIDE.md)**: Grammar authoring guide
- **[PLAN_SCHEMA_GUIDE.md](docs/PLAN_SCHEMA_GUIDE.md)**: Plan schema design
- **[LOSS_BANK.md](docs/LOSS_BANK.md)**: Loss function specifications
- **[DUAL_OPTIMIZER.md](docs/DUAL_OPTIMIZER.md)**: Constraint optimization
- **[PROGRAM_SKILLS.md](docs/PROGRAM_SKILLS.md)**: Skill lifecycle management
- **[WORLD_MODEL.md](docs/WORLD_MODEL.md)**: Planning world model
- **[SAFETY_SPEC.md](docs/SAFETY_SPEC.md)**: Safety invariants

### **Operational Documentation** ✅
- **[RUNTIME_PLAYBOOK.md](docs/RUNTIME_PLAYBOOK.md)**: Runtime operations and deployment
- **[DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md)**: Privacy and compliance
- **[BENCHMARKS.md](docs/BENCHMARKS.md)**: Evaluation and testing standards
- **[OBSERVABILITY.md](docs/OBSERVABILITY.md)**: Monitoring and alerting
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)**: Development guidelines
- **[RELEASE.md](docs/RELEASE.md)**: Release management and deployment

## 🎯 Mission

BuilderBrain aims to transform large language models from passive pattern-matchers into active compositional reasoners. By maintaining pretrained capabilities while adding structured planning and safety constraints, we enable deployment in high-stakes domains like robotics, finance, and social platforms.

**Key Insight:** Composition over memorization. Skills over scripts. Plans over prompts.

---

*This system is designed for researchers and engineers building safe, compositional AI systems. Contributions welcome - see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.*
