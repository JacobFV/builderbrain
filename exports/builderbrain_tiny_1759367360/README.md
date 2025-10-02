---
language: en
license: apache-2.0
tags:
- builderbrain
- compositional-ai
- grammar-constrained
- dual-rail
- pytorch
- transformers
model-index:
- name: builderbrain-tiny
  results: []
---

# BuilderBrain Tiny Model

BuilderBrain is a dual-rail compositional AI system that extends pretrained transformers with learned composition blocks, grammar constraints, and executable plans.

## Model Description

This is a tiny scale BuilderBrain model designed for compositional reasoning tasks with formal guarantees.

### Architecture

- **Base Rail**: Frozen pretrained transformer (gpt2)
- **Builder Rail**: Additional composition layer with 8 discrete program skills
- **Grammar Constraints**: CFG/PEG parsing for structured outputs
- **Plan Validation**: DAG-based plan execution with precondition checking
- **Multi-objective Training**: Lagrangian optimization with constraint satisfaction
- **Safety Monitoring**: Risk energy prediction and violation detection

### Model Specifications

- **Hidden Size**: 768
- **Builder Layers**: 4
- **Program Skills**: 8
- **Alpha Cap**: 0.05
- **Grammar Constraints**: 2 active constraints

### Training

- **Dataset**: Compositional reasoning tasks with structured outputs
- **Loss Functions**: Multi-objective with grammar, plan, and reuse constraints
- **Training Steps**: 5 epochs
- **Batch Size**: 2
- **Learning Rate**: 1e-4

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("builderbrain_tiny_1759367360")
model = AutoModelForCausalLM.from_pretrained("builderbrain_tiny_1759367360")

# Grammar-constrained generation
input_text = "Generate a JSON API call for user registration"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate with grammar constraints and safety monitoring
outputs = model.generate(
    **inputs,
    max_length=150,
    grammar_constraint=True,
    safety_monitoring=True,
    temperature=0.8
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Capabilities

- **Compositional Reasoning**: Combines discrete skills into complex behaviors
- **Grammar Compliance**: Generates syntactically correct structured outputs
- **Safety Awareness**: Monitors and prevents harmful outputs
- **Planning**: Uses world models for multi-step reasoning
- **Constraint Satisfaction**: Maintains formal guarantees during generation

## Limitations

- Requires domain-specific training data for optimal performance
- Grammar constraints may limit creative outputs in unconstrained domains
- Safety monitoring adds computational overhead

## Citation

```bibtex
@misc{builderbrain_tiny,
  title={BuilderBrain: Dual-Rail Compositional AI System},
  author={BuilderBrain Team},
  year={2024},
  url={https://github.com/JacobFV/builderbrain}
}
```
