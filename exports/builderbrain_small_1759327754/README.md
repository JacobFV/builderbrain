---
language: en
license: apache-2.0
tags:
- builderbrain
- compositional-ai
- grammar-constrained
- pytorch
- transformers
model-index:
- name: builderbrain-small
  results: []
---

# BuilderBrain Small Model

BuilderBrain is a dual-rail compositional AI system that extends pretrained transformers with learned composition blocks, grammar constraints, and executable plans.

## Model Description

This is a small scale BuilderBrain model trained for compositional reasoning tasks.

### Architecture

- **Base Model**: GPT-2 based transformer
- **Builder Rail**: Additional composition layer with discrete program skills
- **Grammar Constraints**: CFG/PEG parsing for structured outputs
- **Plan Validation**: DAG-based plan execution with precondition checking
- **Multi-objective Training**: Lagrangian optimization with constraint satisfaction

### Training

- **Dataset**: Compositional reasoning tasks
- **Loss Functions**: Multi-objective with grammar, plan, and reuse constraints
- **Training Steps**: 50 epochs

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("builderbrain_small_1759327754")
model = AutoModelForCausalLM.from_pretrained("builderbrain_small_1759327754")

# Grammar-constrained generation
input_text = "Generate a JSON API call"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate with grammar constraints (implementation specific)
outputs = model.generate(**inputs, max_length=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Limitations

- This is a mock export for demonstration purposes
- In production, models would be trained on domain-specific datasets
- Grammar constraints and plan validation would be fully implemented

## Citation

```bibtex
@misc{builderbrain_small,
  title={BuilderBrain: Dual-Rail Compositional AI System},
  author={BuilderBrain Team},
  year={2024},
  url={https://github.com/JacobFV/builderbrain}
}
```
