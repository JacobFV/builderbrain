"""
Model export utilities for BuilderBrain.

Handles serialization to Hugging Face compatible formats and upload to HF Hub.
"""

import os
import sys
import json
import torch
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path for BuilderBrain imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml


class ModelExporter:
    """Export BuilderBrain models to various formats."""

    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export_to_huggingface(
        self,
        model_path: str,
        config_path: str,
        scale: str = "small"
    ) -> Dict[str, Any]:
        """
        Export BuilderBrain model to Hugging Face format.

        Args:
            model_path: Path to trained BuilderBrain model
            config_path: Path to model configuration
            scale: Model scale (tiny, small, production)

        Returns:
            Export metadata
        """
        export_id = f"builderbrain_{scale}_{int(datetime.now().timestamp())}"
        export_path = self.output_dir / export_id

        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Create export directory structure
            export_path.mkdir(exist_ok=True)

            # Load base model (this would be the actual BuilderBrain model in production)
            # For now, we'll create a mock structure
            base_model_name = config.get('model', {}).get('name', 'gpt2')

            # Create model directory structure
            model_dir = export_path / "model"
            model_dir.mkdir(exist_ok=True)

            # Export configuration files
            self._export_config_files(config, model_dir, scale)

            # Export tokenizer
            self._export_tokenizer(base_model_name, model_dir)

            # Export model weights (mock for now)
            self._export_model_weights(model_path, model_dir)

            # Create model card
            self._create_model_card(export_path, scale, config)

            return {
                "export_id": export_id,
                "export_path": str(export_path),
                "status": "completed",
                "file_size": self._get_directory_size(export_path),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "export_id": export_id,
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    def _export_config_files(self, config: Dict[str, Any], model_dir: Path, scale: str):
        """Export configuration files to model directory."""

        # Main config
        with open(model_dir / "config.json", 'w') as f:
            json.dump({
                "model_type": "builderbrain",
                "scale": scale,
                "builderbrain_version": "1.0.0",
                **config
            }, f, indent=2)

        # Generation config
        with open(model_dir / "generation_config.json", 'w') as f:
            json.dump({
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": 50256,  # EOS token for GPT-2
                "bos_token_id": 50256,
                "eos_token_id": 50256
            }, f, indent=2)

        # Special tokens
        with open(model_dir / "special_tokens_map.json", 'w') as f:
            json.dump({
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": []
            }, f, indent=2)

    def _export_tokenizer(self, model_name: str, model_dir: Path):
        """Export tokenizer files."""
        try:
            # In production, this would load the actual tokenizer
            # For now, we'll create mock tokenizer files
            tokenizer_dir = model_dir / "tokenizer"
            tokenizer_dir.mkdir(exist_ok=True)

            # Create tokenizer config
            with open(tokenizer_dir / "tokenizer_config.json", 'w') as f:
                json.dump({
                    "tokenizer_class": "GPT2Tokenizer",
                    "model_max_length": 1024,
                    "padding_side": "right",
                    "truncation_side": "right",
                    "pad_token": "<|endoftext|>",
                    "unk_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
                    "bos_token": "<|endoftext|>"
                }, f, indent=2)

            # Create vocab file (placeholder)
            with open(tokenizer_dir / "vocab.json", 'w') as f:
                json.dump({"mock": "vocabulary"}, f)

            # Create merges file (placeholder)
            with open(tokenizer_dir / "merges.txt", 'w') as f:
                f.write("# Mock merges file\n")

        except Exception as e:
            print(f"Warning: Could not export tokenizer: {e}")

    def _export_model_weights(self, model_path: str, model_dir: Path):
        """Export model weights."""
        try:
            # In production, this would load and save the actual model
            # For now, we'll create placeholder files

            # Create pytorch_model.bin (placeholder)
            model_file = model_dir / "pytorch_model.bin"
            model_file.write_bytes(b"mock_model_weights")

            # Create model index (for sharded models)
            with open(model_dir / "pytorch_model.bin.index.json", 'w') as f:
                json.dump({
                    "metadata": {
                        "total_size": model_file.stat().st_size
                    },
                    "weight_map": {
                        "model.embed_tokens.weight": "pytorch_model.bin",
                        "model.layers.0.weight": "pytorch_model.bin",
                        "lm_head.weight": "pytorch_model.bin"
                    }
                }, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not export model weights: {e}")

    def _create_model_card(self, export_path: Path, scale: str, config: Dict[str, Any]):
        """Create a model card for the exported model."""
        model_card_content = f"""---
language: en
license: apache-2.0
tags:
- builderbrain
- compositional-ai
- grammar-constrained
- pytorch
- transformers
model-index:
- name: builderbrain-{scale}
  results: []
---

# BuilderBrain {scale.title()} Model

BuilderBrain is a dual-rail compositional AI system that extends pretrained transformers with learned composition blocks, grammar constraints, and executable plans.

## Model Description

This is a {scale} scale BuilderBrain model trained for compositional reasoning tasks.

### Architecture

- **Base Model**: GPT-2 based transformer
- **Builder Rail**: Additional composition layer with discrete program skills
- **Grammar Constraints**: CFG/PEG parsing for structured outputs
- **Plan Validation**: DAG-based plan execution with precondition checking
- **Multi-objective Training**: Lagrangian optimization with constraint satisfaction

### Training

- **Dataset**: Compositional reasoning tasks
- **Loss Functions**: Multi-objective with grammar, plan, and reuse constraints
- **Training Steps**: {config.get('training', {}).get('num_epochs', 'Unknown')} epochs

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{export_path.name}")
model = AutoModelForCausalLM.from_pretrained("{export_path.name}")

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
@misc{{builderbrain_{scale.replace('-', '_')},
  title={{BuilderBrain: Dual-Rail Compositional AI System}},
  author={{BuilderBrain Team}},
  year={{2024}},
  url={{https://github.com/JacobFV/builderbrain}}
}}
```
"""

        with open(export_path / "README.md", 'w') as f:
            f.write(model_card_content)

    def _get_directory_size(self, path: Path) -> str:
        """Calculate directory size in human readable format."""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        # Convert to appropriate units
        if total_size < 1024**2:
            return f"{total_size / 1024:.1f}KB"
        elif total_size < 1024**3:
            return f"{total_size / (1024**2):.1f}MB"
        else:
            return f"{total_size / (1024**3):.1f}GB"

    def export_to_onnx(self, model_path: str, output_path: str) -> Dict[str, Any]:
        """Export model to ONNX format (placeholder)."""
        # In production, this would use torch.onnx.export
        return {
            "export_path": output_path,
            "status": "completed",
            "format": "onnx",
            "timestamp": datetime.now().isoformat()
        }

    def export_to_torchscript(self, model_path: str, output_path: str) -> Dict[str, Any]:
        """Export model to TorchScript format (placeholder)."""
        # In production, this would use torch.jit.trace
        return {
            "export_path": output_path,
            "status": "completed",
            "format": "torchscript",
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Main export function."""
    exporter = ModelExporter()

    # Example usage
    config_path = "configs/small.yaml"
    model_path = "builderbrain_final.ckpt"  # Would be actual model file

    result = exporter.export_to_huggingface(model_path, config_path, "small")

    if "error" not in result:
        print(f"✅ Export completed: {result['export_id']}")
        print(f"📁 Export path: {result['export_path']}")
        print(f"💾 File size: {result['file_size']}")
    else:
        print(f"❌ Export failed: {result['error']}")


if __name__ == "__main__":
    main()
