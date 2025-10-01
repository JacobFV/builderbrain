"""
Main training orchestrator for builderbrain.

Coordinates model training with dual constraints, grammar enforcement,
and multi-objective optimization.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bb_nn.dual_rail import DualRail
from bb_losses.dual_optimizer import DualOptimizer
from bb_losses.loss_functions import CompositeLoss, GrammarLoss, GraphToGraphLoss, ReuseLoss
from bb_priors.grammar_energy import GrammarEnergy
from bb_runtime.plan_checker import PlanChecker


class BuilderBrainTrainer:
    """
    Main trainer for the builderbrain system.

    Orchestrates training across multiple objectives with constraint satisfaction.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize tokenizer first
        self.tokenizer = self._initialize_tokenizer()

        # Initialize model
        self.model = self._initialize_model()

        # Initialize dual optimizer
        self.dual_optimizer = self._initialize_dual_optimizer()

        # Initialize loss functions
        self.loss_functions = self._initialize_loss_functions()

        # Initialize data loader
        self.data_loader = self._initialize_data_loader()

        # Initialize composite loss
        self.composite_loss = self._initialize_composite_loss()

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []

    def _initialize_model(self) -> DualRail:
        """Initialize the dual-rail model with configurable base model."""
        model_config = self.config['model']

        # Load base model based on configuration
        base_model = self._load_base_model(model_config)

        model = DualRail(
            base_model=base_model,
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_programs=model_config['num_programs'],
            alpha_cap=model_config['alpha_cap']
        )

        return model

    def _load_base_model(self, model_config: Dict[str, Any]):
        """Load base transformer model based on configuration."""
        model_type = model_config.get('type', 'gpt2')
        model_name = model_config.get('name', 'gpt2')

        if model_type == 'gpt2':
            from transformers import GPT2LMHeadModel
            print(f"Loading GPT-2 model: {model_name}")
            return GPT2LMHeadModel.from_pretrained(model_name)
        elif model_type == 'gpt_neo':
            from transformers import GPTNeoForCausalLM
            print(f"Loading GPT-Neo model: {model_name}")
            return GPTNeoForCausalLM.from_pretrained(model_name)
        elif model_type == 'llama':
            from transformers import LlamaForCausalLM
            print(f"Loading LLaMA model: {model_name}")
            return LlamaForCausalLM.from_pretrained(model_name)
        elif model_type == 'opt':
            from transformers import OPTForCausalLM
            print(f"Loading OPT model: {model_name}")
            return OPTForCausalLM.from_pretrained(model_name)
        else:
            # Default to GPT-2 for unknown types
            from transformers import GPT2LMHeadModel
            print(f"Unknown model type {model_type}, defaulting to GPT-2")
            return GPT2LMHeadModel.from_pretrained('gpt2')

    def _initialize_dual_optimizer(self) -> DualOptimizer:
        """Initialize the dual constraint optimizer."""
        constraint_configs = self.config['constraints']

        return DualOptimizer(
            constraint_configs=constraint_configs,
            eta_lambda=self.config['training']['eta_lambda'],
            lambda_max=self.config['training']['lambda_max']
        )

    def _initialize_loss_functions(self) -> Dict[str, nn.Module]:
        """Initialize loss functions for constraints."""
        losses = {}

        # Grammar loss
        if self.config['constraints']['grammar']['enabled']:
            from bb_priors.cfg_parser import JSONGrammar
            grammar = JSONGrammar()
            grammar_energy = GrammarEnergy(grammar, self.tokenizer)
            losses['grammar'] = GrammarLoss(grammar_energy)

        # Graph-to-graph loss
        if self.config['constraints']['graph2graph']['enabled']:
            losses['graph2graph'] = GraphToGraphLoss()

        # Reuse loss
        if self.config['constraints'].get('reuse', {}).get('enabled', False):
            losses['reuse'] = ReuseLoss(self.config['model']['num_programs'])

        # Add other losses as needed...

        return losses

    def _initialize_composite_loss(self) -> CompositeLoss:
        """Initialize composite loss with dual optimizer."""
        return CompositeLoss(self.dual_optimizer, self.loss_functions)

    def _initialize_tokenizer(self):
        """Initialize tokenizer for the model."""
        model_config = self.config['model']
        model_type = model_config.get('type', 'gpt2')

        if model_type == 'gpt2':
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer
        elif model_type == 'gpt_neo':
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # GPT-Neo uses GPT-2 tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer
        elif model_type == 'llama':
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_config.get('name', 'huggyllama/llama-7b'))
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer
        elif model_type == 'opt':
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-1.3b')  # OPT uses GPT-2 tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer
        else:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer

    def _initialize_data_loader(self) -> DataLoader:
        """Initialize training data loader."""
        from .data_loader import BuilderBrainDataLoader

        data_loader = BuilderBrainDataLoader(self.config, self.tokenizer)
        return data_loader.get_train_loader()

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        input_ids, targets = batch

        # Forward pass
        model_outputs = self.model(input_ids)

        # Compute task loss (language modeling)
        logits = model_outputs['logits']

        # Handle padding in targets for cross-entropy loss
        if targets.dim() == 2:  # Padded sequences
            # Flatten and ignore padding tokens
            loss_mask = targets != -100  # -100 is ignore_index
            flattened_logits = logits.view(-1, logits.size(-1))
            flattened_targets = targets.view(-1)

            # Create mask for valid tokens (not padding)
            valid_mask = loss_mask.view(-1)

            # Only compute loss on non-padded tokens
            if valid_mask.sum() > 0:  # Make sure we have some valid tokens
                task_loss = F.cross_entropy(
                    flattened_logits[valid_mask],
                    flattened_targets[valid_mask]
                )
            else:
                # Fallback to regular loss if all tokens are padding
                task_loss = F.cross_entropy(
                    flattened_logits,
                    flattened_targets,
                    ignore_index=-100
                )
        else:
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        # Prepare targets for constraint losses
        constraint_targets = {
            'target_graph': {},  # Would be actual target graphs
            'targets': targets,
            'input_ids': input_ids  # Pass input IDs for grammar checking
        }

        # Compute composite loss with constraints
        total_loss, constraint_losses = self.composite_loss(
            task_loss, model_outputs, constraint_targets
        )

        # Update dual variables
        normalized_losses = self.dual_optimizer.constraint_manager.normalize_losses(constraint_losses)
        self.dual_optimizer.update_duals(normalized_losses)

        # Backward pass
        total_loss.backward()

        # Update model parameters
        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), 1.0)
        for param_group in self.model.get_trainable_params():
            if param_group.grad is not None:
                param_group.data.add_(param_group.grad, alpha=-self.config['training']['learning_rate'])

        # Track metrics
        metrics = {
            'total_loss': float(total_loss.item()),
            'task_loss': float(task_loss.item()),
            'constraint_losses': {k: float(v) for k, v in constraint_losses.items()},
            'dual_variables': self.dual_optimizer.get_dual_values(),
            'step': self.step
        }

        self.step += 1
        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = []
        for batch in self.data_loader:
            metrics = self.train_step(batch)
            epoch_losses.append(metrics['total_loss'])

        # Compute epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        epoch_metrics = {
            'epoch': self.epoch,
            'avg_loss': avg_loss,
            'best_loss': self.best_loss,
            'improvement': self.best_loss - avg_loss
        }

        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

        self.epoch += 1
        return epoch_metrics

    def train(self, num_epochs: int = 100) -> Dict[str, List[float]]:
        """Main training loop."""
        print("Starting builderbrain training...")

        start_time = time.time()
        training_history = {
            'total_loss': [],
            'task_loss': [],
            'constraint_losses': {k: [] for k in self.config['constraints'].keys()},
            'dual_variables': []
        }

        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch()

            # Record history
            training_history['total_loss'].append(epoch_metrics['avg_loss'])
            training_history['dual_variables'].append(
                self.dual_optimizer.get_dual_values()
            )

            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_metrics['avg_loss']:.4f}")

                # Print dual variables
                duals = self.dual_optimizer.get_dual_values()
                print(f"  Dual variables: {duals}")

            # Early stopping
            if epoch_metrics['improvement'] < -0.01 and epoch > 50:  # No improvement for 50 epochs
                print(f"Early stopping at epoch {epoch}")
                break

        end_time = time.time()
        training_time = end_time - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        return training_history

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'dual_optimizer_state': self.dual_optimizer.state_dict(),
            'config': self.config,
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state'])
        self.dual_optimizer.load_state_dict(checkpoint['dual_optimizer_state'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']

        print(f"Checkpoint loaded from {path}")


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        'model': {
            'hidden_size': 768,
            'num_layers': 6,
            'num_programs': 32,
            'alpha_cap': 0.1
        },
        'constraints': {
            'grammar': {
                'enabled': True,
                'target': 0.0,
                'normalizer': 'rank'
            },
            'graph2graph': {
                'enabled': True,
                'target': 0.2,
                'normalizer': 'rank'
            }
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'eta_lambda': 1e-2,
            'lambda_max': 50.0,
            'num_epochs': 100
        }
    }


def main():
    """Main training function."""
    config = create_default_config()

    trainer = BuilderBrainTrainer(config)
    history = trainer.train(num_epochs=config['training']['num_epochs'])

    # Save final model
    trainer.save_checkpoint('builderbrain_final.ckpt')

    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
