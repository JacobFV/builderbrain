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

from ..bb_nn.dual_rail import DualRail
from ..bb_losses.dual_optimizer import DualOptimizer
from ..bb_losses.loss_functions import CompositeLoss, GrammarLoss, GraphToGraphLoss
from ..bb_priors.grammar_energy import GrammarEnergy
from ..bb_runtime.plan_checker import PlanChecker


class BuilderBrainTrainer:
    """
    Main trainer for the builderbrain system.

    Orchestrates training across multiple objectives with constraint satisfaction.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize model
        self.model = self._initialize_model()

        # Initialize dual optimizer
        self.dual_optimizer = self._initialize_dual_optimizer()

        # Initialize loss functions
        self.loss_functions = self._initialize_loss_functions()

        # Initialize data loader
        self.data_loader = self._initialize_data_loader()

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []

    def _initialize_model(self) -> DualRail:
        """Initialize the dual-rail model."""
        model_config = self.config['model']

        # Load base model (would be actual pretrained model in practice)
        from transformers import GPT2LMHeadModel
        base_model = GPT2LMHeadModel.from_pretrained('gpt2')

        model = DualRail(
            base_model=base_model,
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_programs=model_config['num_programs'],
            alpha_cap=model_config['alpha_cap']
        )

        return model

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
            from ..bb_priors.cfg_parser import JSONGrammar
            grammar = JSONGrammar()
            grammar_energy = GrammarEnergy(grammar, None)  # Would need tokenizer
            losses['grammar'] = GrammarLoss(grammar_energy)

        # Graph-to-graph loss
        if self.config['constraints']['graph2graph']['enabled']:
            losses['graph2graph'] = GraphToGraphLoss()

        # Add other losses as needed...

        return losses

    def _initialize_data_loader(self) -> DataLoader:
        """Initialize training data loader."""
        # Simplified - in practice would load actual training data
        return DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.randint(0, 1000, (1000, 128)),  # Dummy data
                torch.randint(0, 1000, (1000, 128))
            ),
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        input_ids, targets = batch

        # Forward pass
        model_outputs = self.model(input_ids)

        # Compute task loss (language modeling)
        logits = model_outputs['logits']
        task_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        # Prepare targets for constraint losses
        constraint_targets = {
            'target_graph': {},  # Would be actual target graphs
            'targets': targets
        }

        # Compute composite loss with constraints
        total_loss, constraint_losses = self.loss_functions['composite'](
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
                param_group.data.add_(-self.config['training']['learning_rate'], param_group.grad)

        # Track metrics
        metrics = {
            'total_loss': float(total_loss.item()),
            'task_loss': float(task_loss.item()),
            'constraint_losses': {k: float(v.item()) for k, v in constraint_losses.items()},
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
                print(f"Epoch {epoch}: Loss = {epoch_metrics['avg_loss']".4f"}")

                # Print dual variables
                duals = self.dual_optimizer.get_dual_values()
                print(f"  Dual variables: {duals}")

            # Early stopping
            if epoch_metrics['improvement'] < -0.01 and epoch > 50:  # No improvement for 50 epochs
                print(f"Early stopping at epoch {epoch}")
                break

        end_time = time.time()
        training_time = end_time - start_time

        print(f"Training completed in {training_time".2f"} seconds")

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
