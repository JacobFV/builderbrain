"""
Specific loss functions for builderbrain constraints.

Implements grammar, graph, buildability, and other constraint losses.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from bb_priors.grammar_energy import GrammarEnergy


class GrammarLoss(nn.Module):
    """Grammar compliance loss using CFG energy."""

    def __init__(self, grammar_energy: GrammarEnergy, target: float = 0.0):
        super().__init__()
        self.grammar_energy = grammar_energy
        self.target = target

    def forward(self, model_outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
        """Compute grammar loss."""
        # Get input_ids from model_outputs or targets
        token_ids = model_outputs.get('token_ids') or targets.get('input_ids')
        if token_ids is None:
            return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))

        energies = self.grammar_energy(token_ids)

        # Hinge loss: penalize energies above target
        violations = torch.clamp(energies - self.target, min=0.0)
        return violations.mean()


class GraphToGraphLoss(nn.Module):
    """Graph-to-graph reconstruction loss for plan DAGs."""

    def __init__(self, target_similarity: float = 0.8):
        super().__init__()
        self.target_similarity = target_similarity

    def forward(self, model_outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
        """Compute graph-to-graph loss."""
        # For now, return zero loss since we don't have real graph prediction yet
        # This would be implemented when we add graph prediction heads
        return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))


class BuildabilityLoss(nn.Module):
    """Buildability loss ensuring hidden states are constructible from skills."""

    def __init__(self, projection_dim: int = 256):
        super().__init__()
        self.projection_dim = projection_dim

    def forward(self, model_outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
        """Compute buildability loss."""
        # For now, return zero loss since we don't have composition prediction yet
        return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))


class ReuseLoss(nn.Module):
    """Encourages reuse of existing program skills."""

    def __init__(self, num_programs: int, entropy_weight: float = 1.0):
        super().__init__()
        self.num_programs = num_programs
        self.entropy_weight = entropy_weight

    def forward(self, model_outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
        """Compute reuse loss."""
        program_logits = model_outputs.get('program_logits')
        if program_logits is None:
            return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))

        # Convert logits to probabilities
        probs = F.softmax(program_logits, dim=-1)

        # Entropy regularization (encourage exploration)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

        # Diversity bonus (encourage spread across programs)
        diversity = torch.var(probs.mean(dim=0))  # Variance across batch

        # Combined loss
        loss = -self.entropy_weight * entropy + 0.1 * diversity

        return loss


class CalibrationLoss(nn.Module):
    """Expected calibration error for reliable uncertainty quantification."""

    def __init__(self, num_bins: int = 15):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, model_outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
        """Compute expected calibration error."""
        # For now, return zero loss since we don't have calibration prediction yet
        return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))


class KLLoss(nn.Module):
    """KL divergence constraint for rational inattention."""

    def __init__(self, prior_probs: torch.Tensor, budget: float = 0.05):
        super().__init__()
        self.register_buffer('prior_probs', prior_probs)
        self.budget = budget

    def forward(self, model_outputs: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
        """Compute KL divergence loss."""
        current_probs = model_outputs.get('current_probs')
        if current_probs is None:
            return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else torch.device('cpu'))

        # KL divergence from current to prior
        kl_div = F.kl_div(
            torch.log(current_probs + 1e-8),
            self.prior_probs,
            reduction='batchmean'
        )

        # Hinge loss: penalize KL above budget
        violation = torch.clamp(kl_div - self.budget, min=0.0)

        return violation


class CompositeLoss(nn.Module):
    """Combines multiple constraint losses with dual optimization."""

    def __init__(
        self,
        dual_optimizer: 'DualOptimizer',
        loss_functions: Dict[str, nn.Module]
    ):
        super().__init__()
        self.dual_optimizer = dual_optimizer
        self.loss_functions = loss_functions

    def forward(
        self,
        task_loss: torch.Tensor,
        model_outputs: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute composite loss with all constraints."""
        constraint_losses = {}

        # Compute each constraint loss
        for name, loss_fn in self.loss_functions.items():
            try:
                constraint_losses[name] = loss_fn(model_outputs, targets)
            except Exception as e:
                print(f"Warning: Failed to compute {name} loss: {e}")
                constraint_losses[name] = torch.tensor(0.0)

        # Compute total Lagrangian
        total_loss, normalized_losses = self.dual_optimizer.compute_lagrangian(
            task_loss, constraint_losses
        )

        return total_loss, constraint_losses
