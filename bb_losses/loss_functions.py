"""
Specific loss functions for builderbrain constraints.

Implements grammar, graph, buildability, and other constraint losses.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..bb_priors.grammar_energy import GrammarEnergy


class GrammarLoss(nn.Module):
    """Grammar compliance loss using CFG energy."""

    def __init__(self, grammar_energy: GrammarEnergy, target: float = 0.0):
        super().__init__()
        self.grammar_energy = grammar_energy
        self.target = target

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute grammar loss."""
        energies = self.grammar_energy(token_ids)

        # Hinge loss: penalize energies above target
        violations = torch.clamp(energies - self.target, min=0.0)
        return violations.mean()


class GraphToGraphLoss(nn.Module):
    """Graph-to-graph reconstruction loss for plan DAGs."""

    def __init__(self, target_similarity: float = 0.8):
        super().__init__()
        self.target_similarity = target_similarity

    def forward(
        self,
        predicted_graph: Dict[str, Any],
        target_graph: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute graph-to-graph loss."""
        # Simple graph edit distance approximation
        similarity = self._compute_graph_similarity(predicted_graph, target_graph)

        # Loss as negative similarity (want to maximize similarity)
        loss = 1.0 - similarity

        return loss

    def _compute_graph_similarity(
        self,
        graph1: Dict[str, Any],
        graph2: Dict[str, Any]
    ) -> float:
        """Compute similarity between two graphs."""
        # Simplified similarity based on node/edge overlap
        nodes1 = set(graph1.get('nodes', []))
        nodes2 = set(graph2.get('nodes', []))

        edges1 = set(tuple(edge) for edge in graph1.get('edges', []))
        edges2 = set(tuple(edge) for edge in graph2.get('edges', []))

        # Jaccard similarity for nodes and edges
        node_similarity = len(nodes1 & nodes2) / len(nodes1 | nodes2) if nodes1 | nodes2 else 1.0
        edge_similarity = len(edges1 & edges2) / len(edges1 | edges2) if edges1 | edges2 else 1.0

        # Weighted combination
        return 0.6 * node_similarity + 0.4 * edge_similarity


class BuildabilityLoss(nn.Module):
    """Buildability loss ensuring hidden states are constructible from skills."""

    def __init__(self, projection_dim: int = 256):
        super().__init__()
        self.projection_dim = projection_dim

        # Projection layers for compatibility
        self.projection = nn.Linear(projection_dim, projection_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        composed_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute buildability loss."""
        # Project to common space
        hidden_proj = self.projection(hidden_state)
        composed_proj = self.projection(composed_state)

        # L2 loss between hidden and composed representations
        loss = F.mse_loss(hidden_proj, composed_proj)

        return loss


class ReuseLoss(nn.Module):
    """Encourages reuse of existing program skills."""

    def __init__(self, num_programs: int, entropy_weight: float = 1.0):
        super().__init__()
        self.num_programs = num_programs
        self.entropy_weight = entropy_weight

    def forward(self, program_logits: torch.Tensor) -> torch.Tensor:
        """Compute reuse loss."""
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

    def forward(
        self,
        predictions: torch.Tensor,
        confidences: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute expected calibration error."""
        batch_size = predictions.size(0)

        # Create confidence bins
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        ece = 0.0

        for i in range(self.num_bins):
            # Find samples in this bin
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])

            if mask.sum() > 0:
                # Accuracy in this bin
                bin_predictions = predictions[mask]
                bin_targets = targets[mask]
                bin_accuracy = (bin_predictions == bin_targets).float().mean()

                # Average confidence in this bin
                bin_confidence = confidences[mask].mean()

                # ECE contribution
                ece += (mask.sum() / batch_size) * abs(bin_accuracy - bin_confidence)

        return ece


class KLLoss(nn.Module):
    """KL divergence constraint for rational inattention."""

    def __init__(self, prior_probs: torch.Tensor, budget: float = 0.05):
        super().__init__()
        self.register_buffer('prior_probs', prior_probs)
        self.budget = budget

    def forward(self, current_probs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
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
            if name == 'grammar':
                constraint_losses[name] = loss_fn(model_outputs.get('token_ids', torch.empty(1)))
            elif name == 'graph2graph':
                constraint_losses[name] = loss_fn(
                    model_outputs.get('predicted_graph', {}),
                    targets.get('target_graph', {})
                )
            elif name == 'buildability':
                constraint_losses[name] = loss_fn(
                    model_outputs.get('hidden_state', torch.empty(1)),
                    model_outputs.get('composed_state', torch.empty(1))
                )
            elif name == 'reuse':
                constraint_losses[name] = loss_fn(model_outputs.get('program_logits', torch.empty(1)))
            elif name == 'calibration':
                constraint_losses[name] = loss_fn(
                    model_outputs.get('predictions', torch.empty(1)),
                    model_outputs.get('confidences', torch.empty(1)),
                    targets.get('targets', torch.empty(1))
                )
            elif name == 'kl':
                constraint_losses[name] = loss_fn(model_outputs.get('current_probs', torch.empty(1)))
            else:
                # Default: assume loss function takes outputs and targets
                constraint_losses[name] = loss_fn(model_outputs, targets)

        # Compute total Lagrangian
        total_loss, normalized_losses = self.dual_optimizer.compute_lagrangian(
            task_loss, constraint_losses
        )

        return total_loss, constraint_losses
