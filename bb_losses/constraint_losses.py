"""
Additional constraint loss functions for BuilderBrain.

Contains specialized loss functions for various constraints.
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintLossBank:
    """Collection of constraint loss functions."""

    def __init__(self):
        self.losses = {}

    def add_loss(self, name: str, loss_fn: nn.Module):
        """Add a constraint loss function."""
        self.losses[name] = loss_fn

    def get_loss(self, name: str) -> Optional[nn.Module]:
        """Get a constraint loss function."""
        return self.losses.get(name)

    def compute_all_losses(self, outputs: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute all constraint losses."""
        losses = {}
        for name, loss_fn in self.losses.items():
            try:
                losses[name] = loss_fn(outputs, targets)
            except Exception as e:
                print(f"Warning: Failed to compute {name} loss: {e}")
                losses[name] = torch.tensor(0.0)
        return losses


# Default constraint loss bank
default_constraint_losses = ConstraintLossBank()

# Add any additional constraint losses here as needed
