"""
Fusion gates for controlling base vs builder rail influence.

Learned gating mechanism that controls how much each layer relies on
the builder rail vs the base rail.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionGates(nn.Module):
    """
    Fusion gates that control the alpha values for combining base and builder rails.

    Each layer gets its own gating mechanism to decide the balance between
    base rail (frozen pretrained) and builder rail (learned composition).
    """

    def __init__(self, hidden_size: int, num_layers: int, alpha_cap: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.alpha_cap = alpha_cap

        # Per-layer gating networks
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
            ) for _ in range(num_layers)
        ])

        # Global alpha cap enforcement
        self.register_buffer("max_alpha", torch.tensor(alpha_cap))

    def forward(
        self,
        layer_idx: int,
        base_state: torch.Tensor,
        builder_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alpha gating value for a specific layer.

        Args:
            layer_idx: Index of the layer
            base_state: Hidden state from base rail
            builder_state: Hidden state from builder rail

        Returns:
            Alpha values (batch, 1) in [0, alpha_cap]
        """
        # Combine base and builder states for gating decision
        combined = base_state + builder_state

        # Pool across sequence dimension for global decision per example
        pooled = combined.mean(dim=1)  # (batch, hidden)

        # Get raw alpha from gate network
        raw_alpha = self.gates[layer_idx](pooled)  # (batch, 1)

        # Apply cap and ensure non-negativity
        alpha = torch.clamp(raw_alpha, min=0.0, max=self.alpha_cap)

        return alpha

    def get_gate_gradients(self) -> List[torch.Tensor]:
        """Get gradients for all gate parameters (for analysis)."""
        grads = []
        for gate in self.gates:
            for param in gate.parameters():
                if param.grad is not None:
                    grads.append(param.grad.norm().item())
        return grads

    def enforce_alpha_budget(self, alpha_values: torch.Tensor) -> torch.Tensor:
        """Enforce global alpha budget across layers."""
        # For now, simple per-layer capping
        # Could implement more sophisticated global budgeting
        return torch.clamp(alpha_values, max=self.alpha_cap)

    def compute_alpha_statistics(self, alpha_history: List[torch.Tensor]) -> dict:
        """Compute statistics on alpha values for monitoring."""
        if not alpha_history:
            return {}

        all_alphas = torch.cat([a.flatten() for a in alpha_history])

        return {
            "mean_alpha": float(all_alphas.mean()),
            "std_alpha": float(all_alphas.std()),
            "max_alpha": float(all_alphas.max()),
            "min_alpha": float(all_alphas.min()),
            "sparsity": float((all_alphas < 0.01).float().mean())  # Fraction near zero
        }


class AdaptiveFusionGates(nn.Module):
    """
    Adaptive fusion gates that can adjust alpha caps based on training dynamics.

    Allows the model to request more builder rail influence when beneficial.
    """

    def __init__(self, hidden_size: int, num_layers: int, initial_alpha_cap: float = 0.05):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Base gates (same as FusionGates)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])

        # Adaptive alpha cap (learned, but constrained)
        self.log_alpha_cap = nn.Parameter(torch.log(torch.tensor(initial_alpha_cap)))

        # Running statistics for alpha cap adaptation
        self.register_buffer("alpha_history", torch.tensor([]))

    @property
    def alpha_cap(self) -> float:
        """Current alpha cap value."""
        return float(torch.exp(self.log_alpha_cap).clamp(min=0.001, max=0.5))

    def forward(self, layer_idx: int, base_state: torch.Tensor, builder_state: torch.Tensor) -> torch.Tensor:
        """Compute alpha with current adaptive cap."""
        combined = base_state + builder_state
        pooled = combined.mean(dim=1)

        raw_alpha = self.gates[layer_idx](pooled)
        alpha = torch.clamp(raw_alpha, min=0.0, max=self.alpha_cap)

        return alpha

    def update_alpha_cap(self, performance_improvement: float, safety_margin: float = 0.05):
        """
        Update alpha cap based on training dynamics.

        Args:
            performance_improvement: Recent improvement in task performance
            safety_margin: Minimum margin to maintain for safety
        """
        # Increase cap if performance improved significantly
        if performance_improvement > 0.01:  # 1% improvement threshold
            new_cap = min(self.alpha_cap * 1.1, 0.5)  # Max 50%
        else:
            # Decrease cap if no improvement (to be more conservative)
            new_cap = max(self.alpha_cap * 0.95, 0.01)  # Min 1%

        # Apply safety margin
        new_cap = min(new_cap, 0.5 - safety_margin)

        # Update parameter
        with torch.no_grad():
            self.log_alpha_cap.copy_(torch.log(torch.tensor(new_cap)))

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about gate behavior."""
        return {
            "current_alpha_cap": self.alpha_cap,
            "gate_weight_norms": [float(sum(p.norm() for p in gate.parameters())) for gate in self.gates],
            "alpha_cap_history_length": len(self.alpha_history) if hasattr(self, 'alpha_history') else 0
        }
