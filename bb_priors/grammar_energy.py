"""
Grammar energy computation for soft constraint enforcement.

Provides differentiable energy functions for grammar compliance.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from .cfg_parser import CFGParser


class GrammarEnergy(nn.Module):
    """
    Computes grammar compliance energy for sequences.

    Used as a soft constraint in the loss function.
    """

    def __init__(self, grammar: CFGParser, tokenizer, tau: float = 1.0):
        super().__init__()

        self.grammar = grammar
        self.tokenizer = tokenizer
        self.tau = tau  # Temperature for energy computation

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute grammar energy for batch of token sequences.

        Args:
            token_ids: Token sequences (batch, seq_len)

        Returns:
            Energy values (batch,)
        """
        batch_size = token_ids.size(0)
        energies = torch.zeros(batch_size, device=token_ids.device)

        for batch_idx in range(batch_size):
            # Decode tokens to text (ignore padding)
            tokens = token_ids[batch_idx].tolist()
            # Remove padding tokens
            if self.tokenizer.pad_token_id is not None:
                tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]

            try:
                text = self.tokenizer.decode(tokens)
                energy = self._compute_text_energy(text)
                energies[batch_idx] = energy
            except Exception:
                # High energy for invalid sequences
                energies[batch_idx] = 10.0

        return energies

    def _compute_text_energy(self, text: str) -> float:
        """Compute energy for a single text string."""
        # Split into tokens for grammar evaluation
        tokens = text.split()

        # Get base energy from grammar
        base_energy = self.grammar.sequence_energy(tokens)

        # Additional energy terms for common issues
        additional_energy = 0.0

        # 1. Length penalty (very long or very short sequences)
        if len(tokens) > 100:
            additional_energy += 0.1 * (len(tokens) - 100)
        elif len(tokens) < 3:
            additional_energy += 0.1 * (3 - len(tokens))

        # 2. Repeated token penalty
        if len(tokens) > 1:
            repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
            additional_energy += 0.05 * repeats

        # 3. Invalid character penalty (basic check)
        invalid_chars = sum(1 for char in text if ord(char) < 32 and char not in '\n\r\t')
        additional_energy += 0.1 * invalid_chars

        total_energy = base_energy + additional_energy

        return float(total_energy)

    def compute_violation_types(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute different types of grammar violations.

        Returns:
            Violation scores for different categories (batch, 5)
        """
        batch_size = token_ids.size(0)
        violations = torch.zeros(batch_size, 5, device=token_ids.device)

        for batch_idx in range(batch_size):
            tokens = token_ids[batch_idx].tolist()
            try:
                text = self.tokenizer.decode(tokens)
                violation_scores = self._analyze_violations(text)
                violations[batch_idx] = torch.tensor(violation_scores)
            except Exception:
                violations[batch_idx] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

        return violations

    def _analyze_violations(self, text: str) -> List[float]:
        """Analyze different types of violations in text."""
        tokens = text.split()

        violations = [0.0] * 5

        # 1. Syntax violation (main grammar check)
        violations[0] = self.grammar.sequence_energy(tokens)

        # 2. Structure violation (missing brackets, quotes, etc.)
        open_brackets = text.count('{') + text.count('[')
        close_brackets = text.count('}') + text.count(']')
        violations[1] = abs(open_brackets - close_brackets) * 0.5

        # 3. Quote balance
        quote_count = text.count('"')
        violations[2] = (quote_count % 2) * 0.5

        # 4. Length anomaly
        if len(tokens) > 50 or len(tokens) < 2:
            violations[3] = 1.0

        # 5. Character set violation
        invalid_chars = sum(1 for char in text if ord(char) > 127 or ord(char) < 32)
        violations[4] = min(invalid_chars / len(text), 1.0) if text else 0.0

        return violations

    def get_weighted_energy(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute weighted combination of violation types."""
        base_energy = self(token_ids)
        violations = self.compute_violation_types(token_ids)

        # Weighted combination
        weighted_violations = violations * self.violation_weights.unsqueeze(0)
        total_violation_energy = weighted_violations.sum(dim=1)

        return base_energy + total_violation_energy


class GrammarEnergyScheduler:
    """
    Scheduler for grammar energy targets and weights.

    Allows adaptive adjustment of grammar constraints during training.
    """

    def __init__(self, initial_energy_target: float = 0.1, adaptation_rate: float = 0.01):
        self.initial_target = initial_energy_target
        self.adaptation_rate = adaptation_rate

        self.current_target = initial_energy_target
        self.violation_history = []

    def update_target(self, current_violation_rate: float, step: int):
        """
        Update energy target based on current violation rate.

        Args:
            current_violation_rate: Current grammar violation rate
            step: Training step
        """
        # Annealing schedule: gradually reduce tolerance
        annealing_factor = min(1.0, step / 10000)  # Anneal over 10k steps

        if current_violation_rate > 0.05:  # Too many violations
            # Increase target (be more lenient)
            self.current_target = min(self.current_target * 1.1, 1.0)
        elif current_violation_rate < 0.01:  # Very few violations
            # Decrease target (be more strict)
            self.current_target = max(self.current_target * 0.95, 0.01)

        # Apply annealing
        self.current_target = (self.initial_target * (1 - annealing_factor) +
                              self.current_target * annealing_factor)

    def get_target(self) -> float:
        """Get current energy target."""
        return self.current_target

    def compute_loss(self, energies: torch.Tensor, target: float = None) -> torch.Tensor:
        """Compute grammar constraint loss."""
        if target is None:
            target = self.current_target

        # Hinge loss: penalize energies above target
        violation = torch.clamp(energies - target, min=0.0)
        return violation.mean()
