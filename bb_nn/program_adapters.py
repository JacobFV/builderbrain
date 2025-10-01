"""
Program adapters for discrete skill selection and application.

Implements discrete program tokens using Straight-Through Gumbel-Softmax.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProgramAdapter(nn.Module):
    """
    Program adapter for discrete skill selection and application.

    Maps hidden states to program logits and applies program-specific adaptations.
    """

    def __init__(self, hidden_size: int, num_programs: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_programs = num_programs

        # Program selection head
        self.program_head = nn.Linear(hidden_size, num_programs)

        # Program embeddings (learnable)
        self.program_embeddings = nn.Embedding(num_programs, hidden_size)

        # Program-specific adaptation layers
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size)
            ) for _ in range(num_programs)
        ])

        # Usage tracking for efficiency metrics
        self.usage_counts = torch.zeros(num_programs)

    def get_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Get program selection logits from hidden state."""
        # Average across sequence length for global program selection
        # In practice, might want per-position or hierarchical selection
        pooled = hidden_state.mean(dim=1)  # (batch, hidden)
        return self.program_head(pooled)  # (batch, programs)

    def sample_programs_st_gumbel(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        training: bool = True
    ) -> torch.Tensor:
        """
        Sample discrete programs using Straight-Through Gumbel-Softmax.

        Args:
            logits: Program logits (batch, programs)
            temperature: Sampling temperature
            training: Whether in training mode

        Returns:
            Program probabilities (batch, programs) or one-hot selections
        """
        if training:
            # Gumbel-Softmax with Straight-Through
            y_soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
            y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=self.num_programs).float()

            # Straight-through: forward uses hard, backward uses soft
            program_selection = y_hard - y_soft.detach() + y_soft
        else:
            # Greedy selection at inference
            indices = logits.argmax(dim=-1)
            program_selection = F.one_hot(indices, num_classes=self.num_programs).float()

        return program_selection

    def forward(
        self,
        hidden_state: torch.Tensor,
        program_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply program adapters to hidden state.

        Args:
            hidden_state: Input hidden state (batch, seq, hidden)
            program_probs: Program selection probabilities (batch, seq, programs)
                          If None, computed from hidden_state
        """
        batch_size, seq_len, _ = hidden_state.shape

        if program_probs is None:
            # Compute program logits from hidden state
            logits = self.get_logits(hidden_state)
            # For simplicity, use soft selection during forward
            program_probs = F.softmax(logits, dim=-1)  # (batch, programs)

        # Apply adapters for each program
        adapted_states = []

        for prog_id in range(self.num_programs):
            # Get program embedding
            prog_embed = self.program_embeddings(torch.tensor(prog_id, device=hidden_state.device))
            prog_embed = prog_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

            # Program-specific adaptation
            prog_prob = program_probs[:, prog_id].unsqueeze(-1)  # (batch, 1)
            adapted = self.adapters[prog_id](hidden_state + prog_embed)
            adapted = adapted * prog_prob  # Weight by program probability

            adapted_states.append(adapted)

        # Sum adaptations from all programs
        output_state = sum(adapted_states)

        # Update usage statistics
        if program_probs is not None:
            usage = program_probs.mean(dim=0)  # Average across batch
            self.usage_counts = 0.9 * self.usage_counts + 0.1 * usage.detach()

        return output_state

    def get_program_embedding(self, program_id: int) -> torch.Tensor:
        """Get embedding for a specific program."""
        return self.program_embeddings(torch.tensor(program_id))

    def get_usage_stats(self) -> Dict[int, float]:
        """Get program usage statistics."""
        return {i: float(count) for i, count in enumerate(self.usage_counts)}

    def reset_usage_stats(self):
        """Reset usage tracking."""
        self.usage_counts.zero_()


class HypernetProgramAdapter(nn.Module):
    """
    Hypernetwork-based program adapter for more sophisticated skill modulation.

    Each program generates parameters for a modulation network.
    """

    def __init__(self, hidden_size: int, num_programs: int, hypernet_hidden: int = 128):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_programs = num_programs
        self.hypernet_hidden = hypernet_hidden

        # Program embeddings
        self.program_embeddings = nn.Embedding(num_programs, hidden_size // 2)

        # Hypernetwork that generates modulation parameters
        self.hypernet = nn.Sequential(
            nn.Linear(hidden_size // 2, hypernet_hidden),
            nn.GELU(),
            nn.Linear(hypernet_hidden, hidden_size * hidden_size)
        )

        # Program selection head
        self.program_head = nn.Linear(hidden_size, num_programs)

        # Usage tracking
        self.usage_counts = torch.zeros(num_programs)

    def get_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Get program selection logits."""
        pooled = hidden_state.mean(dim=1)
        return self.program_head(pooled)

    def forward(
        self,
        hidden_state: torch.Tensor,
        program_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply hypernetwork-based program adaptation."""
        batch_size, seq_len, _ = hidden_state.shape

        if program_probs is None:
            logits = self.get_logits(hidden_state)
            program_probs = F.softmax(logits, dim=-1)

        # Generate modulation matrices for each program
        outputs = []

        for prog_id in range(self.num_programs):
            prog_embed = self.program_embeddings(torch.tensor(prog_id, device=hidden_state.device))
            modulation_params = self.hypernet(prog_embed)  # (hidden * hidden,)

            # Reshape to modulation matrix
            modulation_matrix = modulation_params.view(self.hidden_size, self.hidden_size)

            # Apply modulation
            prog_weight = program_probs[:, prog_id].unsqueeze(-1).unsqueeze(-1)
            modulated = torch.matmul(hidden_state, modulation_matrix) * prog_weight

            outputs.append(modulated)

        # Sum all program contributions
        output_state = sum(outputs)

        # Update usage stats
        usage = program_probs.mean(dim=0)
        self.usage_counts = 0.9 * self.usage_counts + 0.1 * usage.detach()

        return output_state
