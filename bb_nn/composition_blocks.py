"""
Composition blocks for the builder rail.

These blocks learn to compose information from the base rail and previous builder states.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionBlock(nn.Module):
    """
    A single composition block in the builder rail.

    Combines information from previous builder state and base rail state
    using cross-attention mechanisms.
    """

    def __init__(self, hidden_size: int, num_programs: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_programs = num_programs

        # Self-attention for builder state
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )

        # Cross-attention from base rail
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        # Program-aware modulation
        self.program_modulation = nn.Linear(num_programs, hidden_size)

    def forward(
        self,
        builder_state: torch.Tensor,
        base_state: torch.Tensor,
        input_ids: torch.Tensor,
        program_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through composition block.

        Args:
            builder_state: Previous builder hidden state (batch, seq, hidden)
            base_state: Base rail state for this layer (batch, seq, hidden)
            input_ids: Input token IDs for position information
            program_probs: Program selection probabilities (batch, seq, programs)
        """
        batch_size, seq_len, _ = builder_state.shape

        # 1. Self-attention on builder state
        residual = builder_state
        builder_state = self.norm1(builder_state)
        attn_out, _ = self.self_attn(builder_state, builder_state, builder_state)
        builder_state = residual + attn_out

        # 2. Cross-attention from base state
        residual = builder_state
        builder_state = self.norm2(builder_state)
        base_state_norm = self.norm2(base_state)  # Use same norm for cross-attention
        attn_out, _ = self.cross_attn(builder_state, base_state_norm, base_state_norm)
        builder_state = residual + attn_out

        # 3. Program modulation (if program probabilities provided)
        if program_probs is not None:
            # Average program probabilities across sequence for global modulation
            program_modulation = program_probs.mean(dim=1)  # (batch, programs)
            modulation = self.program_modulation(program_modulation)  # (batch, hidden)
            modulation = modulation.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, hidden)

            # Apply modulation
            builder_state = builder_state + modulation

        # 4. Feed-forward network
        residual = builder_state
        builder_state = self.norm3(builder_state)
        ff_out = self.feed_forward(builder_state)
        builder_state = residual + ff_out

        return builder_state


class SSMCompositionBlock(nn.Module):
    """
    Composition block using State Space Models (SSM) for efficient long-range modeling.

    Alternative to attention-based composition for better efficiency.
    """

    def __init__(self, hidden_size: int, num_programs: int, state_size: int = 64):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_size = state_size

        # SSM components
        self.A = nn.Parameter(torch.randn(state_size, state_size))
        self.B = nn.Parameter(torch.randn(hidden_size, state_size))
        self.C = nn.Parameter(torch.randn(state_size, hidden_size))

        # Input and output projections
        self.input_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # Cross-attention to base state
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

        # Program modulation
        self.program_modulation = nn.Linear(num_programs, hidden_size)

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        builder_state: torch.Tensor,
        base_state: torch.Tensor,
        input_ids: torch.Tensor,
        program_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through SSM composition block."""
        batch_size, seq_len, _ = builder_state.shape

        # 1. Cross-attention from base state
        residual = builder_state
        builder_state = self.norm1(builder_state)
        base_state_norm = self.norm1(base_state)
        attn_out, _ = self.cross_attn(builder_state, base_state_norm, base_state_norm)
        builder_state = residual + attn_out

        # 2. SSM processing
        residual = builder_state
        builder_state = self.norm2(builder_state)

        # Apply SSM
        # For simplicity, using a basic SSM formulation
        # In practice, would use more sophisticated SSM like S4 or H3
        input_proj = self.input_proj(builder_state)

        # Simple recurrent computation
        states = []
        state = torch.zeros(batch_size, self.state_size, device=builder_state.device)

        for t in range(seq_len):
            # State transition
            state = F.silu(self.A @ state.unsqueeze(-1)).squeeze(-1)

            # Input injection
            input_t = input_proj[:, t, :]
            state = state + self.B.T @ input_t.unsqueeze(-1).squeeze(-1)

            # Output projection
            output_t = self.C @ state.unsqueeze(-1).squeeze(-1)
            states.append(output_t)

        ssm_out = torch.stack(states, dim=1)
        builder_state = residual + ssm_out

        # 3. Program modulation
        if program_probs is not None:
            program_modulation = program_probs.mean(dim=1)
            modulation = self.program_modulation(program_modulation)
            modulation = modulation.unsqueeze(1).expand(-1, seq_len, -1)
            builder_state = builder_state + modulation

        return builder_state
