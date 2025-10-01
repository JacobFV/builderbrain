"""
Dual-rail neural network architecture for builderbrain.

Combines frozen base transformer with learned composition blocks.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .composition_blocks import CompositionBlock
from .program_adapters import ProgramAdapter
from .fusion_gates import FusionGates


class DualRail(nn.Module):
    """
    Dual-rail architecture: frozen base transformer + learned composition builder.

    Args:
        base_model: Frozen pretrained transformer (e.g., GPT-2, LLaMA)
        hidden_size: Hidden dimension size
        num_layers: Number of composition layers
        num_programs: Number of discrete program skills
        alpha_cap: Maximum gating value for builder rail
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_layers: int = 6,
        num_programs: int = 32,
        alpha_cap: float = 0.1
    ):
        super().__init__()

        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_programs = num_programs
        self.alpha_cap = alpha_cap

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Builder rail: stack of composition blocks
        self.builder_layers = nn.ModuleList([
            CompositionBlock(hidden_size, num_programs)
            for _ in range(num_layers)
        ])

        # Program adapters for skill selection
        self.program_adapters = ProgramAdapter(hidden_size, num_programs)

        # Fusion gates to control base vs builder influence
        self.fusion_gates = FusionGates(hidden_size, num_layers, alpha_cap)

        # Final language model head (shared)
        vocab_size = getattr(base_model, 'vocab_size', 50257)  # Default GPT-2 vocab size
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Gates are initialized to favor base rail initially
        for gate in self.fusion_gates.gates:
            # Initialize the Linear layer within the Sequential
            if hasattr(gate, 'weight'):
                nn.init.constant_(gate.weight, -2.0)  # Bias toward alpha = 0

        # LM head initialization
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def forward_base_rail(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through frozen base rail."""
        # Get hidden states from each layer of base model
        hidden_states = []

        def hook_fn(module, input, output):
            hidden_states.append(output)

        # Register hooks to capture hidden states
        hooks = []
        for layer in self.base_model.transformer.h:  # Assuming GPT-style architecture
            hooks.append(layer.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            _ = self.base_model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return hidden_states

    def forward_builder_rail(
        self,
        base_states: List[torch.Tensor],
        input_ids: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through builder rail."""
        batch_size, seq_len = input_ids.shape

        # Initialize builder hidden states
        builder_states = [torch.zeros_like(base_states[0])]

        # Program selection logits (will be used for discrete sampling)
        program_logits = []

        for layer_idx in range(self.num_layers):
            # Get base state for this layer
            base_state = base_states[layer_idx]

            # Composition block
            builder_state = self.builder_layers[layer_idx](
                builder_states[-1],
                base_state,
                input_ids
            )

            # Apply program adapter
            # Sample program for each position (discrete via ST-Gumbel)
            adapter_logits = self.program_adapters.get_logits(builder_state)
            program_logits.append(adapter_logits)

            # For now, use soft program selection
            # In training, we'll use ST-Gumbel for discrete selection
            program_probs = F.softmax(adapter_logits, dim=-1)
            program_selection = program_probs  # Shape: (batch, seq, num_programs)

            # Apply program adapters
            adapted_state = self.program_adapters(builder_state, program_selection)
            builder_states.append(adapted_state)

        # Stack program logits for loss computation
        program_logits = torch.stack(program_logits, dim=1)  # (batch, layers, seq, programs)

        return builder_states, program_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Complete forward pass through dual-rail system."""
        batch_size, seq_len = input_ids.shape

        # Base rail forward pass (frozen)
        base_states = self.forward_base_rail(input_ids)

        # Builder rail forward pass
        builder_states, program_logits = self.forward_builder_rail(base_states, input_ids)

        # Apply fusion gates
        fused_states = []
        alpha_values = []

        for layer_idx in range(self.num_layers):
            base_state = base_states[layer_idx]
            builder_state = builder_states[layer_idx + 1]  # Offset for initial state

            alpha = self.fusion_gates(layer_idx, base_state, builder_state)
            alpha_values.append(alpha)

            # Fuse: alpha * builder + (1 - alpha) * base
            fused = alpha.unsqueeze(-1) * builder_state + (1 - alpha.unsqueeze(-1)) * base_state
            fused_states.append(fused)

        # Use final fused state for LM prediction
        final_state = fused_states[-1]

        # Language model logits
        logits = self.lm_head(final_state)

        return {
            "logits": logits,
            "program_logits": program_logits,
            "alpha_values": torch.stack(alpha_values, dim=1),
            "base_states": base_states,
            "builder_states": builder_states,
            "fused_states": fused_states
        }

    def get_base_states(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Get base rail hidden states for external use."""
        return self.forward_base_rail(input_ids)

    def enable_training_mode(self, train_builder: bool = True, train_gates: bool = True):
        """Control which parts of the model are trainable."""
        # Base model always frozen
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Builder layers
        for layer in self.builder_layers:
            for param in layer.parameters():
                param.requires_grad = train_builder

        # Program adapters
        for param in self.program_adapters.parameters():
            param.requires_grad = train_builder

        # Fusion gates
        for param in self.fusion_gates.parameters():
            param.requires_grad = train_gates

        # LM head (typically shared and trainable)
        for param in self.lm_head.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> List[torch.Tensor]:
        """Get list of trainable parameters."""
        params = []

        # Builder layers
        for layer in self.builder_layers:
            params.extend([p for p in layer.parameters() if p.requires_grad])

        # Program adapters
        params.extend([p for p in self.program_adapters.parameters() if p.requires_grad])

        # Fusion gates
        params.extend([p for p in self.fusion_gates.parameters() if p.requires_grad])

        # LM head
        params.extend([p for p in self.lm_head.parameters() if p.requires_grad])

        return params

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "base_model": sum(p.numel() for p in self.base_model.parameters()),
            "builder_layers": sum(p.numel() for p in self.builder_layers.parameters()),
            "program_adapters": sum(p.numel() for p in self.program_adapters.parameters()),
            "fusion_gates": sum(p.numel() for p in self.fusion_gates.parameters()),
            "lm_head": sum(p.numel() for p in self.lm_head.parameters())
        }
