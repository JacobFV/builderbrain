"""
Runtime decoder for grammar-constrained generation.

Handles syntax-aware decoding with token masking and plan checking.
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F
from bb_priors.token_masks import GrammarMask


class RuntimeDecoder:
    """
    Runtime decoder that applies grammar constraints during generation.

    Supports both hard masking (strict domains) and soft energy (flexible domains).
    """

    def __init__(self, model, tokenizer, grammar_mask: GrammarMask):
        self.model = model
        self.tokenizer = tokenizer
        self.grammar_mask = grammar_mask

    def generate_with_constraints(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        use_hard_mask: bool = True
    ) -> str:
        """
        Generate text with grammar constraints.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            use_hard_mask: Whether to use hard masking

        Returns:
            Generated text respecting grammar constraints
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        generated = input_ids.clone()

        for _ in range(max_length):
            # Get model logits
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs['logits'][:, -1, :]  # Last token logits

            # Apply grammar constraints
            prefix = self.tokenizer.decode(generated[0])
            constrained_logits = self.grammar_mask(
                logits,
                prefix,
                use_hard_mask=use_hard_mask
            )

            # Sample next token
            if temperature == 0:
                # Greedy decoding
                next_token = constrained_logits.argmax(dim=-1, keepdim=True)
            else:
                # Temperature sampling
                constrained_logits = constrained_logits / temperature
                probabilities = F.softmax(constrained_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode final sequence
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def beam_search_with_constraints(
        self,
        prompt: str,
        num_beams: int = 5,
        max_length: int = 100
    ) -> str:
        """
        Beam search with grammar constraints.

        Args:
            prompt: Input prompt
            num_beams: Number of beams
            max_length: Maximum generation length

        Returns:
            Best generated sequence
        """
        # Simplified beam search implementation
        # In practice would implement full beam search with grammar constraints

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # For demo purposes, just use greedy decoding
        return self.generate_with_constraints(prompt, max_length, temperature=0, use_hard_mask=True)

    def validate_generated_sequence(self, text: str) -> Dict[str, Any]:
        """
        Validate that generated sequence complies with grammar.

        Returns validation results with any violations found.
        """
        tokens = text.split()

        # Check grammar compliance
        is_valid = self.grammar_mask.grammar.validate_sequence(tokens)

        violations = []
        if not is_valid:
            # In practice would provide detailed violation information
            violations.append("Grammar violation detected")

        return {
            'valid': is_valid,
            'violations': violations,
            'energy': self.grammar_mask.grammar.sequence_energy(tokens)
        }
