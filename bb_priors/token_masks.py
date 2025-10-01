"""
Token masking system for grammar-constrained generation.

Applies hard masks for strict domains and soft penalties for flexible domains.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
import torch
import torch.nn.functional as F
from .cfg_parser import CFGParser, GrammarRule


class GrammarMask:
    """
    Applies grammar-based token masking for constrained generation.

    Supports both hard masking (strict domains) and soft energy (flexible domains).
    """

    def __init__(self, grammar: CFGParser, tokenizer, strict: bool = False):
        self.grammar = grammar
        self.tokenizer = tokenizer
        self.strict = strict

        # Build mapping from symbols to token IDs
        self.symbol_to_tokens: Dict[str, Set[int]] = {}
        self._build_symbol_mapping()

    def _build_symbol_mapping(self):
        """Build mapping from grammar symbols to tokenizer token IDs."""
        # Map terminals to token IDs
        for terminal in self.grammar.get_terminals():
            token_ids = set()

            # For quoted terminals, try to find exact matches
            if terminal.startswith('"') and terminal.endswith('"'):
                unquoted = terminal[1:-1]
                token_id = self.tokenizer.encode(unquoted)
                if token_id:
                    token_ids.update(token_id)
            else:
                # For other terminals, find tokens that contain or match the symbol
                # This is a simplified approach - in practice would need more sophisticated matching
                tokens = self.tokenizer.get_vocab()
                for token, token_id in tokens.items():
                    if terminal.lower() in token.lower():
                        token_ids.add(token_id)

            if token_ids:
                self.symbol_to_tokens[terminal] = token_ids

        # Map non-terminals (for lookahead)
        for non_terminal in self.grammar.get_non_terminals():
            self.symbol_to_tokens[non_terminal] = set()

    def next_valid_tokens(self, prefix: str) -> Set[int]:
        """
        Get set of valid token IDs that can follow the current prefix.

        Args:
            prefix: Current generated text prefix

        Returns:
            Set of valid token IDs
        """
        # Get valid symbols from grammar
        valid_symbols = self.grammar.next_valid_tokens(prefix)

        # Map symbols to token IDs
        valid_tokens = set()
        for symbol in valid_symbols:
            if symbol in self.symbol_to_tokens:
                valid_tokens.update(self.symbol_to_tokens[symbol])

        # If no specific tokens found, allow common structural tokens
        if not valid_tokens:
            valid_tokens.update(self._get_fallback_tokens())

        return valid_tokens

    def _get_fallback_tokens(self) -> Set[int]:
        """Get fallback tokens when grammar doesn't specify."""
        fallback = [',', ':', '{', '}', '[', ']', '"', ' ']
        token_ids = set()

        for token in fallback:
            token_id = self.tokenizer.encode(token)
            if token_id:
                token_ids.update(token_id)

        return token_ids

    def apply_hard_mask(self, logits: torch.Tensor, prefix: str) -> torch.Tensor:
        """
        Apply hard mask to logits for strict grammar compliance.

        Args:
            logits: Model logits (batch, vocab_size)
            prefix: Current generation prefix

        Returns:
            Masked logits
        """
        valid_tokens = self.next_valid_tokens(prefix)

        # Create mask
        mask = torch.full_like(logits, float('-inf'))
        for token_id in valid_tokens:
            if token_id < logits.size(-1):
                mask[:, token_id] = logits[:, token_id]

        return mask

    def compute_soft_energy(self, tokens: List[int], prefix: str) -> torch.Tensor:
        """
        Compute soft grammar compliance energy.

        Args:
            tokens: Token sequence to evaluate
            prefix: Current prefix (for context)

        Returns:
            Energy tensor (lower = more compliant)
        """
        # Decode tokens to text for grammar evaluation
        try:
            text = self.tokenizer.decode(tokens)
            energy = self.grammar.sequence_energy([text])

            return torch.tensor(float(energy), device=logits.device if 'logits' in locals() else 'cpu')

        except Exception:
            # If decoding fails, return high energy
            return torch.tensor(100.0)

    def __call__(
        self,
        logits: torch.Tensor,
        prefix: str,
        use_hard_mask: bool = None
    ) -> torch.Tensor:
        """
        Apply grammar constraints to logits.

        Args:
            logits: Model logits
            prefix: Current generation prefix
            use_hard_mask: Whether to use hard masking (overrides self.strict)

        Returns:
            Constrained logits
        """
        if use_hard_mask is None:
            use_hard_mask = self.strict

        if use_hard_mask:
            return self.apply_hard_mask(logits, prefix)
        else:
            # For soft constraints, return original logits
            # (energy would be computed separately as auxiliary loss)
            return logits


class MultiGrammarMask:
    """
    Manages multiple grammars for different domains/contexts.

    Allows switching between different grammar constraints based on context.
    """

    def __init__(self, grammars: Dict[str, GrammarMask]):
        self.grammars = grammars
        self.active_grammar = None

    def set_active_grammar(self, grammar_name: str):
        """Set the active grammar for masking."""
        if grammar_name in self.grammars:
            self.active_grammar = self.grammars[grammar_name]
        else:
            raise ValueError(f"Unknown grammar: {grammar_name}")

    def add_grammar(self, name: str, grammar: GrammarMask):
        """Add a new grammar."""
        self.grammars[name] = grammar

    def next_valid_tokens(self, prefix: str) -> Set[int]:
        """Get valid tokens from active grammar."""
        if self.active_grammar is None:
            return set(range(1000))  # Fallback: allow common tokens

        return self.active_grammar.next_valid_tokens(prefix)

    def apply_mask(self, logits: torch.Tensor, prefix: str, use_hard_mask: bool = None) -> torch.Tensor:
        """Apply active grammar constraints."""
        if self.active_grammar is None:
            return logits

        return self.active_grammar(logits, prefix, use_hard_mask)

    def compute_energy(self, tokens: List[int], prefix: str) -> torch.Tensor:
        """Compute grammar energy from active grammar."""
        if self.active_grammar is None:
            return torch.tensor(0.0)

        return self.active_grammar.compute_soft_energy(tokens, prefix)


class AdaptiveMask:
    """
    Adaptive masking that adjusts strictness based on generation context.

    Can switch between hard and soft constraints dynamically.
    """

    def __init__(self, grammar: GrammarMask, adaptation_rate: float = 0.1):
        self.grammar = grammar
        self.adaptation_rate = adaptation_rate

        # Track violation statistics
        self.violation_history = []
        self.strictness_level = 1.0  # 0.0 = soft, 1.0 = hard

    def update_strictness(self, violation_rate: float):
        """Update strictness based on recent violation rate."""
        # Increase strictness if violations are high
        if violation_rate > 0.1:  # 10% violation threshold
            self.strictness_level = min(1.0, self.strictness_level + self.adaptation_rate)
        # Decrease strictness if violations are low (allow more exploration)
        elif violation_rate < 0.01:  # 1% violation threshold
            self.strictness_level = max(0.0, self.strictness_level - self.adaptation_rate)

    def __call__(self, logits: torch.Tensor, prefix: str) -> torch.Tensor:
        """Apply adaptive masking."""
        use_hard_mask = self.strictness_level > 0.5
        return self.grammar(logits, prefix, use_hard_mask)
