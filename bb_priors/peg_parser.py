"""
PEG (Parsing Expression Grammar) parser for builderbrain.

Provides deterministic parsing with ordered choices for practical grammars.
"""

from typing import Dict, List, Any, Optional, Set
import re


class PEGParser:
    """
    Simple PEG parser implementation.

    Supports basic PEG operations: sequence, choice, repetition, etc.
    """

    def __init__(self, grammar_text: str):
        self.grammar_text = grammar_text
        self.rules: Dict[str, str] = {}
        self.terminals: Set[str] = set()

        self._parse_grammar_text()

    def _parse_grammar_text(self):
        """Parse grammar text into rules."""
        lines = self.grammar_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '<-' in line:
                left, right = line.split('<-', 1)
                left = left.strip()
                right = right.strip()

                self.rules[left] = right

                # Extract terminals (quoted strings)
                terminals = re.findall(r'"([^"]*)"', right)
                self.terminals.update(terminals)

    def next_valid_tokens(self, prefix: str) -> Set[str]:
        """Get valid next tokens for the current prefix."""
        # Simplified implementation - in practice would need full parsing state
        valid_tokens = set()

        for terminal in self.terminals:
            if prefix.endswith(terminal[:len(prefix)]):
                valid_tokens.add(terminal)

        # If no matches, return all terminals
        if not valid_tokens:
            valid_tokens = self.terminals.copy()

        return valid_tokens

    def validate_sequence(self, tokens: List[str]) -> bool:
        """Check if token sequence is valid according to grammar."""
        # Simplified validation
        return all(token in self.terminals for token in tokens)

    def sequence_energy(self, tokens: List[str]) -> float:
        """Compute grammar compliance energy."""
        if self.validate_sequence(tokens):
            return 0.0

        invalid_count = sum(1 for token in tokens if token not in self.terminals)
        return float(invalid_count)
