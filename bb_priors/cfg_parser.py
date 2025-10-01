"""
Context-Free Grammar (CFG) parser for builderbrain.

Provides parsing and token masking capabilities for CFG grammars.
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class GrammarRule:
    """Represents a single grammar rule."""
    left: str  # Left-hand side non-terminal
    right: List[str]  # Right-hand side symbols (terminals or non-terminals)


@dataclass
class ParseTree:
    """Represents a parse tree node."""
    symbol: str
    children: List['ParseTree'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class CFGParser:
    """
    Simple top-down CFG parser with token masking capabilities.

    Supports basic CFG parsing for token-level constraints in generation.
    """

    def __init__(self, grammar_text: str):
        self.grammar_text = grammar_text
        self.rules: Dict[str, List[GrammarRule]] = {}
        self.terminals: Set[str] = set()
        self.non_terminals: Set[str] = set()
        self.start_symbol = None

        self._parse_grammar_text()
        self._compute_first_sets()
        self._compute_follow_sets()

    def _parse_grammar_text(self):
        """Parse grammar text into rules."""
        lines = self.grammar_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '->' in line:
                left, right = line.split('->', 1)
                left = left.strip()
                right = right.strip()

                if self.start_symbol is None:
                    self.start_symbol = left

                self.non_terminals.add(left)

                # Parse right-hand side
                alternatives = [alt.strip() for alt in right.split('|')]

                if left not in self.rules:
                    self.rules[left] = []

                for alt in alternatives:
                    symbols = alt.split()
                    rule = GrammarRule(left, symbols)

                    # Identify terminals vs non-terminals
                    for symbol in symbols:
                        if symbol.startswith('"') and symbol.endswith('"'):
                            # Terminal (quoted string)
                            self.terminals.add(symbol[1:-1])
                        elif symbol.isupper() or '_' in symbol:
                            # Non-terminal (uppercase or with underscore)
                            self.non_terminals.add(symbol)

                    self.rules[left].append(rule)

    def _compute_first_sets(self):
        """Compute FIRST sets for all non-terminals."""
        self.first_sets: Dict[str, Set[str]] = {}

        # Initialize with terminals
        for terminal in self.terminals:
            self.first_sets[terminal] = {terminal}

        # Initialize empty for non-terminals
        for nt in self.non_terminals:
            self.first_sets[nt] = set()

        # Iteratively compute FIRST sets
        changed = True
        while changed:
            changed = False

            for nt in self.non_terminals:
                if nt not in self.rules:
                    continue

                for rule in self.rules[nt]:
                    for symbol in rule.right:
                        if symbol in self.terminals:
                            if symbol not in self.first_sets[nt]:
                                self.first_sets[nt].add(symbol)
                                changed = True
                            break  # First terminal found
                        elif symbol in self.non_terminals:
                            # Add all from non-terminal's FIRST set
                            before = len(self.first_sets[nt])
                            self.first_sets[nt].update(self.first_sets[symbol])
                            if len(self.first_sets[nt]) > before:
                                changed = True
                            if '' not in self.first_sets[symbol]:  # No epsilon
                                break

    def _compute_follow_sets(self):
        """Compute FOLLOW sets for all non-terminals."""
        self.follow_sets: Dict[str, Set[str]] = {}

        for nt in self.non_terminals:
            self.follow_sets[nt] = set()

        # Add end marker to start symbol
        if self.start_symbol:
            self.follow_sets[self.start_symbol].add('$')

        # Iteratively compute FOLLOW sets
        changed = True
        while changed:
            changed = False

            for nt in self.non_terminals:
                if nt not in self.rules:
                    continue

                for rule in self.rules[nt]:
                    for i, symbol in enumerate(rule.right):
                        if symbol not in self.non_terminals:
                            continue

                        # Add FOLLOW from symbols after current symbol
                        follow_set = set()
                        for j in range(i + 1, len(rule.right)):
                            next_symbol = rule.right[j]
                            if next_symbol in self.terminals:
                                follow_set.add(next_symbol)
                                break
                            elif next_symbol in self.non_terminals:
                                follow_set.update(self.first_sets[next_symbol])
                                if '' in self.first_sets[next_symbol]:
                                    continue  # Look further
                                else:
                                    break

                        if not follow_set and i == len(rule.right) - 1:
                            # Last symbol in rule, add FOLLOW from left-hand side
                            follow_set.update(self.follow_sets[nt])

                        # Remove epsilon if present
                        follow_set.discard('')

                        before = len(self.follow_sets[symbol])
                        self.follow_sets[symbol].update(follow_set)
                        if len(self.follow_sets[symbol]) > before:
                            changed = True

    def next_valid_tokens(self, prefix: str) -> Set[int]:
        """
        Get set of valid next token IDs that can follow the current prefix.

        This is used for token masking during generation.
        """
        # For simplicity, return all terminals for now
        # In a full implementation, would parse the prefix and determine
        # the current parsing state

        # This is a simplified version - in practice would need to:
        # 1. Parse the prefix to determine current state
        # 2. Determine possible next symbols
        # 3. Map symbols to token IDs

        return self.terminals.copy()

    def validate_sequence(self, tokens: List[str]) -> bool:
        """Check if token sequence is valid according to grammar."""
        try:
            # Simple validation - check if we can parse the sequence
            # In practice, would use a full parser
            return self._can_parse(tokens)
        except:
            return False

    def _can_parse(self, tokens: List[str]) -> bool:
        """Check if sequence can be parsed (simplified implementation)."""
        # This is a very basic implementation
        # A full implementation would use dynamic programming or recursive descent

        if not tokens:
            return True

        # For now, just check if all tokens are valid terminals
        return all(token in self.terminals for token in tokens)

    def sequence_energy(self, tokens: List[str]) -> float:
        """
        Compute grammar compliance energy for sequence.

        Lower energy = more compliant with grammar.
        """
        if self.validate_sequence(tokens):
            return 0.0

        # Simple energy based on invalid token count
        invalid_count = sum(1 for token in tokens if token not in self.terminals)
        return float(invalid_count)

    def get_terminals(self) -> Set[str]:
        """Get all terminal symbols."""
        return self.terminals.copy()

    def get_non_terminals(self) -> Set[str]:
        """Get all non-terminal symbols."""
        return self.non_terminals.copy()

    def export_grammar(self) -> Dict[str, Any]:
        """Export grammar for serialization."""
        return {
            "start_symbol": self.start_symbol,
            "terminals": list(self.terminals),
            "non_terminals": list(self.non_terminals),
            "rules": {
                left: [
                    {"right": rule.right}
                    for rule in rules
                ]
                for left, rules in self.rules.items()
            },
            "first_sets": {k: list(v) for k, v in self.first_sets.items()},
            "follow_sets": {k: list(v) for k, v in self.follow_sets.items()}
        }


class JSONGrammar(CFGParser):
    """Specialized CFG for JSON parsing."""

    def __init__(self):
        json_grammar = '''
        value -> object | array | string | number | "true" | "false" | "null"
        object -> "{" pair ("," pair)* "}"
        pair -> string ":" value
        array -> "[" value ("," value)* "]"
        string -> "\"" (char)* "\""
        number -> "-"? digit+ ("." digit+)? (exp)?
        exp -> ("e"|"E") ("+"|"-")? digit+
        char -> [^"\\\\] | "\\\\" escape
        escape -> ["\\/bfnrt] | "u" hex hex hex hex
        digit -> [0-9]
        hex -> [0-9a-fA-F]
        '''

        super().__init__(json_grammar)

    def next_valid_tokens(self, prefix: str) -> Set[str]:
        """Get valid next tokens for JSON parsing."""
        # Simple state-based token prediction for JSON
        if not prefix.strip():
            return {"{", "[", '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "t", "f", "n"}

        # Basic JSON structure awareness
        tokens = prefix.split()
        if not tokens:
            return self.next_valid_tokens("")

        last_token = tokens[-1]

        if last_token == "{":
            return {'"', '}'}  # Start of object
        elif last_token == '"':
            return {':'}  # Key-value separator
        elif last_token == ':':
            return {"{", "[", '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "t", "f", "n"}
        elif last_token == ',':
            return {'"', '{', '['}  # After comma in object/array
        elif last_token == '[':
            return {"{", "[", '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "t", "f", "n", "]"}
        elif last_token == ']':
            return {',', '}', ']'}  # End of array
        elif last_token == '}':
            return {',', '}', '$'}  # End of object
        else:
            return self.terminals
