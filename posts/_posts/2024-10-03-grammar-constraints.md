---
layout: post
title: "Grammar Constraints: Making AI Output Structured and Reliable"
date: 2024-10-03
categories: ai ml nlp grammars
excerpt: "How BuilderBrain uses formal grammars to guarantee structured outputs like JSON, while maintaining the flexibility of neural generation."
---

## The Problem with Unstructured AI Output

Most language models generate free-form text, which is great for creative writing but problematic when you need:

- **Structured data** (JSON, XML, code)
- **Consistent formatting** (API responses, forms)
- **Safety guarantees** (no malformed outputs)
- **Domain compliance** (specific formats for finance, healthcare, etc.)

## Grammar Constraints to the Rescue

BuilderBrain uses **formal grammars** to constrain generation while maintaining neural flexibility:

```python
# Traditional generation - anything goes
model.generate("Generate JSON")  # Could produce invalid JSON

# Grammar-constrained generation - guarantees valid structure
grammar = JSONGrammar()
mask = GrammarMask(grammar, tokenizer, strict=True)
constrained_logits = mask(logits, prefix)
# Only valid tokens allowed!
```

## How Grammar Constraints Work

### 1. Formal Grammar Definition

We define what "valid" output looks like using **Context-Free Grammars (CFG)**:

```ebnf
value    <- object / array / string / number / boolean / null
object   <- "{" pair ("," pair)* "}"
pair     <- string ":" value
array    <- "[" value ("," value)* "]"
string   <- "\"" char* "\""
number   <- "-"? digit+ ("." digit+)? (exp)?
```

### 2. Real-time Token Masking

At each generation step, the grammar parser determines which tokens are valid:

```python
def next_valid_tokens(self, prefix: str) -> Set[int]:
    """Get token IDs that can legally follow the current prefix."""
    # Parse prefix to determine current state
    # Return only tokens that maintain grammar validity
    return valid_token_ids
```

### 3. Two Constraint Modes

**Hard Constraints (Strict Domains):**
- API responses, code generation, financial data
- Invalid tokens get `-∞` logits
- Guarantees 100% compliance

**Soft Constraints (Flexible Domains):**
- Creative writing, chat, social content
- Invalid tokens get energy penalty
- Allows creativity while encouraging structure

## Code Example: JSON Generation

```python
from bb_priors.cfg_parser import JSONGrammar
from bb_priors.token_masks import GrammarMask
from transformers import GPT2Tokenizer

# Load grammar and tokenizer
grammar = JSONGrammar()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Create grammar mask
mask = GrammarMask(grammar, tokenizer, strict=True)

# Generate with constraints
def generate_constrained_json(prompt: str) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    for _ in range(100):  # Max length
        # Get model logits
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

        # Apply grammar constraints
        prefix = tokenizer.decode(input_ids[0])
        constrained_logits = mask(logits, prefix)

        # Sample next token
        next_token = torch.multinomial(F.softmax(constrained_logits, dim=-1), 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop if we generated closing brace
        if next_token.item() == tokenizer.encode('}')[0]:
            break

    return tokenizer.decode(input_ids[0])
```

## Grammar Energy: Soft Constraints

For flexible domains, we use **grammar energy** instead of hard masking:

```python
energy = grammar.sequence_energy(tokens)
loss = max(0, energy - target_energy)  # Hinge loss
```

This encourages compliance while allowing some flexibility.

## Real-World Applications

### API Agents
```json
{
  "action": "create_user",
  "params": {
    "email": "user@example.com",
    "role": "customer"
  },
  "on_error": "retry"
}
```

### Code Generation
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### Business Logic
```yaml
workflow:
  steps:
    - validate_input
    - process_payment
    - send_confirmation
  on_failure: escalate_to_human
```

## Training with Grammar Constraints

BuilderBrain trains with grammar constraints as part of the multi-objective optimization:

```python
# Grammar constraint in the loss
grammar_loss = GrammarLoss(grammar_energy, target=0.0)
total_loss = task_loss + lambda_grammar * grammar_loss
```

The model learns to generate structured outputs naturally.

## Benefits of Grammar Constraints

1. **Reliability**: Structured outputs are always well-formed
2. **Safety**: Prevents malformed or harmful outputs
3. **Interoperability**: Consistent formats for APIs and tools
4. **Debugging**: Clear structure makes issues easier to identify
5. **Domain Compliance**: Meets specific formatting requirements

## Challenges and Solutions

**Challenge**: Grammar constraints can limit creativity
**Solution**: Use soft constraints for creative domains, hard constraints for structured ones

**Challenge**: Complex grammars slow down generation
**Solution**: Pre-compute valid token sets, use efficient parsing

**Challenge**: Grammar maintenance
**Solution**: Versioned grammars with automated testing

## Next Steps

In the next post, we'll explore [plan execution](/ai/ml/robotics/planning/2024/10/04/plan-execution/) - how BuilderBrain turns generated plans into actual actions.

---

*Grammar constraints represent the bridge between neural creativity and formal reliability. They ensure AI outputs are not just fluent, but also structured, safe, and useful.*
