# GRAMMAR_GUIDE.md — builderbrain

## 0. purpose

Builderbrain requires **formal grammars** to bias the builder rail toward *constructible, compositional languages*. Grammars:

* constrain decoding (hard masks for strict domains).
* provide energy signals (soft penalties for flexible domains).
* supply golden tests for regression safety.

We use CFG/PEG formalisms, compiled to parsers (Earley/GLR/PEG libraries) + token masks.

---

## 1. authoring a grammar

### 1.1 choose the formalism

* **CFG (context-free grammar):** sufficient for JSON, DSLs, macro languages.
* **PEG (parsing expression grammar):** deterministic, handles ordered choices (preferred for practical APIs).

### 1.2 style rules

* Keep grammars **minimal**: only express what is necessary.
* Prefer **explicit tokens** (literals, keywords) over regex-like fuzz.
* Factor optional elements explicitly (use `?`, `*`, `+`).
* Encode **types** (numbers, strings) with precise ranges if possible.

### 1.3 file structure

```
bb_domains/<domain>/grammar.peg
```

* Each domain maintains its own grammar file.
* Versioned; changes must pass golden tests.

### 1.4 example: API JSON

```peg
value    <- object / array / string / number / 'true' / 'false' / 'null'
object   <- '{' pair (',' pair)* '}'
pair     <- string ':' value
array    <- '[' value (',' value)* ']'
string   <- '"' (!['"\\] / escape)* '"'
number   <- '-'? [0-9]+ ('.' [0-9]+)? ([eE][+-]?[0-9]+)?
escape   <- '\\' ['"/bfnrt]
```

---

## 2. golden tests

### 2.1 valid/invalid strings

* **valid set:** minimal canonical examples + complex edge cases.
* **invalid set:** malformed syntax (missing commas, mismatched brackets, bad escapes).

### 2.2 file layout

```
bb_domains/<domain>/tests/grammar_valid.txt
bb_domains/<domain>/tests/grammar_invalid.txt
```

### 2.3 test harness

* Parse every valid → must succeed.
* Parse every invalid → must fail.
* Regression test runs in CI.

---

## 3. decode masking strategies

### 3.1 strict channels (api/json, robot DSL)

* At each decode step, parser returns valid next tokens.
* Invalid tokens masked to (-\infty) in logits.
* Guarantees 100% syntactic validity.

### 3.2 semi-strict (phone callflow tags, ui macros)

* Hard mask structural tokens, soft energy on content.
* e.g., enforce `<tag> ... </tag>` structure, but allow flexible utterance text inside.

### 3.3 soft-only (chat/social)

* No hard mask; compute syntax energy:
  [
  E_{cfs}(x_{1:t}) = -\log P_{CFG}(x_{1:t})
  ]
* Add as auxiliary loss with dual target.

---

## 4. soft-energy calibration

### 4.1 definition

Soft energy is used as a *constraint*, not a hard ban. Normalize via rank/winsor (§ MATH_SPEC). Target slack (c_{cfs}) sets tolerance.

### 4.2 annealing

* Start with high tolerance → allow exploration.
* Gradually tighten → enforce more grammaticality.
* Use dual ascent to regulate automatically.

### 4.3 logging

* Always log `cfs_violation_rate` (fraction of tokens outside grammar).
* Dashboard: rolling average, covariance with other losses.

---

## 5. extending grammars

### 5.1 workflow

1. Edit `grammar.peg` in domain directory.
2. Add new examples to valid/invalid test files.
3. Run `pytest tests/` → all must pass.
4. Commit with message `grammar:<domain>: describe change`.

### 5.2 backwards compatibility

* If old valid examples break, bump grammar version.
* Migration guide required if runtime clients depend on old schema.

### 5.3 complexity budget

* Keep parsing time linear in input length.
* Avoid deep recursion unless bounded.
* For DSLs: prefer explicit repetition bounds.

---

## 6. runtime interfaces

### 6.1 parser API

```python
from bb_priors.cfg import GrammarMask

mask = GrammarMask("bb_domains/api_json/grammar.peg")
valid_tokens = mask.next_valid(prefix)
logits[~valid_tokens] = -inf
```

### 6.2 energy API

```python
from bb_priors.cfg import GrammarEnergy

E = GrammarEnergy("bb_domains/chat/grammar.peg")
energy = E.sequence_energy(tokens)
```

---

## 7. diagnostics + failure smells

* **mask collapse:** if no valid tokens available → grammar too tight.
* **false positives:** if invalid strings parse → grammar too loose.
* **coverage gaps:** if valid user traces fail parse → grammar incomplete.
* **cfs_violation ↑ while ged ↓:** model gaming losses; audit covariance.

---

## 8. ethos

a grammar is a *contract*. write it minimal, test it brutally, enforce it ruthlessly where correctness matters. let soft energy nudge where exploration is valuable. grammars are not vibes; they are rails.
