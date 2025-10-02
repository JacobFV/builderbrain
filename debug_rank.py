#!/usr/bin/env python3

import numpy as np
from bb_core.math_utils import RankNormalizer

# Test rank calculation
normalizer = RankNormalizer(window_size=10)

# Add values [1.0, 3.0, 5.0, 7.0, 9.0]
values = [1.0, 3.0, 5.0, 7.0, 9.0]
for val in values:
    result = normalizer(np.array([val]))
    print(f"Added {val}, history length: {len(normalizer.history)}, history: {normalizer.history}")

print("\nTesting normalization of 5.0:")
result = normalizer(np.array([5.0]))
print(f"Result: {result}")

# Check rank calculation manually
history = normalizer.history
rank = np.sum(np.array(history) <= 5.0) / len(history)
expected = 2 * rank - 1
print(f"Manual calculation: rank = {rank}, expected = {expected}")
print(f"History: {history}")
print(f"Values <= 5.0: {np.sum(np.array(history) <= 5.0)}")

