#!/usr/bin/env python3

import numpy as np
from bb_core.math_utils import RankNormalizer

# Test rank calculation step by step
normalizer = RankNormalizer(window_size=10)

print("Step 1: Single value")
result = normalizer(np.array([5.0]))
print(f"Result: {result}, History: {normalizer.history}")

print("\nStep 2: Add values [1.0, 3.0, 5.0, 7.0, 9.0]")
values = [1.0, 3.0, 5.0, 7.0, 9.0]
for val in values:
    result = normalizer(np.array([val]))
    print(f"Added {val}, Result: {result}, History length: {len(normalizer.history)}")

print(f"\nFinal history: {normalizer.history}")

print("\nStep 3: Normalize 5.0")
result = normalizer(np.array([5.0]))
print(f"Result: {result}")

# Manual calculation
history = normalizer.history
print(f"History length: {len(history)}")
values_le_5 = [h for h in history if h <= 5.0]
print(f"Values <= 5.0: {values_le_5}, count: {len(values_le_5)}")
rank = len(values_le_5) / len(history)
expected = 2 * rank - 1
print(f"Manual: rank = {rank}, expected = {expected}")

# Check what the normalizer actually computes
test_val = 5.0
rank_actual = np.sum(np.array(history) <= test_val) / len(history)
expected_actual = 2 * rank_actual - 1
print(f"Actual calculation: rank = {rank_actual}, expected = {expected_actual}")

