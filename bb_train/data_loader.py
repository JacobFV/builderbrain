"""
Data loading utilities for BuilderBrain training.

Handles loading and preprocessing of training data.
"""

from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    """Dummy dataset for testing purposes."""

    def __init__(self, size: int = 1000, seq_length: int = 128, vocab_size: int = 1000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate dummy input and target sequences
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        targets = torch.randint(0, self.vocab_size, (self.seq_length,))
        return input_ids, targets


class BuilderBrainDataLoader:
    """Data loader for BuilderBrain training."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        dataset = DummyDataset(
            size=1000,
            seq_length=self.config['data']['max_length'],
            vocab_size=self.config['data']['vocab_size']
        )

        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0  # Single-threaded for simplicity
        )

    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        dataset = DummyDataset(
            size=200,
            seq_length=self.config['data']['max_length'],
            vocab_size=self.config['data']['vocab_size']
        )

        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
