"""
Data loading utilities for BuilderBrain training.

Handles loading and preprocessing of real text training data.
"""

from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json


class TextDataset(Dataset):
    """Dataset for text data with proper tokenization."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize text
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)

        # Create input and target sequences
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)  # All but last token
        targets = torch.tensor(tokens[1:], dtype=torch.long)    # All but first token

        return input_ids, targets


class BuilderBrainDataLoader:
    """Data loader for BuilderBrain training with real text data."""

    def __init__(self, config: Dict[str, Any], tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

        # Load training texts
        self.train_texts = self._load_texts('train')
        self.val_texts = self._load_texts('val')

        if self.tokenizer is None:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # Add pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_texts(self, split: str) -> List[str]:
        """Load text data for training/validation."""
        # Try to load from common text datasets
        texts = []

        # Check for common text files
        possible_files = [
            f'data/{split}.txt',
            f'data/{split}.json',
            f'data/{split}.jsonl',
            'train.txt',  # Fallback to simple text file
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                texts = self._load_file(file_path, split)
                if texts:
                    print(f"Loaded {len(texts)} {split} examples from {file_path}")
                    break

        # If no files found, use a small synthetic dataset for testing
        if not texts:
            print(f"No {split} data found, using synthetic data")
            texts = self._get_synthetic_data(split)

        return texts

    def _load_file(self, file_path: str, split: str) -> List[str]:
        """Load texts from various file formats."""
        texts = []

        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = data
                elif isinstance(data, dict) and 'texts' in data:
                    texts = data['texts']

        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'text' in data:
                            texts.append(data['text'])

        return texts

    def _get_synthetic_data(self, split: str) -> List[str]:
        """Generate synthetic text data for testing."""
        if split == 'train':
            return [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming technology.",
                "Natural language processing enables computers to understand text.",
                "Deep learning models require large amounts of data.",
                "Transformer architectures have revolutionized AI.",
                "Attention mechanisms are key to modern neural networks.",
                "Large language models can generate human-like text.",
                "Training neural networks requires significant computational resources.",
                "The field of artificial intelligence continues to advance rapidly.",
                "Computer vision allows machines to interpret visual information."
            ] * 50  # Repeat for more data
        else:  # val
            return [
                "AI systems are becoming increasingly sophisticated.",
                "Neural networks learn patterns from data.",
                "Language models can perform many different tasks.",
                "The future of AI holds great promise.",
                "Research in machine learning is ongoing."
            ] * 10

    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        dataset = TextDataset(
            self.train_texts,
            self.tokenizer,
            self.config['data']['max_length']
        )

        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn
        )

    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        dataset = TextDataset(
            self.val_texts,
            self.tokenizer,
            self.config['data']['max_length']
        )

        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """Collate function for variable-length sequences."""
        input_ids, targets = zip(*batch)

        # Pad sequences to max length in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=-100  # Ignore index for loss
        )

        return input_ids, targets
