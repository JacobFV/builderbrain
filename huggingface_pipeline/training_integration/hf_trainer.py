"""
Hugging Face Trainer integration for BuilderBrain.

Adapts BuilderBrain training pipeline to work with HF Trainer class and datasets.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
import wandb

# Add parent directory to path for BuilderBrain imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from bb_nn.dual_rail import DualRail
from bb_losses.dual_optimizer import DualOptimizer
from bb_train.config import create_default_config


class BuilderBrainDataset(Dataset):
    """Dataset wrapper for BuilderBrain training data."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input
        inputs = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': inputs['input_ids'].squeeze()  # For causal LM
        }


class BuilderBrainTrainer(Trainer):
    """Custom trainer that integrates BuilderBrain's dual optimizer."""

    def __init__(
        self,
        model: DualRail,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        tokenizer: AutoTokenizer = None,
        data_collator: Optional[Callable] = None,
        dual_optimizer: Optional[DualOptimizer] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **kwargs
        )

        self.dual_optimizer = dual_optimizer
        self.step_count = 0

        # Initialize wandb for experiment tracking
        if args.report_to and "wandb" in args.report_to:
            wandb.init(
                project="builderbrain",
                name=args.run_name or "builderbrain-training",
                config=args.to_dict()
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to use dual optimizer."""

        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard cross-entropy loss for causal LM
        labels = inputs.get('labels')
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            loss_fct = nn.CrossEntropyLoss()
            task_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            task_loss = torch.tensor(0.0, device=logits.device)

        # Compute constraint losses if dual optimizer is available
        constraint_losses = {}
        if self.dual_optimizer is not None:
            try:
                # Mock constraint computation (in production, would use actual constraints)
                constraint_losses = {
                    'grammar': torch.tensor(0.05, device=task_loss.device),
                    'graph2graph': torch.tensor(0.03, device=task_loss.device),
                    'reuse': torch.tensor(0.02, device=task_loss.device)
                }
            except Exception as e:
                print(f"Warning: Could not compute constraint losses: {e}")

        # Compute total loss with dual optimizer
        if self.dual_optimizer is not None and constraint_losses:
            total_loss, normalized_losses = self.dual_optimizer.compute_lagrangian(
                task_loss, constraint_losses
            )

            # Update dual variables
            self.dual_optimizer.update_duals(normalized_losses)
        else:
            total_loss = task_loss

        # Log metrics
        self.step_count += 1
        if self.step_count % 10 == 0:  # Log every 10 steps
            metrics = {
                'step': self.step_count,
                'task_loss': float(task_loss.item()),
                'total_loss': float(total_loss.item()),
            }

            if constraint_losses:
                metrics.update({
                    f'constraint_{k}': float(v.item())
                    for k, v in constraint_losses.items()
                })

            if self.dual_optimizer is not None:
                dual_vars = self.dual_optimizer.get_dual_values()
                metrics.update({
                    f'dual_{k}': float(v) for k, v in dual_vars.items()
                })

            # Log to wandb if available
            if wandb.run is not None:
                wandb.log(metrics)

            # Print progress
            if self.step_count % 50 == 0:
                print(f"Step {self.step_count}: Loss = {total_loss.item():.4f}")

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to include dual optimizer metrics."""
        super().log(logs)

        # Add dual optimizer metrics to logs
        if self.dual_optimizer is not None:
            dual_vars = self.dual_optimizer.get_dual_values()
            logs.update({
                f'dual_{k}': float(v) for k, v in dual_vars.items()
            })

        # Log to wandb
        if wandb.run is not None:
            wandb.log(logs)


def create_hf_training_args(
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    logging_steps: int = 10,
    save_steps: int = 1000,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    **kwargs
) -> TrainingArguments:
    """Create training arguments for BuilderBrain HF trainer."""

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Enable mixed precision
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["wandb"],  # Enable wandb logging
        run_name="builderbrain-training",
        **kwargs
    )


def load_builderbrain_model(
    model_name_or_path: str = "gpt2",
    builderbrain_config: Optional[Dict[str, Any]] = None
) -> DualRail:
    """Load or create a BuilderBrain model."""

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Create BuilderBrain configuration
    if builderbrain_config is None:
        builderbrain_config = create_default_config()

    # Create dual-rail model
    model = DualRail(
        base_model=base_model,
        hidden_size=builderbrain_config['model']['hidden_size'],
        num_layers=builderbrain_config['model']['num_layers'],
        num_programs=builderbrain_config['model']['num_programs'],
        alpha_cap=builderbrain_config['model']['alpha_cap']
    )

    return model


def create_mock_dataset(
    tokenizer: AutoTokenizer,
    num_samples: int = 1000,
    max_length: int = 512
) -> List[Dict[str, str]]:
    """Create mock dataset for testing."""

    # Mock compositional reasoning examples
    examples = [
        "Generate a JSON object with user information: name, email, phone",
        "Create an API call to authenticate user with token validation",
        "Plan a robot manipulation sequence: grasp, rotate, place",
        "Design a phone conversation flow for customer support",
        "Compose a structured response for social media interaction",
        "Build a workflow for data processing pipeline",
        "Create a schema for validating user input forms",
        "Design a conversation tree for chatbot responses"
    ]

    dataset = []
    for _ in range(num_samples):
        # Randomly select and modify an example
        example = examples[torch.randint(0, len(examples), (1,)).item()]
        # Add some variation
        example = example.replace("JSON", "structured data").replace("API", "service")

        dataset.append({"text": example})

    return dataset


def main():
    """Main training function with HF integration."""

    # Configuration
    model_name = "gpt2"  # Can be changed to other models
    output_dir = "./builderbrain-hf-training"

    # Create training arguments
    training_args = create_hf_training_args(
        output_dir=output_dir,
        num_train_epochs=1,  # Quick demo
        per_device_train_batch_size=2,
        logging_steps=5,
        save_steps=50,
    )

    # Load model and tokenizer
    model = load_builderbrain_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create mock dataset
    mock_data = create_mock_dataset(tokenizer, num_samples=100)
    train_dataset = BuilderBrainDataset(mock_data[:80], tokenizer)
    eval_dataset = BuilderBrainDataset(mock_data[80:], tokenizer)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # Create dual optimizer
    dual_optimizer = DualOptimizer({
        'grammar': {'target': 0.0, 'normalizer': 'rank'},
        'graph2graph': {'target': 0.2, 'normalizer': 'rank'},
        'reuse': {'target': 0.5, 'normalizer': 'rank'}
    })

    # Create trainer
    trainer = BuilderBrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dual_optimizer=dual_optimizer,
    )

    # Train the model
    print("Starting BuilderBrain training with HF integration...")
    trainer.train()

    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"Training completed! Model saved to {output_dir}")

    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
