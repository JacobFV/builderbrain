#!/usr/bin/env python3
"""
BuilderBrain: A dual-rail extension to pretrained transformers for compositional reasoning.

This is the main entry point for the builderbrain system.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for builderbrain."""
    parser = argparse.ArgumentParser(description="BuilderBrain: Compositional AI System")

    parser.add_argument(
        "--mode",
        choices=["train", "serve", "eval", "demo"],
        default="demo",
        help="Operation mode"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file path"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Model checkpoint path"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "serve":
        run_serving(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


def run_demo():
    """Run a demonstration of the system."""
    print("🤖 BuilderBrain Demo")
    print("=" * 50)

    print("Loading core components...")

    # Import core modules
    from bb_core.math_utils import RankNormalizer, ConstraintManager
    from bb_priors.cfg_parser import JSONGrammar
    from bb_runtime.plan_schemas import RobotPlanSchema

    print("✅ Core interfaces loaded")

    # Demonstrate grammar parsing
    grammar = JSONGrammar()
    print(f"✅ Grammar loaded with {len(grammar.get_terminals())} terminals")

    # Demonstrate plan schema
    schema = RobotPlanSchema()
    print(f"✅ Plan schema loaded with {len(schema.nodes)} node types")

    # Demonstrate normalization
    normalizer = RankNormalizer()
    test_values = [0.1, 0.5, 0.9, 0.3, 0.7]
    normalized = normalizer(np.array(test_values))
    print(f"✅ Normalization: {test_values} -> {normalized}")

    print("\n🎉 Demo completed successfully!")
    print("BuilderBrain is ready for training and inference.")


def run_training(args):
    """Run training pipeline."""
    print("🚂 Starting training...")

    from bb_train.trainer import BuilderBrainTrainer, create_default_config

    config = create_default_config()
    trainer = BuilderBrainTrainer(config)

    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Run training
    history = trainer.train(num_epochs=config['training']['num_epochs'])

    print("✅ Training completed!")
    print(f"Final loss: {history['total_loss'][-1]:.4f}")


def run_serving(args):
    """Run inference server."""
    print("🔄 Starting inference server...")

    # TODO: Implement serving logic
    print("Server functionality not yet implemented")
    print("This would start a web server for inference")


def run_evaluation(args):
    """Run model evaluation."""
    print("📊 Running evaluation...")

    # TODO: Implement evaluation logic
    print("Evaluation functionality not yet implemented")


if __name__ == "__main__":
    main()
