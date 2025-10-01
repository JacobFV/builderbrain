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
        "--scale",
        choices=["tiny", "small", "medium", "large", "production"],
        default="tiny",
        help="Model scale for testing/production"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path (overrides --scale)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Model checkpoint path"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args.scale)
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "serve":
        run_serving(args)
    elif args.mode == "eval":
        run_evaluation(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


def run_demo(scale: str = "tiny"):
    """Run a demonstration of the system."""
    print("🤖 BuilderBrain Demo")
    print("=" * 50)
    print(f"Scale: {scale}")

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

    # Show configuration info for the specified scale
    print(f"✅ Configuration ready for {scale} scale")
    if scale == "tiny":
        print("   Model: Tiny custom (64 hidden, 2 layers)")
        print("   Programs: 8 discrete skills")
        print("   Constraints: Grammar + Graph-to-Graph")
    elif scale == "small":
        print("   Model: GPT-2 (768 hidden, 4 layers)")
        print("   Programs: 16 discrete skills")
        print("   Constraints: Grammar + Graph-to-Graph + Buildability")
    elif scale == "production":
        print("   Model: GPT-Neo 2.7B (2560 hidden, 8 layers)")
        print("   Programs: 32 discrete skills")
        print("   Constraints: All (Grammar, Graph, Buildability, Reuse, Calibration)")

    print("\n🎉 Demo completed successfully!")
    print("BuilderBrain is ready for training and inference.")


def run_training(args):
    """Run training pipeline."""
    print("🚂 Starting training...")

    from bb_train.trainer import BuilderBrainTrainer
    from bb_train.config import get_config_for_scale, load_config

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config_for_scale(args.scale)

    print(f"Training with {args.scale} scale configuration:")
    print(f"  Model: {config['model']['type']} ({config['model']['hidden_size']} hidden)")
    print(f"  Programs: {config['model']['num_programs']}")
    print(f"  Batch size: {config['training']['batch_size']}")

    trainer = BuilderBrainTrainer(config)

    # Run training
    history = trainer.train(num_epochs=config['training']['num_epochs'])

    print("✅ Training completed!")
    print(f"Final loss: {history['total_loss'][-1]:.4f}")

    # Save training history
    import json
    with open(f'training_history_{args.scale}.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to training_history_{args.scale}.json")


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
