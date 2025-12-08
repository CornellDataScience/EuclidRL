#!/usr/bin/env python
"""
Quick start example for GRPO training.

This demonstrates how to set up and run GRPO training
following the DeepSeek-Prover-V1.5 methodology.
"""

import sys
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prover.configs import GRPOConfig
from prover.training import run_grpo


def main():
    """Run GRPO training with example configuration."""

    # Create configuration matching DeepSeek-Prover-V1.5 paper
    config = GRPOConfig(
        # Model settings
        model_name="Qwen/Qwen2.5-Math-1.5B",
        sft_checkpoint="checkpoints/sft/final",  # Must have SFT checkpoint first!
        output_dir="checkpoints/grpo_example",

        # Data
        rollout_file="data/rollouts/train.jsonl",
        max_response_length=2048,

        # GRPO hyperparameters from paper (Section 2.3)
        learning_rate=5e-6,         # Constant learning rate
        batch_size=8,               # Reduced from 512 for demo
        group_size=4,               # Reduced from 32 for demo
        kl_penalty=0.02,            # KL coefficient

        # Training settings
        num_epochs=1,
        log_steps=1,
        save_steps=10,
    )

    print("=" * 80)
    print("GRPO Quick Start Example")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  SFT Checkpoint: {config.sft_checkpoint}")
    print(f"  Group Size: {config.group_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  KL Penalty: {config.kl_penalty}")
    print(f"  Batch Size: {config.batch_size}")
    print()

    # Note: This example uses simplified settings for demonstration
    print("NOTE: This uses simplified settings for demonstration.")
    print("For full training, use the paper's hyperparameters:")
    print("  - batch_size: 512")
    print("  - group_size: 32")
    print("  - See prover/configs/grpo_default.yaml")
    print()

    # Run GRPO
    # Note: You need to have:
    # 1. An SFT checkpoint at the specified path
    # 2. Training data at data/rollouts/train.jsonl
    run_grpo(config, env=None)


if __name__ == "__main__":
    # Check if required files exist
    sft_path = Path("checkpoints/sft/final")
    data_path = Path("data/rollouts/train.jsonl")

    if not sft_path.exists():
        print("ERROR: SFT checkpoint not found!")
        print(f"Expected location: {sft_path}")
        print("\nYou need to run SFT training first:")
        print("  python scripts/train_sft.py --config prover/configs/sft_default.yaml")
        sys.exit(1)

    if not data_path.exists():
        print("ERROR: Training data not found!")
        print(f"Expected location: {data_path}")
        print("\nPlease prepare your training data in JSONL format:")
        print('  {"prompt": "theorem statement...", "theorem": "TheoremName"}')
        sys.exit(1)

    main()
