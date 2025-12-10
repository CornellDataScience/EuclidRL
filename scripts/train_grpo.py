#!/usr/bin/env python
"""
Train a theorem prover using Group Relative Policy Optimization (GRPO).

Based on DeepSeek-Prover-V1.5:
"DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for
Reinforcement Learning and Monte-Carlo Tree Search"
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prover.configs import load_grpo_config
from prover.env import LeanProofEnv
from prover.training import run_grpo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GRPO RL fine-tuning for theorem proving (DeepSeek-Prover-V1.5 style)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to N samples (overrides config)."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use grpo_quick.yaml config (50 samples, small batch/group)."
    )
    parser.add_argument(
        "--lean-repo",
        type=str,
        default=None,
        help="Lean repository URL for LeanDojo (optional)."
    )
    parser.add_argument(
        "--lean-commit",
        type=str,
        default=None,
        help="Commit hash for LeanDojo (optional)."
    )
    parser.add_argument(
        "--theorems",
        type=str,
        nargs="*",
        default=None,
        help="Theorems to practice (optional)."
    )
    args = parser.parse_args()

    # Use quick config if --quick flag is set
    config_path = args.config
    if args.quick and config_path is None:
        config_path = "prover/configs/grpo_quick.yaml"
        print("Quick mode enabled: using grpo_quick.yaml config")

    # Load configuration
    cfg = load_grpo_config(config_path)

    # Override with command-line arguments
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
        print(f"Overriding max_samples: {args.max_samples}")

    # Setup Lean environment if specified
    env = None
    if args.lean_repo and args.lean_commit and args.theorems:
        print(f"Setting up Lean environment:")
        print(f"  Repo: {args.lean_repo}")
        print(f"  Commit: {args.lean_commit}")
        print(f"  Theorems: {args.theorems}")
        env = LeanProofEnv(args.lean_repo, args.lean_commit, args.theorems)

    # Run GRPO training
    run_grpo(cfg, config_path, env)


if __name__ == "__main__":
    main()
