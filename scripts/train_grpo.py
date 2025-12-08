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

    # Load configuration
    cfg = load_grpo_config(args.config)

    # Setup Lean environment if specified
    env = None
    if args.lean_repo and args.lean_commit and args.theorems:
        print(f"Setting up Lean environment:")
        print(f"  Repo: {args.lean_repo}")
        print(f"  Commit: {args.lean_commit}")
        print(f"  Theorems: {args.theorems}")
        env = LeanProofEnv(args.lean_repo, args.lean_commit, args.theorems)

    # Run GRPO training
    run_grpo(cfg, args.config, env)


if __name__ == "__main__":
    main()
