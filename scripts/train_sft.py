#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prover.configs import load_sft_config
from prover.training import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT for Qwen proof bot.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit training data to N samples (overrides config)")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Limit validation data to N samples (overrides config)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: use sft_quick.yaml config (100 samples, 1 epoch)")
    args = parser.parse_args()

    # Use quick config if --quick flag is set
    config_path = args.config
    if args.quick and config_path is None:
        config_path = "prover/configs/sft_quick.yaml"
        print("Quick mode enabled: using sft_quick.yaml config")

    cfg = load_sft_config(config_path)

    # Override with command-line arguments
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples
        print(f"Overriding max_train_samples: {args.max_train_samples}")

    if args.max_val_samples is not None:
        cfg.max_val_samples = args.max_val_samples
        print(f"Overriding max_val_samples: {args.max_val_samples}")

    run_sft(cfg, config_path)


if __name__ == "__main__":
    main()
