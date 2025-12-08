#!/usr/bin/env python3
"""
Download DeepSeek-Prover-V1 training dataset from HuggingFace.

This downloads 27.5k theorem-proof pairs with formal Lean 4 proofs,
suitable for SFT training.

Usage:
    python scripts/download_training_data.py
    python scripts/download_training_data.py --output data/sft/deepseek_v1.jsonl
"""

import argparse
from pathlib import Path


def download_deepseek_v1_data(output_path: Path) -> None:
    """Download DeepSeek-Prover-V1 training data."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found!")
        print("Install it with: pip install datasets")
        return

    print("Downloading DeepSeek-Prover-V1 dataset from HuggingFace...")
    print("This may take a few minutes (19.6 MB download)...")

    # Download dataset
    dataset = load_dataset("deepseek-ai/DeepSeek-Prover-V1")

    print(f"✓ Downloaded {len(dataset['train'])} training examples")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSONL
    print(f"Saving to {output_path}...")
    dataset["train"].to_json(output_path, orient="records", lines=True)

    print(f"✓ Saved to {output_path}")
    print(f"✓ Total examples: {len(dataset['train'])}")
    print()
    print("Dataset format:")
    print("  - name: Problem identifier")
    print("  - split: 'train'")
    print("  - header: Lean 4 imports")
    print("  - formal_statement: Theorem statement")
    print("  - goal: Proof goal")
    print("  - formal_proof: The proof (THIS IS WHAT WE NEED!)")
    print()
    print("Next step: Convert to SFT format using:")
    print("  python scripts/convert_deepseek_v15.py \\")
    print(f"    --inputs {output_path} \\")
    print("    --train-output data/sft/train.jsonl \\")
    print("    --val-output data/sft/val.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DeepSeek-Prover-V1 training data with proofs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/deepseek_v1_train.jsonl"),
        help="Where to save the downloaded data",
    )
    args = parser.parse_args()

    download_deepseek_v1_data(args.output)


if __name__ == "__main__":
    main()
