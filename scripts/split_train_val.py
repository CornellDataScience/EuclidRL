#!/usr/bin/env python3
"""
Split training data into train and validation sets.

Usage:
    python scripts/split_train_val.py
    python scripts/split_train_val.py --val-ratio 0.1
    python scripts/split_train_val.py --train-input data/sft/train.jsonl --val-ratio 0.05
"""

import argparse
import json
import random
from pathlib import Path


def split_train_val(
    train_path: Path,
    val_path: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Split training data into train and validation sets."""

    print(f"Reading training data from {train_path}...")
    with open(train_path) as f:
        all_data = [json.loads(line) for line in f]

    total = len(all_data)
    print(f"Total examples: {total}")

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(all_data)

    # Split
    val_size = int(total * val_ratio)
    train_size = total - val_size

    train_data = all_data[:train_size]
    val_data = all_data[train_size:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_data)} examples ({100*(1-val_ratio):.1f}%)")
    print(f"  Val:   {len(val_data)} examples ({100*val_ratio:.1f}%)")

    # Write train
    print(f"\nWriting train to {train_path}...")
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write val
    print(f"Writing val to {val_path}...")
    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n✓ Done!")
    print(f"  Train: {train_path} ({len(train_data)} examples)")
    print(f"  Val:   {val_path} ({len(val_data)} examples)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split training data into train and validation sets"
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/sft/train.jsonl"),
        help="Input training file (will be overwritten with smaller train set)",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("data/sft/val.jsonl"),
        help="Output validation file",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if not args.train_input.exists():
        print(f"ERROR: Train file not found: {args.train_input}")
        print("\nRun this first:")
        print("  python scripts/download_training_data.py")
        print("  python scripts/convert_deepseek_v15.py \\")
        print("    --inputs data/raw/deepseek_v1_train.jsonl \\")
        print("    --train-output data/sft/train.jsonl \\")
        print("    --val-output data/sft/val.jsonl")
        return

    split_train_val(
        args.train_input,
        args.val_output,
        args.val_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()
