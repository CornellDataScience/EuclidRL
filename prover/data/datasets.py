from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset


@dataclass
class SFTExample:
    prompt: str
    completion: str


def load_jsonl_dataset(
    train_path: str,
    val_path: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[Dataset, Dataset | None]:
    """Load JSONL files into Hugging Face Dataset objects for SFTTrainer.

    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file (optional)
        max_train_samples: Limit training data to N samples (None = use all)
        max_val_samples: Limit validation data to N samples (None = use all)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load training data
    train_data = []
    for line in Path(train_path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            train_data.append({"prompt": row["prompt"], "completion": row["completion"]})

    # Limit training data if specified
    if max_train_samples is not None and max_train_samples < len(train_data):
        print(f"Limiting training data to {max_train_samples} samples (from {len(train_data)})")
        train_data = train_data[:max_train_samples]

    train_ds = Dataset.from_list(train_data)

    # Load validation data if provided
    val_ds = None
    if val_path:
        val_data = []
        for line in Path(val_path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                val_data.append({"prompt": row["prompt"], "completion": row["completion"]})

        # Limit validation data if specified
        if max_val_samples is not None and max_val_samples < len(val_data):
            print(f"Limiting validation data to {max_val_samples} samples (from {len(val_data)})")
            val_data = val_data[:max_val_samples]

        val_ds = Dataset.from_list(val_data)

    return train_ds, val_ds
