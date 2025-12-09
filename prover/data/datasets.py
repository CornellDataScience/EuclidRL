from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset


@dataclass
class SFTExample:
    prompt: str
    completion: str


def load_jsonl_dataset(train_path: str, val_path: str | None = None) -> tuple[Dataset, Dataset | None]:
    """Load JSONL files into Hugging Face Dataset objects for SFTTrainer."""
    # Load training data
    train_data = []
    for line in Path(train_path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            train_data.append({"prompt": row["prompt"], "completion": row["completion"]})

    train_ds = Dataset.from_list(train_data)

    # Load validation data if provided
    val_ds = None
    if val_path:
        val_data = []
        for line in Path(val_path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                val_data.append({"prompt": row["prompt"], "completion": row["completion"]})
        val_ds = Dataset.from_list(val_data)

    return train_ds, val_ds
