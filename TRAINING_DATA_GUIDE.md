# Training Data Guide

## How to Download Training Data

### Quick Answer

Run this command:
```bash
python scripts/download_training_data.py
```

This downloads **27,503 theorem-proof pairs** from DeepSeek-Prover-V1 to `data/raw/deepseek_v1_train.jsonl`.

---

## Detailed Steps

### Step 1: Download Raw Data

```bash
# Download DeepSeek-Prover-V1 training data (19.6 MB)
python scripts/download_training_data.py --output data/raw/deepseek_v1_train.jsonl
```

**Output:**
- File: `data/raw/deepseek_v1_train.jsonl`
- Examples: 27,503
- Format: Raw DeepSeek format with `formal_proof` field

### Step 2: Convert to SFT Format

```bash
# Convert to our training format
python scripts/convert_deepseek_v15.py \
    --inputs data/raw/deepseek_v1_train.jsonl \
    --train-output data/sft/train.jsonl \
    --val-output data/sft/val.jsonl
```

**Output:**
- `data/sft/train.jsonl` - Training examples
- `data/sft/val.jsonl` - Validation examples

### Step 3: Start Training

```bash
# Run SFT training
python scripts/train_sft.py --config prover/configs/sft_default.yaml
```

---

## On Google Colab

The Colab notebook now handles this automatically! Just run the cells in order:

1. **Cell 5**: Downloads DeepSeek-Prover-V1 data
2. **Cell 6**: Converts to SFT format
3. **Cell 8**: Runs SFT training

---

## Data Sources Explained

### ✅ DeepSeek-Prover-V1 (TRAINING DATA - Use This!)

- **Source**: https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1
- **Examples**: 27,503
- **Has proofs**: ✅ YES (`formal_proof` field)
- **Use for**: SFT training
- **Format**:
  ```json
  {
    "name": "thm_0",
    "split": "train",
    "header": "import Mathlib...",
    "formal_statement": "theorem thm_0 : ...",
    "goal": "⊢ ...",
    "formal_proof": "intro h n norm_num [h, n]"  ← THE PROOF!
  }
  ```

### ❌ DeepSeek-Prover-V1.5 Datasets (EVALUATION ONLY - Don't Use for Training!)

- **Source**: `DeepSeek-Prover-V1.5/datasets/`
- **Files**: `minif2f.jsonl`, `proofnet.jsonl`
- **Has proofs**: ❌ NO (`formal_proof` field missing)
- **Use for**: Testing/evaluation only
- **Format**:
  ```json
  {
    "name": "amc12a_2019_p21",
    "split": "test",
    "header": "import Mathlib...",
    "formal_statement": "theorem amc12a_2019_p21 : ...",
    "goal": "⊢ ...",
    ❌ NO formal_proof field!
  }
  ```

---

## Requirements

```bash
pip install datasets  # For downloading from HuggingFace
```

Already included in `requirements.txt`!

---

## File Structure After Download

```
euclidrl-prover/
├── data/
│   ├── raw/
│   │   └── deepseek_v1_train.jsonl    ← Downloaded data (27.5k examples)
│   └── sft/
│       ├── train.jsonl                 ← Converted training data
│       └── val.jsonl                   ← Converted validation data
└── scripts/
    ├── download_training_data.py       ← Download script
    └── convert_deepseek_v15.py         ← Conversion script
```

---

## Troubleshooting

### "datasets library not found"
```bash
pip install datasets
```

### "No examples after conversion"
You're probably using V1.5 eval data instead of V1 training data. Make sure you download from:
- ✅ `deepseek-ai/DeepSeek-Prover-V1` (has proofs)
- ❌ NOT `DeepSeek-Prover-V1.5/datasets/` (no proofs)

### "Authentication required"
Set your HuggingFace token:
```bash
export HF_TOKEN="your_token_here"
```
Or in Python:
```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

---

## Data Statistics

| Metric | Value |
|--------|-------|
| Total examples | 27,503 |
| Download size | 19.6 MB |
| Has proofs | ✅ Yes |
| Format | Lean 4 |
| Source | Mathlib4, synthetic theorems |

---

## Alternative: Manual Download

If you prefer to download manually:

1. Go to https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1
2. Click "Files and versions"
3. Download the Parquet files
4. Load with:
   ```python
   import pandas as pd
   df = pd.read_parquet("train-00000-of-00001.parquet")
   df.to_json("data/raw/deepseek_v1_train.jsonl", orient="records", lines=True)
   ```

---

## Summary

**For Training**: Use DeepSeek-Prover-**V1** (has proofs) ✅
**For Evaluation**: Use DeepSeek-Prover-**V1.5** datasets (test problems only)

**Command**:
```bash
python scripts/download_training_data.py
```

That's it! 🎉
