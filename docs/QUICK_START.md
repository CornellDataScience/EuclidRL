# Quick Start Guide - Fast Training

This guide shows how to run quick experiments without waiting hours for full training.

## TL;DR - Fastest Way to Test

```bash
# SFT: ~5-10 minutes on A100
python scripts/train_sft.py --quick

# GRPO: ~10-20 minutes on A100
python scripts/train_grpo.py --quick
```

## Overview

Training on full datasets takes hours. Use these methods for fast iteration:

| Method | Data Size | Time (A100) | Time (V100) | Use Case |
|--------|-----------|-------------|-------------|----------|
| **Quick Mode** | 50-100 samples | 5-20 min | 15-60 min | Testing setup, debugging |
| **Small Subset** | 500-1000 samples | 30-90 min | 2-4 hours | Hyperparameter tuning |
| **Medium Subset** | 5000 samples | 4-8 hours | 12-20 hours | Initial experiments |
| **Full Dataset** | 10k+ samples | 8-24 hours | 30-60 hours | Final training |

## Quick Mode (Recommended for Testing)

### SFT Quick Mode

```bash
python scripts/train_sft.py --quick
```

**What it does:**
- Uses 100 training samples
- Trains for 1 epoch only
- Saves checkpoint at `checkpoints/sft_quick/`
- Takes ~5-10 minutes on A100

**Config:** [sft_quick.yaml](../prover/configs/sft_quick.yaml)

### GRPO Quick Mode

```bash
python scripts/train_grpo.py --quick
```

**What it does:**
- Uses 50 rollout examples
- Small batch size (4) and group size (8)
- Saves checkpoint at `checkpoints/grpo_quick/`
- Takes ~10-20 minutes on A100

**Config:** [grpo_quick.yaml](../prover/configs/grpo_quick.yaml)

## Command-Line Sample Limiting

Override sample counts without editing configs:

### SFT
```bash
# Use 200 training samples, 50 validation samples
python scripts/train_sft.py \
    --config prover/configs/sft_default.yaml \
    --max-train-samples 200 \
    --max-val-samples 50
```

### GRPO
```bash
# Use 100 samples
python scripts/train_grpo.py \
    --config prover/configs/grpo_default.yaml \
    --max-samples 100
```

## Config-Based Sample Limiting

Edit the config file directly:

### SFT Config
```yaml
# prover/configs/sft_default.yaml
max_train_samples: 500    # Limit to 500 training samples
max_val_samples: 100      # Limit to 100 validation samples
```

### GRPO Config
```yaml
# prover/configs/grpo_default.yaml
max_samples: 500          # Limit to 500 rollout samples
shuffle_data: true        # Shuffle before sampling
```

## Quick Configs Reference

### sft_quick.yaml
```yaml
max_train_samples: 100
max_val_samples: 20
num_train_epochs: 1
batch_size: 2
gradient_accumulation_steps: 8
log_steps: 5
save_steps: 50
```

### grpo_quick.yaml
```yaml
max_samples: 50
batch_size: 4
group_size: 8
log_steps: 1
save_steps: 25
num_epochs: 1
```

### grpo_a100.yaml (Full Training)
```yaml
max_samples: null         # Use all data
batch_size: 1024
group_size: 64
use_flash_attention_2: true
compile_model: true
```

## Recommended Workflow

### 1. Test Setup (5-20 minutes)
```bash
# Quick sanity check
python scripts/train_sft.py --quick
python scripts/train_grpo.py --quick
```

### 2. Hyperparameter Tuning (1-4 hours)
```bash
# Test different learning rates with 500 samples
python scripts/train_sft.py \
    --max-train-samples 500 \
    --max-val-samples 100

python scripts/train_grpo.py --max-samples 500
```

### 3. Validation Run (4-12 hours)
```bash
# Larger subset to validate approach
python scripts/train_sft.py --max-train-samples 5000
python scripts/train_grpo.py --max-samples 5000
```

### 4. Full Training (8-24 hours)
```bash
# Use A100-optimized config for best performance
python scripts/train_sft.py --config prover/configs/sft_default.yaml
python scripts/train_grpo.py --config prover/configs/grpo_a100.yaml
```

## Training Time Estimates

### SFT (Supervised Fine-Tuning)

| Samples | Epochs | A100 (40GB) | V100 (32GB) | T4 (16GB) |
|---------|--------|-------------|-------------|-----------|
| 100 | 1 | ~5 min | ~15 min | ~30 min |
| 500 | 1 | ~20 min | ~1 hour | ~2 hours |
| 1000 | 3 | ~1.5 hours | ~4 hours | ~8 hours |
| 5000 | 3 | ~6 hours | ~16 hours | ~30 hours |
| 10000 | 3 | ~12 hours | ~30 hours | ~60 hours |

### GRPO (Reinforcement Learning)

| Samples | Batch | Group | A100 (40GB) | V100 (32GB) | T4 (16GB) |
|---------|-------|-------|-------------|-------------|-----------|
| 50 | 4 | 8 | ~15 min | ~40 min | ~90 min |
| 100 | 4 | 8 | ~30 min | ~1.5 hours | ~3 hours |
| 500 | 16 | 16 | ~3 hours | ~8 hours | ~16 hours |
| 1000 | 16 | 32 | ~6 hours | ~16 hours | ~32 hours |
| 10000 | 1024 | 64 | **~5-8 hours** | ~40 hours | OOM |

**Note:** GRPO with A100 config (Flash Attention 2) is 3-4x faster.

## Monitoring Training

### Watch GPU Usage
```bash
nvidia-smi -l 1  # Update every second
```

### Check Training Logs
```bash
# SFT logs
tail -f checkpoints/sft/training.log

# GRPO logs show rewards
[GRPO] step=10 | loss/total=0.5234 | rewards/mean=0.12 | rewards/correct_rate=0.15
```

### Validate Checkpoints
```bash
# Check if model saved correctly
ls -lh checkpoints/sft_quick/final/
ls -lh checkpoints/grpo_quick/step_25/
```

## Troubleshooting

### Training Too Slow
1. Use `--quick` flag first
2. Check GPU utilization (`nvidia-smi`)
3. Reduce `max_samples` if testing
4. On A100: use `grpo_a100.yaml` for Flash Attention

### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train_sft.py --quick  # Uses batch_size=2

# Or edit config:
# batch_size: 1
# gradient_accumulation_steps: 16
```

### No Progress / Stuck
- Check if data loaded: Look for "Loaded datasets" message
- Verify GPU is being used: `nvidia-smi`
- Check for errors in terminal output

## Example Session

```bash
# 1. Quick test (10 minutes total)
python scripts/train_sft.py --quick
python scripts/train_grpo.py --quick

# 2. Verify checkpoints exist
ls checkpoints/sft_quick/final/
ls checkpoints/grpo_quick/final/

# 3. If working, try larger subset (2 hours)
python scripts/train_sft.py --max-train-samples 1000
python scripts/train_grpo.py --max-samples 500

# 4. Full training overnight with A100 optimizations
python scripts/train_sft.py
python scripts/train_grpo.py --config prover/configs/grpo_a100.yaml
```

## Advanced: Custom Subsets

### Random Sampling
```python
# Already enabled by default
shuffle_data: true  # in grpo configs
```

### Stratified Sampling
To sample specific theorem types, edit your data file:

```bash
# Keep only first 100 lines
head -n 100 data/sft/train.jsonl > data/sft/train_subset.jsonl

# Then in config:
train_file: data/sft/train_subset.jsonl
```

### Progressive Training
Start small, gradually increase:

```bash
# Week 1: Quick validation
python scripts/train_sft.py --max-train-samples 100

# Week 2: Small scale
python scripts/train_sft.py --max-train-samples 1000

# Week 3: Medium scale
python scripts/train_sft.py --max-train-samples 5000

# Week 4: Full training
python scripts/train_sft.py
```

## See Also

- [A100_OPTIMIZATION.md](A100_OPTIMIZATION.md) - GPU optimization guide
- [README_GRPO.md](../README_GRPO.md) - GRPO training details
- [configs/](../prover/configs/) - All config files
