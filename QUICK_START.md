# Quick Start Guide

## What Was Done

✅ **GRPO Implementation**: Complete implementation of Group Relative Policy Optimization based on DeepSeek-Prover-V1.5 paper
✅ **Colab Notebook Fixed**: Updated for CornellDataScience/EuclidRL repository
✅ **Documentation**: Comprehensive guides and comparisons

---

## Usage Options

### Option 1: Local Training

#### SFT Training
```bash
python scripts/train_sft.py --config prover/configs/sft_default.yaml
```

#### GRPO Training (after SFT)
```bash
python scripts/train_grpo.py --config prover/configs/grpo_default.yaml
```

#### Quick Example
```bash
python examples/grpo_quickstart.py
```

### Option 2: Google Colab

1. Open [colab_sft.ipynb](colab_sft.ipynb) in Google Colab
2. Set your HF_TOKEN in the first code cell
3. Run all cells for SFT training
4. (Optional) Continue to GRPO section for RL training

---

## Key Files

### For Using GRPO
- 📖 **[README_GRPO.md](README_GRPO.md)** - Main usage guide
- 📖 **[docs/GRPO_vs_PPO.md](docs/GRPO_vs_PPO.md)** - Algorithm comparison
- 🔧 **[prover/configs/grpo_default.yaml](prover/configs/grpo_default.yaml)** - Default config
- 📓 **[colab_sft.ipynb](colab_sft.ipynb)** - Colab notebook

### For Understanding Implementation
- 📋 **[GRPO_IMPLEMENTATION_SUMMARY.md](GRPO_IMPLEMENTATION_SUMMARY.md)** - Overview
- 📋 **[FILES_UPDATED_SUMMARY.md](FILES_UPDATED_SUMMARY.md)** - Complete changelog
- 📋 **[COLAB_NOTEBOOK_UPDATES.md](COLAB_NOTEBOOK_UPDATES.md)** - Notebook changes

---

## GRPO vs PPO

| Feature | PPO | GRPO |
|---------|-----|------|
| **Networks** | Policy + Value | Policy only |
| **Advantage** | Learned by critic | Group-relative empirical |
| **Efficiency** | Lower | Higher |
| **Sparse Rewards** | Poor | Excellent |

**For theorem proving**: Use GRPO (better for binary rewards)

---

## Hyperparameters (Paper-Accurate)

```yaml
learning_rate: 5e-6      # Constant (no scheduler)
batch_size: 512          # Total batch size
group_size: 32           # Candidates per theorem
kl_penalty: 0.02         # KL divergence coefficient
max_response_length: 2048 # Max proof tokens
```

For Colab (reduced): `batch_size: 16`, `group_size: 8`

---

## Requirements

### Software
```bash
pip install torch transformers accelerate pyyaml
```

### For Lean 4 Verification (GRPO)
```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### Prerequisites
1. **SFT checkpoint** - Train SFT first
2. **Training data** - JSONL format
3. **GPU** - Recommended for training

---

## Expected Performance

Based on DeepSeek-Prover-V1.5 paper:

- **miniF2F-test**: +2.8% improvement (57.4% → 60.2%)
- **ProofNet**: Comparable performance

GRPO provides "genuine capability enhancement" across all sample budgets.

---

## Programmatic Usage

```python
from prover.configs import GRPOConfig
from prover.training import run_grpo

config = GRPOConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B",
    sft_checkpoint="checkpoints/sft/final",
    learning_rate=5e-6,
    batch_size=512,
    group_size=32,
    kl_penalty=0.02,
)

run_grpo(config)
```

---

## File Structure

```
euclidrl-prover/
├── prover/
│   ├── configs/
│   │   ├── grpo.py                 # GRPO config class
│   │   └── grpo_default.yaml       # Default hyperparameters
│   └── training/
│       └── grpo_trainer.py         # GRPO trainer implementation
├── scripts/
│   └── train_grpo.py               # Training script
├── examples/
│   └── grpo_quickstart.py          # Quick start example
├── docs/
│   └── GRPO_vs_PPO.md             # Algorithm comparison
├── colab_sft.ipynb                 # Colab notebook (updated)
├── README_GRPO.md                  # Main usage guide
└── GRPO_IMPLEMENTATION_SUMMARY.md  # Implementation overview
```

---

## Troubleshooting

### Issue: Keras 3 import error
**Solution**: `pip install tf-keras`
**Note**: This is a local environment issue, not a code problem

### Issue: ModuleNotFoundError: prover.models
**Solution**: Make sure you're in the repo root directory
**Check**: `ls prover/` should show `models/` directory

### Issue: No SFT checkpoint found
**Solution**: Train SFT first with `python scripts/train_sft.py`

### Issue: Lean 4 verification fails
**Solution**: Install elan and Lean 4 (see Requirements section)

---

## Status

✅ **Implementation**: Complete and ready to use
✅ **Colab Notebook**: Fixed for CornellDataScience/EuclidRL
✅ **Documentation**: Comprehensive guides available
✅ **Paper Accuracy**: Matches DeepSeek-Prover-V1.5 specifications

---

## References

1. **DeepSeek-Prover-V1.5**: Xin et al., 2024, arXiv:2408.08152
2. **GRPO Paper**: Shao et al., 2024 - DeepSeekMath
3. **Repository**: [CornellDataScience/EuclidRL](https://github.com/CornellDataScience/EuclidRL)

---

**Ready to start!** 🚀

For detailed information, see [README_GRPO.md](README_GRPO.md)
