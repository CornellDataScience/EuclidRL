# Colab Notebook Updates Summary

## Changes Made to `colab_sft.ipynb`

### 1. Repository URL Updated
- **Changed from**: `Datboy0127/EuclidRL`
- **Changed to**: `CornellDataScience/EuclidRL`
- **Cell**: b38d2e64

### 2. Removed Obsolete Cell
- **Removed**: Cell bba8df30 (unzip euclidrl-prover.zip)
- **Reason**: Obsolete workflow, using git clone instead

### 3. Improved HF Token Setup
- **Cell**: 11fe8818
- **Added**: Warning message about setting HF_TOKEN
- **Cleaned up**: Duplicate token assignment lines

### 4. Better Checkpoint Handling
- **Cell**: da60f497 (SFT checkpoints)
- **Added**: Path existence check before zipping
- **Added**: Informative error messages

### 5. Added GRPO Training Section

**New cells added (5 total):**

#### a) GRPO Section Header (Markdown)
- **Cell ID**: grpo-section
- **Content**: Explains GRPO training, requirements, and what it does
- **Links to**: README_GRPO.md for details

#### b) Install Lean 4
- **Cell ID**: grpo-install-lean
- **Purpose**: Install Lean 4 prover for proof verification
- **Commands**:
  - Downloads elan installer
  - Installs Lean 4 stable version

#### c) Prepare GRPO Data
- **Cell ID**: grpo-prepare-data
- **Purpose**: Set up training data for GRPO
- **Features**:
  - Creates `data/rollouts/` directory
  - Copies SFT data if available
  - Shows helpful error messages

#### d) Run GRPO Training
- **Cell ID**: grpo-train
- **Purpose**: Execute GRPO training
- **Settings**:
  - Reduced batch_size (16) for Colab
  - Reduced group_size (8) for Colab
  - Uses prover/configs/grpo_default.yaml
  - Follows DeepSeek-Prover-V1.5 methodology

#### e) Zip GRPO Checkpoints
- **Cell ID**: grpo-zip-checkpoints
- **Purpose**: Package GRPO checkpoints for download
- **Features**: Path validation and error handling

### 6. Updated Title Cell
- **Cell**: cell-0 (markdown header)
- **Content**:
  - Updated description to mention both SFT and GRPO
  - Added repository link
  - Clearer step-by-step instructions

## Notebook Structure (Final)

```
📓 colab_sft.ipynb (15 cells total)

Setup Section:
├─ 0. Header (markdown) - Title and instructions
├─ 1. HF Token - Set authentication
├─ 2. GPU Check - nvidia-smi
├─ 3. Clone Repo - CornellDataScience/EuclidRL
├─ 4. Install Dependencies - pip install
├─ 5. Convert Data - DeepSeek-V1.5 format (optional)
└─ 6. Model Download - Validation check

Part 1: SFT Training:
├─ 7. SFT Section Header (markdown)
├─ 8. Run SFT Training
└─ 9. Zip SFT Checkpoints

Part 2: GRPO Training (NEW):
├─ 10. GRPO Section Header (markdown)
├─ 11. Install Lean 4
├─ 12. Prepare GRPO Data
├─ 13. Run GRPO Training
└─ 14. Zip GRPO Checkpoints
```

## How to Use the Updated Notebook

### For SFT Only:
1. Set HF_TOKEN in cell 1
2. Run cells 0-9 (stops after SFT checkpoint zip)

### For SFT + GRPO:
1. Set HF_TOKEN in cell 1
2. Run cells 0-9 (SFT training)
3. Continue with cells 10-14 (GRPO training)

## Key Features

✅ **Single source of truth**: CornellDataScience/EuclidRL repository
✅ **Clear sections**: SFT and GRPO training separated
✅ **Error handling**: Helpful messages when files are missing
✅ **Colab-optimized**: Reduced batch/group sizes for T4 GPU
✅ **Complete workflow**: From setup to checkpoint download
✅ **Documentation**: Links to README_GRPO.md for details

## Testing Checklist

- [ ] Clone works from CornellDataScience/EuclidRL
- [ ] Dependencies install correctly
- [ ] SFT training runs (with proper data)
- [ ] GRPO training runs (with SFT checkpoint + Lean 4)
- [ ] Checkpoints zip correctly
- [ ] All cells have clear error messages

## Notes

- GRPO requires SFT checkpoint first
- Lean 4 installation is required for GRPO
- Batch/group sizes reduced for Colab (can increase on better hardware)
- Full paper settings: batch_size=512, group_size=32
