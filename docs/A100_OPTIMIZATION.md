# A100 GPU Optimization Guide

This guide explains how to optimize GRPO training for NVIDIA A100 GPUs to achieve 2-3x faster training compared to V100.

## Quick Start

For A100 GPUs, use the optimized config:

```bash
python scripts/train_grpo.py --config prover/configs/grpo_a100.yaml
```

The Colab notebook automatically detects A100 and uses the optimized config.

## Performance Improvements

| Optimization | Speedup | Notes |
|-------------|---------|-------|
| Flash Attention 2 | 2-3x | Native A100 support, most impactful |
| BF16 precision | ~20% | A100 has native BF16 hardware |
| torch.compile | 20-30% | PyTorch 2.0+ graph optimization |
| Larger batches | 15-25% | Better GPU utilization |
| DataLoader workers | 10-15% | Parallel data loading |
| **Total** | **~3-4x** | Combined effect |

## Configuration Differences

### Default Config (V100/T4)
```yaml
batch_size: 512
gradient_accumulation_steps: 16
group_size: 32
use_flash_attention_2: false
compile_model: false
```

### A100 Config
```yaml
batch_size: 1024              # Doubled (A100 has more memory)
gradient_accumulation_steps: 8    # Halved (faster updates)
group_size: 64                # Doubled (better variance reduction)
use_flash_attention_2: true   # 2-3x speedup
compile_model: true           # 20-30% speedup
num_workers: 8                # Parallel data loading
```

## Detailed Optimizations

### 1. Flash Attention 2 (Most Important)

Flash Attention 2 provides 2-3x speedup on A100 by optimizing the attention mechanism.

**Enable in config:**
```yaml
use_flash_attention_2: true
```

**Install:**
```bash
pip install flash-attn --no-build-isolation
```

**Verification:**
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    attn_implementation="flash_attention_2"
)
```

### 2. BF16 Native Precision

A100 has native BF16 tensor cores (no accuracy loss, faster than FP16).

**Enable in config:**
```yaml
mixed_precision: bf16
```

### 3. Model Compilation (PyTorch 2.0+)

`torch.compile` optimizes the model graph for 20-30% speedup.

**Enable in config:**
```yaml
compile_model: true
```

**Requirements:**
- PyTorch >= 2.0
- Works best with PyTorch 2.1+

### 4. Larger Batch Sizes

A100 has 40GB-80GB memory vs V100's 16-32GB. Use larger batches for better GPU utilization.

**Guidelines:**
- 40GB A100: batch_size=1024, group_size=64
- 80GB A100: batch_size=2048, group_size=128

**Monitor memory:**
```bash
nvidia-smi -l 1  # Watch GPU memory usage
```

### 5. DataLoader Optimization

Parallel data loading prevents CPU bottlenecks.

**Enable in config:**
```yaml
num_workers: 8
dataloader_pin_memory: true
```

## Memory Optimization (If OOM)

If you encounter Out-Of-Memory errors on 40GB A100:

### Option 1: Reduce Batch/Group Size
```yaml
batch_size: 512
group_size: 32
```

### Option 2: Gradient Checkpointing
```yaml
gradient_checkpointing: true  # Trade compute for memory
```

### Option 3: Enable LoRA
```yaml
use_lora: true
lora_r: 16
lora_alpha: 32
```

## Benchmarks

### Expected Throughput (examples/hour)

| GPU | Default Config | A100 Config | Speedup |
|-----|---------------|-------------|---------|
| T4 (16GB) | 10-15 | N/A (OOM) | - |
| V100 (32GB) | 30-50 | N/A | - |
| A100 40GB | 100-150 | 120-180 | ~3x |
| A100 80GB | 100-150 | 150-250 | ~4x |

### Training Time Estimates (10k examples)

| GPU | Default Config | A100 Config |
|-----|---------------|-------------|
| T4 | ~150-200 hours | N/A |
| V100 | ~40-60 hours | N/A |
| A100 40GB | ~15-20 hours | **~5-8 hours** |
| A100 80GB | ~15-20 hours | **~4-6 hours** |

## Troubleshooting

### Flash Attention Not Available

**Error:** `Flash attention is not available`

**Solutions:**
1. Install flash-attn:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. If installation fails, disable in config:
   ```yaml
   use_flash_attention_2: false
   ```

### Model Compilation Errors

**Error:** `torch.compile` failures

**Solutions:**
1. Upgrade PyTorch:
   ```bash
   pip install --upgrade torch
   ```

2. Disable compilation:
   ```yaml
   compile_model: false
   ```

### OOM Errors

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size:
   ```yaml
   batch_size: 512  # or 256
   ```

2. Reduce group size:
   ```yaml
   group_size: 32  # or 16
   ```

3. Enable gradient checkpointing:
   ```yaml
   gradient_checkpointing: true
   ```

### Slow DataLoader

**Symptom:** GPU utilization < 80%

**Solutions:**
1. Increase workers:
   ```yaml
   num_workers: 16  # Try 8, 12, 16
   ```

2. Enable pinned memory:
   ```yaml
   dataloader_pin_memory: true
   ```

## Advanced Tips

### 1. Multi-GPU Training

For multiple A100s, use Accelerate:

```bash
accelerate config  # Configure multi-GPU
accelerate launch scripts/train_grpo.py --config prover/configs/grpo_a100.yaml
```

### 2. Mixed Batch Processing

Process different batch sizes based on proof length:

```yaml
# Short proofs (< 512 tokens): batch_size=2048
# Long proofs (> 512 tokens): batch_size=512
```

### 3. Dynamic Group Size

Adjust group size based on validation performance:

```python
# If correct_rate < 0.1: increase group_size
# If correct_rate > 0.5: decrease group_size
```

## Validation

To verify optimizations are working:

1. **Check Flash Attention:**
   ```python
   print(model.config._attn_implementation)
   # Should print: "flash_attention_2"
   ```

2. **Check Compilation:**
   ```python
   print(hasattr(model, '_orig_mod'))
   # Should print: True (if compiled)
   ```

3. **Monitor GPU Utilization:**
   ```bash
   nvidia-smi dmon -s u
   # Should show ~95%+ GPU utilization
   ```

4. **Check Throughput:**
   Look for log message:
   ```
   [GRPO] step=100 | examples_per_sec=42.3
   ```

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [PyTorch 2.0 Compile](https://pytorch.org/get-started/pytorch-2.0/)
- [A100 Tensor Core Performance](https://www.nvidia.com/en-us/data-center/a100/)
- [DeepSeek-Prover V1.5 Paper](https://arxiv.org/abs/2408.08152)
