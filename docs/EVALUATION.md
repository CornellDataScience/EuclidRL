# Model Evaluation Guide

This guide shows how to evaluate your trained models on the validation set.

## Quick Start

### Evaluate GRPO Model

```bash
# Quick test on 50 validation examples
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-samples 50

# Full validation set (2,750 examples)
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final
```

### Evaluate SFT Model

```bash
# Test SFT model
python scripts/eval_model.py \
    --model checkpoints/sft/final \
    --max-samples 100
```

### Generate Multiple Proof Attempts

```bash
# Generate 4 proof attempts per theorem
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --num-samples 4 \
    --max-samples 50
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | Path to model checkpoint |
| `--val-file` | `data/sft/val.jsonl` | Validation data file |
| `--max-samples` | All | Limit to N validation examples |
| `--num-samples` | 1 | Number of proof attempts per theorem |
| `--max-new-tokens` | 512 | Max tokens to generate per proof |
| `--temperature` | 0.7 | Sampling temperature (lower = more deterministic) |
| `--output` | None | Save results to JSONL file |

## Understanding the Output

### Summary Statistics

The script outputs:
- **Total examples evaluated**: Number of theorems tested
- **Samples per theorem**: How many proofs generated per theorem
- **Total proofs generated**: Total number of proof attempts

### Example Output

```
================================================================================
EVALUATION SUMMARY
================================================================================
Total examples evaluated: 50
Samples per theorem: 4
Total proofs generated: 200

================================================================================
EXAMPLE GENERATIONS (first 3)
================================================================================

--- Example 1 ---
PROMPT (first 200 chars):
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem thm_9860 (a b c d : ℤ) (h₀ : a * a + b * c + b * d + c * d = 7) :
  ∃ (a b c d : ℤ), a * a + b ...

EXPECTED COMPLETION:

refine ⟨a, b, c, d,?_⟩
  exact h₀

GENERATED COMPLETIONS:
  [1] refine ⟨a, b, c, d, ?_⟩
  exact h₀
  [2] use a, b, c, d
  exact h₀
  [3] refine ⟨a, b, c, d, h₀⟩
  [4] exists a, b, c, d
```

## Saving Results

Save all generated proofs to a file for later analysis:

```bash
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-samples 100 \
    --num-samples 4 \
    --output results/grpo_eval_results.jsonl
```

Output format (one JSON object per line):
```json
{
  "prompt": "theorem statement...",
  "expected": "expected proof...",
  "generated": ["proof 1", "proof 2", "proof 3", "proof 4"]
}
```

## Evaluation Strategies

### 1. Quick Sanity Check (5 minutes)

```bash
# Test on 20 examples with single proof per theorem
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-samples 20 \
    --temperature 0.2
```

Use this to:
- Verify model loads correctly
- Check if proofs look reasonable
- Quick quality check

### 2. Pass@k Evaluation (30 minutes)

```bash
# Generate multiple proofs per theorem
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-samples 100 \
    --num-samples 8 \
    --output results/pass_at_k.jsonl
```

Use this to:
- Measure pass@1, pass@4, pass@8 rates
- Understand model diversity
- Compare with baselines

### 3. Full Validation (2-3 hours)

```bash
# Evaluate on entire validation set
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --num-samples 4 \
    --output results/full_validation.jsonl
```

Use this for:
- Final model evaluation
- Publication/reporting results
- Comparing SFT vs GRPO performance

## Comparing Models

### Side-by-Side Comparison (Recommended)

Use the dedicated comparison script to see outputs from multiple models simultaneously:

```bash
# Compare Base Qwen vs SFT vs GRPO (Quick test - 20 examples)
python scripts/compare_models.py \
    --models Qwen/Qwen2.5-Math-1.5B checkpoints/sft/final checkpoints/grpo_quick/final \
    --labels "Base Qwen" "SFT" "GRPO" \
    --max-samples 20

# Compare SFT vs GRPO (Full comparison - 100 examples)
python scripts/compare_models.py \
    --models checkpoints/sft/final checkpoints/grpo_quick/final \
    --labels "SFT" "GRPO" \
    --max-samples 100 \
    --output results/sft_vs_grpo_comparison.json
```

**Benefits:**
- See all model outputs side-by-side for each theorem
- Easily spot which model generates better proofs
- Save results for later analysis
- Perfect for understanding training impact

### Individual Model Evaluation (Alternative)

Evaluate each model separately:

```bash
# Evaluate base Qwen model (no training)
python scripts/eval_model.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --max-samples 100 \
    --output results/base_qwen_eval.jsonl

# Evaluate SFT model
python scripts/eval_model.py \
    --model checkpoints/sft/final \
    --max-samples 100 \
    --output results/sft_eval.jsonl

# Evaluate GRPO model
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-samples 100 \
    --output results/grpo_eval.jsonl
```

### Temperature Sweep

Test different sampling temperatures:

```bash
# Greedy (deterministic)
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-samples 50 \
    --temperature 0.1 \
    --output results/temp_0.1.jsonl

# Balanced
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-samples 50 \
    --temperature 0.7 \
    --output results/temp_0.7.jsonl

# Creative
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-samples 50 \
    --temperature 1.0 \
    --output results/temp_1.0.jsonl
```

## Tips

### Memory Management

If you run out of GPU memory:

```bash
# Reduce max_new_tokens
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-new-tokens 256 \
    --max-samples 50
```

### Speed Optimization

For faster evaluation:

```bash
# Use lower temperature (faster sampling)
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --temperature 0.1 \
    --max-samples 100
```

### Quality Analysis

Look for these in the generated proofs:
- **Syntactic correctness**: Does it follow Lean 4 syntax?
- **Tactic diversity**: Does it use various tactics (ring, linarith, simp, etc.)?
- **Reasoning quality**: Does it match the expected proof structure?
- **Completeness**: Does it fully solve the theorem?

## Next Steps

After evaluation:

1. **Analyze results**: Look at common failure patterns
2. **Iterate training**: Adjust hyperparameters based on results
3. **Verify with Lean 4**: Use actual Lean 4 compiler to check correctness
4. **Compare baselines**: Test against DeepSeek-Prover or other models

## Troubleshooting

### Model Not Found

```
Error: Can't load model from checkpoints/grpo_quick/final
```

**Solution**: Check that training completed and checkpoint exists:
```bash
ls -lh checkpoints/grpo_quick/final/
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or max tokens:
```bash
python scripts/eval_model.py \
    --model checkpoints/grpo_quick/final \
    --max-new-tokens 256 \
    --max-samples 20
```

### Empty Generations

If proofs are empty or very short, try:
- Increasing temperature: `--temperature 0.8`
- Checking if model was trained properly
- Verifying validation data format matches training data
