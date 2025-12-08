# GRPO Implementation for Theorem Proving

This is an implementation of **Group Relative Policy Optimization (GRPO)** based on the DeepSeek-Prover-V1.5 paper:

> "DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search"

## What is GRPO?

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm that:

1. **Does NOT require a critic/value model** (unlike PPO) - making it more efficient
2. **Samples groups of candidate outputs** for each input
3. **Optimizes based on relative performance within each group**
4. **Uses binary rewards** from the Lean 4 verifier (1 if correct, 0 if incorrect)

## Key Differences from PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Value Model | ✅ Required | ❌ Not needed |
| Training Efficiency | Lower | Higher |
| Reward Signal | Absolute | Group-relative |
| Implementation Complexity | Higher | Lower |

## Implementation Details

Based on DeepSeek-Prover-V1.5 paper (Section 2.3):

### Hyperparameters

- **Learning Rate**: 5e-6 (constant)
- **Batch Size**: 512
- **Group Size**: 32 candidate proofs per theorem
- **KL Penalty**: 0.02
- **Max Response Length**: 2048 tokens
- **Rewards**: Binary (1 for correct, 0 for incorrect)

### Algorithm

For each training batch:
1. Sample a **group of 32 candidate proofs** for each theorem
2. Verify each proof using **Lean 4 prover** (binary reward: 0 or 1)
3. Compute **group-relative advantages**: reward - group_mean_reward
4. Optimize policy to increase probability of better-performing proofs
5. Apply **KL penalty** to prevent divergence from reference model

## Files Created

```
prover/
├── configs/
│   ├── grpo.py              # GRPO configuration dataclass
│   └── grpo_default.yaml    # Default GRPO hyperparameters
└── training/
    └── grpo_trainer.py      # GRPO trainer implementation

scripts/
└── train_grpo.py            # Training script
```

## Usage

### Basic Training

```bash
python scripts/train_grpo.py --config prover/configs/grpo_default.yaml
```

### With Lean Environment

```bash
python scripts/train_grpo.py \
  --config prover/configs/grpo_default.yaml \
  --lean-repo https://github.com/leanprover-community/mathlib4.git \
  --lean-commit <commit-hash> \
  --theorems Theorem1 Theorem2
```

### Custom Configuration

Create your own YAML config:

```yaml
# my_grpo_config.yaml
model_name: Qwen/Qwen2.5-Math-1.5B
sft_checkpoint: checkpoints/sft/final
output_dir: checkpoints/my_grpo_run

learning_rate: 5.0e-6
batch_size: 512
group_size: 32
kl_penalty: 0.02

rollout_file: data/rollouts/train.jsonl
max_response_length: 2048
```

Then run:

```bash
python scripts/train_grpo.py --config my_grpo_config.yaml
```

## Code Structure

### GRPOTrainer Class

The main trainer class implements:

- **`generate_group()`**: Generate group_size candidate proofs per prompt
- **`compute_rewards()`**: Verify proofs with Lean 4 and assign binary rewards
- **`compute_grpo_loss()`**: Calculate group-relative policy gradient loss
- **`train_step()`**: Execute one training iteration
- **`train()`**: Main training loop

### Loss Function

```python
# Compute advantages relative to group mean
group_mean_rewards = rewards.mean(dim=1, keepdim=True)
advantages = (rewards - group_mean_rewards) / (rewards.std(dim=1, keepdim=True) + 1e-8)

# GRPO loss: advantage-weighted log prob - KL penalty
policy_loss = -(advantages * log_probs).mean()
kl_loss = kl_penalty * (log_probs - ref_log_probs).mean()
total_loss = policy_loss + kl_loss
```

## Requirements

Make sure you have the required dependencies:

```bash
pip install torch transformers accelerate pyyaml
```

For Lean 4 verification:
```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

## Expected Performance

According to the DeepSeek-Prover-V1.5 paper:

- **miniF2F-test**: 60.2% pass rate (single-pass), 63.5% (with RMaxTS)
- **ProofNet**: 22.6% pass rate (single-pass), 25.3% (with RMaxTS)

GRPO training provides a **genuine enhancement** of fundamental capabilities, improving performance across all sample budgets (not just TopK).

## Paper Reference

```bibtex
@article{xin2024deepseekproverv15,
  title={DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search},
  author={Huajian Xin and Z.Z. Ren and Junxiao Song and Zhihong Shao and others},
  journal={arXiv preprint arXiv:2408.08152},
  year={2024}
}
```

## Comparison with TRL Library

The `trl` library (used in DeepSeek-Prover-V1.5) also provides GRPO:
- `trl.GRPOTrainer` - official implementation
- `trl.GRPOConfig` - configuration

Our implementation follows the same principles but is tailored for theorem proving with:
- Lean 4 proof verification
- Binary reward signals
- Group-based candidate sampling

## Notes

1. **This is inference on the SFT model** - GRPO requires a good SFT checkpoint
2. **Lean 4 verification is slow** - consider using parallelization
3. **Group size trades off diversity vs. computation** - 32 is from the paper
4. **No value model needed** - this is the key efficiency gain over PPO

## License

This implementation follows the same license as the euclidrl-prover repository.
