# GRPO vs PPO: A Comparison for Theorem Proving

## Overview

This document compares **GRPO (Group Relative Policy Optimization)** and **PPO (Proximal Policy Optimization)** in the context of theorem proving, based on the DeepSeek-Prover-V1.5 paper.

## Key Quote from the Paper

> "We employ the Group Relative Policy Optimization (GRPO; Shao et al., 2024) as our RL algorithm, which has demonstrated **superior effectiveness and efficiency compared to PPO** (Schulman et al., 2017), **primarily because it eliminates the necessity of training an additional critic model**."

## Architectural Differences

### PPO Architecture

```
┌─────────────────────────────────────┐
│  Policy Network (Actor)             │
│  - Generates actions (proof steps)  │
│  - Updated via policy gradient      │
└─────────────────────────────────────┘
              ↕
┌─────────────────────────────────────┐
│  Value Network (Critic)             │
│  - Estimates state values           │
│  - Provides baseline for advantages │
│  - REQUIRES SEPARATE TRAINING       │
└─────────────────────────────────────┘
```

### GRPO Architecture

```
┌─────────────────────────────────────┐
│  Policy Network (Actor)             │
│  - Generates actions (proof steps)  │
│  - Updated via group-relative grad  │
└─────────────────────────────────────┘
              ↕
┌─────────────────────────────────────┐
│  Reference Model (Frozen)           │
│  - Copy of SFT model                │
│  - Only for KL divergence           │
│  - NO TRAINING NEEDED               │
└─────────────────────────────────────┘
```

## Detailed Comparison

| Feature | PPO | GRPO |
|---------|-----|------|
| **Models Needed** | Policy + Value (2 networks) | Policy + Reference (1 trained) |
| **Value Estimation** | Learned by critic network | Group-relative rewards |
| **Training Complexity** | High (2 networks to optimize) | Low (1 network to optimize) |
| **Memory Usage** | ~2x model size | ~2x model size |
| **Computation** | Higher (critic forward/backward) | Lower (reference is frozen) |
| **Advantage Calculation** | `reward - value_estimate` | `reward - group_mean_reward` |
| **Reward Signal** | Absolute rewards | Relative rewards within group |

## Algorithm Pseudocode

### PPO Algorithm

```python
for batch in dataloader:
    # Sample trajectories
    states, actions, rewards = rollout_policy(batch)

    # Compute value estimates (CRITIC)
    values = value_network(states)

    # Compute advantages
    advantages = rewards - values

    # Update policy (ACTOR)
    policy_loss = -log_prob(actions) * advantages
    policy_loss += kl_penalty(policy, ref_policy)

    # Update value network (CRITIC)
    value_loss = MSE(values, rewards)

    # Two optimization steps
    optimize(policy_loss)
    optimize(value_loss)
```

### GRPO Algorithm

```python
for batch in dataloader:
    # Sample GROUP of trajectories per prompt
    for prompt in batch:
        candidates = [sample_policy(prompt) for _ in range(group_size)]
        rewards = [verify_proof(c) for c in candidates]

        # Compute GROUP-RELATIVE advantages (NO CRITIC)
        group_mean = mean(rewards)
        advantages = [r - group_mean for r in rewards]

    # Update policy only (NO VALUE UPDATE)
    policy_loss = -log_prob(actions) * advantages
    policy_loss += kl_penalty(policy, ref_policy)

    # One optimization step
    optimize(policy_loss)
```

## Advantage Calculation

### PPO Advantages

```python
# Requires trained value network
value_estimates = value_network(states)
advantages = rewards - value_estimates
```

**Problem**: Value network must learn to predict future rewards, which is difficult in sparse-reward environments like theorem proving.

### GRPO Advantages

```python
# Uses empirical group statistics
group_mean_rewards = rewards.mean(dim=1)  # Mean within each group
advantages = rewards - group_mean_rewards
advantages = advantages / (rewards.std(dim=1) + 1e-8)  # Normalize
```

**Benefit**: No learning required - advantages are computed directly from empirical rewards within each group.

## Why GRPO Works Better for Theorem Proving

### 1. **Sparse Rewards**
- Theorem proving has binary rewards (0 or 1)
- PPO's value network struggles to learn in sparse-reward settings
- GRPO doesn't need to learn value function - uses empirical statistics

### 2. **Sample Efficiency**
- PPO needs many samples to train both policy and value networks
- GRPO only trains policy, using groups for variance reduction
- Group-relative advantages are more stable than learned values

### 3. **Computational Efficiency**
```
PPO Computation per step:
  - Policy forward pass
  - Policy backward pass
  - Value forward pass
  - Value backward pass
  Total: 4 passes

GRPO Computation per step:
  - Policy forward pass (×group_size for sampling)
  - Policy backward pass
  - Reference forward pass (no gradients)
  Total: 2 passes (with gradients)
```

### 4. **No Hyperparameter Tuning for Critic**
- PPO requires tuning: value loss coefficient, advantage normalization, etc.
- GRPO has simpler hyperparameters: just group_size and kl_penalty

## Performance Results (from DeepSeek-Prover-V1.5 Paper)

### Training Results

| Metric | SFT | RL (GRPO) | Improvement |
|--------|-----|-----------|-------------|
| miniF2F-test (Pass@128) | 50.4% | 51.6% | +1.2% |
| miniF2F-test (Pass@16×6400) | 57.4% | 60.2% | +2.8% |
| ProofNet (Pass@128) | 15.9% | 18.2% | +2.3% |
| ProofNet (Pass@4×6400) | 22.9% | 22.6% | -0.3% |

**Key Finding**: GRPO provides "genuine enhancement of fundamental capabilities" - improves performance across **all sample budgets**, not just TopK.

## Hyperparameters

### PPO Typical Settings

```yaml
learning_rate: 1e-5
batch_size: 128
ppo_epochs: 4
value_loss_coef: 0.5
clip_range: 0.2
kl_penalty: 0.1
gae_lambda: 0.95  # For advantage estimation
```

### GRPO Settings (DeepSeek-Prover-V1.5)

```yaml
learning_rate: 5e-6     # Lower than PPO
batch_size: 512         # Larger than PPO
group_size: 32          # NEW: samples per prompt
kl_penalty: 0.02        # Lower than PPO
# NO value_loss_coef, clip_range, or gae_lambda!
```

## Code Complexity

### PPO Training Loop
```python
# Complex: Two networks, two losses
for batch in dataloader:
    # 1. Rollout
    responses = generate(prompts)
    rewards = verify(responses)

    # 2. Compute values (REQUIRES VALUE NETWORK)
    values = value_net(states)
    advantages = compute_gae(rewards, values)

    # 3. Update policy
    for _ in range(ppo_epochs):
        ratio = prob_new / prob_old
        clipped = clip(ratio, 1-eps, 1+eps)
        policy_loss = -min(ratio*adv, clipped*adv)
        optimize_policy(policy_loss)

    # 4. Update value (SEPARATE OPTIMIZATION)
    value_loss = MSE(values, returns)
    optimize_value(value_loss)
```

### GRPO Training Loop
```python
# Simple: One network, one loss
for batch in dataloader:
    # 1. Generate group
    for prompt in batch:
        responses = [generate(prompt) for _ in range(group_size)]
        rewards = [verify(r) for r in responses]

        # 2. Group-relative advantages (NO VALUE NETWORK)
        advantages = rewards - rewards.mean()

    # 3. Update policy (SINGLE OPTIMIZATION)
    policy_loss = -(advantages * log_probs).mean()
    kl_loss = kl_penalty * (log_probs - ref_log_probs).mean()
    optimize(policy_loss + kl_loss)
```

## When to Use Each

### Use PPO When:
- ✅ You have dense reward signals
- ✅ You have lots of compute for value network training
- ✅ You need per-step value estimates
- ✅ Environment has long episodes with continuous rewards

### Use GRPO When:
- ✅ You have sparse binary rewards (like theorem proving!)
- ✅ You want simpler, more efficient training
- ✅ You can sample multiple candidates per input
- ✅ Group-relative performance is meaningful

## References

1. **GRPO Paper**: Shao et al., 2024 - "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
2. **DeepSeek-Prover-V1.5**: Xin et al., 2024 - "Harnessing Proof Assistant Feedback for RL and MCTS"
3. **PPO Paper**: Schulman et al., 2017 - "Proximal Policy Optimization Algorithms"

## Conclusion

For theorem proving:
- **GRPO is superior** due to sparse binary rewards
- **No critic network** = simpler and more efficient
- **Group-relative advantages** = more stable than learned values
- **Better performance** on DeepSeek-Prover-V1.5 benchmarks

The DeepSeek team's choice of GRPO over PPO is well-justified for this domain!
