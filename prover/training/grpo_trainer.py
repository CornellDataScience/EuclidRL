"""
Group Relative Policy Optimization (GRPO) Trainer for Theorem Proving.

Implementation based on DeepSeek-Prover-V1.5 paper:
"DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for
Reinforcement Learning and Monte-Carlo Tree Search"

Key implementation details from Section 2.3:
- Uses GRPO algorithm (Shao et al., 2024) instead of PPO
- No critic model needed (more efficient than PPO)
- Samples groups of candidate proofs per theorem
- Optimizes based on relative rewards within each group
- Binary rewards from Lean 4 verifier: 1 if correct, 0 otherwise
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, List, Tuple
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from accelerate import Accelerator
import numpy as np

from prover.configs.grpo import GRPOConfig
from prover.env import LeanProofEnv


class TheoremDataset(Dataset):
    """Dataset of theorem statements for GRPO training."""

    def __init__(
        self,
        path: str,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        self.items: List[dict] = []
        for line in Path(path).read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                self.items.append({
                    "prompt": data["prompt"],
                    "theorem": data.get("theorem"),
                })

        # Shuffle and limit dataset size
        if shuffle:
            import random
            random.shuffle(self.items)

        if max_samples is not None and max_samples < len(self.items):
            self.items = self.items[:max_samples]
            print(f"Limited dataset to {max_samples} samples (from {len(self.items)} total)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    Unlike PPO, GRPO does not require a separate value/critic model.
    Instead, it optimizes based on the relative performance of outputs
    within a sampled group.
    """

    def __init__(
        self,
        config: GRPOConfig,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        dataset: TheoremDataset,
        env: Optional[LeanProofEnv] = None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model  # Reference model for KL penalty
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.env = env

        # Setup accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Prepare model and dataloader
        self.model = self.accelerator.prepare(self.model)
        self.ref_model = self.accelerator.prepare(self.ref_model)

        # Set model to training mode (needed for gradients)
        self.model.train()

        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

        # Setup dataloader with A100 optimizations
        num_workers = getattr(config, 'num_workers', 0)
        pin_memory = getattr(config, 'dataloader_pin_memory', True)

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # Return list of dicts
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        self.dataloader = self.accelerator.prepare(self.dataloader)

        self.global_step = 0

    def generate_group(
        self,
        prompts: List[str],
        group_size: int,
    ) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
        """
        Generate a group of candidate proofs for each prompt.

        Returns:
            responses: List[List[str]] - group_size responses per prompt
            response_log_probs: Tensor of shape (batch_size, group_size)
            ref_log_probs: Tensor of shape (batch_size, group_size)
        """
        batch_size = len(prompts)
        # Tokenize prompts ONCE (outside the loop)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
        ).to(self.accelerator.device)

        all_responses = [[] for _ in range(batch_size)]
        all_response_log_probs = []
        all_ref_log_probs = []

        # Generate group_size samples for each prompt
        for _ in range(group_size):
            # Generate from current policy (no_grad for generation only)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_response_length,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                # Detach and clone to ensure it's a standalone tensor
                generated_sequences = outputs.sequences.detach()

            # Decode responses
            responses = self.tokenizer.batch_decode(
                generated_sequences[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Store responses
            for i, resp in enumerate(responses):
                all_responses[i].append(resp)

            # Compute log probs WITH gradients for the policy model
            # The key insight: we pass generated_sequences as DATA (indices),
            # and compute a fresh forward pass through the model WITH gradients
            response_log_probs = self._compute_log_probs(
                inputs.input_ids,
                generated_sequences,
                self.model,
                requires_grad=True,  # Need gradients for policy optimization
            )

            # Reference model (no gradients)
            with torch.no_grad():
                ref_log_probs = self._compute_log_probs(
                    inputs.input_ids,
                    generated_sequences,
                    self.ref_model,
                    requires_grad=False,  # Reference model is frozen
                )

            all_response_log_probs.append(response_log_probs)
            all_ref_log_probs.append(ref_log_probs)

        # Stack log probs: (batch_size, group_size)
        response_log_probs_tensor = torch.stack(all_response_log_probs, dim=1)
        ref_log_probs_tensor = torch.stack(all_ref_log_probs, dim=1)

        # Debug: Check if gradients are attached
        if response_log_probs_tensor.requires_grad:
            print(f"  [DEBUG] response_log_probs has gradients: requires_grad={response_log_probs_tensor.requires_grad}")
        else:
            print(f"  [WARNING] response_log_probs MISSING gradients! requires_grad={response_log_probs_tensor.requires_grad}")
            print(f"  [DEBUG] Individual log_probs requires_grad: {[lp.requires_grad for lp in all_response_log_probs]}")

        return all_responses, response_log_probs_tensor, ref_log_probs_tensor

    def _compute_log_probs(
        self,
        input_ids: torch.Tensor,
        full_sequences: torch.Tensor,
        model: AutoModelForCausalLM,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Compute log probabilities of generated tokens.

        Args:
            input_ids: Input prompt tokens
            full_sequences: Full sequences (prompt + generated)
            model: Model to use for computing log probs
            requires_grad: Whether to compute gradients (True for policy, False for reference)
        """
        # Compute forward pass
        if not requires_grad:
            # For reference model: no gradients needed
            with torch.no_grad():
                outputs = model(full_sequences, return_dict=True)
                logits = outputs.logits
        else:
            # For current policy: need gradients - no context manager!
            outputs = model(full_sequences, return_dict=True)
            logits = outputs.logits

        # Get log probs for the generated tokens only
        prompt_length = input_ids.shape[1]
        gen_logits = logits[:, prompt_length-1:-1, :]  # Shift by 1
        gen_tokens = full_sequences[:, prompt_length:]

        # Compute log probabilities
        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=gen_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probs over sequence length
        # Mask padding tokens
        if not requires_grad:
            mask = (gen_tokens != self.tokenizer.pad_token_id).float()
        else:
            # When computing gradients, don't detach anything
            mask = (gen_tokens != self.tokenizer.pad_token_id).float()

        sequence_log_prob = (token_log_probs * mask).sum(dim=1)

        # Debug gradient flow
        if requires_grad and not sequence_log_prob.requires_grad:
            print(f"  [ERROR] _compute_log_probs: sequence_log_prob missing gradients!")
            print(f"    logits.requires_grad: {logits.requires_grad}")
            print(f"    log_probs.requires_grad: {log_probs.requires_grad}")
            print(f"    token_log_probs.requires_grad: {token_log_probs.requires_grad}")

        return sequence_log_prob

    def compute_rewards(
        self,
        responses_batch: List[List[str]],
        examples: List[dict],
    ) -> torch.Tensor:
        """
        Compute binary rewards using Lean 4 verifier.

        Returns:
            rewards: Tensor of shape (batch_size, group_size)
        """
        batch_size = len(responses_batch)
        group_size = len(responses_batch[0])
        rewards = torch.zeros(batch_size, group_size)

        for i, (responses, example) in enumerate(zip(responses_batch, examples)):
            for j, response in enumerate(responses):
                # Verify proof with Lean 4
                is_correct = self._verify_proof(response, example)

                # Binary reward: 1 if correct, 0 otherwise
                rewards[i, j] = (
                    self.config.reward_correct if is_correct
                    else self.config.reward_incorrect
                )

        return rewards.to(self.accelerator.device)

    def _verify_proof(self, response: str, example: dict) -> bool:
        """Verify a proof using Lean 4 prover."""
        if self.env is None or example.get("theorem") is None:
            # Fallback: simple heuristic reward
            return len(response.strip()) > 0

        try:
            state = self.env.reset(example["theorem"])
            for raw_tac in response.split(";"):
                tactic = raw_tac.strip()
                if not tactic:
                    continue
                _, reward, done, _ = self.env.step(tactic)
                if done and reward > 0:
                    return True
            return False
        except Exception:
            return False

    def compute_grpo_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute GRPO loss based on relative rewards within each group.

        Key idea: Optimize based on which samples in the group performed better.

        Args:
            rewards: (batch_size, group_size)
            log_probs: (batch_size, group_size) - current policy
            ref_log_probs: (batch_size, group_size) - reference policy

        Returns:
            loss: scalar tensor
            stats: dict of logging statistics
        """
        batch_size, group_size = rewards.shape

        # Compute advantages: rewards relative to group mean
        # This is the key difference from PPO - we use group-relative advantages
        group_mean_rewards = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - group_mean_rewards  # (batch_size, group_size)

        # Normalize advantages per group for stability
        group_std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = advantages / group_std

        # Compute KL divergence penalty
        kl_divergence = log_probs - ref_log_probs  # (batch_size, group_size)

        # GRPO objective: maximize advantage-weighted log prob - KL penalty
        # This encourages the model to increase probability of high-reward samples
        # relative to low-reward samples in the same group
        policy_loss = -(advantages * log_probs).mean()
        kl_loss = self.config.kl_penalty * kl_divergence.mean()

        total_loss = policy_loss + kl_loss

        # Logging statistics
        stats = {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/kl": kl_loss.item(),
            "rewards/mean": rewards.mean().item(),
            "rewards/max": rewards.max().item(),
            "rewards/min": rewards.min().item(),
            "rewards/correct_rate": (rewards > 0.5).float().mean().item(),
            "kl/mean": kl_divergence.mean().item(),
            "advantages/mean": advantages.mean().item(),
            "advantages/std": advantages.std().item(),
        }

        return total_loss, stats

    def train_step(self, batch: List[dict]) -> dict:
        """Execute one training step."""
        prompts = [item["prompt"] for item in batch]

        # Generate group of candidates for each prompt
        responses, log_probs, ref_log_probs = self.generate_group(
            prompts,
            self.config.group_size,
        )

        # Compute rewards from Lean 4 verifier
        rewards = self.compute_rewards(responses, batch)

        # Compute GRPO loss
        self.optimizer.zero_grad()

        loss, stats = self.compute_grpo_loss(rewards, log_probs, ref_log_probs)

        # Backward pass with gradient accumulation
        self.accelerator.backward(loss)

        # Gradient clipping
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.global_step += 1

        stats["step"] = self.global_step

        return stats

    def train(self) -> None:
        """Main training loop."""
        print(f"Starting GRPO training with {len(self.dataset)} examples")
        print(f"Group size: {self.config.group_size}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"KL penalty: {self.config.kl_penalty}")

        self.model.train()

        for epoch in range(self.config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")

            for batch in self.dataloader:
                stats = self.train_step(batch)

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    log_str = " | ".join(
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in stats.items()
                    )
                    print(f"[GRPO] {log_str}")

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        # Final checkpoint
        self.save_checkpoint("final")

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        save_dir = Path(self.config.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap model from accelerator
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save model and tokenizer
        unwrapped_model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))

        # Save config
        config_path = save_dir / "grpo_config.yaml"
        config_path.write_text(
            "\n".join(f"{k}: {v}" for k, v in asdict(self.config).items())
        )

        print(f"Checkpoint saved to {save_dir}")


def run_grpo(
    cfg: GRPOConfig,
    config_path: Optional[str] = None,
    env: Optional[LeanProofEnv] = None,
) -> None:
    """
    Run GRPO training for theorem proving.

    This follows the training procedure described in DeepSeek-Prover-V1.5 paper.
    """
    print("=" * 80)
    print("Group Relative Policy Optimization (GRPO) Training")
    print("Based on DeepSeek-Prover-V1.5")
    print("=" * 80)
    print(f"Config: {config_path or 'defaults'}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use left padding for decoder-only models during generation
    tokenizer.padding_side = "left"

    # Check for A100 optimizations
    use_flash_attn = getattr(cfg, 'use_flash_attention_2', False)
    compile_model = getattr(cfg, 'compile_model', False)

    # Load policy model (initialized from SFT checkpoint)
    print(f"\nLoading policy model from: {cfg.sft_checkpoint or cfg.model_name}")
    if use_flash_attn:
        print("  Using Flash Attention 2 (2-3x speedup on A100)")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.sft_checkpoint or cfg.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )

    # Compile model for PyTorch 2.0+ (20-30% speedup)
    if compile_model and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile (20-30% speedup)")
        model = torch.compile(model, mode="reduce-overhead")

    # Load reference model (frozen copy of SFT model for KL penalty)
    print(f"Loading reference model from: {cfg.sft_checkpoint or cfg.model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.sft_checkpoint or cfg.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )

    # Load dataset
    print(f"Loading dataset from: {cfg.rollout_file}")
    max_samples = getattr(cfg, 'max_samples', None)
    shuffle_data = getattr(cfg, 'shuffle_data', True)

    dataset = TheoremDataset(
        cfg.rollout_file,
        max_samples=max_samples,
        shuffle=shuffle_data,
    )
    print(f"Dataset size: {len(dataset)} examples")
    if max_samples is not None:
        print(f"  (Limited to {max_samples} samples for faster training)")

    # Create trainer
    trainer = GRPOTrainer(
        config=cfg,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        env=env,
    )

    # Train
    trainer.train()

    print("\n" + "=" * 80)
    print("GRPO Training Complete!")
    print("=" * 80)
