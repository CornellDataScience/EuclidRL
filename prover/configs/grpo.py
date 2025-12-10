from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class GRPOConfig:
    """
    Configuration for Group Relative Policy Optimization (GRPO) training.
    Based on DeepSeek-Prover-V1.5 paper (Section 2.3).
    """
    # Model settings
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    sft_checkpoint: Optional[str] = "checkpoints/sft/final"
    output_dir: str = "checkpoints/grpo"

    # Data settings
    rollout_file: str = "data/rollouts/train.jsonl"
    max_prompt_length: int = 512
    max_response_length: int = 2048  # Paper uses 2048 for max length

    # Training hyperparameters (from paper Section 2.3)
    learning_rate: float = 5e-6  # Paper: constant learning rate of 5e-6
    batch_size: int = 512  # Paper: training batch size is 512
    gradient_accumulation_steps: int = 16

    # GRPO-specific settings
    group_size: int = 32  # Paper: "sample a group of 32 candidate proofs"
    kl_penalty: float = 0.02  # Paper: "KL penalty coefficient is set to 0.02"

    # Logging and checkpointing
    log_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 200

    # Mixed precision and optimization
    mixed_precision: Optional[str] = "bf16"

    # A100-specific optimizations
    use_flash_attention_2: bool = False  # 2-3x speedup on A100
    compile_model: bool = False  # PyTorch 2.0+ compile (20-30% speedup)
    num_workers: int = 0  # DataLoader workers for parallel loading
    dataloader_pin_memory: bool = True  # Faster CPU->GPU transfer

    # LoRA settings (optional fine-tuning)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Reward settings
    # Paper uses binary rewards: 1 for correct proofs, 0 otherwise
    reward_correct: float = 1.0
    reward_incorrect: float = 0.0

    # Other settings
    seed: int = 42
    num_epochs: int = 1

    # Prompt settings (Paper Section 2.3)
    # "Each theorem is prefixed with both CoT and non-CoT guiding prompts"
    use_cot_prompts: bool = True
    use_non_cot_prompts: bool = True


def load_grpo_config(path: Optional[str]) -> GRPOConfig:
    """Load GRPO config from YAML file or return defaults."""
    if path is None:
        return GRPOConfig()
    data = yaml.safe_load(Path(path).read_text())
    return GRPOConfig(**data)
