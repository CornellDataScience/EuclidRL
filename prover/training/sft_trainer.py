from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback
from trl import SFTConfig as TRLSFTConfig, SFTTrainer

# Suppress tokenization mismatch warnings
warnings.filterwarnings("ignore", message="Mismatch between tokenized prompt")

from prover.configs import SFTConfig
from prover.data import load_jsonl_dataset
from prover.models import get_tokenizer, load_qwen_policy
from prover.models.qwen_policy import PolicyInitConfig


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs:
            print(f"[SFT] step={state.global_step} logs={logs}")


def run_sft(cfg: SFTConfig, config_path: Optional[str] = None) -> None:
    print("Launching SFT with config:", config_path or "defaults")
    policy_cfg = PolicyInitConfig(
        model_name=cfg.model_name,
        load_in_4bit=False,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
    )
    model, tokenizer = load_qwen_policy(policy_cfg)

    max_train = getattr(cfg, 'max_train_samples', None)
    max_val = getattr(cfg, 'max_val_samples', None)

    train_ds, val_ds = load_jsonl_dataset(
        cfg.train_file,
        cfg.val_file,
        max_train_samples=max_train,
        max_val_samples=max_val,
    )
    print(f"Loaded datasets - Train: {len(train_ds)} examples, Val: {len(val_ds) if val_ds else 0} examples")

    # Handle validation
    disable_val = getattr(cfg, 'disable_validation', False)
    if disable_val and val_ds is not None:
        print("Validation disabled via config (disable_validation=True)")
        val_ds = None
    elif val_ds is not None:
        print("Validation enabled - will evaluate every eval_steps")
        print("  (Using completion_only_loss=False to avoid filtering issues)")

    # Use TRL's SFTConfig - it handles prompt+completion datasets automatically
    args = TRLSFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if val_ds is not None else None,
        eval_strategy="steps" if val_ds is not None else "no",
        save_strategy="steps",
        bf16=cfg.mixed_precision == "bf16",
        fp16=cfg.mixed_precision == "fp16",
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to="none",
        seed=cfg.seed,
        # SFT-specific parameters
        max_length=cfg.max_seq_length,
        packing=False,
        # Don't use completion_only_loss - it can filter out all validation examples
        # if tokenization doesn't align perfectly
        completion_only_loss=False,
    )

    # SFTTrainer handles prompt+completion datasets automatically - no custom collator needed
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.add_callback(LogCallback())
    trainer.save_model(cfg.output_dir + "/init")
    trainer.train()
    trainer.save_state()
    trainer.save_model(cfg.output_dir + "/final")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, "sft_config_used.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in asdict(cfg).items())
    )
