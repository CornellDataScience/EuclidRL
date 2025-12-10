#!/usr/bin/env python
"""
Quick primer on SFT and GRPO for this repository.

Run:
    python scripts/explain_sft_grpo.py
"""


def main() -> None:
    lines = [
        "=" * 80,
        "SFT (Supervised Fine-Tuning)",
        "=" * 80,
        "- Goal: teach the base model to imitate reference proofs.",
        "- Data: prompt/completion JSONL in data/sft/train.jsonl (and val).",
        "- How: minimize cross-entropy on proof completions.",
        "- Command: python scripts/train_sft.py --config prover/configs/sft_default.yaml",
        "- Output: checkpoints/sft/ (final used as starting point for GRPO).",
        "",
        "=" * 80,
        "GRPO (Group Relative Policy Optimization)",
        "=" * 80,
        "- Goal: improve the SFT model with reinforcement learning on proof rewards.",
        "- Data: rollout prompts in data/rollouts/train.jsonl; reward from proof success.",
        "- How: sample k responses per prompt, score them, update with group-relative advantages.",
        "- Needs: a trained SFT checkpoint as both policy init and reference model.",
        "- Command: python scripts/train_grpo.py --config prover/configs/grpo_default.yaml",
        "- Output: checkpoints/grpo/ (or checkpoints/grpo_quick/ for --quick).",
        "",
        "=" * 80,
        "Typical Flow",
        "=" * 80,
        "1) Prepare SFT data (prompt/completion JSONL).",
        "2) Run SFT → get checkpoints/sft/final.",
        "3) Run GRPO pointing sft_checkpoint to checkpoints/sft/final.",
        "4) Evaluate or demo with scripts/demo_prover.py using the GRPO checkpoint.",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
