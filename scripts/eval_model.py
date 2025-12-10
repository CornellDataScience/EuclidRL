#!/usr/bin/env python
"""
Evaluate a trained model on the validation set.

Usage:
    python scripts/eval_model.py --model checkpoints/grpo_quick/final --max-samples 100
    python scripts/eval_model.py --model checkpoints/sft/final --val-file data/sft/val.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_validation_data(val_file: str, max_samples: int | None = None) -> List[Dict]:
    """Load validation dataset from JSONL file."""
    data = []
    with open(val_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break
    return data


def generate_proof(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_samples: int = 1,
) -> List[str]:
    """Generate proof completions for a given prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the generated part (excluding the prompt)
    prompt_length = inputs.input_ids.shape[1]
    completions = []
    for seq in outputs:
        completion = tokenizer.decode(seq[prompt_length:], skip_special_tokens=True)
        completions.append(completion)

    return completions


def evaluate_model(
    model_path: str,
    val_file: str,
    max_samples: int | None = None,
    num_samples_per_theorem: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device_map: str = "auto",
    offload_dir: str | None = None,
    output_file: str | None = None,
) -> Dict:
    """Evaluate model on validation set."""
    print(f"Loading model from: {model_path}")

    # Check if this is a local path or HuggingFace model ID
    # Strategy: Default to local UNLESS it clearly looks like a HF model ID
    # HuggingFace IDs: "org/model" where org doesn't contain dots/special chars
    is_local = True  # Default to local

    if "/" in model_path and model_path.count("/") == 1:
        # Exactly one slash - could be "org/model" (HF) or "checkpoints/final" (local)
        first_part = model_path.split("/")[0]
        # If first part looks like a HF org (no dots, not common dir names), treat as HF
        if first_part not in ["checkpoints", "models", "output", ".", ".."] and "." not in first_part:
            is_local = False  # Likely HuggingFace ID like "Qwen/Qwen2.5-Math-1.5B"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        offload_folder=offload_dir,
        trust_remote_code=True,
        local_files_only=is_local,  # Only use local files for local paths
    )
    model.eval()

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading validation data from: {val_file}")
    val_data = load_validation_data(val_file, max_samples)
    print(f"Validation examples: {len(val_data)}")

    results = []
    total_examples = 0

    print(f"\nGenerating proofs ({num_samples_per_theorem} per theorem)...")
    with torch.no_grad():
        for example in tqdm(val_data, desc="Evaluating"):
            prompt = example["prompt"]
            expected_completion = example.get("completion", "")

            # Generate completions
            completions = generate_proof(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_samples=num_samples_per_theorem,
            )

            result = {
                "prompt": prompt,
                "expected": expected_completion,
                "generated": completions,
            }
            results.append(result)
            total_examples += 1

    # Save results if output file specified
    if output_file:
        print(f"\nSaving results to: {output_file}")
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

    # Print summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total examples evaluated: {total_examples}")
    print(f"Samples per theorem: {num_samples_per_theorem}")
    print(f"Total proofs generated: {total_examples * num_samples_per_theorem}")

    # Print a few examples
    print("\n" + "=" * 80)
    print("EXAMPLE GENERATIONS (first 3)")
    print("=" * 80)
    for i, result in enumerate(results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"PROMPT (first 200 chars):")
        print(result["prompt"][:200] + "...")
        print(f"\nEXPECTED COMPLETION:")
        print(result["expected"])
        print(f"\nGENERATED COMPLETIONS:")
        for j, gen in enumerate(result["generated"]):
            print(f"  [{j+1}] {gen}")
        print()

    stats = {
        "total_examples": total_examples,
        "num_samples_per_theorem": num_samples_per_theorem,
        "total_proofs_generated": total_examples * num_samples_per_theorem,
    }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on validation set."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., checkpoints/grpo_quick/final or your configured output_dir)",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/sft/val.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit to N validation examples (default: use all)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of proof attempts per theorem (default: 1)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per proof (default: 512)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Device map to use for loading (e.g., "auto", "cpu")',
    )
    parser.add_argument(
        "--offload-dir",
        type=str,
        default=None,
        help="Directory to offload weights when using device_map=auto (required if layers are offloaded)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, lower = more deterministic)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSONL file (optional)",
    )
    args = parser.parse_args()

    stats = evaluate_model(
        model_path=args.model,
        val_file=args.val_file,
        max_samples=args.max_samples,
        num_samples_per_theorem=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map=args.device_map,
        offload_dir=args.offload_dir,
        output_file=args.output,
    )

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
