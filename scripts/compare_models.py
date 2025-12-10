#!/usr/bin/env python
"""
Compare multiple models on the validation set.

Usage:
    # Compare SFT vs GRPO vs Base Qwen model
    python scripts/compare_models.py \
        --models Qwen/Qwen2.5-Math-1.5B checkpoints/sft/final checkpoints/grpo_quick/final \
        --labels "Base Qwen" "SFT" "GRPO" \
        --max-samples 50

    # Quick comparison
    python scripts/compare_models.py \
        --models checkpoints/sft/final checkpoints/grpo_quick/final \
        --labels "SFT" "GRPO" \
        --max-samples 20
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


def load_model_and_tokenizer(model_path: str, device_map: str = "auto", offload_dir: str | None = None):
    """Load model and tokenizer."""
    print(f"  Loading from: {model_path}")

    # Check if this is a local path or HuggingFace model ID
    # Strategy: Default to local UNLESS it clearly looks like a HF model ID
    is_local = True  # Default to local

    if "/" in model_path and model_path.count("/") == 1:
        # Exactly one slash - could be "org/model" (HF) or "checkpoints/final" (local)
        first_part = model_path.split("/")[0]
        # If first part looks like a HF org (no dots, not common dir names), treat as HF
        if first_part not in ["checkpoints", "models", "output", ".", ".."] and "." not in first_part:
            is_local = False  # Likely HuggingFace ID

    offload_kwargs = {}
    if offload_dir:
        offload_kwargs = {"offload_folder": offload_dir}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        low_cpu_mem_usage=True,
        **offload_kwargs,
        trust_remote_code=True,
        local_files_only=is_local,  # Only use local files for local paths
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def generate_proof(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a single proof completion."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    prompt_length = inputs.input_ids.shape[1]
    completion = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return completion


def compare_models(
    model_paths: List[str],
    labels: List[str],
    val_file: str,
    max_samples: int | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device_map: str = "auto",
    offload_dir: str | None = None,
    output_file: str | None = None,
) -> None:
    """Compare multiple models on validation set."""

    # Load validation data
    print(f"\nLoading validation data from: {val_file}")
    val_data = load_validation_data(val_file, max_samples)
    print(f"Validation examples: {len(val_data)}")

    # Load all models
    print(f"\nLoading {len(model_paths)} models...")
    models_and_tokenizers = []
    for model_path, label in zip(model_paths, labels):
        print(f"\n[{label}]")
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            device_map=device_map,
            offload_dir=offload_dir,
        )
        models_and_tokenizers.append((model, tokenizer, label))

    # Evaluate all models
    all_results = {label: [] for label in labels}

    print(f"\nGenerating proofs from all models...")
    for example in tqdm(val_data, desc="Evaluating"):
        prompt = example["prompt"]
        expected = example.get("completion", "")

        # Generate from each model
        for model, tokenizer, label in models_and_tokenizers:
            completion = generate_proof(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            all_results[label].append({
                "prompt": prompt,
                "expected": expected,
                "generated": completion,
            })

    # Save results
    if output_file:
        print(f"\nSaving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump({
                "labels": labels,
                "results": all_results,
            }, f, indent=2)

    # Print comparison
    print("\n" + "=" * 100)
    print("MODEL COMPARISON - EXAMPLE OUTPUTS")
    print("=" * 100)

    for i in range(min(5, len(val_data))):
        print(f"\n{'=' * 100}")
        print(f"EXAMPLE {i+1}")
        print(f"{'=' * 100}")

        # Show prompt (truncated)
        prompt = all_results[labels[0]][i]["prompt"]
        print(f"\nPROMPT (first 150 chars):")
        print(prompt[:150] + "...")

        print(f"\nEXPECTED:")
        expected = all_results[labels[0]][i]["expected"]
        print(expected if expected else "(no expected completion)")

        # Show each model's output
        for label in labels:
            print(f"\n[{label}] GENERATED:")
            generated = all_results[label][i]["generated"]
            # Truncate long outputs
            if len(generated) > 300:
                print(generated[:300] + "...[truncated]")
            else:
                print(generated)

    # Print summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    for label in labels:
        results = all_results[label]
        total_chars = sum(len(r["generated"]) for r in results)
        avg_length = total_chars / len(results) if results else 0

        print(f"\n[{label}]")
        print(f"  Total examples: {len(results)}")
        print(f"  Avg proof length: {avg_length:.1f} characters")

        # Count non-empty completions
        non_empty = sum(1 for r in results if len(r["generated"].strip()) > 0)
        print(f"  Non-empty completions: {non_empty}/{len(results)} ({100*non_empty/len(results):.1f}%)")

    print("\n" + "=" * 100)
    print("Comparison complete!")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple models on validation set."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model checkpoints (space-separated)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Labels for each model (space-separated, same order as --models)",
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
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per proof (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
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
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file (optional)",
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.models) != len(args.labels):
        print("Error: Number of models and labels must match!")
        print(f"  Models: {len(args.models)}")
        print(f"  Labels: {len(args.labels)}")
        sys.exit(1)

    compare_models(
        model_paths=args.models,
        labels=args.labels,
        val_file=args.val_file,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map=args.device_map,
        offload_dir=args.offload_dir,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
