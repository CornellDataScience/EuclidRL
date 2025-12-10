#!/usr/bin/env python
"""
Interactive theorem prover demo.

Run a trained model interactively to generate proofs for custom theorems.

Usage:
    # Interactive mode
    python scripts/demo_prover.py --model checkpoints/grpo_quick/final

    # Single proof
    python scripts/demo_prover.py \
        --model checkpoints/grpo_quick/final \
        --theorem "Prove that for all natural numbers n, n + 0 = n"
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def is_local_path(model_path: str) -> bool:
    """Determine if model_path is a local path or HuggingFace model ID."""
    # Default to local unless it clearly looks like a HF model ID
    is_local = True

    if "/" in model_path and model_path.count("/") == 1:
        # Exactly one slash - could be "org/model" (HF) or "checkpoints/final" (local)
        first_part = model_path.split("/")[0]
        # If first part looks like a HF org, treat as HF
        if first_part not in ["checkpoints", "models", "output", ".", ".."] and "." not in first_part:
            is_local = False

    return is_local


def load_model_and_tokenizer(model_path: str, device_map: str = "auto", offload_dir: str | None = None):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    is_local = is_local_path(model_path)
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
        local_files_only=is_local,
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


def create_lean_prompt(theorem_statement: str, theorem_name: str = "custom_theorem") -> str:
    """Create a Lean 4 proof prompt from a natural language theorem statement."""
    # Standard imports used in training data
    header = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

"""

    # Convert natural language to Lean theorem format
    # This is a simple template - for production, you'd want better NL -> Lean conversion
    formal_statement = f"theorem {theorem_name} : {theorem_statement} := by\n"

    prompt = header + formal_statement + "\n-- Provide a Lean 4 proof script that discharges the goal above."

    return prompt


def generate_proof(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    num_samples: int = 1,
) -> list[str]:
    """Generate proof completions."""
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
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    prompt_length = inputs.input_ids.shape[1]
    proofs = []
    for seq in outputs:
        proof = tokenizer.decode(seq[prompt_length:], skip_special_tokens=True)
        proofs.append(proof)

    return proofs


def interactive_mode(model, tokenizer, temperature: float, max_new_tokens: int, num_samples: int):
    """Run interactive proof generation."""
    print("\n" + "=" * 80)
    print("INTERACTIVE THEOREM PROVER")
    print("=" * 80)
    print("\nEnter theorem statements in Lean 4 syntax.")
    print("Examples:")
    print("  - ∀ n : ℕ, n + 0 = n")
    print("  - ∀ a b : ℝ, a + b = b + a")
    print("  - ∀ n : ℕ, n * 2 = n + n")
    print("\nType 'quit' or 'exit' to quit.")
    print("Type 'help' for examples.")
    print("=" * 80)

    while True:
        print("\n" + "-" * 80)
        theorem = input("\nTheorem statement: ").strip()

        if theorem.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if theorem.lower() == "help":
            print("\n" + "=" * 80)
            print("EXAMPLE THEOREMS")
            print("=" * 80)
            print("\n1. Simple arithmetic:")
            print("   ∀ n : ℕ, n + 0 = n")
            print("\n2. Commutativity:")
            print("   ∀ a b : ℝ, a + b = b + a")
            print("\n3. Doubling:")
            print("   ∀ n : ℕ, n * 2 = n + n")
            print("\n4. With hypothesis:")
            print("   ∀ a b c : ℤ, a = b → b = c → a = c")
            continue

        if not theorem:
            continue

        # Create full Lean prompt
        prompt = create_lean_prompt(theorem)

        print(f"\n[Generating {num_samples} proof(s) with temperature={temperature}...]")

        proofs = generate_proof(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )

        print("\n" + "=" * 80)
        print("GENERATED PROOF(S)")
        print("=" * 80)

        for i, proof in enumerate(proofs, 1):
            print(f"\n--- Proof {i} ---")
            print(proof.strip())

        print("\n" + "=" * 80)


def single_proof_mode(
    model, tokenizer, theorem: str, temperature: float, max_new_tokens: int, num_samples: int
):
    """Generate proof for a single theorem."""
    prompt = create_lean_prompt(theorem)

    print("\n" + "=" * 80)
    print("THEOREM")
    print("=" * 80)
    print(theorem)

    print("\n" + "=" * 80)
    print(f"GENERATING {num_samples} PROOF(S)")
    print("=" * 80)

    proofs = generate_proof(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_samples=num_samples,
    )

    for i, proof in enumerate(proofs, 1):
        print(f"\n--- Proof {i} ---")
        print(proof.strip())

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive theorem prover demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode
    python scripts/demo_prover.py --model checkpoints/grpo_quick/final

    # Single theorem
    python scripts/demo_prover.py \\
        --model checkpoints/grpo_quick/final \\
        --theorem "∀ n : ℕ, n + 0 = n"

    # Generate multiple proofs
    python scripts/demo_prover.py \\
        --model checkpoints/grpo_quick/final \\
        --theorem "∀ a b : ℝ, a + b = b + a" \\
        --num-samples 5
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--theorem",
        type=str,
        default=None,
        help="Theorem statement in Lean 4 syntax (optional, interactive mode if not provided)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of proof attempts to generate (default: 3)",
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
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate (default: 512)",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device_map=args.device_map,
        offload_dir=args.offload_dir,
    )

    print(f"\nModel loaded successfully!")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Samples per theorem: {args.num_samples}")

    if args.theorem:
        # Single proof mode
        single_proof_mode(
            model,
            tokenizer,
            args.theorem,
            args.temperature,
            args.max_tokens,
            args.num_samples,
        )
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args.temperature, args.max_tokens, args.num_samples)


if __name__ == "__main__":
    main()
