#!/usr/bin/env python
"""
Advanced Lean proof examples - Actually interesting theorems!

These are non-trivial mathematical results.

Usage:
    python advanced_examples.py              # Run all
    python advanced_examples.py 1            # Run example 1
    python advanced_examples.py pythagoras   # Run by name
"""

import sys
import subprocess

EXAMPLES = {
    "bernoulli_variance": {
        "name": "Bernoulli Variance Bound",
        "theorem": "∀ p : ℝ, 0 ≤ p → p ≤ 1 → p * (1 - p) ≤ 1/4",
        "proof": """intro p hp0 hp1
have h : 0 ≤ (p - 1/2)^2 := sq_nonneg _
have : p - p^2 ≤ (1/4 : ℝ) := by
  nlinarith
nlinarith""",
        "difficulty": "★★☆☆☆",
        "description": "Variance of a Bernoulli(p) is maximized at p = 1/2",
    },

    "am_gm": {
        "name": "AM-GM Inequality",
        "theorem": "∀ a b : ℝ, a ≥ 0 → b ≥ 0 → Real.sqrt (a * b) ≤ (a + b) / 2",
        "proof": """intro a b ha hb
have h : 0 ≤ (a - b)^2 := sq_nonneg _
calc Real.sqrt (a * b)
    ≤ Real.sqrt ((a + b)^2 / 4) := by
      apply Real.sqrt_le_sqrt
      nlinarith
  _ = (a + b) / 2 := by
      rw [Real.sqrt_div (sq_nonneg _)]
      rw [Real.sqrt_sq (by linarith : 0 ≤ a + b)]
      ring""",
        "difficulty": "★★★★☆",
        "description": "Arithmetic mean ≥ Geometric mean",
    },

    "cauchy_schwarz": {
        "name": "Cauchy-Schwarz Inequality",
        "theorem": "∀ a b c d : ℝ, (a*c + b*d)^2 ≤ (a^2 + b^2) * (c^2 + d^2)",
        "proof": """intro a b c d
have h : ∀ t : ℝ, 0 ≤ (a*t - c)^2 + (b*t - d)^2 := by
  intro t
  nlinarith [sq_nonneg (a*t - c), sq_nonneg (b*t - d)]
have key : ∀ t : ℝ, 0 ≤ (a^2 + b^2)*t^2 - 2*(a*c + b*d)*t + (c^2 + d^2) := by
  intro t
  have := h t
  nlinarith
by_cases ha : a^2 + b^2 = 0
· simp [ha]; nlinarith [sq_nonneg a, sq_nonneg b]
· have := key ((a*c + b*d) / (a^2 + b^2))
  nlinarith""",
        "difficulty": "★★★★★",
        "description": "Fundamental inequality in linear algebra",
    },

    "bernoulli": {
        "name": "Bernoulli's Inequality",
        "theorem": "∀ n : ℕ, ∀ x : ℝ, x ≥ -1 → (1 + x)^n ≥ 1 + n * x",
        "proof": """intro n x hx
induction n with
| zero =>
  simp
  linarith
| succ n ih =>
  calc (1 + x)^(n + 1)
      = (1 + x)^n * (1 + x) := by ring
    _ ≥ (1 + n * x) * (1 + x) := by
        apply mul_le_mul_of_nonneg_right ih
        linarith
    _ = 1 + (n + 1) * x + n * x^2 := by ring
    _ ≥ 1 + (n + 1) * x := by nlinarith [sq_nonneg x]""",
        "difficulty": "★★★☆☆",
        "description": "(1+x)ⁿ ≥ 1 + nx for x ≥ -1",
    },

    "geometric_series": {
        "name": "Geometric Series",
        "theorem": "∀ n : ℕ, ∀ r : ℝ, r ≠ 1 → ∑ i in Finset.range (n + 1), r^i = (1 - r^(n + 1)) / (1 - r)",
        "proof": """intro n r hr
induction n with
| zero =>
  simp
  field_simp
  ring
| succ n ih =>
  rw [Finset.sum_range_succ, ih]
  field_simp
  ring""",
        "difficulty": "★★★☆☆",
        "description": "1 + r + r² + ... + rⁿ = (1-rⁿ⁺¹)/(1-r)",
    },

    "binomial": {
        "name": "Binomial Theorem (n=2)",
        "theorem": "∀ a b : ℝ, (a + b)^2 = a^2 + 2*a*b + b^2",
        "proof": """intro a b
ring""",
        "difficulty": "★☆☆☆☆",
        "description": "Simple case, but foundation for Pascal's triangle",
    },

    "binomial_3": {
        "name": "Binomial Theorem (n=3)",
        "theorem": "∀ a b : ℝ, (a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3",
        "proof": """intro a b
ring""",
        "difficulty": "★★☆☆☆",
        "description": "Cube expansion",
    },

    "triangle_inequality": {
        "name": "Triangle Inequality",
        "theorem": "∀ a b : ℝ, |a + b| ≤ |a| + |b|",
        "proof": """intro a b
by_cases ha : 0 ≤ a <;> by_cases hb : 0 ≤ b
· simp [abs_of_nonneg ha, abs_of_nonneg hb, abs_of_nonneg (by linarith : 0 ≤ a + b)]
· simp [abs_of_nonneg ha, abs_of_neg (by linarith : b < 0)]
  by_cases hab : 0 ≤ a + b
  · simp [abs_of_nonneg hab]; linarith
  · simp [abs_of_neg (by linarith : a + b < 0)]; linarith
· simp [abs_of_neg (by linarith : a < 0), abs_of_nonneg hb]
  by_cases hab : 0 ≤ a + b
  · simp [abs_of_nonneg hab]; linarith
  · simp [abs_of_neg (by linarith : a + b < 0)]; linarith
· simp [abs_of_neg (by linarith : a < 0), abs_of_neg (by linarith : b < 0)]
  simp [abs_of_neg (by linarith : a + b < 0)]
  linarith""",
        "difficulty": "★★★★☆",
        "description": "Fundamental inequality for absolute values",
    },

    "factorial_positive": {
        "name": "Factorial is Always Positive",
        "theorem": "∀ n : ℕ, 0 < n! ",
        "proof": """intro n
exact Nat.factorial_pos n""",
        "difficulty": "★☆☆☆☆",
        "description": "n! > 0 for all n",
    },

    "trig_identity": {
        "name": "Trig Pythagorean Identity",
        "theorem": "∀ x : ℝ, Real.sin x ^ 2 + Real.cos x ^ 2 = 1",
        "proof": """intro x
have h := Real.sin_sq_add_cos_sq x
simpa [pow_two] using h""",
        "difficulty": "★★☆☆☆",
        "description": "The basic sin² + cos² = 1 identity",
    },

    "exp_nonneg": {
        "name": "Exponential is Nonnegative",
        "theorem": "∀ x : ℝ, 0 ≤ Real.exp x",
        "proof": """intro x
exact le_of_lt (Real.exp_pos x)""",
        "difficulty": "★☆☆☆☆",
        "description": "exp(x) > 0 for all real x",
    },

    "cos_bound": {
        "name": "Sine and Cosine Bounded by 1",
        "theorem": "∀ x : ℝ, |Real.sin x| ≤ 1 ∧ |Real.cos x| ≤ 1",
        "proof": """intro x
constructor
· have h := Real.abs_sin_le_one x
  simpa using h
· have h := Real.abs_cos_le_one x
  simpa using h""",
        "difficulty": "★★☆☆☆",
        "description": "Elementary analytic bounds for sin and cos",
    },
}


def run_example(key):
    """Run a single example."""
    if key not in EXAMPLES:
        print(f"Error: Example '{key}' not found")
        print(f"Available: {', '.join(EXAMPLES.keys())}")
        return

    ex = EXAMPLES[key]

    print("=" * 80)
    print(f"📐 {ex['name']}")
    print("=" * 80)
    print(f"Difficulty: {ex['difficulty']}")
    print(f"Description: {ex['description']}")
    print(f"\n🎯 Theorem:\n{ex['theorem']}\n")
    print("📝 Proof:\n")
    for line in ex['proof'].split('\n'):
        print(f"    {line}")
    print()

    # Run the explainer
    result = subprocess.run(
        ['python', 'explain_proof.py', ex['proof']],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print()


def list_examples():
    """List all examples with difficulty."""
    print("\n" + "🎓 ADVANCED LEAN PROOF EXAMPLES ".center(80, "="))
    print("\n Available Examples:\n")

    for i, (key, ex) in enumerate(EXAMPLES.items(), 1):
        print(f"  {i:2d}. {key:20s} {ex['difficulty']} - {ex['name']}")

    print("\n" + "=" * 80)
    print("\nUsage:")
    print("  python advanced_examples.py              # Run all")
    print("  python advanced_examples.py 3            # Run #3")
    print("  python advanced_examples.py pythagoras   # Run by name")
    print("  python advanced_examples.py list         # Show this list")
    print()


def main():
    if len(sys.argv) == 1:
        # Run all examples
        print("\n" + "🎓 ADVANCED LEAN PROOF EXAMPLES ".center(80, "="))
        print()

        for i, (key, ex) in enumerate(EXAMPLES.items(), 1):
            print(f"\n{'━' * 80}")
            print(f"Example {i}/{len(EXAMPLES)}: {ex['name']} {ex['difficulty']}")
            print('━' * 80)
            run_example(key)

            if i < len(EXAMPLES):
                input("Press Enter for next example...")

    elif sys.argv[1] == "list":
        list_examples()

    elif sys.argv[1].isdigit():
        # Run by number
        num = int(sys.argv[1])
        if 1 <= num <= len(EXAMPLES):
            key = list(EXAMPLES.keys())[num - 1]
            run_example(key)
        else:
            print(f"Error: Example number must be 1-{len(EXAMPLES)}")
            list_examples()

    else:
        # Run by name
        run_example(sys.argv[1])


if __name__ == "__main__":
    main()
