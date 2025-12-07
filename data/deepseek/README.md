# DeepSeek-Prover-V1.5 datasets (RL/SFT)

Copies of the small JSONL datasets referenced by the DeepSeek-Prover-V1.5 release for supervised fine-tuning and RL evaluation/search experiments:

- `minif2f.jsonl`: miniF2F benchmark samples with Lean statements/goals.
- `proofnet.jsonl`: ProofNet benchmark samples with Lean statements/goals.
- `minif2f_valid_few_shot.jsonl`: miniF2F valid split few-shot examples.

The full DeepSeek-Prover-V1.5 training/eval JSONLs include `header`, `formal_statement`, `goal`, and `formal_proof`. The copies here do **not** contain `formal_proof`; use the official release with proofs when running `scripts/convert_deepseek_v15.py`.

Source: https://github.com/deepseek-ai/DeepSeek-Prover-V1.5 (see LICENSE-MODEL/LICENSE-CODE in that repo for terms). Do not commit downloaded model weights; only these small metadata files are tracked here.
