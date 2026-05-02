# CSE 493S/599S HW 1

This repository contains our code and artifacts for HW 1. The Part 1 code trains small GPT-style transformers on modular arithmetic; Part 2 contains inference-time scaling experiments for reasoning models.

Run commands from this directory:

```bash
cd cse493s-spring26-hw1
```

## Environment

Part 1 was written for Python 3 and PyTorch. A CPU environment is enough for the sanity checks, but CUDA is recommended for the longer modular arithmetic runs.

Minimal Part 1 install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Part 2 additionally uses Hugging Face datasets/transformers and vLLM:

```bash
pip install datasets transformers vllm
```

If using gated Hugging Face models or datasets, set:

```bash
export HF_TOKEN=<your_token>
```

## Important Files

- `model.py`: GPT-style decoder-only transformer.
- `tokenizer.py`: whitespace tokenizer with BOS/EOS/UNK/PAD tokens.
- `data_generation.py`: modular arithmetic train/val/test data generation.
- `train.py`: generic trainer with masked loss, checkpointing, metrics logging, AdamW settings, and optional progress bar.
- `inference.py`: checkpoint loading and greedy generation helpers.
- `sanity_checks.py`: Part 0.1 sanity checks.
- `train_plot_add_subtract.ipynb`: trains/loads four addition/subtraction models and plots combined/add/sub metrics.
- `train_plot_division.ipynb`: division grokking run for Part 1.3.
- `grokking_ablations.ipynb`: Part 1.4 ablations.
- `infer_add_subtract.py`: evaluates saved add/subtract checkpoints on test files and writes predictions plus test loss/accuracy.
- `run_eval.py`, `part_2_starter.ipynb`: Part 2 reasoning-model inference experiments.

## Data Generation

`generate_data` returns `(train, val, test)` and optionally writes files:

```python
from data_generation import generate_data

train, val, test = generate_data(
    p=97,
    train_split=0.64,
    val_split=0.16,
    ops=["+", "-"],
    seed=42,
    output_dir="add_subtract",
)
```

This writes:

```text
add_subtract/train_97.txt
add_subtract/val_97.txt
add_subtract/test_97.txt
```

The same convention is used for `p=113`.

## Sanity Checks

Run:

```bash
python3 sanity_checks.py
```

This trains two tiny models:

- all-token next-token memorization of `"I love machine learning"`
- suffix prediction after masking the first 3 input positions

Outputs are written under:

```text
sanity_checks/
```

The log file is:

```text
sanity_checks/sanity_checks.log
```

## Training Part 1 Models

The main trainer is `train.train(...)`. It saves each run into `output_dir` if provided, otherwise into a timestamped directory under `runs/`.

Each run directory contains:

```text
ckpt_<epochs>_epochs.pt
metrics.json
```

`metrics.json` has:

```json
{
  "combined": {
    "train_losses": [],
    "train_accs": [],
    "val_losses": [],
    "val_accs": []
  },
  "by_operator": {
    "+": {},
    "-": {},
    "/": {}
  },
  "run_dir": "..."
}
```

The `by_operator` section is present when the default modulo answer mask is used.

### Addition/Subtraction Experiments

Open and run:

```text
train_plot_add_subtract.ipynb
```

The notebook trains or loads four models:

```text
add_subtract/p97_seed42
add_subtract/p97_seed43
add_subtract/p113_seed42
add_subtract/p113_seed43
```

It also writes the generated data files under `add_subtract/` and plots:

- combined loss/accuracy
- addition-only loss/accuracy
- subtraction-only loss/accuracy

### Division Grokking

Open and run:

```text
train_plot_division.ipynb
```

The existing saved division artifacts are under:

```text
division_grokking/
```

### Ablations

Open and run:

```text
grokking_ablations.ipynb
```

This notebook runs the Part 1.4 ablations over training fraction and learning rate.

## Add/Subtract Test-Set Inference

After the add/subtract checkpoints and test files exist, run:

```bash
python3 infer_add_subtract.py
```

The script reads:

```text
add_subtract/test_97.txt
add_subtract/test_113.txt
```

and writes predictions plus aggregate test metrics to:

```text
add_subtract/p97_seed42/test_outputs.txt
add_subtract/p97_seed43/test_outputs.txt
add_subtract/p113_seed42/test_outputs.txt
add_subtract/p113_seed43/test_outputs.txt
```

It also prints each model's test loss and accuracy.

## Part 2

Part 2 code is in:

```text
part_2_starter.ipynb
run_eval.py
```

Generated Part 2 figures and JSON outputs are saved under:

```text
part2/
```

`run_eval.py` uses `Qwen/Qwen3-4B`, `OpenRLHF/aime-2024`, `transformers`, `datasets`, and `vllm`. This part should be run on a GPU machine with enough memory for the selected model.

## Notes For Graders

- Checkpoints include both model weights and tokenizer state.
- Part 1 data is whitespace-tokenized, so modular arithmetic examples have fixed token length.
