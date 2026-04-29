"""
CSE 493S/599S HW2: interface for Part 0 and Part 1.

We will be using an autograder for this part. For ease of grading, please fill in
these functions to evaluate your trained models. Do not rename the functions
or change their signatures.

You may import from other files in your repo. You may add helper functions.
Just make sure the three functions below work as specified.
"""

from pathlib import Path

import torch

from inference import load_model_and_tokenizer as load_checkpoint


def _resolve_checkpoint(checkpoint_dir: str) -> Path:
    path = Path(checkpoint_dir)
    if path.is_file():
        return path

    ckpt = path / "ckpt.pt"
    if ckpt.exists():
        return ckpt

    candidates = list(path.glob("ckpt_*_epochs.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    def epoch_num(candidate: Path) -> int:
        try:
            return int(candidate.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=epoch_num)


def load_model_and_tokenizer(checkpoint_dir: str):
    """
    Load a trained model and its tokenizer from a checkpoint directory.

    Args:
        checkpoint_dir: Path to a directory containing your saved model
            and any tokenizer files you need.

    Returns:
        A tuple (model, tokenizer). The model should be ready for inference
        (in eval mode, on an appropriate device). The tokenizer should be
        whatever object your predict_answer / generate_sanity_check functions
        expect — we do not constrain its type.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_checkpoint(str(_resolve_checkpoint(checkpoint_dir)), device)


def get_bos_token(tokenizer=None):
    """
    Get the BOS token for the tokenizer, for part 0 of the assignment.
    """
    return "<s>"


def predict_answer(model, tokenizer, a: int, b: int, op: str, p: int) -> int:
    """
    Predict the answer to a modular arithmetic problem.

    Args:
        model: The model returned by load_model_and_tokenizer.
        tokenizer: The tokenizer returned by load_model_and_tokenizer.
        a: First operand, integer in [0, p).
        b: Second operand, integer in [0, p).
        op: One of '+', '-', '/'.
        p: The modulus (97 or 113).

    Returns:
        The model's predicted answer as an integer in [0, p).
        You are responsible for formatting the input according to your
        training scheme and parsing the model's output back to an integer.
    """
    device = next(model.parameters()).device
    token_ids = [
        tokenizer.vocab_index_mapping.get(token, tokenizer.vocab_index_mapping["<UNK>"])
        for token in ("<s>", str(a), op, str(b), "=")
    ]
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        pred_id = model(tokens)[:, -1, :].argmax(dim=-1).item()

    pred = tokenizer.index_vocab_mapping[pred_id]
    return int(pred)
