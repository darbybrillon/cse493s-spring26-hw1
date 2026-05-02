from pathlib import Path

import torch
import torch.nn.functional as F

from inference import load_model_and_tokenizer


BASE_DIR = Path("add_subtract")
EPOCHS = 500
RUNS = [
    {"p": 97, "seed": 42, "run_dir": BASE_DIR / "p97_seed42"},
    {"p": 97, "seed": 43, "run_dir": BASE_DIR / "p97_seed43"},
    {"p": 113, "seed": 42, "run_dir": BASE_DIR / "p113_seed42"},
    {"p": 113, "seed": 43, "run_dir": BASE_DIR / "p113_seed43"},
]


def final_checkpoint(run_dir, epochs):
    return Path(run_dir) / f"ckpt_{epochs}_epochs.pt"


def read_split(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def score_example(model, tok, example):
    prompt, answer = example.rsplit("=", 1)
    prompt = prompt.strip() + " ="
    answer = answer.strip()

    ids = [tok.vocab_index_mapping["<s>"]]
    ids += [
        tok.vocab_index_mapping.get(token, tok.vocab_index_mapping["<UNK>"])
        for token in prompt.lower().split()
    ]
    target = tok.vocab_index_mapping[answer]

    device = next(model.parameters()).device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    target_tensor = torch.tensor([target], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tokens)[:, -1, :]
        loss = F.cross_entropy(logits, target_tensor).item()
        pred_id = logits.argmax(dim=-1).item()

    pred = tok.index_vocab_mapping[pred_id]
    return prompt, pred, loss, int(pred == answer)


def write_test_outputs(run, test_path, output_path):
    checkpoint_path = final_checkpoint(run["run_dir"], EPOCHS)

    test_set = read_split(test_path)
    model, tok = load_model_and_tokenizer(str(checkpoint_path))
    loss_sum = 0.0
    correct = 0

    with open(output_path, "w") as f:
        for example in test_set:
            prompt, output, loss, is_correct = score_example(model, tok, example)
            loss_sum += loss
            correct += is_correct
            f.write(f"input: {prompt}\n")
            f.write(f"target: {example}\n")
            f.write(f"prediction: {output}\n\n")

        avg_loss = loss_sum / len(test_set)
        accuracy = correct / len(test_set)
        f.write(f"test_loss: {avg_loss}\n")
        f.write(f"test_accuracy: {accuracy}\n")

    print(f"{run['run_dir']}: test_loss={avg_loss:.6f}, test_accuracy={accuracy:.4f}")
    print(f"wrote {output_path}")


def main():
    test_files = {
        97: BASE_DIR / "test_97.txt",
        113: BASE_DIR / "test_113.txt",
    }

    for run in RUNS:
        write_test_outputs(run, test_files[run["p"]], run["run_dir"] / "test_outputs.txt")


if __name__ == "__main__":
    main()
