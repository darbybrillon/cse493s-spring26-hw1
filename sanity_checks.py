from pathlib import Path

from inference import generate_output, load_model_and_tokenizer
from tokenizer import tokenizer
from train import all_tokens_mask, suffix_after_n_input_tokens, train


SANITY_TEXT = "I love machine learning"


def _build_tokenizer(data):
    tok = tokenizer()
    tok.build_vocab(data)
    return tok


def _final_checkpoint(run_dir, epochs):
    return Path(run_dir) / f"ckpt_{epochs}_epochs.pt"


def run_all_tokens_sanity():
    data = [SANITY_TEXT]
    tok = _build_tokenizer(data)
    epochs = 1000

    val_losses, train_losses, train_accs, val_accs, run_dir = train(
        tok=tok,
        train_data=data,
        val_data=data,
        batch_size=1,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=64,
        dropout=0.0,
        bias=True,
        learning_rate=1e-2,
        epochs=epochs,
        seed=0,
        mask_fn=all_tokens_mask,
        output_dir="sanity_checks/all_toks",
        intermediate_ckpt=False,
        verbose=True,
    )

    model, tok = load_model_and_tokenizer(str(_final_checkpoint(run_dir, epochs)))
    generated = generate_output(model, tok, "", output_size=5)

    return {
        "name": "all_tokens",
        "run_dir": run_dir,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "generated": generated,
    }


def run_suffix_sanity():
    data = [SANITY_TEXT]
    tok = _build_tokenizer(data)
    epochs = 1000

    val_losses, train_losses, train_accs, val_accs, run_dir = train(
        tok=tok,
        train_data=data,
        val_data=data,
        batch_size=1,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=64,
        dropout=0.0,
        bias=True,
        learning_rate=1e-2,
        epochs=epochs,
        seed=1,
        mask_fn=suffix_after_n_input_tokens(3),
        output_dir="sanity_checks/suffix3",
        intermediate_ckpt=False,
        verbose=True,
    )

    model, tok = load_model_and_tokenizer(str(_final_checkpoint(run_dir, epochs)))
    generated = generate_output(model, tok, "I love machine", output_size=2)

    return {
        "name": "suffix_after_3",
        "run_dir": run_dir,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "generated": generated,
    }


def main():
    for result in (run_all_tokens_sanity(), run_suffix_sanity()):
        print(f"{result['name']}:")
        print(f"  run_dir: {result['run_dir']}")
        print(f"  final_train_loss: {result['final_train_loss']:.6f}")
        print(f"  final_val_loss: {result['final_val_loss']:.6f}")
        print(f"  final_train_acc: {result['final_train_acc']:.4f}")
        print(f"  final_val_acc: {result['final_val_acc']:.4f}")
        print(f"  generated: {result['generated']}")


if __name__ == "__main__":
    main()
