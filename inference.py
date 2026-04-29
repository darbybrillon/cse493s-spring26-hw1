import torch
from model import GPT, GPTConfig
from tokenizer import tokenizer


"""
Assumes model is saved as:
torch.save({"model_config": config_dict, "model": model.state_dict(), "tokenizer": tokenizer_state}, "ckpt.pt")
"""

def load_model(checkpoint_loc: str, device="cpu") -> GPT:
    checkpoint = torch.load(checkpoint_loc, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model

def load_tokenizer(checkpoint_loc: str) -> tokenizer:
    checkpoint = torch.load(checkpoint_loc, map_location="cpu")
    tok = tokenizer()
    state = checkpoint.get("tokenizer")
    if state is None:
        return tok
    tok.vocab = set(state["vocab"])
    tok.vocab_index_mapping = state["vocab_index_mapping"]
    tok.index_vocab_mapping = state["index_vocab_mapping"]
    return tok

def load_model_and_tokenizer(checkpoint_loc: str, device="cpu") -> tuple[GPT, tokenizer]:
    return load_model(checkpoint_loc, device), load_tokenizer(checkpoint_loc)

def generate_output(
    model: GPT,
    tokenizer,
    input: str,
    output_size: int,
    add_eos_to_prompt: bool = False,
) -> str: # may need to increase output size by two to account for SOS and EOS tokens
    tokens = [tokenizer.vocab_index_mapping["<s>"]]
    tokens += [
        tokenizer.vocab_index_mapping.get(token, tokenizer.vocab_index_mapping["<UNK>"])
        for token in input.lower().split()
    ]
    if add_eos_to_prompt:
        tokens.append(tokenizer.vocab_index_mapping["</s>"])
    device = next(model.parameters()).device
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(output_size):
            output = model(tokens)
            logits = output[:, -1, :]
            prediction = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, prediction], dim=1)
    return tokenizer.decoding(tokens[0].tolist())

def main() -> None:
    file_name = "file name"
    input = "I love machine learning"
    model, tokenizer = load_model_and_tokenizer(file_name)
    output = generate_output(model, tokenizer, input, 4)
    print(output)   

if __name__ == "__main__":
    main()
