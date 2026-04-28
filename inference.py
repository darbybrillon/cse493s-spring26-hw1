import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
from pathlib import Path
from tokenizer import tokenizer


"""
Assumes model is saved as: torch.save({"model_config": config_dict, "model": model.state_dict()}, "ckpt.pt")
"""

def load_model(checkpoint_loc: str, device="cpu") -> GPT:
    checkpoint = torch.load(checkpoint_loc, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model

def generate_output(model: GPT, tokenizer, input: str, output_size: int) -> list[str]: # may need to increase output size by two to account for SOS and EOS tokens
    tokens = tokenizer.encode(input)
    tokens = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        for _ in range(output_size):
            output = model(tokens)
            prediction = torch.multinomial(F.softmax(output[:, -1, :], dim=-1), num_samples=1)
            tokens = torch.cat([tokens, prediction], dim=1)
    return tokenizer.decode(tokens[0].tolist())

def main() -> None:
    file_name = "file name"
    input = "I love machine learning"
    tokenizer = tokenizer()
    model = load_model(file_name)
    output = generate_output(model, tokenizer, input, 4)
    print(output)   

if __name__ == "__main__":
    main()