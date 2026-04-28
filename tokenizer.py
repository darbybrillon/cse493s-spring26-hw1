import numpy as np

"""
<s> = begining of sentence token
</s> = end of sentence token
<UNK> = unknown token
<PAD> = padding token
"""

class tokenizer:
    def __init__(self) -> None:
        self.vocab = set()
        self.vocab_index_mapping = None
        self.index_vocab_mapping = None

    def build_vocab(self, data: list[str]) -> None:
        self.vocab = set()
        special_tokens = ["<s>", "</s>", "<UNK>", "<PAD>"]
        for string in data:
            self.vocab.update(string.lower().split())
        self.vocab = sorted(self.vocab)
        self.vocab_index_mapping = {token: i for i, token in enumerate(self.vocab, start=4)}
        for i in range(len(special_tokens)):
            self.vocab_index_mapping[special_tokens[i]] = i
        self.index_vocab_mapping = {i: token for token, i in self.vocab_index_mapping.items()}
    
    def encoding(self, input: str) -> list[int]:
        encoded_input = [self.vocab_index_mapping["<s>"]] + [self.vocab_index_mapping.get(token, self.vocab_index_mapping["<UNK>"]) for token in input.lower().split()] + [self.vocab_index_mapping["</s>"]]
        return encoded_input

    def decoding(self, encoded_input: list[int]) -> str:
        decoded_input = "" 
        for index in encoded_input:
            decoded_input += self.index_vocab_mapping[index] + " "
        return decoded_input[:-1] 
        #tokens = [self.index_vocab_mapping[i] for i in encoded_input if self.index_vocab_mapping[i] not in {"<s>", "</s>", "<PAD>"}]
        #return " ".join(tokens)