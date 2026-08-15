import torch
import tiktoken


def get_tokenizer():
    """Return the GPT-2 BPE tokenizer used by the experiments."""
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer=None, device=None):
    tokenizer = tokenizer or get_tokenizer()
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def token_ids_to_text(token_ids, tokenizer=None):
    tokenizer = tokenizer or get_tokenizer()
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
