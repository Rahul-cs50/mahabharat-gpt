import argparse
import json

import torch

from model import GPTModel
from tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text


def generate(model, idx, max_new_tokens, context_size, temperature=0.8, top_k=40, eos_id=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

        if temperature and temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def load_model(config_path, checkpoint_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = GPTModel(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained Mahabharata GPT checkpoint.")
    parser.add_argument("--config", default="configs/32m_config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="ॐ")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    model, cfg = load_model(args.config, args.checkpoint, device)
    ids = text_to_token_ids(args.prompt, tokenizer, device=device)
    out = generate(model, ids, args.max_new_tokens, cfg["context_length"], args.temperature, args.top_k)
    print(token_ids_to_text(out, tokenizer))


if __name__ == "__main__":
    main()
