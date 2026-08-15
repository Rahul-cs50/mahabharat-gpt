import argparse
import json
import math

import torch
import torch.nn.functional as F

from dataset import create_dataloader
from generate import load_model
from tokenizer import get_tokenizer


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= num_batches:
                break
            total_loss += calc_loss_batch(x, y, model, device).item()
    return total_loss / num_batches


def perplexity(loss):
    return math.exp(loss) if loss < 20 else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Mahabharata GPT checkpoint.")
    parser.add_argument("--config", default="configs/32m_config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = get_tokenizer()
    model, _ = load_model(args.config, args.checkpoint, device)
    loader = create_dataloader(
        text,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=False,
        drop_last=False,
    )
    loss = calc_loss_loader(loader, model, device, args.num_batches)
    print(f"loss={loss:.4f}")
    print(f"perplexity={perplexity(loss):.2f}")


if __name__ == "__main__":
    main()
