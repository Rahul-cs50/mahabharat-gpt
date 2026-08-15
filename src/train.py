import argparse
import json
import os
import time

import torch
from torch.optim import AdamW

from dataset import create_dataloader
from evaluate import calc_loss_batch, calc_loss_loader, perplexity
from generate import generate
from model import GPTModel, count_parameters
from tokenizer import get_tokenizer, text_to_token_ids, token_ids_to_text


def save_checkpoint(path, model, optimizer, epoch, global_step, tokens_seen, train_losses, val_losses):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        path,
    )


def train(model, train_loader, val_loader, optimizer, device, cfg, epochs, eval_every, checkpoint_every):
    train_losses, val_losses = [], []
    global_step = 0
    tokens_seen = 0
    start_time = time.time()
    os.makedirs("checkpoints", exist_ok=True)

    model.train()
    for epoch in range(epochs):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_every == 0:
                tr = calc_loss_loader(train_loader, model, device, num_batches=min(5, len(train_loader)))
                va = calc_loss_loader(val_loader, model, device, num_batches=min(5, len(val_loader)))
                train_losses.append(tr)
                val_losses.append(va)
                print(
                    f"step={global_step:6d} | "
                    f"train={tr:.4f} (ppl {perplexity(tr):.2f}) | "
                    f"val={va:.4f} (ppl {perplexity(va):.2f})"
                )

            if global_step > 0 and global_step % checkpoint_every == 0:
                path = os.path.join("checkpoints", f"checkpoint_step_{global_step}.pt")
                save_checkpoint(path, model, optimizer, epoch, global_step, tokens_seen, train_losses, val_losses)
                print("Saved:", path)

            global_step += 1

        tokenizer = get_tokenizer()
        ids = text_to_token_ids("ॐ", tokenizer, device=device)
        out = generate(model, ids, max_new_tokens=50, context_size=cfg["context_length"], temperature=0.8, top_k=40)
        print("Sample:", token_ids_to_text(out, tokenizer).replace("\n", " "))

    print(f"Training time: {(time.time() - start_time) / 60:.2f} min")
    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Train Mahabharata GPT from scratch.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/32m_config.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(args.data, "r", encoding="utf-8") as f:
        text_data = f.read()

    split_idx = int(args.train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    tokenizer = get_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    print("Device:", device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    train_loader = create_dataloader(
        train_data,
        tokenizer,
        args.batch_size,
        cfg["context_length"],
        cfg["context_length"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_data,
        tokenizer,
        args.batch_size,
        cfg["context_length"],
        cfg["context_length"],
        shuffle=False,
        drop_last=False,
    )

    train(model, train_loader, val_loader, optimizer, device, cfg, args.epochs, args.eval_every, args.checkpoint_every)


if __name__ == "__main__":
    main()
