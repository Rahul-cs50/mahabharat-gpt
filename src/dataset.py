import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import get_tokenizer


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length], dtype=torch.long))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1], dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text,
    tokenizer=None,
    batch_size=8,
    max_length=256,
    stride=256,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tokenizer or get_tokenizer()
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
