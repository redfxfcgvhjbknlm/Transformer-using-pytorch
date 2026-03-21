from datasets import load_dataset
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("t5-small")

def load_data(split="train", max_len=128):
    dataset = load_dataset("code_search_net", "python")[split]

    def encode(example):
        src = tokenizer(
            example["docstring"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
        tgt = tokenizer(
            example["code"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
        return {"src": src["input_ids"], "tgt": tgt["input_ids"]}

    dataset = dataset.map(encode, remove_columns=dataset.column_names)
    return dataset


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx]["src"]),
            torch.tensor(self.data[idx]["tgt"])
        )
