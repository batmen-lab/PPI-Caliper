"""Adapt sequence-pair CSV files to the official MINT collator."""
import random

import pandas as pd
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, path, max_seq_length=2048):
        self.frame = pd.read_csv(path)
        self.max_residues = max_seq_length - 2
        self.has_labels = "label" in self.frame.columns

    def __len__(self):
        return len(self.frame)

    def sequence(self, value):
        sequence = value.replace("*", "").replace("f", "")
        if len(sequence) > self.max_residues:
            start = random.randint(0, len(sequence) - self.max_residues)
            sequence = sequence[start:start + self.max_residues]
        return sequence

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return self.sequence(row["query"]), self.sequence(row["text"])
