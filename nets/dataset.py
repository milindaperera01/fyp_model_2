import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class StressEEGDataset(Dataset):
    """
    Loads multiple NPZ files.
    Each NPZ file is treated as one domain (subject/session).
    """

    def __init__(self, npz_files, input_key="X_RAW"):
        self.X = []
        self.y = []
        self.domains = []

        for domain_id, path in enumerate(npz_files):
            data = np.load(path)

            X = data[input_key]              # (N, 32, T)
            y = data["stress"].astype(int)   # (N,)

            self.X.append(X)
            self.y.append(y)
            self.domains.append(
                np.full(len(y), domain_id, dtype=np.int64)
            )

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        self.domains = np.concatenate(self.domains, axis=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            {
                "inputs": torch.tensor(self.X[idx], dtype=torch.float64),
                "domains": torch.tensor(self.domains[idx], dtype=torch.long),
            },
            torch.tensor(self.y[idx], dtype=torch.float64),
        )


def split_dataset(dataset, val_ratio=0.2):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])
