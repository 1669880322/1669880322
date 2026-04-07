from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AmplitudeDataset(Dataset):
    def __init__(self, path: Path):
        arr = np.load(path)
        self.time_grid = torch.from_numpy(arr["time_grid"]).float()
        self.amplitude = torch.from_numpy(arr["amplitude_sequence"]).float()
        self.theta = torch.from_numpy(arr["theta_true"]).float().unsqueeze(-1)
        self.raw_q2 = torch.from_numpy(arr["raw_q2"]).float()

    def __len__(self) -> int:
        return self.amplitude.shape[0]

    def __getitem__(self, idx: int):
        return {
            "time_grid": self.time_grid,
            "amplitude_sequence": self.amplitude[idx],
            "theta_true": self.theta[idx],
            "raw_q2": self.raw_q2[idx],
        }


def build_dataloaders(paths: Dict[str, Path], batch_size: int, num_workers: int = 0) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = AmplitudeDataset(paths[split])
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders
