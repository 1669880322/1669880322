import numpy as np
import torch

from utils import load_json


def load_training_data(train_csv_path, metadata_path, device):
    arr = np.loadtxt(train_csv_path, delimiter=",", skiprows=1)
    t = torch.tensor(arr[:, 0:1], dtype=torch.float32, device=device)
    w_obs = torch.tensor(arr[:, 1:2], dtype=torch.float32, device=device)

    metadata = load_json(metadata_path)
    w0 = torch.tensor([[metadata["w0"]]], dtype=torch.float32, device=device)
    w0_dot = metadata.get("w0_dot", None)
    w0_dot = torch.tensor([[w0_dot]], dtype=torch.float32, device=device) if w0_dot is not None else None
    return t, w_obs, w0, w0_dot, metadata
