import torch
import torch.nn as nn

from src.models.sequence_encoder import SequenceEncoder


class PhysicsInformedIdentifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = SequenceEncoder()
        self.correction_gain = nn.Parameter(torch.tensor(1.0))
        self.cfg = cfg

    def forward(self, amplitude_sequence: torch.Tensor) -> torch.Tensor:
        delta = self.encoder(amplitude_sequence)
        c_hat = self.cfg.c_init + self.correction_gain * delta
        return torch.clamp(c_hat, min=self.cfg.c_min, max=self.cfg.c_max)
