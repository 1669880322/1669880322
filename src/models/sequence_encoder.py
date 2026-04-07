import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    """1D CNN + GRU encoder for full amplitude sequence."""

    def __init__(self, in_channels: int = 1, cnn_channels: int = 32, gru_hidden: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=cnn_channels, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, amplitude_sequence: torch.Tensor) -> torch.Tensor:
        # amplitude_sequence: [B, T]
        x = amplitude_sequence.unsqueeze(1)  # [B,1,T]
        x = self.cnn(x)  # [B,C,T]
        x = x.transpose(1, 2)  # [B,T,C]
        _, h = self.gru(x)
        z = h[-1]
        return self.head(z)
