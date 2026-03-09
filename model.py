import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, fourier_dim: int, scale: float):
        super().__init__()
        B = torch.randn(in_dim, fourier_dim) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2.0 * torch.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINN(nn.Module):
    def __init__(self, cfg_model, t_min: float, t_max: float):
        super().__init__()
        self.normalize_time = cfg_model.normalize_time
        self.t_min = t_min
        self.t_max = t_max

        self.use_fourier = cfg_model.use_fourier_features
        if self.use_fourier:
            self.ff = FourierFeatures(cfg_model.in_dim, cfg_model.fourier_dim, cfg_model.fourier_scale)
            in_dim = cfg_model.fourier_dim * 2
        else:
            in_dim = cfg_model.in_dim

        layers = []
        act = nn.Tanh()
        prev = in_dim
        for _ in range(cfg_model.hidden_layers):
            layers += [nn.Linear(prev, cfg_model.hidden_width), act]
            prev = cfg_model.hidden_width
        layers += [nn.Linear(prev, cfg_model.out_dim)]
        self.net = nn.Sequential(*layers)

    def _normalize_t(self, t):
        if not self.normalize_time:
            return t
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

    def forward(self, t):
        x = self._normalize_t(t)
        if self.use_fourier:
            x = self.ff(x)
        out = self.net(x)
        return out[:, :1], out[:, 1:2]
