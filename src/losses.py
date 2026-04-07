from typing import Dict

import torch
import torch.nn.functional as F


def amplitude_from_q2(q2: torch.Tensor, scale_um: float, eps_amp: float) -> torch.Tensor:
    return scale_um * torch.sqrt(q2**2 + eps_amp)


def compute_losses(
    a_hat: torch.Tensor,
    a_obs: torch.Tensor,
    q2: torch.Tensor,
    q2_dot: torch.Tensor,
    q2_ddot: torch.Tensor,
    c2: torch.Tensor,
    k2: torch.Tensor,
    c_hat: torch.Tensor,
    c_true: torch.Tensor,
    cfg,
    loss_mode: str,
) -> Dict[str, torch.Tensor]:
    lamp = F.mse_loss(a_hat, a_obs)
    residual = q2_ddot + c2 * q2_dot + k2 * q2 + cfg.kappa * k2 * q2**3
    lphys = torch.mean(residual**2)
    lparam = F.mse_loss(c_hat, c_true)

    lic = F.mse_loss(q2[:, :1], torch.full_like(q2[:, :1], cfg.q20_fixed))
    lic = lic + F.mse_loss(q2_dot[:, :1], torch.full_like(q2_dot[:, :1], cfg.q2d0_fixed))

    if loss_mode == "physics_only":
        total = cfg.w_amp * lamp + cfg.w_phys * lphys + cfg.w_ic * lic
    elif loss_mode == "supervised_synthetic":
        total = cfg.w_amp * lamp + cfg.w_param * lparam + cfg.w_ic * lic
    else:  # hybrid
        total = cfg.w_amp * lamp + cfg.w_phys * lphys + cfg.w_param * lparam + cfg.w_ic * lic

    return {
        "total": total,
        "Lamp": lamp,
        "Lphys": lphys,
        "Lparam": lparam,
        "Lic": lic,
        "residual": residual,
    }
