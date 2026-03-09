import math
import torch


def compute_coefficients(t: torch.Tensor, u0: torch.Tensor, cfg):
    pi = math.pi
    u2_t = u0 * (1.0 + cfg.A * torch.cos(cfg.omega1 * t))

    conv_term = cfg.phi1 * cfg.u + cfg.lambda_ * cfg.phi2 * (cfg.u + cfg.u1) + (cfg.u - u2_t)
    stiff_term = (
        cfg.gamma / 2.0
        - cfg.alpha * cfg.Omega_tilde**2
        - cfg.phi1 * cfg.u**2
        - cfg.lambda_ * cfg.phi2 * (cfg.u + cfg.u1) ** 2
        - (cfg.u - u2_t) ** 2
    )

    c1 = (cfg.c + 2.0 * cfg.alpha * cfg.Omega_tilde * pi**2) / (1.0 + cfg.alpha * pi**2)
    c2 = (cfg.c + 8.0 * cfg.alpha * cfg.Omega_tilde * pi**2) / (1.0 + 4.0 * cfg.alpha * pi**2)

    c11 = 32.0 * conv_term * math.sqrt(cfg.beta) / (3.0 * (1.0 + cfg.alpha * pi**2))
    c22 = 16.0 * conv_term * math.sqrt(cfg.beta) / (3.0 * (1.0 + 4.0 * cfg.alpha * pi**2))

    g1 = 4.0 * cfg.g0 / ((1.0 + cfg.alpha * pi**2) * pi)

    k1 = (stiff_term + pi**2) * pi**2 / (1.0 + cfg.alpha * pi**2)
    k2 = 4.0 * (stiff_term + 4.0 * pi**2) * pi**2 / (1.0 + 4.0 * cfg.alpha * pi**2)

    return {"c1": c1, "c2": c2, "c11": c11, "c22": c22, "g1": g1, "k1": k1, "k2": k2, "u2_t": u2_t}


def observation_operator(q1: torch.Tensor, q2: torch.Tensor, xi_obs: float):
    pi = math.pi
    return torch.sin(torch.tensor(pi * xi_obs, device=q1.device, dtype=q1.dtype)) * q1 + torch.sin(
        torch.tensor(2.0 * pi * xi_obs, device=q1.device, dtype=q1.dtype)
    ) * q2
