from typing import Dict

import torch


def compute_second_mode_coefficients(
    t: torch.Tensor,
    c_struct: torch.Tensor,
    cfg,
) -> Dict[str, torch.Tensor]:
    """Compute reduced-model coefficients with graph connectivity preserved.

    Args:
        t: shape [T] or [B, T]
        c_struct: shape [] or [B, 1]
    """
    pi = torch.pi
    if t.dim() == 1:
        t_eff = t.unsqueeze(0)
    else:
        t_eff = t

    if c_struct.dim() == 0:
        c_eff = c_struct.view(1, 1)
    elif c_struct.dim() == 1:
        c_eff = c_struct.view(-1, 1)
    else:
        c_eff = c_struct

    u2_t = cfg.u0_fixed * (1.0 + cfg.A * torch.cos(cfg.omega1 * t_eff))
    conv_term = (
        cfg.phi1 * cfg.u
        + cfg.lambda_ * cfg.phi2 * (cfg.u + cfg.u1)
        + (cfg.u - u2_t)
    )
    stiff_term = (
        cfg.gamma / 2.0
        - cfg.alpha * cfg.Omega_tilde**2
        - cfg.phi1 * cfg.u**2
        - cfg.lambda_ * cfg.phi2 * (cfg.u + cfg.u1) ** 2
        - (cfg.u - u2_t) ** 2
    )

    c1 = (c_eff + 2.0 * cfg.alpha * cfg.Omega_tilde * pi**2) / (1.0 + cfg.alpha * pi**2)
    c2 = (c_eff + 8.0 * cfg.alpha * cfg.Omega_tilde * pi**2) / (1.0 + 4.0 * cfg.alpha * pi**2)

    c11 = 32.0 * conv_term * torch.sqrt(torch.tensor(cfg.beta, dtype=t_eff.dtype, device=t_eff.device))
    c11 = c11 / (3.0 * (1.0 + cfg.alpha * pi**2))
    c22 = 16.0 * conv_term * torch.sqrt(torch.tensor(cfg.beta, dtype=t_eff.dtype, device=t_eff.device))
    c22 = c22 / (3.0 * (1.0 + 4.0 * cfg.alpha * pi**2))

    g1 = 4.0 * cfg.g0 / ((1.0 + cfg.alpha * pi**2) * pi)

    k1 = (stiff_term + pi**2) * pi**2 / (1.0 + cfg.alpha * pi**2)
    k2 = 4.0 * (stiff_term + 4.0 * pi**2) * pi**2 / (1.0 + 4.0 * cfg.alpha * pi**2)

    return {
        "u2_t": u2_t,
        "conv_term": conv_term,
        "stiff_term": stiff_term,
        "c1": c1.expand(-1, t_eff.size(1)),
        "c2": c2.expand(-1, t_eff.size(1)),
        "c11": c11,
        "c22": c22,
        "g1": torch.full_like(t_eff, g1),
        "k1": k1,
        "k2": k2,
    }


def second_mode_acceleration(
    q2: torch.Tensor,
    q2_dot: torch.Tensor,
    t: torch.Tensor,
    c_struct: torch.Tensor,
    cfg,
) -> torch.Tensor:
    coeffs = compute_second_mode_coefficients(t=t, c_struct=c_struct, cfg=cfg)
    c2 = coeffs["c2"]
    k2 = coeffs["k2"]
    return -(c2 * q2_dot + k2 * q2 + cfg.kappa * k2 * q2**3)


def coefficient_sanity_check(cfg, device: torch.device = torch.device("cpu")) -> Dict[str, tuple]:
    t = torch.linspace(cfg.time_start, cfg.time_end, 10, device=device)
    c = torch.tensor(cfg.c_true, dtype=torch.float32, device=device, requires_grad=True)
    coeffs = compute_second_mode_coefficients(t=t, c_struct=c, cfg=cfg)
    scalar = coeffs["c2"].mean() + coeffs["k2"].mean()
    scalar.backward()
    return {k: tuple(v.shape) for k, v in coeffs.items()}
