from typing import Tuple

import torch

from src.physics.second_mode_ode import second_mode_acceleration


def rk4_second_mode(
    t_grid: torch.Tensor,
    c_struct: torch.Tensor,
    q20: torch.Tensor,
    q2d0: torch.Tensor,
    cfg,
    state_clip: float = 1e3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable RK4 integration for q2 and q2_dot.

    Shapes:
        t_grid: [T]
        c_struct: [B,1]
        q20/q2d0: [B,1]
    Returns:
        q2, q2_dot, q2_ddot all [B,T]
    """
    bsz = c_struct.size(0)
    T = t_grid.size(0)
    dt = t_grid[1] - t_grid[0]

    q = torch.zeros(bsz, T, device=t_grid.device, dtype=t_grid.dtype)
    v = torch.zeros_like(q)
    a = torch.zeros_like(q)

    q[:, 0] = q20.squeeze(-1)
    v[:, 0] = q2d0.squeeze(-1)

    for i in range(T - 1):
        t_i = t_grid[i].expand(bsz, 1)
        qi = q[:, i:i+1]
        vi = v[:, i:i+1]

        def f(_q, _v, _t):
            return _v, second_mode_acceleration(_q, _v, _t, c_struct, cfg)

        k1_q, k1_v = f(qi, vi, t_i)
        k2_q, k2_v = f(qi + 0.5 * dt * k1_q, vi + 0.5 * dt * k1_v, t_i + 0.5 * dt)
        k3_q, k3_v = f(qi + 0.5 * dt * k2_q, vi + 0.5 * dt * k2_v, t_i + 0.5 * dt)
        k4_q, k4_v = f(qi + dt * k3_q, vi + dt * k3_v, t_i + dt)

        q_next = qi + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        v_next = vi + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        q[:, i + 1] = torch.clamp(q_next.squeeze(-1), -state_clip, state_clip)
        v[:, i + 1] = torch.clamp(v_next.squeeze(-1), -state_clip, state_clip)

    t_batch = t_grid.unsqueeze(0).expand(bsz, -1)
    a = second_mode_acceleration(q, v, t_batch, c_struct, cfg)
    return q, v, a
