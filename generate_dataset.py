import math
import os
import numpy as np
from scipy.integrate import solve_ivp

from config import DYNAMICS, OBSERVATION, TIME
from equations import compute_coefficients
from utils import ensure_dirs, save_json
import torch


def _ode_rhs(t, y, cfg, u0_true):
    q1, q2, q1_dot, q2_dot = y
    t_torch = torch.tensor([[t]], dtype=torch.float64)
    u0_torch = torch.tensor(float(u0_true), dtype=torch.float64)
    coeff = compute_coefficients(t_torch, u0_torch, cfg)

    c1 = float(coeff["c1"])
    c2 = float(coeff["c2"])
    c11 = float(coeff["c11"].item())
    c22 = float(coeff["c22"].item())
    g1 = float(coeff["g1"])
    k1 = float(coeff["k1"].item())
    k2 = float(coeff["k2"].item())

    q1_ddot = -(c1 * q1_dot + k1 * q1 + cfg.kappa * k1 * (q1**3) + c11 * q2_dot + g1)
    q2_ddot = -(c2 * q2_dot + k2 * q2 + cfg.kappa * k2 * (q2**3) + c22 * q1_dot)
    return [q1_dot, q2_dot, q1_ddot, q2_ddot]


def generate_synthetic_dataset(data_dir="data"):
    ensure_dirs([data_dir])
    t_eval = np.arange(TIME.time_start, TIME.time_end + TIME.dt / 2, TIME.dt)

    y0 = [0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(
        lambda t, y: _ode_rhs(t, y, DYNAMICS, DYNAMICS.u0_true),
        (TIME.time_start, TIME.time_end),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    q1_true, q2_true, q1_dot_true, q2_dot_true = sol.y
    xi = OBSERVATION.xi_obs
    w_obs = np.sin(math.pi * xi) * q1_true + np.sin(2.0 * math.pi * xi) * q2_true
    if TIME.noise_std > 0:
        w_obs = w_obs + np.random.normal(scale=TIME.noise_std, size=w_obs.shape)

    train_path = os.path.join(data_dir, "train_data.csv")
    np.savetxt(train_path, np.column_stack([sol.t, w_obs]), delimiter=",", header="t,w_obs", comments="")

    hidden_path = os.path.join(data_dir, "hidden_truth.csv")
    np.savetxt(
        hidden_path,
        np.column_stack([sol.t, q1_true, q2_true, q1_dot_true, q2_dot_true]),
        delimiter=",",
        header="t,q1_true,q2_true,q1_dot_true,q2_dot_true",
        comments="",
    )

    w0 = float(w_obs[0])
    w0_dot = float(np.sin(math.pi * xi) * q1_dot_true[0] + np.sin(2.0 * math.pi * xi) * q2_dot_true[0])
    metadata = {
        "w0": w0,
        "w0_dot": w0_dot,
        "u0_true": DYNAMICS.u0_true,
        "xi_obs": xi,
        "time_start": TIME.time_start,
        "time_end": TIME.time_end,
        "dt": TIME.dt,
        "num_points": int(sol.t.shape[0]),
        "noise_std": TIME.noise_std,
    }
    save_json(os.path.join(data_dir, "metadata.json"), metadata)
    return train_path, hidden_path


if __name__ == "__main__":
    generate_synthetic_dataset()
