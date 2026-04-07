import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from config import Config
from src.physics.second_mode_ode import compute_second_mode_coefficients
from src.utils import ensure_dirs, save_json, set_seed
import torch


def _ode_rhs(t: float, y: np.ndarray, c_struct: float, cfg: Config) -> np.ndarray:
    qt, vt = y
    t_tensor = torch.tensor([t], dtype=torch.float64)
    c_tensor = torch.tensor(c_struct, dtype=torch.float64)
    coeffs = compute_second_mode_coefficients(t_tensor, c_tensor, cfg)
    c2 = coeffs["c2"].detach().cpu().numpy()[0, 0]
    k2 = coeffs["k2"].detach().cpu().numpy()[0, 0]
    qdd = -(c2 * vt + k2 * qt + cfg.kappa * k2 * qt**3)
    return np.array([vt, qdd], dtype=np.float64)


def generate_single_trajectory(cfg: Config, c_value: float) -> Tuple[np.ndarray, np.ndarray, str]:
    t_eval = np.arange(cfg.time_start, cfg.time_end + cfg.dt, cfg.dt)
    y0 = np.array([cfg.q20_fixed, cfg.q2d0_fixed], dtype=np.float64)

    method_used = "RK45"
    sol = solve_ivp(
        fun=lambda t, y: _ode_rhs(t, y, c_value, cfg),
        t_span=(cfg.time_start, cfg.time_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    if (not sol.success) or np.any(~np.isfinite(sol.y)):
        method_used = "Radau (fallback)"
        sol = solve_ivp(
            fun=lambda t, y: _ode_rhs(t, y, c_value, cfg),
            t_span=(cfg.time_start, cfg.time_end),
            y0=y0,
            t_eval=t_eval,
            method="Radau",
            rtol=1e-7,
            atol=1e-9,
        )

    if (not sol.success) or np.any(~np.isfinite(sol.y)):
        raise RuntimeError(f"IVP failed even with Radau fallback: {sol.message}")

    return t_eval, sol.y[0], method_used


def build_dataset(cfg: Config, regenerate: bool = False, seed: int = 42) -> Dict[str, Path]:
    set_seed(seed)
    ensure_dirs(cfg.data_dir)
    train_path = cfg.data_dir / "train.npz"
    val_path = cfg.data_dir / "val.npz"
    test_path = cfg.data_dir / "test.npz"
    meta_path = cfg.data_dir / "metadata.json"

    if all(p.exists() for p in [train_path, val_path, test_path, meta_path]) and not regenerate:
        return {"train": train_path, "val": val_path, "test": test_path, "meta": meta_path}

    c_values = np.random.uniform(cfg.c_min, cfg.c_max, size=(cfg.n_samples,)).astype(np.float64)
    c_values[0] = cfg.c_true

    all_q2 = []
    all_amp = []
    methods = []
    t_ref = None

    for c in c_values:
        t_grid, q2, method = generate_single_trajectory(cfg, float(c))
        methods.append(method)
        t_ref = t_grid
        all_q2.append(q2)

    all_q2 = np.stack(all_q2, axis=0)

    raw_amp = np.sqrt(all_q2**2 + cfg.eps_amp)
    p95 = np.percentile(raw_amp, 95)
    target_p95 = 3.0
    dynamic_scale = target_p95 / max(p95, 1e-8)
    cfg.scale_um = float(dynamic_scale)

    all_amp = cfg.scale_um * raw_amp
    if cfg.noise_std > 0:
        all_amp = all_amp + np.random.normal(0.0, cfg.noise_std, size=all_amp.shape)

    lo, hi = float(np.min(all_amp)), float(np.max(all_amp))
    if hi < 1.5 or lo > 5.0:
        raise RuntimeError(f"Amplitude range [{lo:.3f}, {hi:.3f}] out of requested engineering window 1.5-5 um")

    idx = np.arange(cfg.n_samples)
    np.random.shuffle(idx)
    n_train = int(cfg.n_samples * cfg.train_ratio)
    n_val = int(cfg.n_samples * cfg.val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def save_split(path: Path, ids: np.ndarray) -> None:
        np.savez_compressed(
            path,
            time_grid=t_ref.astype(np.float32),
            amplitude_sequence=all_amp[ids].astype(np.float32),
            theta_true=c_values[ids].astype(np.float32),
            raw_q2=all_q2[ids].astype(np.float32),
        )

    save_split(train_path, train_idx)
    save_split(val_path, val_idx)
    save_split(test_path, test_idx)

    metadata = {
        "q20_fixed": cfg.q20_fixed,
        "q2d0_fixed": cfg.q2d0_fixed,
        "c_true": cfg.c_true,
        "u0_fixed": cfg.u0_fixed,
        "time_start": cfg.time_start,
        "time_end": cfg.time_end,
        "dt": cfg.dt,
        "scale_um": cfg.scale_um,
        "noise_std": cfg.noise_std,
        "solver_primary": "RK45",
        "solver_fallback": "Radau",
        "fallback_count": int(sum(1 for m in methods if "fallback" in m)),
    }
    save_json(meta_path, metadata)
    return {"train": train_path, "val": val_path, "test": test_path, "meta": meta_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    cfg = Config()
    output = build_dataset(cfg=cfg, regenerate=args.regenerate)
    print(output)
