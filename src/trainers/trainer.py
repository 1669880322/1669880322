from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.losses import amplitude_from_q2, compute_losses
from src.physics.differentiable_integrator import rk4_second_mode
from src.physics.second_mode_ode import compute_second_mode_coefficients
from src.utils import save_json


class Trainer:
    def __init__(self, model, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.history: Dict[str, List[float]] = {
            "train_total": [],
            "val_total": [],
            "Lamp": [],
            "Lphys": [],
            "Lparam": [],
            "Lic": [],
            "c_est": [],
            "c_rel_err": [],
        }

    def _run_epoch(self, loader, optimizer=None) -> Dict[str, float]:
        train_mode = optimizer is not None
        self.model.train(train_mode)
        acc = {"total": 0.0, "Lamp": 0.0, "Lphys": 0.0, "Lparam": 0.0, "Lic": 0.0}
        n = 0

        for batch in loader:
            amp = batch["amplitude_sequence"].to(self.device)
            t_grid = batch["time_grid"][0].to(self.device)
            c_true = batch["theta_true"].to(self.device)

            if train_mode:
                optimizer.zero_grad()

            c_hat = self.model(amp)
            q20 = torch.full((amp.size(0), 1), self.cfg.q20_fixed, device=self.device)
            q2d0 = torch.full((amp.size(0), 1), self.cfg.q2d0_fixed, device=self.device)
            q2, q2_dot, q2_ddot = rk4_second_mode(t_grid, c_hat, q20, q2d0, self.cfg)
            a_hat = amplitude_from_q2(q2, self.cfg.scale_um, self.cfg.eps_amp)

            coeffs = compute_second_mode_coefficients(
                t=t_grid.unsqueeze(0).expand(amp.size(0), -1),
                c_struct=c_hat,
                cfg=self.cfg,
            )
            losses = compute_losses(
                a_hat=a_hat,
                a_obs=amp,
                q2=q2,
                q2_dot=q2_dot,
                q2_ddot=q2_ddot,
                c2=coeffs["c2"],
                k2=coeffs["k2"],
                c_hat=c_hat,
                c_true=c_true,
                cfg=self.cfg,
                loss_mode=self.cfg.loss_mode,
            )

            if train_mode:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                optimizer.step()

            bsz = amp.size(0)
            n += bsz
            for k in acc:
                acc[k] += float(losses[k].detach().cpu()) * bsz

        return {k: v / max(1, n) for k, v in acc.items()}

    def _clamp_model(self):
        self.model.eval()
        with torch.no_grad():
            for p in self.model.parameters():
                if p.ndim == 0:
                    p.clamp_(min=-10.0, max=10.0)

    def train(self, loaders, profile: Dict[str, int], ckpt_path: Path) -> Dict[str, List[float]]:
        adam = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr_adam)

        for _ in range(profile["adam_epochs"]):
            tr = self._run_epoch(loaders["train"], optimizer=adam)
            self._clamp_model()
            vl = self._run_epoch(loaders["val"], optimizer=None)
            self._record(tr, vl, loaders["val"])

        lbfgs = torch.optim.LBFGS(self.model.parameters(), lr=self.cfg.lr_lbfgs, max_iter=10)
        train_loader = loaders["train"]
        for _ in range(profile["lbfgs_epochs"]):
            batch = next(iter(train_loader))

            def closure():
                lbfgs.zero_grad()
                amp = batch["amplitude_sequence"].to(self.device)
                t_grid = batch["time_grid"][0].to(self.device)
                c_true = batch["theta_true"].to(self.device)
                c_hat = self.model(amp)
                c_hat = torch.clamp(c_hat, self.cfg.c_min, self.cfg.c_max)
                q20 = torch.full((amp.size(0), 1), self.cfg.q20_fixed, device=self.device)
                q2d0 = torch.full((amp.size(0), 1), self.cfg.q2d0_fixed, device=self.device)
                q2, q2_dot, q2_ddot = rk4_second_mode(t_grid, c_hat, q20, q2d0, self.cfg)
                a_hat = amplitude_from_q2(q2, self.cfg.scale_um, self.cfg.eps_amp)
                coeffs = compute_second_mode_coefficients(
                    t=t_grid.unsqueeze(0).expand(amp.size(0), -1),
                    c_struct=c_hat,
                    cfg=self.cfg,
                )
                losses = compute_losses(
                    a_hat, amp, q2, q2_dot, q2_ddot, coeffs["c2"], coeffs["k2"], c_hat, c_true, self.cfg, self.cfg.loss_mode
                )
                losses["total"].backward()
                return losses["total"]

            lbfgs.step(closure)
            self._clamp_model()
            tr = self._run_epoch(loaders["train"], optimizer=None)
            vl = self._run_epoch(loaders["val"], optimizer=None)
            self._record(tr, vl, loaders["val"])

        torch.save(self.model.state_dict(), ckpt_path)
        return self.history

    def _record(self, tr: Dict[str, float], vl: Dict[str, float], val_loader) -> None:
        self.history["train_total"].append(tr["total"])
        self.history["val_total"].append(vl["total"])
        self.history["Lamp"].append(tr["Lamp"])
        self.history["Lphys"].append(tr["Lphys"])
        self.history["Lparam"].append(tr["Lparam"])
        self.history["Lic"].append(tr["Lic"])

        batch = next(iter(val_loader))
        amp = batch["amplitude_sequence"].to(self.device)
        c_true = batch["theta_true"].to(self.device)
        with torch.no_grad():
            c_hat = torch.clamp(self.model(amp), self.cfg.c_min, self.cfg.c_max)
        c_est = float(c_hat.mean().cpu())
        rel = abs(c_est - float(c_true.mean().cpu())) / max(1e-8, abs(float(c_true.mean().cpu())))
        self.history["c_est"].append(c_est)
        self.history["c_rel_err"].append(rel)

    def evaluate_split(self, loader) -> Dict[str, float]:
        out = self._run_epoch(loader, optimizer=None)
        batch = next(iter(loader))
        amp = batch["amplitude_sequence"].to(self.device)
        c_true = batch["theta_true"].to(self.device)
        t_grid = batch["time_grid"][0].to(self.device)
        with torch.no_grad():
            c_hat = torch.clamp(self.model(amp), self.cfg.c_min, self.cfg.c_max)
            q20 = torch.full((amp.size(0), 1), self.cfg.q20_fixed, device=self.device)
            q2d0 = torch.full((amp.size(0), 1), self.cfg.q2d0_fixed, device=self.device)
            q2, q2_dot, q2_ddot = rk4_second_mode(t_grid, c_hat, q20, q2d0, self.cfg)
            coeffs = compute_second_mode_coefficients(
                t=t_grid.unsqueeze(0).expand(amp.size(0), -1), c_struct=c_hat, cfg=self.cfg
            )
            residual = q2_ddot + coeffs["c2"] * q2_dot + coeffs["k2"] * q2 + self.cfg.kappa * coeffs["k2"] * q2**3
        mae = torch.mean(torch.abs(c_hat - c_true)).item()
        rmse = torch.sqrt(torch.mean((c_hat - c_true) ** 2)).item()
        out.update(
            {
                "parameter_MAE": mae,
                "parameter_RMSE": rmse,
                "physics_residual_mean": float(residual.abs().mean().cpu()),
                "physics_residual_std": float(residual.std().cpu()),
            }
        )
        return out

    def save_history(self, path: Path) -> None:
        serializable = {k: [float(x) for x in v] for k, v in self.history.items()}
        save_json(path, serializable)
