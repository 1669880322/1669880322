from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.losses import amplitude_from_q2
from src.physics.differentiable_integrator import rk4_second_mode


def plot_training_curves(history: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = range(len(history["train_total"]))

    plt.figure(figsize=(8, 4))
    plt.plot(x, history["train_total"], label="train")
    plt.plot(x, history["val_total"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, history["Lamp"], label="Lamp")
    plt.plot(x, history["Lphys"], label="Lphys")
    plt.plot(x, history["Lparam"], label="Lparam")
    plt.plot(x, history["Lic"], label="Lic")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "component_losses.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, history["c_est"], label="c_est")
    plt.xlabel("epoch")
    plt.ylabel("c_struct estimate")
    plt.tight_layout()
    plt.savefig(out_dir / "c_est_evolution.png", dpi=150)
    plt.close()


def plot_reconstruction_examples(model, loader, cfg, device, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader))
    amp = batch["amplitude_sequence"].to(device)
    t_grid = batch["time_grid"][0].to(device)
    with torch.no_grad():
        c_hat = model(amp)
        q20 = torch.full((amp.size(0), 1), cfg.q20_fixed, device=device)
        q2d0 = torch.full((amp.size(0), 1), cfg.q2d0_fixed, device=device)
        q2, _, _ = rk4_second_mode(t_grid, c_hat, q20, q2d0, cfg)
        a_hat = amplitude_from_q2(q2, cfg.scale_um, cfg.eps_amp)

    n = min(3, amp.size(0))
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(t_grid.cpu().numpy(), amp[i].cpu().numpy(), label="a_obs")
        plt.plot(t_grid.cpu().numpy(), a_hat[i].cpu().numpy(), "--", label="a_hat")
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_fit_examples.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t_grid.cpu().numpy(), amp[0].cpu().numpy(), label="a_obs")
    plt.plot(t_grid.cpu().numpy(), a_hat[0].cpu().numpy(), label="a_hat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sample_reconstruction.png", dpi=150)
    plt.close()
