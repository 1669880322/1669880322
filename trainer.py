import csv
import os
import torch

from losses import total_loss


def train(model, t_data, w_obs, w0, w0_dot, cfg_bundle, profile_cfg, device):
    inv = cfg_bundle["inv"]
    dyn = cfg_bundle["dyn"]

    u0_param = torch.nn.Parameter(torch.tensor(float(inv.u0_init), dtype=torch.float32, device=device))
    params = list(model.parameters()) + [u0_param]

    adam = torch.optim.Adam(params, lr=profile_cfg["adam_lr"])
    lbfgs = torch.optim.LBFGS(params, lr=profile_cfg["lbfgs_lr"], max_iter=profile_cfg["lbfgs_max_iter"])

    hist = []
    t_colloc = t_data.clone().detach().requires_grad_(True)

    def run_epoch(epoch_idx, optimizer, stage):
        def closure():
            optimizer.zero_grad()
            total, l_data, l_phys, l_ic, _, _ = total_loss(
                model, t_data, w_obs, t_colloc, u0_param, cfg_bundle, w0, w0_dot
            )
            total.backward()
            return total

        if stage == "lbfgs":
            total_val = optimizer.step(closure)
        else:
            total_val = closure()
            optimizer.step()

        with torch.no_grad():
            u0_param.clamp_(inv.u0_min, inv.u0_max)

        total, l_data, l_phys, l_ic, _, _ = total_loss(
                model, t_data, w_obs, t_colloc, u0_param, cfg_bundle, w0, w0_dot
            )
        rel_err = abs((u0_param.item() - dyn.u0_true) / dyn.u0_true)
        hist.append(
            {
                "epoch": epoch_idx,
                "stage": stage,
                "total": float(total.item()),
                "l_data": float(l_data.item()),
                "l_phys": float(l_phys.item()),
                "l_ic": float(l_ic.item()),
                "u0_est": float(u0_param.item()),
                "u0_rel_err": float(rel_err),
            }
        )
        if epoch_idx % profile_cfg["print_every"] == 0:
            print(
                    f"[{stage}] epoch={epoch_idx} total={total.item():.4e} "
                    f"data={l_data.item():.4e} phys={l_phys.item():.4e} ic={l_ic.item():.4e} "
                    f"u0={u0_param.item():.6f} rel_err={rel_err:.4e}"
                )

    for epoch in range(profile_cfg["adam_epochs"]):
        run_epoch(epoch, adam, "adam")

    start = profile_cfg["adam_epochs"]
    for i in range(profile_cfg["lbfgs_epochs"]):
        run_epoch(start + i, lbfgs, "lbfgs")

    return u0_param, hist


def save_training_log(path, hist):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hist[0].keys())
        writer.writeheader()
        writer.writerows(hist)
