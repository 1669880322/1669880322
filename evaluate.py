import numpy as np
import torch

from equations import observation_operator


def evaluate_model(model, t, w_obs, u0_est, u0_true, xi_obs):
    model.eval()
    with torch.no_grad():
        q1_hat, q2_hat = model(t)
        w_hat = observation_operator(q1_hat, q2_hat, xi_obs)

    w_mse = torch.mean((w_hat - w_obs) ** 2).item()
    rel_err = abs((u0_est - u0_true) / u0_true)
    return {
        "w_mse": w_mse,
        "u0_est": float(u0_est),
        "u0_true": float(u0_true),
        "u0_rel_err": float(rel_err),
        "q1_hat": q1_hat.cpu().numpy().reshape(-1),
        "q2_hat": q2_hat.cpu().numpy().reshape(-1),
        "w_hat": w_hat.cpu().numpy().reshape(-1),
        "w_obs": w_obs.cpu().numpy().reshape(-1),
        "t": t.cpu().numpy().reshape(-1),
    }
