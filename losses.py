import torch
from equations import compute_coefficients, observation_operator


def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]


def compute_physics_residuals(model, t, u0_param, dyn_cfg):
    q1, q2 = model(t)
    q1_dot = grad(q1, t)
    q2_dot = grad(q2, t)
    q1_ddot = grad(q1_dot, t)
    q2_ddot = grad(q2_dot, t)

    coeff = compute_coefficients(t, u0_param, dyn_cfg)
    r1 = q1_ddot + coeff["c1"] * q1_dot + coeff["k1"] * q1 + dyn_cfg.kappa * coeff["k1"] * q1**3 + coeff["c11"] * q2_dot + coeff["g1"]
    r2 = q2_ddot + coeff["c2"] * q2_dot + coeff["k2"] * q2 + dyn_cfg.kappa * coeff["k2"] * q2**3 + coeff["c22"] * q1_dot
    return r1, r2, q1, q2


def total_loss(model, t_data, w_obs, t_colloc, u0_param, cfg, w0, w0_dot=None):
    r1, r2, q1_c, q2_c = compute_physics_residuals(model, t_colloc, u0_param, cfg["dyn"])
    l_phys = torch.mean(r1**2 + r2**2)

    q1_d, q2_d = model(t_data)
    w_hat = observation_operator(q1_d, q2_d, cfg["obs"].xi_obs)
    l_data = torch.mean((w_hat - w_obs) ** 2)

    t0 = t_data[:1].clone().detach().requires_grad_(True)
    q1_0, q2_0 = model(t0)
    w0_hat = observation_operator(q1_0, q2_0, cfg["obs"].xi_obs)
    l_ic = torch.mean((w0_hat - w0) ** 2)
    if w0_dot is not None:
        dw0_hat = grad(w0_hat, t0)
        l_ic = l_ic + torch.mean((dw0_hat - w0_dot) ** 2)

    total = cfg["loss"].w_data * l_data + cfg["loss"].w_phys * l_phys + cfg["loss"].w_ic * l_ic
    return total, l_data, l_phys, l_ic, w_hat, (q1_d, q2_d)
