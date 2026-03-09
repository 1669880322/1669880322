import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(hist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.array([h["epoch"] for h in hist])
    total = np.array([h["total"] for h in hist])
    l_data = np.array([h["l_data"] for h in hist])
    l_phys = np.array([h["l_phys"] for h in hist])
    l_ic = np.array([h["l_ic"] for h in hist])
    u0 = np.array([h["u0_est"] for h in hist])

    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, total)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_total.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, l_data, label="L_data")
    plt.semilogy(epochs, l_phys, label="L_phys")
    plt.semilogy(epochs, l_ic, label="L_ic")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss components")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_components.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, u0)
    plt.xlabel("Epoch")
    plt.ylabel("u0_est")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u0_evolution.png"), dpi=200)
    plt.close()


def plot_predictions(eval_dict, out_dir, hidden_truth_path=None):
    os.makedirs(out_dir, exist_ok=True)
    t = eval_dict["t"]

    plt.figure(figsize=(9, 5))
    plt.plot(t, eval_dict["w_obs"], label="w_obs", linewidth=2)
    plt.plot(t, eval_dict["w_hat"], "--", label="w_hat", linewidth=1.5)
    plt.legend()
    plt.xlabel("t (dimensionless)")
    plt.ylabel("displacement")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "w_obs_vs_w_hat.png"), dpi=200)
    plt.close()

    if hidden_truth_path and os.path.exists(hidden_truth_path):
        arr = np.loadtxt(hidden_truth_path, delimiter=",", skiprows=1)
        t_true, q1_true, q2_true = arr[:, 0], arr[:, 1], arr[:, 2]

        plt.figure(figsize=(9, 5))
        plt.plot(t_true, q1_true, label="q1_true")
        plt.plot(t, eval_dict["q1_hat"], "--", label="q1_hat")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "q1_true_vs_q1_hat.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(t_true, q2_true, label="q2_true")
        plt.plot(t, eval_dict["q2_hat"], "--", label="q2_hat")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "q2_true_vs_q2_hat.png"), dpi=200)
        plt.close()
