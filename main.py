import argparse
import os
import torch

from config import DYNAMICS, OBSERVATION, TIME, INVERSION, LOSSES, MODEL, get_profile
from generate_dataset import generate_synthetic_dataset
from data_loader import load_training_data
from model import PINN
from trainer import train, save_training_log
from evaluate import evaluate_model
from plot_results import plot_training_history, plot_predictions
from utils import ensure_dirs, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="smoke_test", choices=["smoke_test", "full_train"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--data-mode", default="synthetic", choices=["synthetic", "file"])
    p.add_argument("--train-csv", default="data/train_data.csv")
    p.add_argument("--metadata", default="data/metadata.json")
    p.add_argument("--hidden-truth", default="data/hidden_truth.csv")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    ensure_dirs(["data", "logs", "checkpoints", "results"])

    if args.data_mode == "synthetic" and (args.regenerate or not os.path.exists(args.train_csv)):
        generate_synthetic_dataset("data")

    device = torch.device(args.device)
    t_data, w_obs, w0, w0_dot, metadata = load_training_data(args.train_csv, args.metadata, device)

    model = PINN(MODEL, TIME.time_start, TIME.time_end).to(device)
    profile_cfg = get_profile(args.profile)

    cfg_bundle = {"dyn": DYNAMICS, "obs": OBSERVATION, "inv": INVERSION, "loss": LOSSES}
    u0_param, hist = train(model, t_data, w_obs, w0, w0_dot, cfg_bundle, profile_cfg, device)

    torch.save(
        {"model_state_dict": model.state_dict(), "u0_est": u0_param.item(), "profile": args.profile},
        "checkpoints/best_model.pt",
    )
    save_training_log("logs/train_log.csv", hist)
    plot_training_history(hist, "results")

    eval_dict = evaluate_model(model, t_data, w_obs, u0_param.item(), DYNAMICS.u0_true, OBSERVATION.xi_obs)
    plot_predictions(eval_dict, "results", hidden_truth_path=args.hidden_truth)

    save_json(
        "results/metrics.json",
        {
            "u0_true": DYNAMICS.u0_true,
            "u0_est": float(u0_param.item()),
            "u0_rel_err": eval_dict["u0_rel_err"],
            "w_mse": eval_dict["w_mse"],
            "profile": args.profile,
            "data_mode": args.data_mode,
            "metadata": metadata,
        },
    )
    print("Training complete. Results written to results/, logs/, checkpoints/.")


if __name__ == "__main__":
    main()
