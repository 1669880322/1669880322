import argparse
from pathlib import Path

import torch

from config import Config, PROFILES
from src.data.data_loader import build_dataloaders
from src.data.generate_amplitude_dataset import build_dataset
from src.evaluate import evaluate_all
from src.models.physics_informed_identifier import PhysicsInformedIdentifier
from src.plot_results import plot_reconstruction_examples, plot_training_curves
from src.physics.second_mode_ode import coefficient_sanity_check
from src.trainers.trainer import Trainer
from src.utils import ensure_dirs, resolve_device, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=str, default="smoke_test", choices=list(PROFILES.keys()))
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--loss-mode", type=str, default="hybrid", choices=["supervised_synthetic", "physics_only", "hybrid"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.loss_mode = args.loss_mode
    set_seed(42)
    ensure_dirs(cfg.data_dir, cfg.log_dir, cfg.ckpt_dir, cfg.result_dir)
    device = resolve_device(args.device)

    dataset_paths = build_dataset(cfg, regenerate=args.regenerate)
    loaders = build_dataloaders(dataset_paths, batch_size=cfg.batch_size)

    sanity = coefficient_sanity_check(cfg, device=device)
    save_json(cfg.log_dir / "coefficient_sanity.json", {k: list(v) for k, v in sanity.items()})

    model = PhysicsInformedIdentifier(cfg).to(device)
    trainer = Trainer(model=model, cfg=cfg, device=device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (cfg.ckpt_dir / f"best_{args.profile}.pt")

    if not args.test_only:
        history = trainer.train(loaders=loaders, profile=PROFILES[args.profile], ckpt_path=ckpt_path)
        trainer.save_history(cfg.log_dir / "train_history.json")
        plot_training_curves(history, cfg.result_dir)
    else:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    metrics = evaluate_all(trainer=trainer, loaders=loaders)
    save_json(cfg.result_dir / "metrics.json", metrics)
    plot_reconstruction_examples(model, loaders["test"], cfg, device, cfg.result_dir)

    save_json(
        cfg.result_dir / "synthetic_truth.json",
        {"c_true": cfg.c_true, "c_min": cfg.c_min, "c_max": cfg.c_max},
    )

    print("Done. Metrics:", metrics)


if __name__ == "__main__":
    main()
