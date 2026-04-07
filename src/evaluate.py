from typing import Dict

from src.trainers.trainer import Trainer


def evaluate_all(trainer: Trainer, loaders) -> Dict[str, float]:
    train_m = trainer.evaluate_split(loaders["train"])
    val_m = trainer.evaluate_split(loaders["val"])
    test_m = trainer.evaluate_split(loaders["test"])
    return {
        "train_loss": train_m["total"],
        "val_loss": val_m["total"],
        "test_loss": test_m["total"],
        "parameter_MAE": test_m["parameter_MAE"],
        "parameter_RMSE": test_m["parameter_RMSE"],
        "amplitude_reconstruction_error": test_m["Lamp"],
        "physics_residual_mean": test_m["physics_residual_mean"],
        "physics_residual_std": test_m["physics_residual_std"],
    }
