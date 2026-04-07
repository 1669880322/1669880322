from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class Config:
    # Dimensionless hardcoded physics parameters (v1 baseline)
    alpha: float = 0.005
    Omega_tilde: float = 2.5
    phi1: float = 0.4
    phi2: float = 0.15
    lambda_: float = 0.1
    beta: float = 0.02
    u: float = 0.5
    u1: float = 1.5
    gamma: float = 3.0
    g0: float = 0.5
    A: float = 0.2
    omega1: float = 1.0
    kappa: float = 0.01
    u0_fixed: float = 3.5

    # Inverse target parameter config
    c_true: float = 0.05
    c_init: float = 0.02
    c_min: float = 0.005
    c_max: float = 0.15

    # Time-domain and data generation
    time_start: float = 0.0
    time_end: float = 12.0
    dt: float = 0.01
    noise_std: float = 0.0

    # Loss weights
    w_amp: float = 1.0
    w_phys: float = 1.0
    w_param: float = 0.2
    w_ic: float = 0.5

    # Initial conditions (fixed v1)
    q20_fixed: float = 0.01
    q2d0_fixed: float = 0.0

    # Amplitude construction
    scale_um: float = 1.0
    eps_amp: float = 1e-8

    # Data split and size
    n_samples: int = 120
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Training
    loss_mode: str = "hybrid"  # supervised_synthetic | physics_only | hybrid
    batch_size: int = 16
    lr_adam: float = 1e-3
    lr_lbfgs: float = 0.5
    grad_clip: float = 1.0

    # Paths
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)
    result_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.root_dir / "data"
        self.log_dir = self.root_dir / "logs"
        self.ckpt_dir = self.root_dir / "checkpoints"
        self.result_dir = self.root_dir / "results"


PROFILES: Dict[str, Dict[str, int]] = {
    "smoke_test": {"adam_epochs": 500, "lbfgs_epochs": 50},
    "full_train": {"adam_epochs": 15000, "lbfgs_epochs": 3000},
}
