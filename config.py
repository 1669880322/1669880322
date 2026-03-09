from dataclasses import dataclass


@dataclass
class DynamicsConfig:
    c: float = 0.05
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
    u0_true: float = 3.5


@dataclass
class ObservationConfig:
    xi_obs: float = 0.33


@dataclass
class TimeConfig:
    time_start: float = 0.0
    time_end: float = 12.0
    dt: float = 0.01
    noise_std: float = 0.0


@dataclass
class InversionConfig:
    u0_init: float = 2.5
    u0_min: float = 2.0
    u0_max: float = 5.0


@dataclass
class LossConfig:
    w_data: float = 1.0
    w_phys: float = 1.0
    w_ic: float = 1.0


@dataclass
class ModelConfig:
    in_dim: int = 1
    out_dim: int = 2
    hidden_layers: int = 4
    hidden_width: int = 64
    activation: str = "tanh"
    use_fourier_features: bool = False
    fourier_dim: int = 16
    fourier_scale: float = 1.0
    normalize_time: bool = True


TRAINING_PROFILES = {
    "smoke_test": {
        "adam_epochs": 500,
        "lbfgs_epochs": 50,
        "print_every": 50,
        "adam_lr": 1e-3,
        "lbfgs_lr": 1.0,
        "lbfgs_max_iter": 20,
    },
    "full_train": {
        "adam_epochs": 15000,
        "lbfgs_epochs": 5000,
        "print_every": 200,
        "adam_lr": 1e-3,
        "lbfgs_lr": 1.0,
        "lbfgs_max_iter": 20,
    },
}


def get_profile(name: str):
    if name not in TRAINING_PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    return TRAINING_PROFILES[name]


DYNAMICS = DynamicsConfig()
OBSERVATION = ObservationConfig()
TIME = TimeConfig()
INVERSION = InversionConfig()
LOSSES = LossConfig()
MODEL = ModelConfig()
