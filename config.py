"""Configuration for PID-Lagrangian Safe SAC training on EV2Gym."""

CONFIG = {
    # Environment
    "config_file": "ev2gym/example_config_files/PID_Lagrangian.yaml",
    "ev2gym_path": "E:/cjx12363/PID/EV2Gym",

    # Training
    "total_episodes": 60,
    "gradient_steps_per_episode": 100,
    "batch_size": 256,
    "buffer_capacity": int(1e6),
    "eval_every": 20,
    "eval_episodes": 5,

    # SAC
    "gamma": 0.99,
    "tau": 0.005,
    "lr": 1e-4,
    "alpha_lr": 3e-4,
    "device": "cuda",

    # PID-Lagrangian
    "pid": {
        "K_P": 0.10,
        "K_I": 0.01,
        "K_D": 0.05,
        "d": 1.0,
    },
    "cost_scale": 1.0,

    # Logging
    "log_dir": "./logs",
    "save_dir": "./checkpoints",
}

# Baseline configs
LAGRANGIAN_CONFIG = {**CONFIG, "pid": {"K_P": 1.0, "K_I": 0.01, "K_D": 0.0, "d": 1.0}}
UNCONSTRAINED_CONFIG = {**CONFIG, "pid": {"K_P": 0.0, "K_I": 0.0, "K_D": 0.0, "d": 1e9}}
