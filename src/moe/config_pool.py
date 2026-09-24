from ray import tune

CONFIG_POOL = {
    "NSX": {
        # --- shared NeuralForecast training knobs (MLP-style) ---
        # "input_size_multiplier": tune.choice([1, 2, 3]),
        "input_size_multiplier": tune.choice([2]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(["robust", "standard"]),
        "max_steps": tune.choice([1500, 2000, 2500, 3000]),
        "batch_size": tune.choice([64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512]),
        "random_seed": tune.randint(1, 20),

        # --- NSX architecture ---
        "num_experts": tune.choice([7, 10, 15, 20]),
        # "expert_arch": tune.choice(["mlp", "kan", "nbeats"]),
        "expert_arch": tune.choice(["mlp"]),
        "gate": tune.choice(['linear','attention','mlp']),
        "pooling": tune.choice(["dense", "sparse", "soft"]),
        "k": tune.randint(1, 6),
        "gate_loss_type": tune.choice(
            ["ib_softmax_mse", "softmax_mse", "ib_softmax_mse_grad", "kl"]
        ),
    },
}
