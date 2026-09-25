from ray import tune

CONFIG_POOL = {
    "NSX": {
        # --- shared NeuralForecast training knobs (MLP-style) ---
        # "input_size_multiplier": tune.choice([1, 2, 3]),
        "input_size_multiplier": tune.choice([2]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        # "scaler_type": tune.choice(["robust", "standard"]),
        "scaler_type": tune.choice(["standard"]),
        # "max_steps": tune.choice([1500, 2000, 2500, 3000]),
        "max_steps": tune.choice([2000]),
        # "batch_size": tune.choice([64, 128, 256]),
        "batch_size": tune.choice([64]),
        # "windows_batch_size": tune.choice([128, 256, 512]),
        "windows_batch_size": tune.choice([256]),
        # "random_seed": tune.randint(1, 20),
        "random_seed": tune.choice([18]),

        # --- NSX architecture ---
        "num_experts": tune.choice([7, 10, 15, 20]),
        # "expert_arch": tune.choice(["mlp", "kan", "nbeats"]),
        "expert_arch": tune.choice(["mlp"]),
        "pooling": tune.choice(["dense", "sparse"]),
        "k": tune.randint(1, 6),
        "pooled_combined_loss": tune.choice([True, False]),
        "gate_loss_type": tune.choice(
            [
                "ib_softmax_mse",
                "softmax_mse",
                "ib_softmax_mse_grad",
                "kl",
                "ib_softmax_mse_window",
                "ib_softmax_mse_grad_window",
            ]
        ),
    },
}
