from ray import tune

CONFIG_POOL = {
    "NSX": {
        # --- shared NeuralForecast training knobs (MLP-style) ---
        # "input_size_multiplier": tune.choice([1, 2, 3]),
        "input_size_multiplier": tune.choice([2]),
        # "learning_rate": tune.loguniform(1e-4, 1e-1),
        "learning_rate": tune.choice([0.001]),
        # "scaler_type": tune.choice(["robust", "standard"]),
        "scaler_type": tune.choice(["standard"]),
        # "max_steps": tune.choice([1500, 2000, 2500, 3000]),
        "max_steps": tune.choice([2500]),
        # "batch_size": tune.choice([64, 128, 256]),
        "batch_size": tune.choice([128]),
        # "windows_batch_size": tune.choice([128, 256, 512]),
        "windows_batch_size": tune.choice([256]),
        # "random_seed": tune.randint(1, 20),
        "random_seed": tune.choice([18]),

        # --- NSX architecture ---
        # "num_experts": tune.choice([7, 10, 15, 20]),
        "num_experts": tune.choice([10]),
        # "expert_arch": tune.choice(["mlp", "kan", "nbeats"]),
        "expert_arch": tune.choice(["mlp"]),
        "pooling": tune.choice(["sparse"]),
        # "pooling": tune.choice(["dense", "sparse"]),
        # "k": tune.randint(2, 6),
        "k": tune.choice([4]),
        "gate": tune.choice(["linear_bias"]),
        # "gate": tune.choice(["linear", "linear_bias", "mlp"]),
        "online_eg": tune.choice([False]),
        "series_state": tune.choice([False, True]),
        "specialize": tune.choice([False, True]),
        "gate_loss_type": tune.choice(
            [
                "ib_softmax_mse",
                "softmax_mse",
                "ib_softmax_mse_grad",
                "ib_softmax_mse_window",
                "ib_softmax_mse_grad_window",
            ]
        ),
    },
}
