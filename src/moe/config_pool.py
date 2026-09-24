from ray import tune

CONFIG_POOL = {
    "NSX": {
        # --- shared NeuralForecast training knobs (MLP-style) ---
        "input_size_multiplier": tune.choice([1, 2, 3]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(["robust", "standard"]),
        "max_steps": tune.choice([1500, 2000, 2500]),
        "batch_size": tune.choice([64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512]),
        "random_seed": tune.randint(1, 20),

        # --- NSX architecture ---
        "num_experts": tune.choice([7, 10, 15, 20]),
        # "expert_arch": tune.choice(["mlp", "kan", "nbeats"]),
        "expert_arch": tune.choice(["mlp"]),
        "gate": tune.choice(["linear"]),
        "pooling": tune.choice(["dense", "sparse", "soft"]),
        # used by SparsePooling; clamped to min(k, num_experts) at runtime
        "k": tune.randint(1, 6),
        # used by SoftPooling
        # "temperature": tune.loguniform(0.1, 5.0),

        # --- NSX gate / total loss ---
        # kl uses softmax(-expert_loss), same ranking as softmax_mse
        "gate_loss_type": tune.choice(
            ["ib_softmax_mse", "softmax_mse", "ib_softmax_mse_grad", "kl"]
        ),
        # implemented: "random", "annealing", "sum"
        # "total_loss_type": tune.choice(["random", "annealing", "sum"]),
        "total_loss_type": tune.choice(["annealing", "sum"]),
        "anneal_temperature": tune.choice([True, False]),
        # step horizon for total_loss annealing and get_temperature()
        "annealing_temperature": tune.choice([1000, 2000]),

        "add_balance_loss": tune.choice([False]),
        "balance_factor": tune.loguniform(1e-3, 0.1),
    },
}
