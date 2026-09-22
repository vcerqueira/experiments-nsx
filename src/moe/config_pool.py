from ray import tune

CONFIG_POOL = {
    "NSX": {
        # --- shared NeuralForecast training knobs (MLP-style) ---
        "input_size_multiplier": tune.choice([1, 2, 3]),
        # "dropout": tune.choice([0.0, 0.1, 0.2, 0.3]),
        "loss": tune.choice(['mae', 'maeg']),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(["identity", "robust", "standard"]),
        "max_steps": tune.choice([500, 1000, 2000]),
        "start_padding_enabled": tune.choice([True, False]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(1, 20),

        # --- NSX architecture ---
        "num_experts": tune.choice([3, 5, 7, 10, 15]),
        "expert_arch": tune.choice(["mlp", "kan", "nbeats"]),
        "gate": tune.choice(["mlp", "attention", "linear", "rnn"]),
        "pooling": tune.choice(["dense", "sparse", "soft", "ste"]),
        # used by SparsePooling; clamped to min(k, num_experts) at runtime
        "k": tune.randint(1, 6),
        # used by SoftPooling (and STE internally, though NSX currently
        # constructs STE without passing this)
        "temperature": tune.loguniform(0.1, 5.0),

        # --- NSX gate / total loss ---
        # kl uses softmax(-expert_loss), same ranking as softmax_mse
        "gate_loss_type": tune.choice(
            ["ib_softmax_mse", "softmax_mse", "ib_softmax_mse_grad", "kl"]
        ),
        "detach_gate_targets": tune.choice([True, False]),
        # scales the MAEGrad expert term by that expert's gate weight
        "scale_maeg_by_gate": tune.choice([True, False]),
        # implemented: "random", "annealing", "sum" (comment says "sumsqr")
        "total_loss_type": tune.choice(["random", "annealing", "sum"]),
        "include_combined_loss": tune.choice([True, False]),
        "anneal_temperature": tune.choice([True, False]),
        # step horizon for total_loss annealing and get_temperature()
        "annealing_temperature": tune.choice([250, 500, 1000, 2000]),

        # --- auxiliary losses (factors only apply when the flag is True) ---
        "add_specialization_loss": tune.choice([True, False]),
        "specialization_factor": tune.loguniform(0.01, 0.5),
        "add_ncl_loss": tune.choice([True, False]),
        "ncl_factor": tune.loguniform(0.01, 0.5),
        "add_balance_loss": tune.choice([True, False]),
        "balance_factor": tune.loguniform(1e-3, 0.1),
    },
}