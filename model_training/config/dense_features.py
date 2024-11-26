"""Feautures and pipeline parameters for dense layer."""

DENSE_FEATURES = ["batch_size", "input_size", "output_size"]

DENSE_PIPELINE = {
    "power": {
        "degree": 3,
        "scaler": "standard",
        "lasso_params": {
            "max_iter": 10000,
            "n_alphas": 500,
            "fit_intercept": True,
            "positive": True,
        },
    },
    "runtime": {
        "degree": 3,
        "scaler": "standard",
        "lasso_params": {
            "max_iter": 10000,
            "n_alphas": 500,
            "fit_intercept": True,
            "positive": True,
        },
    },
}
