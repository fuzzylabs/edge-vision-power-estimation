"""Feautures and pipeline parameters for pooling layer."""

POOLING_FEATURES = [
    "batch_size",
    "input_size_0",
    "input_size_1",
    "input_size_2",
    "output_size_0",
    "output_size_1",
    "output_size_2",
    "kernel_0",
    "kernel_1",
    "stride_0",
    "stride_1",
]


TOTAL_POOLING_INPUT_FEATURES = [
    "batch_size",
    "input_size_0",
    "input_size_1",
    "input_size_2",
]

TOTAL_POOLING_OUTPUT_FEATURES = [
    "batch_size",
    "output_size_0",
    "output_size_1",
    "output_size_2",
]

TOTAL_POOLING_NO_OPS = [
    "batch_size",
    "input_size_0",
    "input_size_1",
    "input_size_2",
    "kernel_0",
    "kernel_1",
]

POOLING_PIPELINE = {
    "power": {
        "is_log": False,
        "degree": 3,
        "special_terms_list": [
            TOTAL_POOLING_INPUT_FEATURES,
            TOTAL_POOLING_OUTPUT_FEATURES,
            TOTAL_POOLING_NO_OPS,
        ],
        "scaler": "robust",
        "lasso_params": {
            "max_iter": 10000,
            "n_alphas": 500,
            "fit_intercept": True,
            "positive": True,
        },
    },
    "runtime": {
        "is_log": False,
        "degree": 3,
        "special_terms_list": [
            TOTAL_POOLING_INPUT_FEATURES,
            TOTAL_POOLING_OUTPUT_FEATURES,
            TOTAL_POOLING_NO_OPS,
        ],
        "scaler": "standard",
        "lasso_params": {
            "max_iter": 10000,
            "n_alphas": 500,
            "fit_intercept": True,
            "positive": False,
        },
    },
}
