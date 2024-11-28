"""Feautures and pipeline parameters for convolutional layer."""

CONV_FEATURES = [
    "batch_size",
    "input_size_0",
    "input_size_1",
    "input_size_2",
    "output_size_0",
    "output_size_1",
    "output_size_2",
    "kernel_0",
    "kernel_1",
    "padding_0",
    "padding_1",
    "stride_0",
    "stride_1",
]


TOTAL_CONV_OPS_PER_INPUT = [
    "input_size_0",
    "input_size_1",
    "input_size_2",
    "kernel_0",
    "kernel_1",
    "output_size_2",
]

TOTAL_CONV_OPS_PER_BATCH = TOTAL_CONV_OPS_PER_INPUT + ["batch_size"]


CONVOLUTION_PIPELINE = {
    "power": {
        "is_log": True,
        "degree": 3,
        "special_terms_list": [TOTAL_CONV_OPS_PER_INPUT, TOTAL_CONV_OPS_PER_BATCH],
        "scaler": "standard",
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
        "special_terms_list": [TOTAL_CONV_OPS_PER_INPUT, TOTAL_CONV_OPS_PER_BATCH],
        "scaler": "standard",
        "lasso_params": {
            "max_iter": 10000,
            "n_alphas": 500,
            "fit_intercept": True,
            "positive": True,
        },
    },
}
