# Hyperparameter Tuning

[Optuna](https://optuna.readthedocs.io/en/stable/) library is used to find optimal hyperparameter for power and runtime models for 3 layer types.

The optimal parameter found in the notebook are used to update the `lasso_params` key for each of 3 layer configuration `*_features.py` inside [config](./config/) folder.
