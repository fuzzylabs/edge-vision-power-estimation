"""Tune model parameters using Optuna."""

import numpy as np
from pipeline.trainer import Trainer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from model.model_builder import ModelBuilder

RANDOM_STATE = 42
"""Random state for LassoCV model."""


class OptunaOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, model_builder: ModelBuilder):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_builder = model_builder

    def get_transformers(self, trial, features_mapping=None, special_terms_list=None):
        """Get transformers to be applied to input features.

        It creates a Polynomial model applies input transformation such as
        log scaling or min-max, standard or robust scaling to the input features.

        Args:
            trial: Optuna trial
            features_mapping: Mapping of feature names to indices.
            Defaults to None.
            special_terms_list: Definitions of special polynomial terms.
            Defaults to None.

        Returns:
            _description_
        """
        degree = trial.suggest_int("degree", 1, 4)
        log_scale = trial.suggest_categorical("log_scale", [True, False])
        special_features = trial.suggest_categorical("special_features", [True, False])

        transformers = [PolynomialFeatures(degree=degree)]

        if log_scale:
            transformers.append(FunctionTransformer(np.log1p))

        if special_features and special_terms_list and features_mapping:
            special_terms_transformer = (
                self.model_builder.get_special_polynomial_terms_transformer(
                    features_mapping, special_terms_list
                )
            )
            transformers.append(special_terms_transformer)

        return transformers

    def get_scaler(self, trial):
        """Get scaler for input features preprocessing.

        Args:
            trial: Optuna trial

        Returns:
            Sklearn scaler.
        """
        scalers = trial.suggest_categorical("scalers", ["minmax", "standard", "robust"])
        if scalers == "minmax":
            return MinMaxScaler()
        elif scalers == "standard":
            return StandardScaler()
        elif scalers == "robust":
            return RobustScaler()

    def get_lasso(self, trial, cross_validation: int):
        """Get lasso CV model

        Args:
            trial: Optuna trial
            cross_validation: Number of cross valiation

        Returns:
            LassoCV model
        """
        max_iter = trial.suggest_int("max_iter", 1000, 50000, log=True)
        n_alphas = trial.suggest_int("n_alphas", 100, 1000, log=True)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        positive = trial.suggest_categorical("positive", [True, False])

        return LassoCV(
            cv=cross_validation,
            max_iter=max_iter,
            n_alphas=n_alphas,
            positive=positive,
            fit_intercept=fit_intercept,
            random_state=RANDOM_STATE,
        )

    def objective(
        self,
        trial,
        cross_validation: int = 10,
        features_mapping=None,
        special_terms_list=None,
    ) -> float:
        """Create a optuna objective to optimize.

        It maximises R^2 score on test dataset.

        Args:
            trial: Optuna trial
            cross_validation: Number of cross valiation. Defaults to 10.
            features_mapping: Mapping of feature names to indices.
            Defaults to None.
            special_terms_list: Definitions of special polynomial terms.
            Defaults to None.

        Returns:
            R^2 score on test dataset.
        """
        transformers = self.get_transformers(
            trial, features_mapping, special_terms_list
        )
        scaler = self.get_scaler(trial)
        lasso = self.get_lasso(trial, cross_validation)

        steps = transformers + [scaler, lasso]
        pipeline = make_pipeline(*steps)

        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)

        metrics = Trainer.eval_metrics(actual=self.y_test, pred=y_pred)
        for metric, value in metrics.items():
            trial.set_user_attr(metric, value)

        return metrics["testing_r2_score"]
