"""Model Builder."""

from typing import Any

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.linear_model import LassoCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

RANDOM_STATE = 42
"""Random state for LassoCV model."""


class ModelBuilder:
    def __init__(self, cv: int = 10):
        self.cv = cv

    def _create_pipeline(
        self, scaler: str, transformer: TransformerMixin, lasso_params: dict
    ) -> Pipeline:
        """Create a neural power pipeline with given transformer."""
        return Pipeline(
            [
                ("transformer", transformer),
                ("scaler", self.get_scaler(scaler)),
                (
                    "lasso",
                    LassoCV(cv=self.cv, random_state=RANDOM_STATE, **lasso_params),
                ),
            ]
        )

    def get_scaler(self, scaler: str):
        """Get a scaler to transform input features.

        Args:
            scaler: Name of scaler to use.

        Returns:
            Sklearn preprocessing scaler.
        """
        if scaler == "minmax":
            return MinMaxScaler()
        elif scaler == "standard":
            return StandardScaler()
        elif scaler == "robust":
            return RobustScaler()

    def create_pipeline(
        self,
        features_mapping: dict[str, int],
        polynomial_degree: int,
        scaler: str,
        is_log: bool = False,
        special_terms_list: list[list[str]] | None = None,
        lasso_params: dict[str, Any] = {},
    ) -> Pipeline:
        """Create a neural power pipeline.

        Args:
            features_mapping: Mapping of feature names to indices.
            polynomial_degree: Polynomial degree of regular polynomial terms.
            scaler: Name of sklearn preprocessing scaler
            is_log: Whether to log1p input features.
            special_terms_list: Definitions of special polynomial terms.
            lasso_params: Parameters added to LassoCV sklearn model

        Returns:
            Pipeline: neural power pipeline
        """
        regular_terms_transformer = self.get_regular_polynomial_terms_transformer(
            degree=polynomial_degree, is_log=is_log
        )
        if special_terms_list is not None and len(special_terms_list) > 0:
            special_terms_transformer = self.get_special_polynomial_terms_transformer(
                features_mapping,
                special_terms_list,
            )
            transformer = FeatureUnion(
                [
                    ("regular_polynomial", regular_terms_transformer),
                    ("special_polynomial", special_terms_transformer),
                ]
            )
        else:
            transformer = regular_terms_transformer

        return self._create_pipeline(
            transformer=transformer, scaler=scaler, lasso_params=lasso_params
        )

    def multiply_columns(
        self, input_arr: np.ndarray, column_indices: list[int]
    ) -> np.ndarray:
        """Multiple all specified columns row-wise."""
        return np.multiply.reduce(input_arr[:, column_indices], axis=1)

    def get_regular_polynomial_terms_transformer(
        self, degree: int, is_log: bool = False
    ) -> Pipeline | TransformerMixin:
        """Create a polynomial terms transformer."""
        polynomial_transformer = PolynomialFeatures(degree=degree)
        if is_log:
            return Pipeline(
                [
                    # log2 produces -inf if any input feature (e.g. padding_0 or padding_1) is zero
                    ("log1p", FunctionTransformer(np.log1p)),
                    ("polynomial_features", polynomial_transformer),
                ]
            )
        else:
            return polynomial_transformer

    def get_special_polynomial_terms_transformer(
        self, features_mapping: dict[str, int], terms_list: list[list[str]]
    ) -> TransformerMixin:
        """Create a transformer for special polynomial terms."""

        def _build_special_terms(input_arr: np.ndarray) -> np.ndarray:
            """Build special terms based on the mappings and the terms list."""
            result = []
            for terms in terms_list:
                column_indices = [features_mapping[t] for t in terms]
                result.append(self.multiply_columns(input_arr, column_indices))

            return np.stack(result).T

        return FunctionTransformer(_build_special_terms)
