"""Model Builder."""

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.linear_model import LassoCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)


class ModelBuilder:
    def __init__(self, cv: int, max_iter: int = 80000, n_alphas: int = 500):
        self.cv = cv
        self.max_iter = max_iter
        self.n_alphas = n_alphas

    def _create_pipeline(self, transformer: TransformerMixin) -> Pipeline:
        """Create a neural power pipeline with given transformer."""
        return Pipeline(
            [
                ("transformer", transformer),
                ("scaler", StandardScaler()),
                (
                    "lasso",
                    LassoCV(cv=self.cv, max_iter=self.max_iter, n_alphas=self.n_alphas),
                ),
            ]
        )

    def create_pipeline(
        self,
        features_mapping: dict[str, int],
        polynomial_degree: int,
        is_log: bool = False,
        special_terms_list: list[list[str]] | None = None,
    ) -> Pipeline:
        """Create a neural power pipeline.

        Args:
            features_mapping (dict[str, int]): Mapping of feature names to indices.
            polynomial_degree (int): Polynomial degree of regular polynomial terms.
            is_log (bool): Whether to log1p input features.
            special_terms_list (list[list[str]]): Definitions of special polynomial terms.

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

        return self._create_pipeline(transformer)

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
