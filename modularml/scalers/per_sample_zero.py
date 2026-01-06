import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PerSampleZeroStart(BaseEstimator, TransformerMixin):
    """
    Shifts each sample independently so that the first value is 0.

    For each sample x:
        x_scaled = x - x[0]
    """

    def __init__(self):
        super().__init__()
        self.x0_ = None

    def get_params(self, deep=True):  # noqa: FBT002
        params = super().get_params(deep)
        return params

    def fit(self, X, y=None):
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)

        # store offsets for later inverse
        self.x0_ = X[:, [0]]

    def transform(self, X):
        self.fit(X)
        return X - self.x0_

    def inverse_transform(self, X):
        if X.ndim != 2:
            msg = f"Expected 2D array (n_samples, n_features), got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)
        return X + self.x0_
