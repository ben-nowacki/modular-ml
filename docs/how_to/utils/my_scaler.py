import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PerSampleStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize each sample to zero mean and unit variance.

    Unlike sklearn's ``StandardScaler``, statistics are computed per sample
    at transform time; no global state is learned from the training set.
    """

    def __init__(self):
        self._sample_mean = None
        self._sample_std = None

    def fit(self, X, y=None):
        # No global statistics to learn; all computation is deferred to transform
        return self

    def transform(self, X):
        if X.ndim != 2:
            msg = f"Expected 2D array, got shape {X.shape}"
            raise ValueError(msg)
        X = np.asarray(X)
        self._sample_mean = X.mean(axis=1, keepdims=True)
        self._sample_std = X.std(axis=1, keepdims=True)
        # Guard against constant samples (std = 0)
        self._sample_std = np.where(self._sample_std == 0, 1.0, self._sample_std)
        return (X - self._sample_mean) / self._sample_std

    def inverse_transform(self, X):
        if self._sample_mean is None:
            raise RuntimeError("Scaler has not been applied yet.")
        return X * self._sample_std + self._sample_mean
