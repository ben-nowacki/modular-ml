
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class PerSampleMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scales each sample independently to the given feature_range.

    For each sample x:
        x_scaled = (x - min) / (max - min) * (feature_range[1] - feature_range[0]) + feature_range[0]

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_, self.max_ = self.feature_range
        
        self._sample_min = None
        self._sample_range = None

    def fit(self, X, y=None):
        # No global fitting needed for per-sample normalization
        return self

    def transform(self, X):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")

        sample_min = np.min(X, axis=1, keepdims=True)
        sample_max = np.max(X, axis=1, keepdims=True)
        sample_range = np.where((sample_max - sample_min) == 0, 1, sample_max - sample_min)

        self._sample_min = sample_min
        self._sample_range = sample_range
        
        scale = self.max_ - self.min_
        return (X - sample_min) / sample_range * scale + self.min_
    
    def inverse_transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")

        scale = self.max_ - self.min_
        
        return (((X - self.min_) / scale) * self._sample_range) + self._sample_min
    
    