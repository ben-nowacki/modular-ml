
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Absolute(BaseEstimator, TransformerMixin):
    """
    Takes the absolute value.
    """
    def __init__(self):
        super().__init__()
        self._mask = None
        
    def fit(self, X, y=None):
        # No global fitting needed for per-sample normalization
        return self

    def transform(self, X):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")    
        X = np.asarray(X)
        self._mask = np.sign(X)
        return X * self._mask
    
    def inverse_transform(self, X):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")    
        X = np.asarray(X)
        return X * self._mask
    
    