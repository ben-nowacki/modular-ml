


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
        self._x0 = None
    
    def fit(self, X, y=None):
        # No global fitting needed for per-sample normalization
        return self

    def transform(self, X):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")    
        X = np.asarray(X)
        
        # store offsets for later inverse
        self._x0 = X[:, [0]]
        return X - self._x0
    
    def inverse_transform(self, X):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")    
        X = np.asarray(X)
        return X + self._x0
    
    