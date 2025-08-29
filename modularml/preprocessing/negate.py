




import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Negate(BaseEstimator, TransformerMixin):
    """
    Multiplies the input by -1.
    """
    
    def fit(self, X, y=None):
        # No global fitting needed for per-sample normalization
        return self

    def transform(self, X):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")    
        X = np.asarray(X)
        
        return -X
    
    def inverse_transform(self, X):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")    
        X = np.asarray(X)
        return -X
    
    