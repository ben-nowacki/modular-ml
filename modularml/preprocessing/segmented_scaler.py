from typing import Any, Tuple, List, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class SegmentedScaler(BaseEstimator, TransformerMixin):
    """
    Applies an independent scaler to each segment of the input feature array.

    Example:
        If boundaries = [0, 30, 40, 60, 100], then segments are:
            - feature[:, 0:30]
            - feature[:, 30:40]
            - feature[:, 40:60]
            - feature[:, 60:100]

    Parameters:
        boundaries (tuple): List of segment boundary indices.
        scaler (sklearn transformer): A scaler class or instance (e.g., MinMaxScaler). A new copy will be created per segment.
    """
    def __init__(self, boundaries: Tuple[int], scaler: Optional[Any] = None):
        self.boundaries = boundaries
        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self._segment_scalers: List[Any] = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self._segment_scalers.clear()

        for i in range(len(self.boundaries) - 1):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = X[:, start:end]
            scaler = self._clone_scaler()
            scaler.fit(segment)
            self._segment_scalers.append(scaler)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        segments = []
        for i, scaler in enumerate(self._segment_scalers):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = X[:, start:end]
            transformed = scaler.transform(segment)
            segments.append(transformed)
        return np.concatenate(segments, axis=1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        segments = []
        for i, scaler in enumerate(self._segment_scalers):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = X[:, start:end]
            inverse = scaler.inverse_transform(segment)
            segments.append(inverse)
        return np.concatenate(segments, axis=1)

    def _clone_scaler(self):
        """Clone the scaler instance."""
        return self.scaler.__class__(**self.scaler.get_params())
    
    