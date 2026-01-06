from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from modularml.core.transforms.scaler import Scaler


class SegmentedScaler(BaseEstimator, TransformerMixin):
    """
    Applies an independent scaler to each segment of the input feature array.

    Example:
        If boundaries = [0, 30, 40, 60, 100], then segments are:
            - feature[:, 0:30]
            - feature[:, 30:40]
            - feature[:, 40:60]
            - feature[:, 60:100]

    Arguments:
        boundaries (tuple): list of segment boundary indices.
        scaler (sklearn transformer): A scaler class or instance (e.g., MinMaxScaler). A new copy will be created per segment.

    """

    def __init__(
        self,
        boundaries: tuple[int],
        scaler: Scaler | str | Any,
        scaler_kwargs: dict[str, Any] | None = None,
    ):
        # Validate boundaries
        if 0 not in boundaries:
            raise ValueError("Boundaries must start at 0.")
        if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ValueError("Boundaries must be strictly increasing.")

        # Normalize scaler input
        # Accepts: scaler instance, scaler name, sklearn-like object
        if isinstance(scaler, Scaler):
            self.scaler_template = scaler
        else:
            self.scaler_template = Scaler(
                scaler=scaler,
                scaler_kwargs=scaler_kwargs,
            )

        # Store all init args exactly as name (requirement of BaseEstimator)
        self.boundaries = boundaries
        self.scaler = self.scaler_template.scaler_name
        self.scaler_kwargs = self.scaler_template.scaler_kwargs

        # Runtime-only scaler (one per defined boundary)
        self._segment_scalers: list[Scaler] = []

    def get_params(self, deep=True):  # noqa: FBT002
        params = super().get_params(deep)
        params["boundaries"] = self.boundaries
        params["scaler"] = self.scaler
        params["scaler_kwargs"] = self.scaler_kwargs
        return params

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self._segment_scalers.clear()

        if X.shape[1] != self.boundaries[-1]:
            msg = f"Last boundary does not match feature length: {self.boundaries[-1]} != {X.shape[1]}"
            raise ValueError(msg)
        for i in range(len(self.boundaries) - 1):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = X[:, start:end]

            scaler: Scaler = self.scaler_template.clone_unfitted()
            scaler.fit(segment)
            self._segment_scalers.append(scaler)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._segment_scalers:
            raise RuntimeError("SegmentedScaler has not been fit.")

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
