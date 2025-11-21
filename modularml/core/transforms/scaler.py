from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modularml.core.transforms.scaler_registry import SCALER_REGISTRY
from modularml.utils.serialization import SerializableMixin

if TYPE_CHECKING:
    import numpy as np


class Scaler(SerializableMixin):
    """
    Wrapper for feature scaling and transformation operations.

    Description:
        Provides a standardized interface for initializing, fitting, transforming, \
        and serializing feature scaling objects. The scaler can be specified by name \
        (from the global SCALER_REGISTRY) or provided as an existing sklearn-like \
        transformer instance.

    Example:
        ```python
        ft = Scaler("standard")
        ft.fit(X_train)
        X_scaled = ft.transform(X_test)
        ```

    """

    def __init__(
        self,
        scaler: str | Any = "StandardScaler",
        scaler_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize a ModularML Scaler wrapper.

        Description:
            It is better to initiallize with the name of a registered scaler \
            (eg, `"StandardScaler"`) and kwargs (eg, `{"with_mean: True}`) than to \
            provide an instance of a scaler class.
            While providing an instance is supported, it may not be possible \
            to seriallize and reproduce its state.

            If an instance is provided, this constructor attempts to:
                1. Find a matching class name in SCALER_REGISTRY
                2. Extract constructor parameters using method signatures \
                    and reachable parameters
                3. Store the recovered `scaler_name` and `scaler_kwargs`

        Args:
            scaler (str | Any, optional):
                Name of a registered scaler (preferred), or an sklearn-like \
                transformer instance. Defaults to `"StandardScaler"`.
            scaler_kwargs (dict[str, Any] | None, optional):
                Keyword arguments used when constructing the scaler from its name. \
                Ignored if a pre-instantiated instance is provided.

        """
        # Case 1: scaler given by name
        if isinstance(scaler, str):
            if scaler not in SCALER_REGISTRY:
                msg = (
                    f"Scaler '{scaler}' not recognized. Run `Scaler.get_supported_scalers()` to see supported scalers."
                )
                raise ValueError(msg)

            self.scaler_name = scaler
            self.scaler_kwargs = scaler_kwargs or {}
            self._scaler = SCALER_REGISTRY[scaler](**self.scaler_kwargs)
            self._is_fit = False

        # Case 2: scaler given as instance
        else:
            extracted_kwargs: dict[str, Any] = {}
            if hasattr(scaler, "get_params"):
                extracted_kwargs = scaler.get_params()
            else:
                import inspect

                try:
                    sig = inspect.signature(scaler.__class__.__init__)
                    for p_name in sig.parameters:
                        if p_name == "self":
                            continue
                        if hasattr(scaler, p_name):
                            extracted_kwargs[p_name] = getattr(scaler, p_name)
                except Exception:  # noqa: BLE001
                    extracted_kwargs = {}

            # Store names/kwargs or fall back to class name
            cls_name = scaler.__class__.__name__
            self.scaler_name = SCALER_REGISTRY.get_original_key(cls_name) or cls_name
            self.scaler_kwargs = scaler_kwargs or extracted_kwargs or {}

            # Store instance for now, but it will be lost during serialization
            self._scaler = scaler
            self._is_fit = False

        # Validate scaler
        self._validate_scaler()

    @classmethod
    def get_supported_scalers(cls) -> dict[str, Any]:
        """
        Return the registry of supported scalers.

        Returns:
            dict[str, Any]:
                Mapping of registered scaler names to their corresponding classes.

        """
        # Ensure all scalers are registered
        import modularml.preprocessing  # noqa: F401

        return SCALER_REGISTRY

    def _validate_scaler(self):
        if not hasattr(self._scaler, "fit"):
            raise AttributeError("Underlying scaler instance does not have a `fit()` method.")
        if not hasattr(self._scaler, "transform"):
            raise AttributeError("Underlying scaler instance does not have a `transform()` method.")

    # ==========================================
    # Core logic
    # ==========================================
    def fit(self, data: np.ndarray):
        """
        Fit the scaler to input data.

        Args:
            data (np.ndarray):
                Input data of shape `(n_samples, n_features)` used to compute \
                the transformation parameters.

        """
        self._scaler.fit(data)
        self._is_fit = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the fitted transformation to new data.

        Args:
            data (np.ndarray):
                Input data to transform. Must have the same feature layout \
                as the data used during fitting.

        Returns:
            np.ndarray:
                Transformed data array.

        Raises:
            RuntimeError:
                If the scaler has not been fit before calling this method.

        """
        if not self._is_fit:
            raise RuntimeError("Scaler has not been fit yet.")
        return self._scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data in a single step.

        Args:
            data (np.ndarray):
                Input data of shape `(n_samples, n_features)`.

        Returns:
            np.ndarray:
                Transformed data after fitting the scaler.

        """
        if hasattr(self._scaler, "fit_transform"):
            out = self._scaler.fit_transform(data)
            self._is_fit = True
            return out
        self._scaler.fit(data)
        self._is_fit = True
        return self._scaler.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse the transformation applied by the scaler, if supported.

        Args:
            data (np.ndarray):
                Transformed data to invert.

        Returns:
            np.ndarray:
                Original-scale data.

        Raises:
            NotImplementedError:
                If the underlying scaler does not implement `inverse_transform`.

        """
        if not self._is_fit:
            raise RuntimeError("Scaler has not been fit yet.")
        if hasattr(self._scaler, "inverse_transform"):
            return self._scaler.inverse_transform(data)
        msg = f"{self.scaler_name} does not support inverse_transform."
        raise NotImplementedError(msg)

    # ==========================================
    # SerializableMixin
    # ==========================================
    def get_state(self) -> dict:
        """Returns config of Scaler."""
        state = {
            "version": "1.0",
            "scaler_name": self.scaler_name,
            "scaler_kwargs": self.scaler_kwargs,
            "_is_fit": self._is_fit,
        }
        if self._is_fit:
            # Save learned attributes only
            learned = {
                k: v
                for k, v in self._scaler.__dict__.items()
                if k.endswith("_")  # sklearn learned attributes convention
            }
            state["learned_attributes"] = learned

        return state

    def set_state(self, state: dict) -> None:
        """Reverse operation of get_state()."""
        # Ensure all scalers are registered
        import modularml.preprocessing  # noqa: F401

        if state["version"] == "1.0":
            self.scaler_name = state["scaler_name"]
            self.scaler_kwargs = state["scaler_kwargs"]
        else:
            msg = f"Not implemented for version: {state['version']}"
            raise NotImplementedError(msg)

        # Rebuild fresh scaler from registry
        self._scaler = SCALER_REGISTRY[self.scaler_name](**self.scaler_kwargs)
        self._is_fit = False

        # Restore learned attributes
        if "learned_attributes" in state:
            for attr, val in state["learned_attributes"].items():
                setattr(self._scaler, attr, val)
            self._is_fit = True

    @classmethod
    def from_state(cls, state: dict) -> Scaler:
        # Ensure all scalers are registered
        import modularml.preprocessing  # noqa: F401

        obj = cls.__new__(cls)
        obj.set_state(state)
        return obj
