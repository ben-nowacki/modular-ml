from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

from modularml.utils.backend import Backend


class BaseModel(ABC):
    def __init__(self, config: dict[str, Any], backend: Backend):
        super().__init__()
        self.config = config
        self._backend = backend
        self._built = False

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...] | None:
        """
        Shape of the expected input to this mode.

        Returns:
            tuple[int, ...] | None: Input shape tuple (e.g., (32, 64)) or None if unknown.

        """

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...] | None:
        """
        Shape of the output produced by the model.

        Returns:
            tuple[int, ...] | None: Output shape tuple (e.g., (32, 10)) or None if unknown.

        """

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def is_built(self) -> bool:
        return self._built

    @abstractmethod
    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
    ):
        """Build the internal model layers given an input shape."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Run a forward pass."""

    # def get_config(self) -> Dict[str, Any]:
    #     """Return a serializable config dictionary, including `_target_`."""
    #     return {
    #         "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
    #         "backend": str(self._backend.value),
    #         **self.config,
    #     }

    # @classmethod
    # def from_config(cls, config: Dict[str, Any]) -> "BaseModel":
    #     """
    #     Dynamically reconstructs a model from config.
    #     Prioritizes `_target_` path, falls back to ModelRegistry.
    #     """
    #     config = config.copy()
    #     backend = Backend(config.pop("backend")) if "backend" in config else None
    #     target = config.pop("_target_", None)

    #     # Try dynamic import from _target_ path
    #     if target is not None:
    #         module_path, class_name = target.rsplit(".", 1)
    #         try:
    #             module = import_module(module_path)
    #             model_class = getattr(module, class_name)
    #             return model_class(**config, backend=backend)
    #         except Exception as e:
    #             raise ImportError(f"Failed to import model from _target_='{target}': {e}")

    #     return model_class(**config, backend=backend)
