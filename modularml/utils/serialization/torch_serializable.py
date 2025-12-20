from __future__ import annotations

import importlib

from modularml.utils.environment.optional_imports import ensure_torch
from modularml.utils.nn.backend import Backend
from modularml.utils.serialization.serializable_mixin import SerializableMixin


class TorchSerializableMixin(SerializableMixin):
    """
    A lightweight reusable serialization mixin for torch models.

    Description:
        The model must already have:
            - `self.config`: a dict of constructor kwargs
            - `self._input_shape`
            - `self._output_shape`

        This mixin only handles the generic logic. Your model must:
            - Provide a constructor `__init__(..., **config)`
            - Provide `.config` dict describing the model
            - Implement `build(input_shape, output_shape)`
    """

    # Get structured state
    def get_state(self) -> dict:
        return {
            "class": self.__class__.__module__ + "." + self.__class__.__qualname__,
            "config": self.config,  # constructor args
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "weights": self.get_weights(),  # numpy arrays
        }

    def set_state(self, state: dict):
        # Step 1: reconstruct model
        self.config = state["config"]
        self._input_shape = state["input_shape"]
        self._output_shape = state["output_shape"]

        # Build internal torch layers
        self.build(self._input_shape, self._output_shape)

        # Step 2: load weights
        self.set_weights(state["weights"])

    @classmethod
    def from_state(cls, state: dict):
        from modularml.models.base_model import BaseModel

        torch = ensure_torch()

        # Dynamically import class
        full_name = state["class"]
        module_name, class_name = full_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        klass = getattr(module, class_name)

        # Create empty instance
        obj = klass.__new__(klass)

        # Initialize base classes manually
        torch.nn.Module.__init__(obj)
        BaseModel.__init__(obj, backend=Backend.TORCH)

        # Restore state
        obj.set_state(state)
        return obj

    # Weight handling
    def get_weights(self) -> dict:
        if not self.is_built:
            return {}
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict):
        torch = ensure_torch()
        torch_state = {k: torch.tensor(v) for k, v in weights.items()}
        self.load_state_dict(torch_state, strict=True)
