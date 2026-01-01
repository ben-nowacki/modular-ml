from abc import ABC
from typing import Any

import numpy as np
import torch

from modularml.core.models.base_model import BaseModel
from modularml.utils.nn.backend import Backend


class TorchBaseModel(BaseModel, torch.nn.Module, ABC):
    """
    Base class for all ModularML-native PyTorch models.

    This class is intended for framework-owned models
    (e.g., SequentialMLP, SequentialCNN).

    User-defined torch.nn.Module objects should can subclass this,
    but it is easier to use the TorchModelWrapper.
    """

    def __init__(self, **init_args: Any):
        # torch.nn.Module must be initialized first
        torch.nn.Module.__init__(self)

        # BaseModel handles backend + built flag
        _ = init_args.pop("backend", None)
        super().__init__(backend=Backend.TORCH, **init_args)

    # ================================================
    # Model Weights (Stateful)
    # ================================================
    def get_weights(self) -> dict[str, np.ndarray]:
        """PyTorch weights are returned via the internal state_dict."""
        if not self.is_built:
            return {}
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Restore weights retrieved from `get_weights`."""
        if not weights:
            return
        torch_state = {k: torch.as_tensor(v) for k, v in weights.items()}
        self.load_state_dict(torch_state, strict=True)
