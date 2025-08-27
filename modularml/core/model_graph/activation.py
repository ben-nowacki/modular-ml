
from typing import Any, Callable, Dict

from modularml.utils.backend import Backend
from modularml.utils.exceptions import ActivationError, BackendNotSupportedError


class Activation:
    def __init__(self, name: str, backend: Backend):
        """Initiallize an Activation function

        Args:
            name (str): Activation function to use (e.g., 'relu')
            backend (Backend): Backend to use (e.g., Backend.TORCH)
        """
        self.name = name.lower()
        self.backend = backend
        self.layer = self._resolve()
        
    def _resolve(self) -> Callable:
        avail_acts = {}
        if self.backend == Backend.TORCH:
            import torch.nn as nn
            avail_acts = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(),
                "sigmoid": nn.Sigmoid(),
                "tanh": nn.Tanh(),
                "gelu": nn.GELU(),
                "elu": nn.ELU(),
            }
        elif self.backend == Backend.TENSORFLOW:
            import tensorflow as tf
            avail_acts = {
                "relu": tf.keras.layers.ReLU(),
                "leaky_relu": tf.keras.layers.LeakyReLU(),
                "sigmoid": tf.keras.layers.Activation("sigmoid"),
                "tanh": tf.keras.layers.Activation("tanh"),
                "gelu": tf.keras.layers.Activation("gelu"),
                "elu": tf.keras.layers.ELU(),
            }
        else:
            raise BackendNotSupportedError(backend=self.backend, method="Activation._resolve()")
    
        act = avail_acts.get(self.name)
        if act is None:
            raise ActivationError(
                f"Unknown activation name (`{self.name}`) for `{self.backend}` backend."
                f"Available activations: {avail_acts.keys()}"
            )
        return act
    
    def get_layer(self):
        return self.layer
    
    def __repr__(self):
        return f"Activation `{self.name}` ({self.backend})"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "backend": str(self.backend.value)
        }
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(name=config["name"], backend=Backend(config["backend"]))
    