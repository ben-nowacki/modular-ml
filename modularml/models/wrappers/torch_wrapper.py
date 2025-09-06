

import torch as torch

from modularml.models.base import BaseModel



class TorchModelWrapper(BaseModel, torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def infer_output_shape(self, input_shape):
        dummy = torch.randn(1, *input_shape)
        return tuple(self.model(dummy).shape[1:])