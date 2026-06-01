from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tensorflow as tf
    import torch

    TorchTensor = torch.Tensor
    TorchModule = torch.nn.Module

    TensorFlowTensor = tf.Tensor
else:
    TorchTensor = Any
    TorchModule = Any

    TensorFlowTensor = Any
