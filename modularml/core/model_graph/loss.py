

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
import torch as torch
import inspect

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat, convert_dict_to_format, get_data_format_for_backend, to_numpy
from modularml.utils.exceptions import BackendNotSupportedError, LossError


class Loss:
    def __init__(
        self,
        name: Optional[str] = None,
        backend: Optional[Backend] = None,
        loss_function: Optional[Callable] = None,
        reduction: str = 'none',
    ):
        """
        Initiallizes a new Loss term. Loss objects must be defined with \
        either a custom `loss_function` or both a `name` and `backend`.

        Args:
            name (optional, str): Name of the loss function to use (e.g., "mse")
            backend (optional, Backend): The backend to use (e.g., Backend.TORCH)
            loss_function (optional, Callable): A custom loss function.
            reduction (optional, str): Defaults to 'none'.
            
        Examples:
            ``` python
            from sklearn.metrics import mean_squared_error 
            from modularml import Backend
            loss1 = Loss(loss_function = mean_squared_error)
            loss2 = Loss(name='mse', backend=Backend.TORCH)
            ```
        """
        self.name = name.lower() if name else None
        self.backend = backend
        self.reduction = reduction
        
        if loss_function is not None:
            self.loss_function: Callable = loss_function
            mod = inspect.getmodule(loss_function)
            if self.name is None: self.name = mod.__name__
            
            # TODO: how to infer backend?
            if 'torch' in mod.__name__:
                self.backend = Backend.TORCH
            elif 'tensorflow' in mod.__name__:
                self.backend = Backend.TENSORFLOW
            else:
                self.backend = Backend.NONE
                
        elif name and backend:
            self.loss_function: Callable = self._resolve()
        else:
            raise LossError(f"Loss cannot be initiallized. You must specify either `loss_function` or both `name` and `backend`.")
        
    def _resolve(self) -> Callable:
        avail_losses = {}
        if self.backend == Backend.TORCH:
            avail_losses = {
                "mse": torch.nn.MSELoss(reduction=self.reduction),
                "mae": torch.nn.L1Loss(reduction=self.reduction),
                "cross_entropy": torch.nn.CrossEntropyLoss(reduction=self.reduction),
                "bce": torch.nn.BCELoss(reduction=self.reduction),
                "bce_logits": torch.nn.BCEWithLogitsLoss(reduction=self.reduction),
            }
        elif self.backend == Backend.TENSORFLOW:
            avail_losses = {
                "mse": tf.keras.losses.MeanSquaredError(reduction=self.reduction),
                "mae": tf.keras.losses.MeanAbsoluteError(reduction=self.reduction),
                "cross_entropy": tf.keras.losses.CategoricalCrossentropy(reduction=self.reduction),
                "bce": tf.keras.losses.BinaryCrossentropy(reduction=self.reduction),
            }
        else:
            raise BackendNotSupportedError(backend=self.backend, method="Loss._resolve()")
            
        loss = avail_losses.get(self.name)
        if loss is None:
            raise LossError(
                f"Unknown loss name (`{self.name}`) for `{self.backend}` backend."
                f"Available losses: {avail_losses.keys()}"
            )
        return loss
    
    @property
    def allowed_keywords(self) -> List[str]:
        # Get the signature object
        sig = inspect.signature(self.loss_function)
        # Iterate through the parameters in the signature
        arg_names = [param.name for param in sig.parameters.values()]
        return arg_names
    
    def __call__(self, *args, **kwargs):
        try:
            return self.loss_function(*args, **kwargs)
        except Exception as e:
            raise LossError(f"Failed to call loss function: {e}")
        
    def __repr__(self):
        if self.name:
            return f"Loss(name='{self.name}', backend='{self.backend.name}', reduction='{self.reduction}')"
        return f"Loss(custom_function={self.loss_function})"
    


@dataclass
class LossResult:
    label: str              # for logging (e.g. 'mse_regression')
    value: Any              # raw loss tensor or float (backend-dependent)
    # weight: float           # scalar weight
    # n_samples: int          # number of samples used in loss calc

class AppliedLoss:
    def __init__(
        self,
        loss: Loss,
        inputs: Dict[str, str],
        weight: float = 1.0,
        label: Optional[str] = None
    ):
        """
        An applied loss term that maps data from the ModelGraph to keyword arguments of a `Loss` function.

        Args:
            loss (Loss): A loss function wrapper (e.g., MSE, triplet, cross-entropy).
            inputs (Dict[str, str]): Maps each argument of the loss function to a specific data source. \
                Keys are the expected loss argument names (e.g., "true", "pred", "anchor", etc). Positional \
                arguments are supported via "0" or "1" keys. 
                Values are dot-strings of the form:
                
                - "FeatureSet.targets"          # use `targets` from a FeatureSet batch (default role)
                - "FeatureSet.features.anchor"  # use `features` from a FeatureSet batch with role="anchor"
                - "Encoder.output.pos"          # use `output` from a ModelStage with role="pos"

                The three components are always:
                - source node (FeatureSet or ModelStage)
                - attribute: "features", "targets", or "output"
                - role: optional; defaults to "default" if not specified.

            weight (float, optional): A scalar weight to apply to the final loss value. \
                Useful when combining multiple loss terms. Default is 1.0.

            label (str, optional): Optional name to assign to this loss for logging or visualization. \
                If not provided, defaults to the `Loss.name`.

        Example:
            For a triplet loss requiring `anchor`, `positive`, and `negative`:
            ```python
            AppliedLoss(
                loss=triplet_loss,
                inputs={
                    "anchor": "Encoder.output.anchor",
                    "positive": "Encoder.output.pos",
                    "negative": "Encoder.output.neg"
                }
            )
            ```

            For an MSE loss requiring `pred` and `true`:
            ```python
            AppliedLoss(
                loss=triplet_loss,
                inputs={
                    "pred": "Encoder.output",
                    "true": "PulseFeatures.features",
                }
            )
            ```
        """
        
        self.loss : Loss = loss
        self.inputs : Dict[str, Tuple[str, str, str]] = {
            str(k): self._parse_input_spec(p) 
            for k,p in inputs.items()
        }
        self.weight = float(weight)
        self.label = label if label is not None else loss.name
        
    @property
    def backend(self) -> Backend:
        return self.loss.backend
    
    @property
    def parsed_inputs(self) -> Dict[str, Tuple[str, str, str]]:
        """
        Returns the required Loss input specs. Returned list could include:
            - "pred": (Encoder, output, default)
            - "true": (PulseFeatures, features, default)
            - "anchor": (Encoder, output, anchor)

        Returns:
            Dict[str, Tuple[str, str, str]]
        """
        return self.inputs
    
    def _parse_input_spec(self, spec: str) -> Tuple[str, str, str]:
        """
        Parses a string specifying a data source for a loss argument.

        Accepted formats:
            - "Node.attribute"         # role defaults to "default"
            - "Node.attribute.role"    # explicitly specify the role

        Where:
            - Node: name of the FeatureSet or ModelStage
            - Attribute: one of "features", "targets", or "output"
            - Role: (optional) name of the role (e.g., "anchor", "pos")

        Returns:
            Tuple[str, str, str]: (node, attribute, role)
        """
        
        node, attribute, role = None, None, None
        
        parts = spec.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid `AppliedLoss.inputs` spec: {spec}")
        elif len(parts) == 2:
            node, attribute = parts
            role = 'default'        # use default role if not specified
        elif len(parts) == 3:
            node, attribute, role = parts
        else:
            raise ValueError(f"Invalid `AppliedLoss.inputs` spec: {spec}")

        allowed_attrs = ['features', "targets", "output"]
        if not attribute in allowed_attrs:
            raise ValueError(
                f"Invalid `AppliedLoss.inputs` spec: {spec}. "
                f"Attribute must be one of the following: {allowed_attrs}. "
                f"Received: {attribute}"
            )

        return node, attribute, role
        
    def compute(self, batch_input: Dict[str, Batch], model_outputs: Dict[str, BatchOutput]) -> LossResult:
        """
        Computes the loss value using the specified input mappings and provided data.

        Args:
            batch_input (Dict[str, Batch]):
                A dictionary of FeatureSet batch_input keyed by FeatureSet label.
                Each Batch contains samples and per-role sample weights.

            model_outputs (Dict[str, BatchOutput]):
                A dictionary of ModelStage outputs keyed by ModelStage label.
                Each output is treated as a BatchOutput where `.features` holds the model output values.

        Returns:
            LossResult:
                Contains:
                    - label (str): loss name for logging
                    - value (Any): backend-dependent scalar or tensor (raw loss output)
                    - weight (float): scalar weight applied to the loss
                    - sample_weights (np.ndarray): shape (n_samples,), averaged across inputs
                    - inputs (Dict[str, Any]): raw tensors passed to the loss function (for debugging)

        Notes:
            - If multiple roles are mapped to the loss function (e.g., "anchor", "pos", "neg"),
              the sample weights from each input are averaged per sample.
            - Loss inputs are converted to the appropriate backend format before evaluation.
        """
        
        kwargs = {}
        sample_weights = {}
        for k,input in self.inputs.items():
            # Ex. values of input: ('PulseFeatures', 'targets', 'default')
            node, attribute, role = input
            
            # Get FeatureSet data 
            if attribute in ['features', 'targets']:
                # Check that node label exists in batches (eg, "PulseFeatures" or "Encoder")
                if node not in batch_input.keys():
                    raise ValueError(
                        f"Required AppliedLoss input (`{node}`) is missing from batch data: {batch_input.keys()}"
                    )
                # Check that role exists in batch (eg, "default" or "anchor")
                if role not in batch_input[node].available_roles:
                    raise ValueError(
                        f"Required AppliedLoss input (`{role}`) is missing from batch data: {batch_input[node].available_roles}"
                    )
                
                # Get sample data
                sample_coll = batch_input[node].role_samples[role]
                n_samples = len(sample_coll)
                sample_weights[k] = batch_input[node].role_sample_weights[role]
                if attribute == 'features':
                    kwargs[k] = sample_coll.get_all_features(
                        format=get_data_format_for_backend(backend=self.backend)
                    )
                else:
                    kwargs[k] = sample_coll.get_all_targets(
                        format=get_data_format_for_backend(backend=self.backend)
                    )
            
            # Get model output data
            elif attribute == 'output':
                # Check that node label exists in model output (eg, "Encoder")
                if node not in model_outputs.keys():
                    raise ValueError(
                        f"Required AppliedLoss input (`{node}`) is missing from model_outputs data: {model_outputs.keys()}"
                    )
                # Check that role exists in model_outputs (eg, "default" or "anchor")
                if role not in model_outputs[node].available_roles:
                    raise ValueError(
                        f"Required AppliedLoss input (`{role}`) is missing from model_outputs data: {model_outputs[node].available_roles}"
                    )
                    
                # Get sample data (don't convert data formats - will break pytorch autograd)
                kwargs[k] = model_outputs[node].features[role]
                
            else:
                raise ValueError(f"Unsupported attribute value: {attribute}")
        
        # Average all sample weights (per-sample weights across all inputs)
        mean_weights = None
        if sample_weights:
            sample_weights = [to_numpy(v) for v in sample_weights.values()]
            # Avg across roles (retain len = len(samples))
            mean_weights = np.mean(np.stack(sample_weights, axis=0), axis=0).reshape(-1)    # shape: (n_samples, )
            
        # Call loss function (convert to positional args if needed)
        if all(k.isdigit() for k in kwargs.keys()):
            args = [kwargs[str(i)] for i in range(len(kwargs))]
            loss_res = self.loss(*args)
        else:
            loss_res = self.loss(**kwargs)
        
        
        # Apply sample weighting
        # Convert mean_weights to correct backend tensor
        weighted_loss = None
        if self.backend == Backend.TORCH:
            # Ensure loss has shape (batch_size, )
            loss_res = loss_res.view(-1)
            mean_weights_tensor = torch.as_tensor(mean_weights, device=loss_res.device)
            weighted_loss = torch.sum(loss_res * mean_weights_tensor) * self.weight


        elif self.backend == Backend.TENSORFLOW:
            # Ensure loss has shape (batch_size, )
            loss_res = tf.reshape(loss_res, [-1])
            mean_weights_tensor = tf.convert_to_tensor(mean_weights, dtype=loss_res.dtype)
            weighted_loss = tf.reduce_sum(loss_res * mean_weights_tensor) * self.weight
            
        else:
            # Assume NumPy
            loss_res = np.reshape(loss_res, (-1,))
            mean_weights = np.reshape(mean_weights, (-1,))
            weighted_loss = np.sum(loss_res * mean_weights) * self.weight
        
        return LossResult(
            label=self.label,
            value=weighted_loss,
        )
    
    

    def __repr__(self) -> str:
        return (
            f"AppliedLoss("
            f"label={self.label}, "
            # f"loss={self.loss}, "
            # f"inputs={self.inputs}, "
            f"weight={self.weight})"
        )