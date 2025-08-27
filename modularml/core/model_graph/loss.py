

from abc import ABC
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import tensorflow as tf
import torch as torch
import inspect

from modularml.core.data_structures.batch import Batch, BatchComponentSelector
from modularml.core.data_structures.sample import Sample
from modularml.utils.backend import Backend
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
            loss1 = Loss(loss_function: mean_squared_error)
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
    
    def __call__(self, *args, **kwargs):
        try:
            return self.loss_function(*args, **kwargs)
        except Exception as e:
            raise LossError(f"Failed to call loss function: {e}")
        
    def __repr__(self):
        if self.name:
            return f"Loss(name='{self.name}', backend='{self.backend.name}', reduction='{self.reduction}')"
        return f"Loss(custom_function={self.loss_function})"
    
    
    # ==========================================
    #   State/Config Management Methods
    # ==========================================	
    def get_config(self) -> Dict[str, Any]:
        if self.name is None or self.backend is None:
            # TODO: figure out how to serialize/register custom Losses
            raise LossError("Loss object created from callable is not serializable.")
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "backend": str(self.backend.value),
            "reduction": self.reduction,
        }
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Loss":
        name = config.pop("name")
        backend = config.pop("backend")
        reduction = config.pop("reduction")
        return cls(name=name, backend=backend, reduction=reduction)
    
    
class AppliedLoss:
    def __init__(
        self,
        loss: Loss,
        loss_kw_selector: Dict[str, BatchComponentSelector],
        weight: float = 1.0,
        label: Optional[str] = None
    ):
        """
        A base class for different Loss applications (e.g. simple, constrastive, triplet, ...).

        Args:
            loss (Loss): The Loss function being applied.
            loss_kw_selector (Dict[str, BatchComponentSelector]): Maps keyword arguments of `loss` \
                to a specific selection of the batch data. See the examples below.
            weight (float, optional): An optional weight to apply to this Loss. This is useful when \
                balancing multiple loss terms in the larger ModelGraph. Defaults to 1.0.
            label (Optional[str], optional): An optional label assigned to this Loss instance. \
                This label will be used when logging/print/plotting. Defaults to None.
                
        Examples:
            Say you have a triplet_loss function that requires keyword arguments of `anchor`, \
            `negative`, and `positive`. There's also a FeatureSampler with `.available_roles` of \
            `['ref_samples', 'high_soh', 'low_soh']`. We can map the sampler roles to the require \
            loss keywords with:
            ``` python
                def triplet_loss(*, anchor, negative, positive): 
                    ...
                FeatureSampler.available_roles
                >>> ['ref_samples', 'high_soh', 'low_soh']
                
                AppliedLoss(
                    loss=triplet_loss,
                    loss_kw_selector={
                        'anchor': BatchComponentSelector('ref_samples', 'features'),
                        'negative': BatchComponentSelector('low_soh', 'features'),
                        'positive': BatchComponentSelector('high_soh', 'features'),
                    }
                )
            ```
            This maps the features from samples under the role 'ref_samples' to 'anchor', 'low_soh' \
            to 'negative', and 'high_soh' to 'positive'.
        """
        
        self.loss = loss
        self.loss_kw_selector : Dict[str, BatchComponentSelector] = loss_kw_selector
        self.weight = weight
        self.label = label if label is not None else loss.name
        
    @property
    def backend(self) -> Backend:
        return self.loss.backend
    
    
    # ==========================================
    #   State/Config Management Methods
    # ==========================================	
    def get_config(self) -> Dict[str, Any]:
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "loss": self.loss.get_config(),
            "loss_kw_selector": {k: v.get_config() for k,v in self.loss_kw_selector.items()},
            "weight": self.weight,
            "label": self.label,
        }
  
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AppliedLoss":
        return cls(
            loss=Loss.from_config(config['loss']),
            loss_kw_selector={
                k: BatchComponentSelector.from_config(v)
                for k,v in config['loss_kw_selector'].items()
            },
            weight=config['weight'],
            label=config['label'],        
        )
    
    
    def compute(self, batch: Batch, model_outputs: Batch) -> Dict[str, Any]:
        """
        Computes the AppliedLoss on the provided batch data and subsequent model outputs.

        Args:
            batch (Batch): The batch data input to the model.
            model_outputs (Batch): The batch data output by the model.
            
        Returns:
            Dict[str, Any]: Keys include: `'loss', 'unweighted', 'label'`
        """
        
        
        
        
        
        pass
    
    
    
    def compute(self, **kwargs) -> Dict[str, Any]:
        """
        Computes the AppliedLoss on the provided data. The required keyword argument differ \
            based on the value of `self.between`. 

        Args:
            kwargs (Dict[str, Any]): Must contain an `'outputs'` key in addition to requirements \
                defined by `self.between` (see below). The following keys will also be used if \
                provided: `['weights', 'tags']`
                * `between='target'`: `data.keys()` must contain `['outputs', 'targets']`
                * `between='feature'`: `data.keys()` must contain `['outputs', 'features']`
                * `between='sample'`: `data.keys()` must contain `'outputs'` and `'outputs'` must \
                    be a dictionary with keys matching the values defined by `self.sample_key`.
        Returns:
            Dict[str, Any]: Keys include: `'loss', 'unweighted', 'label'`
        """
        
        if not 'outputs' in kwargs:
            raise LossError(f"AppliedLoss.compute() requires an `outputs` keyword.")
        
        loss_fnc_inputs = []
        if self.between == 'target':            
            if not "targets" in kwargs:
                raise LossError(f"AppliedLoss.compute() requires an `targets` keyword for `between='target'`.")            
            loss_fnc_inputs = [kwargs['outputs'], kwargs['targets']]

        elif self.between == 'feature':
            if not "features" in kwargs:
                raise LossError(f"AppliedLoss.compute() requires an `features` keyword for `between='feature'`.")
            loss_fnc_inputs = [kwargs['outputs'], kwargs['features']]
        
        elif self.between == 'sample':
            if not isinstance(kwargs['outputs'], dict):
                raise TypeError(
                    f"Expected `kwargs['outputs']` to be a dictionary with keys equal to `self.sample_key`."
                    f"Received: {type(kwargs['outputs'])}"
                )
            missing_sample_key = [
                k for k in self.sample_key
                if k not in kwargs['outputs'].keys()
            ]
            if missing_sample_key:
                raise ValueError(f"`kwargs['outputs']` is missing required sample keys: {missing_sample_key}")
            loss_fnc_inputs = [kwargs['outputs'][k] for k in self.sample_key]
            
        else:
            raise LossError(f"Unknown `between` value: {self.between}")
        
        loss_value = 0.0
        sample_weights = kwargs.get("weights", None)
        
        if sample_weights.shape[0] != loss_fnc_inputs[0].shape[0]:
            raise LossError("Sample weights must match batch size.")

        if self.backend == Backend.TORCH:
            if sample_weights is not None:
                # Case 1: Loss returns per-sample value (shape of [batch_size, ])
                if getattr(self.loss, "reduction", None) == 'none':
                    raw_loss = self.loss(*loss_fnc_inputs)  # shape: [batch_size, ]
                    loss_value = (raw_loss * sample_weights).mean()
                
                # Case 2: Loss returns scalar, must iterate over samples
                else:
                    batch_size = loss_fnc_inputs[0].size(0)
                    for i in range(batch_size):
                        inputs = [x[i].unsqueeze(0) for x in loss_fnc_inputs]
                        loss_value += self.loss(*inputs) * sample_weights[i]
                    loss_value /= batch_size            
            else:
                # No sample weighting
                loss_value = self.loss(*loss_fnc_inputs).mean()
                 
        elif self.backend == Backend.TENSORFLOW:
            if sample_weights is not None:
                # Case 1: Loss returns per-sample value (shape of [batch_size, ])
                if getattr(self.loss, "reduction", None) == 'none':
                    raw_loss = self.loss(*loss_fnc_inputs)  # shape: [batch_size, ]
                    loss_value = tf.reduce_mean(raw_loss * sample_weights)
                
                # Case 2: Loss returns scalar, must iterate over samples
                else:
                    batch_size = tf.shape(loss_fnc_inputs[0])[0]
                    for i in range(batch_size):
                        inputs = [x[i:i+1] for x in loss_fnc_inputs]  # keep dims
                        loss_value += self.loss(*inputs) * sample_weights[i]
                    loss_value /= tf.cast(batch_size, tf.float32)       
            else:
                # No sample weighting
                loss_value = tf.reduce_mean(self.loss(*loss_fnc_inputs))
            
        else:
            raise BackendNotSupportedError(backend=self.backend, method="AppliedLoss.compute()")

        return {
            "loss": loss_value * self.weight,    # Apply overall loss weight
            "unweighted": loss_value,
            "label": self.label or self.loss.name or "custom_loss",
        }




    
    
 


class AppliedLoss:
    def __init__(
        self,
        loss: Loss,
        between: Literal['target', 'feature', 'sample'], 
        sample_key: Union[str, List[str]],
        feature_key: Optional[str] = None,
        weight: float = 1.0,
        label: Optional[str] = None,
    ):
        """
        A base class for different Loss applications (e.g. simple, constrastive, triplet, ...).

        Args:
            loss (Loss): The Loss function being applied.
            between (Literal['target', 'feature', 'sample']): How the loss is computed.
                * 'target': the loss is applied between the model output and the true output.
                * 'feature': the loss is applied between the model output and the original feature.
                * 'sample': the loss is applied between the model's output for multiple samples.
                
            sample_key (Union[str, List[str]]): Defines which sample to use when computing the loss. \
                If `between='target'` or `between='feature'`, `sample_key` must be a single string \
                matching a sample key name in FeatureSampler (e.g., 'anchor'). If `between='sample'`, \
                `sample_key` must be a list of at least 2 strings (e.g.,. ['anchor', 'positive', 'negative']).
            
            feature_key (str, optional): Required if `between=feature`. Defines which feature source \
                to use when computing the loss. 
            weight (float, optional): An optional weight to apply to this Loss. \
                This is useful when utilizing multiple loss terms in the ModelGraph. \
                Defaults to 1.0.
            label (Optional[str], optional): An optional label to assigned to this Loss instance. \
                This label will be used when logging/print/plotting. Defaults to None.
        """
        
        self.loss = loss
        self.weight = weight
        self.label = label
        
        valid_between = ['target', 'feature', 'sample']
        if not between in valid_between:
            raise LossError(f"`AppliedLoss.between` must be one of the following: {valid_between}")
        self.between = between
        
        self.feature_key = feature_key
        self.sample_key = sample_key
        
        self._validate_definitions()
        
    def _validate_definitions(self):
        if self.between in ['target', 'feature']:
            if not isinstance(self.sample_key, str):
                raise LossError(
                    f"`sample_key` must be a single string when `between='feature'` or `between='target'`"
                )
            if self.between == 'feature' and self.feature_key is None:
                raise LossError(f"`feature_key` must be provided when `between='feature'`")
        
        elif self.between == 'sample':
            if not isinstance(self.sample_key, list):
                raise LossError(f"`sample_key` must be a list of strings when `between='sample'`")
            if len(self.sample_key) < 2:
                raise LossError(f"`sample_key` must have at least 2 elements when `between='sample'`")
                    
    @property
    def backend(self) -> Backend:
        return self.loss.backend
    
    def compute(self, **kwargs) -> Dict[str, Any]:
        """
        Computes the AppliedLoss on the provided data. The required keyword argument differ \
            based on the value of `self.between`. 

        Args:
            kwargs (Dict[str, Any]): Must contain an `'outputs'` key in addition to requirements \
                defined by `self.between` (see below). The following keys will also be used if \
                provided: `['weights', 'tags']`
                * `between='target'`: `data.keys()` must contain `['outputs', 'targets']`
                * `between='feature'`: `data.keys()` must contain `['outputs', 'features']`
                * `between='sample'`: `data.keys()` must contain `'outputs'` and `'outputs'` must \
                    be a dictionary with keys matching the values defined by `self.sample_key`.
        Returns:
            Dict[str, Any]: Keys include: `'loss', 'unweighted', 'label'`
        """
        
        if not 'outputs' in kwargs:
            raise LossError(f"AppliedLoss.compute() requires an `outputs` keyword.")
        
        loss_fnc_inputs = []
        if self.between == 'target':            
            if not "targets" in kwargs:
                raise LossError(f"AppliedLoss.compute() requires an `targets` keyword for `between='target'`.")            
            loss_fnc_inputs = [kwargs['outputs'], kwargs['targets']]

        elif self.between == 'feature':
            if not "features" in kwargs:
                raise LossError(f"AppliedLoss.compute() requires an `features` keyword for `between='feature'`.")
            loss_fnc_inputs = [kwargs['outputs'], kwargs['features']]
        
        elif self.between == 'sample':
            if not isinstance(kwargs['outputs'], dict):
                raise TypeError(
                    f"Expected `kwargs['outputs']` to be a dictionary with keys equal to `self.sample_key`."
                    f"Received: {type(kwargs['outputs'])}"
                )
            missing_sample_key = [
                k for k in self.sample_key
                if k not in kwargs['outputs'].keys()
            ]
            if missing_sample_key:
                raise ValueError(f"`kwargs['outputs']` is missing required sample keys: {missing_sample_key}")
            loss_fnc_inputs = [kwargs['outputs'][k] for k in self.sample_key]
            
        else:
            raise LossError(f"Unknown `between` value: {self.between}")
        
        loss_value = 0.0
        sample_weights = kwargs.get("weights", None)
        
        if sample_weights.shape[0] != loss_fnc_inputs[0].shape[0]:
            raise LossError("Sample weights must match batch size.")

        if self.backend == Backend.TORCH:
            if sample_weights is not None:
                # Case 1: Loss returns per-sample value (shape of [batch_size, ])
                if getattr(self.loss, "reduction", None) == 'none':
                    raw_loss = self.loss(*loss_fnc_inputs)  # shape: [batch_size, ]
                    loss_value = (raw_loss * sample_weights).mean()
                
                # Case 2: Loss returns scalar, must iterate over samples
                else:
                    batch_size = loss_fnc_inputs[0].size(0)
                    for i in range(batch_size):
                        inputs = [x[i].unsqueeze(0) for x in loss_fnc_inputs]
                        loss_value += self.loss(*inputs) * sample_weights[i]
                    loss_value /= batch_size            
            else:
                # No sample weighting
                loss_value = self.loss(*loss_fnc_inputs).mean()
                 
        elif self.backend == Backend.TENSORFLOW:
            if sample_weights is not None:
                # Case 1: Loss returns per-sample value (shape of [batch_size, ])
                if getattr(self.loss, "reduction", None) == 'none':
                    raw_loss = self.loss(*loss_fnc_inputs)  # shape: [batch_size, ]
                    loss_value = tf.reduce_mean(raw_loss * sample_weights)
                
                # Case 2: Loss returns scalar, must iterate over samples
                else:
                    batch_size = tf.shape(loss_fnc_inputs[0])[0]
                    for i in range(batch_size):
                        inputs = [x[i:i+1] for x in loss_fnc_inputs]  # keep dims
                        loss_value += self.loss(*inputs) * sample_weights[i]
                    loss_value /= tf.cast(batch_size, tf.float32)       
            else:
                # No sample weighting
                loss_value = tf.reduce_mean(self.loss(*loss_fnc_inputs))
            
        else:
            raise BackendNotSupportedError(backend=self.backend, method="AppliedLoss.compute()")

        return {
            "loss": loss_value * self.weight,    # Apply overall loss weight
            "unweighted": loss_value,
            "label": self.label or self.loss.name or "custom_loss",
        }

    # ==========================================
    #   State/Config Management Methods
    # ==========================================	
    def get_config(self) -> Dict[str, Any]:
        return {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "loss": self.loss.get_config(),
            "between": self.between,
            "sample_key": self.sample_key,
            "feature_key": self.feature_key,
            "weight": self.weight,
            "label": self.label,
        }
  
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AppliedLoss":
        return cls(
            loss=Loss.from_config(config['loss']),
            between=config['between'],
            sample_key=config['sample_key'],
            feature_key=config['feature_key'],
            weight=config['weight'],
            label=config['label'],
        )
 
