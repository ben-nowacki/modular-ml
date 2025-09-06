import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
import torch

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.utils.backend import Backend
from modularml.utils.data_format import get_data_format_for_backend, to_numpy
from modularml.utils.exceptions import BackendNotSupportedError, LossError


class Loss:
    """
    A backend-agnostic wrapper around loss functions used in model training.

    This class allows the use of built-in loss functions from supported backends (PyTorch, TensorFlow)
    or custom-defined loss functions (e.g., scikit-learn, numpy-based) and ensures compatibility with
    the modular training workflow.
    """

    def __init__(
        self,
        name: str | None = None,
        backend: Backend | None = None,
        loss_function: Callable | None = None,
        reduction: str = "none",
    ):
        """
        A backend-agnostic wrapper around loss functions used in model training.

        This class allows the use of built-in loss functions from supported backends (PyTorch, TensorFlow)
        or custom-defined loss functions (e.g., scikit-learn, numpy-based) and ensures compatibility with
        the modular training workflow.

        Args:
            name (str | None): Name of the built-in loss function (e.g., "mse", "mae").
            backend (Backend | None): Backend to use (e.g., Backend.TORCH or Backend.TENSORFLOW).
            loss_function (Callable | None): A custom user-defined loss function.
            reduction (str): Reduction strategy (e.g., "mean", "sum", "none"). Defaults to "none".

        Raises:
            LossError: If neither `loss_function` nor both `name` and `backend` are provided.
            BackendNotSupportedError: If backend resolution is attempted on an unsupported backend.

        Examples:
            ```python
            # Using built-in loss
            loss1 = Loss(name="mse", backend=Backend.TORCH)

            # Using custom loss function
            from sklearn.metrics import mean_squared_error

            loss2 = Loss(loss_function=mean_squared_error)
            ```

        """
        self.name = name.lower() if name else None
        self.backend = backend
        self.reduction = reduction

        if loss_function is not None:
            self.loss_function: Callable = loss_function
            mod = inspect.getmodule(loss_function)
            if self.name is None:
                self.name = mod.__name__

            # TODO: how to infer backend?
            if "torch" in mod.__name__:
                self.backend = Backend.TORCH
            elif "tensorflow" in mod.__name__:
                self.backend = Backend.TENSORFLOW
            else:
                self.backend = Backend.NONE

        elif name and backend:
            self.loss_function: Callable = self._resolve()
        else:
            msg = "Loss cannot be initiallized. You must specify either `loss_function` or both `name` and `backend`."
            raise LossError(msg)

    def _resolve(self) -> Callable:
        """
        Resolves the appropriate loss function object from the selected backend using the given name.

        Returns:
            Callable: A callable loss function.

        Raises:
            BackendNotSupportedError: If the backend is not supported.
            LossError: If the loss name is not recognized.

        """
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
            msg = (
                f"Unknown loss name (`{self.name}`) for `{self.backend}` backend."
                f"Available losses: {avail_losses.keys()}"
            )
            raise LossError(msg)
        return loss

    @property
    def allowed_keywords(self) -> list[str]:
        """
        Returns the list of valid keyword arguments for the current loss function.

        Returns:
            List[str]: A list of argument names accepted by the loss function.

        """
        # Get the signature object
        sig = inspect.signature(self.loss_function)
        # Iterate through the parameters in the signature
        arg_names = [param.name for param in sig.parameters.values()]
        return arg_names

    def __call__(self, *args, **kwargs):
        """
        Invokes the underlying loss function with the provided arguments.

        Returns:
            Any: Output of the loss function.

        Raises:
            LossError: If the loss function fails during execution.

        """
        try:
            return self.loss_function(*args, **kwargs)
        except Exception as e:
            raise LossError("Failed to call loss function.") from e

    def __repr__(self):
        if self.name:
            return f"Loss(name='{self.name}', backend='{self.backend.name}', reduction='{self.reduction}')"
        return f"Loss(custom_function={self.loss_function})"


@dataclass
class LossResult:
    """
    A container for storing the result of computing a loss value.

    Attributes:
        label (str): Identifier for the loss (e.g., "mse_loss", "triplet_margin").
        value (Any): Computed loss value (typically a backend-specific tensor or scalar).

    """

    label: str
    value: Any


class AppliedLoss:
    """
    Encapsulates a loss function with explicit data mappings from the model graph.

    Description:
        `AppliedLoss` binds a `Loss` function to specific inputs from a modular model graph.
        Each input to the loss function is defined as a string reference of the form
        `"Node.attribute"` or `"Node.attribute.role"`, where:

        - `Node` is a FeatureSet or ModelStage label
        - `attribute` is one of: 'features', 'targets', 'output'
        - `role` is optional (defaults to "default")

        This enables flexible training configurations (e.g., supervised, multitask, contrastive)
        across any backend (Torch, TensorFlow, NumPy).

    Example:
        ```python
        AppliedLoss(
            loss=Loss(name="mse", backend=Backend.TORCH),
            all_inputs={"pred": "Regressor.output", "true": "Inputs.targets"},
        )
        ```

    """

    def __init__(self, loss: Loss, all_inputs: dict[str, str], weight: float = 1.0, label: str | None = None):
        """
        Initialize an AppliedLoss instance.

        Args:
            loss (Loss): Loss function object, including backend and callable.
            all_inputs (dict[str, str]): Dictionary mapping loss argument names (e.g., "pred", "true") \
                to graph references like "Node.attribute" or "Node.attribute.role".
            weight (float, optional): Scalar multiplier applied to the loss result. Defaults to 1.0.
            label (str, optional): Custom name for this loss. Defaults to the loss's name.

        """
        self.loss: Loss = loss
        self.all_inputs: dict[str, tuple[str, str, str]] = {
            str(k): self._parse_input_spec(p) for k, p in all_inputs.items()
        }
        self.weight = float(weight)
        self.label = label if label is not None else loss.name

    @property
    def backend(self) -> Backend:
        """
        Backend of the underlying loss function.

        Returns:
            Backend: The backend (Torch, TF, NumPy) used by the loss function.

        """
        return self.loss.backend

    @property
    def parsed_inputs(self) -> dict[str, tuple[str, str, str]]:
        """
        Get the parsed input mappings for this AppliedLoss instance.

        Description:
            This property returns the dictionary of parsed loss input mappings.
            Each entry maps a loss argument (e.g., "pred", "true", "anchor") to a
            3-tuple of the form:

                (node_label, attribute_name, role_name)

            Where:
                - `node_label` is the label of a FeatureSet or ModelStage in the model graph.
                - `attribute_name` is one of: "features", "targets", or "output".
                - `role_name` is a string indicating the input role (e.g., "default", "anchor").

            These mappings are used internally by the `compute()` method to extract the appropriate
            tensors from input batches or model outputs and pass them to the loss function.

        Example:
            ```python
            {
                "pred":   ("Encoder", "output", "default"),
                "true":   ("InputFeatures", "targets", "default"),
                "anchor": ("Encoder", "output", "anchor")
            }
            ```

        Returns:
            dict[str, tuple[str, str, str]]: Dictionary mapping loss argument names to \
            (node_label, attribute_name, role_name) tuples.

        """
        return self.all_inputs

    def _parse_input_spec(self, spec: str) -> tuple[str, str, str]:
        """
        Parse a dot-separated input spec into components.

        Args:
            spec (str): A string in the form "Node.attribute" or "Node.attribute.role".

        Returns:
            tuple[str, str, str]: Parsed (node, attribute, role) tuple.

        Raises:
            ValueError: If the input is malformed or contains invalid attributes.

        """
        node, attribute, role = None, None, None

        parts = spec.split(".")
        if len(parts) < 2:
            msg = f"Invalid `AppliedLoss.inputs` spec: {spec}"
            raise ValueError(msg)
        if len(parts) == 2:
            node, attribute = parts
            role = "default"  # use default role if not specified
        elif len(parts) == 3:
            node, attribute, role = parts
        else:
            msg = f"Invalid `AppliedLoss.inputs` spec: {spec}"
            raise ValueError(msg)

        allowed_attrs = ["features", "targets", "output"]
        if attribute not in allowed_attrs:
            msg = (
                f"Invalid `AppliedLoss.inputs` spec: {spec}. "
                f"Attribute must be one of the following: {allowed_attrs}. "
                f"Received: {attribute}"
            )
            raise ValueError(msg)

        return node, attribute, role

    def compute(self, batch_input: dict[str, Batch], model_outputs: dict[str, BatchOutput]) -> LossResult:
        """
        Compute the loss value given input batches and model outputs.

        Args:
            batch_input (dict[str, Batch]): Mapping of FeatureSet label to input batch.
            model_outputs (dict[str, BatchOutput]): Mapping of ModelStage label to output.

        Returns:
            LossResult: Contains the computed loss value and metadata (label, weight).

        Raises:
            ValueError: If input spec references unknown nodes or missing roles.

        Notes:
            - Each loss argument (e.g., "pred", "true", "anchor") is fetched from either batch_input \
              or model_outputs depending on its node label.
            - All sample weights are averaged across inputs for per-sample weighting.
            - Loss function is called with raw backend tensors (not Data or Batch).

        """
        kwargs = {}
        sample_weights = {}
        for k, parsed_input in self.all_inputs.items():
            # Ex. values of input: ('PulseFeatures', 'targets', 'default') or ('Encoder', 'output', 'default')
            node, attribute, role = parsed_input

            # ComputationNodes support the following attributes:
            # - 'features' or 'output': gets the BatchOutput.features data
            # - 'targets': gets the BatchOutput.targets data

            # FeatureSets support the following attributes:
            # - 'features': gets the FeatureSet.features data
            # - 'targets': gets the BatchOutput.targets data

            # Check the loss spec only maps to either the input data or model outputs, but not both
            if node in batch_input and node in model_outputs:
                msg = (
                    f"Ambiguous AppliedLoss definition. Input key exists in both batch_input and model_outputs: {node}"
                )
                raise ValueError(msg)

            # Collect input data
            if node in batch_input:
                # Ensure that role exists in batch (eg, "default" or "anchor")
                if role not in batch_input[node].available_roles:
                    msg = f"Required AppliedLoss role (`{role}`) is missing from batch_input: {batch_input[node].available_roles}"
                    raise ValueError(msg)

                sample_coll = batch_input[node].role_samples[role]
                sample_weights[k] = batch_input[node].role_sample_weights[role]
                if attribute == "features":
                    kwargs[k] = sample_coll.get_all_features(format=get_data_format_for_backend(backend=self.backend))
                elif attribute == "targets":
                    kwargs[k] = sample_coll.get_all_targets(format=get_data_format_for_backend(backend=self.backend))
                else:
                    msg = f"Invalid AppliedLoss input attribute for batch_input: {attribute}"
                    raise ValueError(msg)

            # Collect output data
            elif node in model_outputs:
                # Ensure that role exists in batch (eg, "default" or "anchor")
                if role not in model_outputs[node].available_roles:
                    msg = f"Required AppliedLoss role (`{role}`) is missing from model_outputs: {model_outputs[node].available_roles}"
                    raise ValueError(msg)

                # Get attribute data (don't convert data format, will break pytorch autograd)
                if attribute in ["features", "output"]:
                    kwargs[k] = model_outputs[node].features[role]
                elif attribute == "targets":
                    kwargs[k] = model_outputs[node].targets[role]
                else:
                    msg = f"Invalid AppliedLoss input attribute for model_outputs: {attribute}"
                    raise ValueError(msg)

            else:
                msg = f"AppliedLoss input key does not exist in batch_input or model_outputs: {node}"
                raise ValueError(msg)

        # Average all sample weights (per-sample weights across all inputs)
        mean_weights = None
        if sample_weights:
            sample_weights = [to_numpy(v) for v in sample_weights.values()]
            # Avg across roles (retain len = len(samples))
            mean_weights = np.mean(np.stack(sample_weights, axis=0), axis=0).reshape(-1)  # shape: (n_samples, )

        # Call loss function (convert to positional args if needed)
        if all(k.isdigit() for k in kwargs):
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
        return f"AppliedLoss(label={self.label}, loss={self.loss}, all_inputs={self.all_inputs}, weight={self.weight})"

    def __str__(self):
        return f"AppliedLoss ('{self.label}')"
