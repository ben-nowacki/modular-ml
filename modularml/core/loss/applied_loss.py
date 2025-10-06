from typing import Any

import numpy as np
import tensorflow as tf
import torch

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.core.loss.loss import Loss
from modularml.utils.backend import Backend
from modularml.utils.data_conversion import to_numpy
from modularml.utils.data_format import get_data_format_for_backend


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

    def __init__(
        self,
        loss: Loss,
        all_inputs: list[str] | dict[str, str],
        weight: float = 1.0,
        label: str | None = None,
    ):
        """
        Initialize an AppliedLoss instance.

        Args:
            loss (Loss): Loss function object, including backend and callable.
            all_inputs (list[str] | dict[str, str]): A list of loss arguments or a dictionary \
                mapping loss argument names (e.g., "pred", "true") to graph references like \
                "Node.attribute" or "Node.attribute.role".
            weight (float, optional): Scalar multiplier applied to the loss result. Defaults to 1.0.
            label (str, optional): Custom name for this loss. Defaults to the loss's name.

        """
        self.loss: Loss = loss
        self.all_inputs: dict[str, tuple[str, str, str]] = {}
        if isinstance(all_inputs, list):
            for i, p in enumerate(all_inputs):
                self.all_inputs[f"{i}"] = self._parse_input_spec(p)
        elif isinstance(all_inputs, dict):
            for k, p in all_inputs.items():
                self.all_inputs[str(k)] = self._parse_input_spec(p)
        else:
            msg = f"`all_inputs` must be a list or dict. Received: {type(all_inputs)}."
            raise TypeError(msg)

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

    def compute(
        self,
        batch_input: dict[str, Batch],
        model_outputs: dict[str, BatchOutput],
    ) -> Any:
        """
        Compute the loss value given input batches and model outputs.

        Args:
            batch_input (dict[str, Batch]): Mapping of FeatureSet label to input batch.
            model_outputs (dict[str, BatchOutput]): Mapping of ModelStage label to output.

        Returns:
            Any: The computed loss value in the data format of the loss function backend.

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
                    kwargs[k] = sample_coll.get_all_features(fmt=get_data_format_for_backend(backend=self.backend))
                elif attribute == "targets":
                    kwargs[k] = sample_coll.get_all_targets(fmt=get_data_format_for_backend(backend=self.backend))
                else:
                    msg = f"Invalid AppliedLoss input attribute for batch_input: {attribute}"
                    raise ValueError(msg)

            # Collect output data
            elif node in model_outputs:
                # Ensure that role exists in batch (eg, "default" or "anchor")
                if role not in model_outputs[node].available_roles:
                    msg = f"Required AppliedLoss role (`{role}`) is missing from model_outputs: {model_outputs[node].available_roles}"
                    raise ValueError(msg)

                sample_weights[k] = model_outputs[node].sample_weights[role]

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
            raw_loss = self.loss(*args)
        else:
            raw_loss = self.loss(**kwargs)

        # Apply sample weighting
        # Convert mean_weights to correct backend tensor
        weighted_raw_loss = None
        if self.backend == Backend.TORCH:
            # Ensure loss has shape (batch_size, )
            raw_loss = raw_loss.view(-1)
            mean_weights_tensor = torch.as_tensor(mean_weights, device=raw_loss.device)
            weighted_raw_loss = torch.sum(raw_loss * mean_weights_tensor) * self.weight

        elif self.backend == Backend.TENSORFLOW:
            # Ensure loss has shape (batch_size, )
            raw_loss = tf.reshape(raw_loss, [-1])
            mean_weights_tensor = tf.convert_to_tensor(mean_weights, dtype=raw_loss.dtype)
            weighted_raw_loss = tf.reduce_sum(raw_loss * mean_weights_tensor) * self.weight

        else:
            # Assume NumPy
            raw_loss = np.reshape(raw_loss, (-1,))
            mean_weights = np.reshape(mean_weights, (-1,))
            weighted_raw_loss = np.sum(raw_loss * mean_weights) * self.weight

        # returns the raw weighted loss without changing data types (to preserve auto-grad)
        return weighted_raw_loss

    def __repr__(self) -> str:
        return f"AppliedLoss(label={self.label}, loss={self.loss}, all_inputs={self.all_inputs}, weight={self.weight})"

    def __str__(self):
        return f"AppliedLoss ('{self.label}')"
