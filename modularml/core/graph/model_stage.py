from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import tensorflow as tf
import torch

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.core.data_structures.data import Data
from modularml.core.data_structures.step_result import StepResult
from modularml.core.graph.computation_node import ComputationNode
from modularml.core.graph.graph_node import GraphNode
from modularml.core.graph.mixins import EvaluableMixin, TrainableMixin
from modularml.core.loss.loss_collection import LossCollection
from modularml.core.loss.loss_record import LossRecord
from modularml.models.wrappers import wrap_model
from modularml.utils.backend import Backend
from modularml.utils.data_format import (
    DataFormat,
    convert_to_format,
    get_data_format_for_backend,
)
from modularml.utils.exceptions import (
    BackendMismatchError,
    BackendNotSupportedError,
    OptimizerNotSetError,
)

if TYPE_CHECKING:
    from modularml.core.loss.applied_loss import AppliedLoss
    from modularml.core.loss.loss_record import LossResult
    from modularml.core.optimizer.optimizer import Optimizer
    from modularml.models.base import BaseModel


class ModelStage(ComputationNode, TrainableMixin, EvaluableMixin):
    """
    A ModelStage represents a single learnable or non-learnable transformation block in a ModelGraph.

    It wraps a backend-specific model (e.g., PyTorch, TensorFlow, or Scikit-learn) and optionally includes an Optimizer.
    A ModelStage receives data from a single input source (FeatureSet or another ModelStage), and produces an output
    which can be consumed by downstream stages or used directly for loss computation.

    If an optimizer is attached, `train_step()` and `eval_step()` can be called directly for this stage. Otherwise,
    training and evaluation should be managed by a parent ModelGraph that handles multiple stages.
    """

    def __init__(
        self,
        label: str,
        model: BaseModel | Any,
        upstream_node: str | GraphNode,
        optimizer: Optimizer | None = None,
    ):
        """
        Initialize a ModelStage.

        Args:
            label (str): Unique name identifying this stage within the model graph.
            model (Union[BaseModel, Any]): A backend-specific model instance or config.
            upstream_node (Union[str, FeatureSet, ModelStage]): The upstream node providing input.
            optimizer (Optional[Optimizer]): Optimizer to use during training (optional).

        """
        ups_node = None
        if isinstance(upstream_node, str):
            ups_node: str | GraphNode = upstream_node
        elif isinstance(upstream_node, GraphNode):
            ups_node = upstream_node.label
        elif isinstance(upstream_node, list | tuple):
            msg = f"ModelStage only accepts a single input. Received: {upstream_node}"
            raise TypeError(msg)
        else:
            msg = f"Unknown input: {upstream_node}"
            raise TypeError(msg)
        if label == ups_node:
            raise ValueError(
                "The upstream_node cannot have the same label as this ModelStage.",
            )

        super().__init__(
            label=label,
            upstream_nodes=ups_node,
        )

        # Set model (cast to BaseModel if explicit subclass not provided)
        self._model: BaseModel = wrap_model(model)
        self._freeze = False  # make stage trainable as default

        # Error checking on optimizer (can be None)
        self._optimizer = optimizer
        self._check_valid_optimizer(required=False)

    # ==========================================
    # ComputationNode Interface
    # ==========================================
    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Returns the input shape of the model after building.

        Returns:
            tuple[int, ...]: Input shape (excluding batch dimension).

        """
        if self._model.input_shape is None:
            if self.is_built:
                raise RuntimeError("Input shape is None after model building.")
            raise RuntimeError("Model must be built before accessing input_shape")
        return self._model.input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Returns the output shape of the model after building.

        Returns:
            tuple[int, ...]: Output shape (excluding batch dimension).

        """
        if self._model.output_shape is None:
            if self.is_built:
                raise RuntimeError("Output shape is None after model building.")
            raise RuntimeError("Model must be built before accessing output_shape")
        return self._model.output_shape

    @property
    def max_upstream_nodes(self) -> int:
        return 1

    @property
    def is_built(self) -> bool:
        """
        Checks if the model has been built (i.e., instantiated with input/output shape).

        Returns:
            bool: True if built, False otherwise.

        """
        return self._model.is_built

    def infer_output_shapes(
        self,
        input_shapes: list[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """
        Infer the expected output shape of this ModelStage without building the backend model.

        This method is used during graph construction to determine the shape of data
        produced by this stage, based on its input shape(s) and internal configuration.

        Expected behavior:
        - If `self.output_shape` is already defined, it will be returned directly.
        - If the underlying `BaseModel` defines a static method `infer_output_shape(input_shape)`,
          it will be used to compute the output shape from the given input.
        - If the output shape cannot be inferred, an error is raised.

        Notes:
            - This method assumes the stage only supports a single input; an error will be raised
              if multiple input shapes are provided.
            - This does NOT instantiate or build the backend model.

        Args:
            input_shapes (list[tuple[int, ...]]): A list of input shapes from upstream stages.
                Must contain exactly one element for ModelStage.

        Returns:
            list[tuple[int, ...]]: The inferred output shapes.

        Raises:
            ValueError: If multiple input shapes are provided or output shape cannot be inferred.

        """
        if self.max_upstream_nodes is not None and len(input_shapes) > self.max_upstream_nodes:
            msg = f"ModelStage only support a single input. Received: {input_shapes}"
            raise ValueError(msg)

        # Return output_shape is already known
        if self.output_shape is not None:
            return [self.output_shape]

        # Pass inferencing task to BaseModel (if supports it)
        if hasattr(self._model, "infer_output_shapes"):
            return self._model.infer_output_shapes(input_shapes[0])
        if hasattr(self._model, "infer_output_shape"):
            return [self._model.infer_output_shape(input_shapes[0])]

        # Otherwise, raise error
        msg = f"Cannot infer output shape for ModelStage `{self.label}`."
        raise ValueError(msg)

    def build(
        self,
        input_shapes: list[tuple[int, ...]] | None = None,
        output_shapes: list[tuple[int, ...]] | None = None,
        *,
        force: bool = False,
    ):
        """
        Build the ModelStage by initializing the underlying BaseModel and its optimizer.

        This method:
        - Validates input shape constraints (only a single input supported).
        - Delegates to the underlying BaseModel's `build()` method using the provided shapes.
        - Constructs the optimizer if defined, based on the model backend.

        Args:
            input_shapes (list[tuple[int, ...]]):
                A list of input shapes from upstream stages. Must contain exactly one element.

            output_shapes (list[tuple[int, ...]] | None, optional):
                The expected output shapes of this stage. If provided, it must contain exactly
                one element. If not provided, the BaseModel must be capable of inferring it
                internally or during construction.
                
            force (bool): If model is already instantiated it will not be re-instantiated unless \
                `force=True`. Defaults to False.

        Raises:
            ValueError: If more than one input shape is provided (ModelStage supports a single input).
            BackendNotSupportedError: If the backend is unrecognized during optimizer construction.

        Notes:
            - For PyTorch and TensorFlow, optimizers are built after the model is initialized.
            - Scikit-learn models typically do not require external optimizers.
            - This method assumes that shape inference and merge logic (if needed) has already
              been resolved upstream by the ModelGraph.

        """
        if self.max_upstream_nodes is not None and len(input_shapes) > self.max_upstream_nodes:
            msg = f"ModelStage only support a single input. Received: {input_shapes}"
            raise ValueError(msg)
        input_shape = input_shapes[0]

        if (
            output_shapes is not None
            and self.max_downstream_nodes is not None
            and len(output_shapes) > self.max_downstream_nodes
        ):
            msg = f"ModelStage only supports a single output. Received: {output_shapes}"
            raise ValueError(msg)
        output_shape = output_shapes[0] if output_shapes is not None else None

        # Build underlying BaseModel if not already built
        if (not self._model.is_built) or force:
            self._model.build(input_shape=input_shape, output_shape=output_shape, force=force)

        # Build optimizer if defined
        if self._optimizer is not None:
            if self.backend == Backend.TORCH:
                self._optimizer.build(
                    parameters=self._model.parameters(),
                    backend=self.backend,
                )
            elif self.backend == Backend.TENSORFLOW:
                self._optimizer.build(backend=self.backend)
            elif self.backend == Backend.SCIKIT:
                # Scikit-learn optimizers are typically fit internally
                pass
            else:
                raise BackendNotSupportedError(
                    backend=self.backend,
                    message="Unknown backend for optimizer building",
                )

    @overload
    def forward(self, batch: Batch, **kwargs) -> BatchOutput: ...
    @overload
    def forward(self, batch: BatchOutput, **kwargs) -> BatchOutput: ...
    @overload
    def forward(self, data: Data, **kwargs) -> Data: ...
    def forward(
        self,
        x: Data | Batch | BatchOutput,
        **kwargs,
    ) -> Data | BatchOutput:
        """
        Performs a forward pass through the model using inputs from Data, Batch, or BatchOutput.

        This method preserves raw tensor outputs to maintain backend autograd support.
        It returns a `BatchOutput` or `Data` object containing model predictions.

        Args:
            x (Union[Data, Batch, BatchOutput]): Input data to the model.
            **kwargs: Any additional keyword arguments to provide to BaseModel.forward

        Returns:
            Union[Data, BatchOutput]: Outputs from the model.

        """
        if isinstance(x, Batch):
            all_outputs = {}
            all_targets = {}
            sample_uuids = {}
            tags = {}
            for role, samples in x.role_samples.items():
                # Format features for this backend
                features = samples.get_all_features(
                    fmt=get_data_format_for_backend(self.backend),
                )
                all_outputs[role] = self._model(features, **kwargs)
                all_targets[role] = samples.get_all_targets(fmt=DataFormat.NUMPY)
                sample_uuids[role] = samples.sample_uuids
                tags[role] = samples.get_all_tags(fmt=DataFormat.DICT_NUMPY)

            # In order to preserve auto-grad for pytorch, we cannot modify underlying
            # data format until after loss computation and optimizer stepping
            return BatchOutput(
                features=all_outputs,  # preserve backend-specific tensors
                targets=all_targets,  # pass targets unmodified
                sample_uuids=sample_uuids,
                tags=tags,
            )

        if isinstance(x, BatchOutput):
            all_outputs = {}
            for role in x.available_roles:
                # Format any-format to this backend (obj is unchanged if in correct format)
                features = convert_to_format(
                    x.features[role],
                    fmt=get_data_format_for_backend(self.backend),
                )
                all_outputs[role] = self._model(features, **kwargs)

            # In order to preserve auto-grad for pytorch, we cannot modify underlying
            # data format until after loss computation and optimizer stepping
            return BatchOutput(
                features=all_outputs,  # preserve backend-specific tensors
                targets=x.targets,  # pass targets unmodified
                sample_uuids=x.sample_uuids,
                tags=x.tags,
            )

        if isinstance(x, Data):
            x = x.to_backend(target=self.backend)
            return Data(self._model(x))

        msg = f"Input must be of type Data or Batch. Received: {type(x)}"
        raise TypeError(msg)

    # ==========================================
    # ModelStages Properties & Dunders
    # ==========================================
    def __repr__(self):
        return (
            f"ModelStage(label='{self.label}', "
            f"upstream_nodes={self._upstream_nodes}, "
            f"downstream_nodes={self._downstream_nodes}, "
            f"model={self._model!r}, "
            f"optimizer={self._optimizer}, "
            f"backend={self.backend})"
        )

    def __str__(self):
        return f"ModelStage ('{self.label}')"

    @overload
    def __call__(self, batch: Batch, **kwargs) -> Batch: ...
    @overload
    def __call__(self, batch: BatchOutput, **kwargs) -> Batch: ...
    @overload
    def __call__(self, data: Data, **kwargs) -> Data: ...
    def __call__(self, x: Data | Batch, **kwargs) -> Data | Batch:
        """
        Shorthand to call `forward()` on input.

        Args:
            x (Union[Data, Batch]): Input to model.
            **kwargs: Additional arguments to pass to BaseModel.__call__

        Returns:
            Union[Data, Batch]: Model output.

        """
        return self.forward(x=x, **kwargs)

    # ==========================================
    # Error Checking Methods
    # ==========================================
    def _check_valid_optimizer(self, *, required: bool = True):
        """
        Verifies that the optimizer is compatible with the model's backend.

        Args:
            required (bool): Whether an optimizer is required. Default is True.

        Raises:
            OptimizerNotSetError: If required and optimizer is None.
            BackendMismatchError: If optimizer and model backends differ.

        """
        if self._optimizer is None and required:
            raise OptimizerNotSetError(model_stage_name=self.label)

        if self._optimizer is not None:
            if self._optimizer.backend is None:
                self._optimizer.backend = self.backend
            elif self._optimizer.backend != self.backend:
                raise BackendMismatchError(
                    expected=self.backend,
                    received=self._optimizer.backend,
                    message=f"Optimizer backend does not match model backend: {self._optimizer.backend} != {self.backend}",
                )

    def _validate_source(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Validates that the required sources for input and losses are present in batch_input.

        Args:
            batch_input (Dict[str, Batch]): Mapping of source labels to Batches.
            losses (list[AppliedLoss], optional): Losses to compute.

        Raises:
            ValueError: If any expected input or loss role is missing.

        """
        # Check that self.upstream_node exists in batch_input
        if self.upstream_node not in batch_input:
            msg = f"batch_input missing required upstream_node: {self.upstream_node}"
            raise ValueError(msg)

        # Check that all required loss roles exist in batch_input
        if losses is not None:
            for loss in losses:
                # node: FeatureSet or ModelStage label
                # attribute: "features", "targets", or "output"
                # role: default or a custom string (eg, 'anchor')
                for node_lbl, attribute, role in loss.parsed_inputs.values():
                    if attribute != "output":
                        # Check that node exists in batch_input
                        if node_lbl not in batch_input:
                            msg = f"batch_input missing required key: {node_lbl}"
                            raise ValueError(msg)
                        # Check that role exists
                        if role not in batch_input[node_lbl].available_roles:
                            msg = f"batch_input missing required `{role}` role for key: {node_lbl}"
                            raise ValueError(msg)

    # ==========================================
    # Trainable Mixin
    # ==========================================
    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def backend(self) -> Backend:
        """
        Returns the backend associated with the wrapped model.

        Returns:
            Backend: TORCH, TENSORFLOW, SCIKIT, ...

        """
        return self._model.backend

    @property
    def freeze(self) -> bool:
        """
        Indicates whether this stage is frozen (not trainable).

        Returns:
            bool: True if frozen, False if trainable.

        """
        return self._freeze

    @freeze.setter
    def freeze(self, value: bool):
        """
        Sets the stage to be frozen (non-trainable) or trainable.

        Args:
            value (bool): True to freeze, False to unfreeze.

        Raises:
            ValueError: If value is not a boolean.

        """
        if not isinstance(value, bool):
            msg = f"Freeze must be a boolean, got {type(value)}"
            raise TypeError(msg)
        self._freeze = value

    def get_input_batch(self, all_batch_data: dict[str, Batch]) -> Batch:
        return all_batch_data[self.upstream_node]

    def _train_step_torch(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss],
    ) -> StepResult:
        """
        Runs a training step using PyTorch: forward, loss, backward, optimizer.

        Args:
            batch_input (Dict[str, Batch]): Input batches.
            losses (list[AppliedLoss]): Loss functions to apply.

        Returns:
            StepResult: Losses and model outputs.

        """
        # Set optimizer and train mode
        self._model.train()
        self._optimizer.zero_grad()

        loss_records: list[LossRecord] = []
        outputs: dict[str, BatchOutput] = {}

        # Forward pass
        outputs[self.label] = self.forward(self.get_input_batch(batch_input))

        # Compute losses
        if losses is not None:
            for loss in losses:
                weighted_raw_loss = loss.compute(batch_input=batch_input, model_outputs=outputs)
                lr = LossRecord(
                    value=weighted_raw_loss,
                    label=loss.label,
                    contributes_to_update=True,
                )
                loss_records.append(lr)

        # Backward + opt step
        lc = LossCollection(records=loss_records)
        lc.trainable.backward()
        self._optimizer.step()

        return StepResult(losses=lc, node_outputs=outputs)

    def _train_step_tensorflow(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss],
    ) -> StepResult:
        """
        Runs a training step using Tensorflow: forward, loss, backward, optimizer.

        Args:
            batch_input (Dict[str, Batch]): Input batches.
            losses (list[AppliedLoss]): Loss functions to apply.

        Returns:
            StepResult: Losses and model outputs.

        """
        # Zero optimizer gradients
        self._optimizer.zero_grad()

        loss_records: list[LossRecord] = []
        outputs: dict[str, BatchOutput] = {}

        # Track gradients over forward passes & loss computation
        with tf.GradientTape() as tape:
            # Forward pass (with training=True)
            outputs: dict[str, BatchOutput] = {}
            outputs[self.label] = self.forward(
                self.get_input_batch(batch_input),
                training=True,
            )

        # Compute losses
        if losses is not None:
            for loss in losses:
                weighted_raw_loss = loss.compute(batch_input=batch_input, model_outputs=outputs)
                lr = LossRecord(
                    value=weighted_raw_loss,
                    label=loss.label,
                    contributes_to_update=True,
                )
                loss_records.append(lr)

        # Backward + opt step
        lc = LossCollection(records=loss_records)
        grads = tape.gradient(lc.total, self._model.trainable_variables)
        self._optimizer.step(grads=grads, variables=self._model.trainable_variables)

        return StepResult(losses=lc, node_outputs=outputs)

    def _train_step_scikit(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss],
    ) -> StepResult:
        # TODO: implement sklean.fit inplace of training
        raise NotImplementedError("Training for scikit model not implemented yet.")

    def train_step(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss],
    ) -> StepResult:
        """
        Performs a training step (forward, loss, backward, optimizer step) for this stage.

        Only callable if this stage has an optimizer and is not frozen. Otherwise, training
        must be delegated to `ModelGraph`.

        Args:
            batch_input (Dict[str, Batch]): Mapping from upstream sources to input batches.
            losses (list[AppliedLoss]): list of loss functions to compute and backpropagate.

        Returns:
            StepResult: Contains loss values and model outputs.

        Raises:
            RuntimeError: If stage is frozen or optimizer is missing.

        """
        # If stage is frozen, raise error
        if self.freeze:
            raise RuntimeError(
                "`train_step` called with `freeze=True`. Use `eval_step` instead.",
            )

        # Ensure batch_input matches expected data in losses
        self._validate_source(batch_input=batch_input, losses=losses)

        # Ensure optimizer is set and matches model backend
        self._check_valid_optimizer(required=True)

        if self.backend == Backend.TORCH:
            return self._train_step_torch(batch_input=batch_input, losses=losses)

        if self.backend == Backend.TENSORFLOW:
            return self._train_step_tensorflow(batch_input=batch_input, losses=losses)

        if self.backend == Backend.SCIKIT:
            return self._train_step_scikit(batch_input=batch_input, losses=losses)

        msg = f"Unknown backend: {self.backend}"
        raise ValueError(msg)

    # ==========================================
    # Evaluable Mixin
    # ==========================================
    def _eval_step_torch(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss] | None,
    ) -> StepResult:
        """
        Runs an evaluation step using PyTorch: forward + loss (no gradients).

        Args:
            batch_input (Dict[str, Batch]): Input batches.
            losses (list[AppliedLoss]): Losses to apply.

        Returns:
            StepResult: Loss values and outputs.

        """
        self._model.eval()

        loss_records: list[LossRecord] = []
        outputs: dict[str, BatchOutput] = {}

        with torch.no_grad():
            # Forward pass
            outputs[self.label] = self.forward(self.get_input_batch(batch_input))

            # Compute losses
            if losses is not None:
                for loss in losses:
                    weighted_raw_loss = loss.compute(batch_input=batch_input, model_outputs=outputs)
                    lr = LossRecord(
                        value=weighted_raw_loss,
                        label=loss.label,
                        contributes_to_update=False,  # not used in optimizer stepping
                    )
                    loss_records.append(lr)

        lc = LossCollection(records=loss_records)
        return StepResult(losses=lc, node_outputs=outputs)

    def _eval_step_tensorflow(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss] | None,
    ) -> StepResult:
        """
        Runs an evaluation step using Tensorflow: forward + loss (no gradients).

        Args:
            batch_input (Dict[str, Batch]): Input batches.
            losses (list[AppliedLoss]): Losses to apply.

        Returns:
            StepResult: Loss values and outputs.

        """
        loss_records: list[LossRecord] = []
        outputs: dict[str, BatchOutput] = {}

        # Forward pass (with training=True)
        outputs[self.label] = self.forward(
            self.get_input_batch(batch_input),
            training=True,
        )

        # Compute losses
        if losses is not None:
            for loss in losses:
                weighted_raw_loss = loss.compute(batch_input=batch_input, model_outputs=outputs)
                lr = LossRecord(
                    value=weighted_raw_loss,
                    label=loss.label,
                    contributes_to_update=False,  # not used in optimizer stepping
                )
                loss_records.append(lr)

        lc = LossCollection(records=loss_records)
        return StepResult(losses=lc, node_outputs=outputs)

    def _eval_step_scikit(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss],
    ) -> StepResult:
        # TODO: perform sklearn.predict
        raise NotImplementedError("Evaluation for scikit model not implemented yet.")

    def eval_step(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss],
    ) -> StepResult:
        """
        Performs an evaluation step (forward pass and loss computation) for this stage.

        Only callable if this stage is frozen. No gradient tracking is performed.

        Args:
            batch_input (Dict[str, Batch]): Mapping from upstream sources to input batches.
            losses (list[AppliedLoss]): list of loss functions to compute.

        Returns:
            StepResult: Contains loss values and model outputs.

        Raises:
            RuntimeError: If stage is not frozen.

        """
        # If stage is not frozen, raise error
        if not self.freeze:
            raise RuntimeError(
                "`eval_step` called with `freeze=False`. Use `train_step` for training.",
            )

        self._validate_source(batch_input=batch_input, losses=losses)

        if self.backend == Backend.TORCH:
            return self._eval_step_torch(batch_input=batch_input, losses=losses)

        if self.backend == Backend.TENSORFLOW:
            return self._eval_step_tensorflow(batch_input=batch_input, losses=losses)

        if self.backend == Backend.SCIKIT:
            return self._eval_step_scikit(batch_input=batch_input, losses=losses)

        msg = f"Unknown backend: {self.backend}"
        raise ValueError(msg)
