
from platform import node
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, overload
import torch
import tensorflow as tf

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.core.data_structures.data import Data
from modularml.core.data_structures.step_result import StepResult
from modularml.core.model_graph.computation_node import ComputationNode
from modularml.core.model_graph.graph_node import GraphNode
from modularml.core.model_graph.loss import AppliedLoss, LossResult
from modularml.core.model_graph.optimizer import Optimizer
from modularml.models.base import BaseModel
from modularml.models.wrappers import wrap_model
from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat, convert_to_format, get_data_format_for_backend
from modularml.utils.exceptions import BackendMismatchError, BackendNotSupportedError, OptimizerNotSetError
from modularml.utils.modeling import make_dummy_data


if TYPE_CHECKING:
    from modularml.core.data_structures.feature_set import FeatureSet



class ModelStage(ComputationNode):
    """
    A modular wrapper around a backend-specific model (e.g., PyTorch, TensorFlow, Scikit-learn),
    used as a single stage in a ModelGraph.

    A ModelStage can optionally have an optimizer attached. If an optimizer is present,
    `train_step()` and `eval_step()` can be called directly. If no optimizer is set,
    the stage is assumed to be trained externally via `ModelGraph`.
    """
    def __init__(
        self,
        label: str,
        model: Union[BaseModel, Any],
        input: Union[str, "FeatureSet", "ModelStage"],
        optimizer: Optional[Optimizer] = None,
    ):
        """
        A modular wrapper around a backend-specific model (e.g., PyTorch, TensorFlow, Scikit-learn),
        used as a single stage in a ModelGraph.

        A ModelStage can optionally have an optimizer attached. If an optimizer is present,
        `train_step()` and `eval_step()` can be called directly. If no optimizer is set,
        the stage is assumed to be trained externally via `ModelGraph`.

        Attributes:
            label (str): Unique label for the stage within the ModelGraph.
            model (Union[BaseModel, Any]): The underlying model object (Torch/TF/Sklearn compatible).
            input (Union[str, "FeatureSet", "ModelStage"]): The input to this ModelStage. Can be a \
                string (label of FeatureSet or ModelStage) or the FeatureSet/ModelStage instance itself.
            optimizer (Optional[Optimizer]): Optimizer used during training (can be None).
            freeze (bool): Whether this stage is frozen during training.
        """
        src = None
        if isinstance(input, str):
            src = input
        elif isinstance(input, GraphNode):
            src = input.label
        else:
            raise TypeError(f"Unknown input: {input}")
        if label == src:
            raise ValueError("The source cannot have the same label as this ModelStage.")
        
        super().__init__(self, label=label, inputs=src,)
        
        # Set model (cast to BaseModel if explicit subclass not provided)
        self._model : BaseModel = wrap_model(model)
        self._freeze = False        # make stage trainable as default
        
        # Error checking on optimizer (can be None)
        self._optimizer = optimizer
        self._check_valid_optimizer(required=False)
    
    
    # ==========================================
    # ComputationNode Methods
    # ==========================================
    @property
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """
        Returns the input shape of the model after building.

        Returns:
            Optional[Tuple[int, ...]]: Input shape (excluding batch dimension).
        """
        inp_shape = self._model.input_shape
        if inp_shape is None:
            if self.is_built:
                raise RuntimeError(f"Input shape is None after model building.")
            else:
                raise RuntimeError(f"Model must be built before accessing input_shape")
        inp_shape : list = convert_to_format(inp_shape, format=DataFormat.LIST)
        return tuple(inp_shape)
    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """
        Returns the output shape of the model after building.

        Returns:
            Optional[Tuple[int, ...]]: Output shape (excluding batch dimension).
        """
        out_shape = self._model.output_shape
        if out_shape is None:
            if self.is_built:
                raise RuntimeError(f"Output shape is None after model building.")
            else:
                raise RuntimeError(f"Model must be built before accessing output_shape")
        out_shape : list = convert_to_format(out_shape, format=DataFormat.LIST)
        return tuple(out_shape)
    @property
    def max_inputs(self) -> int:
        """
        Maximum number of input connections allowed for this stage.

        Returns:
            int: Always 1 for ModelStage.
        """
        return 1
    @property
    def backend(self) -> Backend:
        """
        Returns the backend associated with the wrapped model.

        Returns:
            Backend: TORCH, TENSORFLOW, SCIKIT, ...
        """
        return self._model.backend
    
    @property
    def is_built(self) -> bool:
        """
        Checks if the model has been built (i.e., instantiated with input/output shape).

        Returns:
            bool: True if built, False otherwise.
        """
        return self._model.is_built
        
    def build(self, input_shape: Optional[Tuple[int]] = None, output_shape: Optional[Tuple[int]] = None):
        """
        Builds the underlying model and optimizer (if present), using specified input/output shapes.

        Args:
            input_shape (Optional[Tuple[int]]): Shape of model input (excluding batch dim).
            output_shape (Optional[Tuple[int]]): Optional shape of model output.
        """
        
        # Build underlying BaseModel if not already built
        if not self._model.is_built:
            self._model.build(input_shape=input_shape, output_shape=output_shape)
        
        # Build optimizer if defined
        if self._optimizer is not None:
            if self.backend == Backend.TORCH:
                self._optimizer.build(parameters=self._model.parameters())
            elif self.backend == Backend.TENSORFLOW:
                self._optimizer.build()
            elif self.backend == Backend.SCIKIT:
                # Scikit-learn optimizers are typically fit internally
                pass
            else:
                raise BackendNotSupportedError(backend=self.backend, message="Unknown backend for optimizer building")
   
    @overload
    def forward(self, batch:Batch, **kwargs) -> BatchOutput: ...
    @overload
    def forward(self, batch:BatchOutput, **kwargs) -> BatchOutput: ...
    @overload
    def forward(self, data:Data, **kwargs) -> Data: ...
    def forward(self, x: Union[Data, Batch, BatchOutput], **kwargs) -> Union[Data, BatchOutput]:
        """
        Performs a forward pass through the model using inputs from Data, Batch, or BatchOutput.

        This method preserves raw tensor outputs to maintain backend autograd support.
        It returns a `BatchOutput` or `Data` object containing model predictions.

        Args:
            x (Union[Data, Batch, BatchOutput]): Input data to the model.

        Returns:
            Union[Data, BatchOutput]: Outputs from the model.
        """
                
        if isinstance(x, Batch):
            all_outputs = {}
            sample_uuids = {}
            for role, samples in x.role_samples.items():
                # Format features for this backend
                features = samples.get_all_features(format=get_data_format_for_backend(self.backend))
                all_outputs[role] = self._model(features, **kwargs)
                sample_uuids[role] = samples.sample_uuids
                
            # In order to preserve auto-grad for pytorch, we cannot modify underlying
            # data format until after loss computation and optimizer stepping
            return BatchOutput(
                features=all_outputs,       # Preserves tensors
                sample_uuids=sample_uuids
            )  
            
        elif isinstance(x, BatchOutput):
            all_outputs = {}
            sample_uuids = {}
            for role in x.available_roles:
                # Format any-format to this backend (obj is unchanged if in correct format)
                features = convert_to_format(x.features[role], format=get_data_format_for_backend(self.backend))
                all_outputs[role] = self._model(features, **kwargs)
                sample_uuids[role] = x.sample_uuids[role]
            
            # In order to preserve auto-grad for pytorch, we cannot modify underlying
            # data format until after loss computation and optimizer stepping
            return BatchOutput(
                features=all_outputs,       # Preserves tensors
                sample_uuids=sample_uuids
            )  
                
        elif isinstance(x, Data):
            x = x.to_backend(target=self.backend)
            self._model(x)
            return Data(self._model(x))
        
        else:
            raise TypeError(f"Input must be of type Data or Batch. Received: {type(x)}")
    
    def infer_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Runs a dummy forward pass to infer the output shape of the model.

        Args:
            input_shape (Tuple[int, ...]): Shape of model input (excluding batch).

        Returns:
            Tuple[int, ...]: Output shape (excluding batch).
        """
        # Make dummy input for ModelStage (add batch_dim of 1)
        X: Data = make_dummy_data(shape=(1, *input_shape))
    
        # Collect model output
        y = self.forward(X)
        
        # Drop batch dimension
        return tuple(int(dim) for dim in y.shape[1:])
   
    
    # ==========================================
    # ModelStages Properties & Dunders
    # ==========================================
    def __repr__(self):
        return self.description_long()
        
    def description_short(self) -> str:
        return (
            f"ModelStage (label='{self.label}', "
            f"model={self._model.__class__.__name__}, "
            f"inputs={self.get_inputs()}, "
            f"optimizer={self._optimizer}, "
            f"backend={self.backend})"
        )
        
    def description_long(self) -> str:
        return (
            f"ModelStage: `{self.label}`\n"
            f" + Model: {self._model.__class__.__name__} ({self.backend})\n"
            f" + Inputs: {self.get_inputs()}\n"
            f" + Optimizer: {self._optimizer}"
        )
      
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
            raise ValueError(f"Freeze must be a boolean, got {type(value)}")
        self._freeze = value
    
    @overload
    def __call__(self, batch:Batch, **kwargs) -> Batch: ...
    @overload
    def __call__(self, batch:BatchOutput, **kwargs) -> Batch: ...
    @overload
    def __call__(self, data:Data, **kwargs) -> Data: ...
    def __call__(self, x: Union[Data, Batch], **kwargs) -> Union[Data, Batch]:
        """
        Shorthand to call `forward()` on input.

        Args:
            x (Union[Data, Batch]): Input to model.

        Returns:
            Union[Data, Batch]: Model output.
        """  
        return self.forward(x=x, **kwargs)
    
    
    # ==========================================
    # Error Checking Methods
    # ==========================================
    def _check_valid_optimizer(self, required:bool=True):
        """
        Verifies that the optimizer is compatible with the model's backend.

        Args:
            required (bool): Whether an optimizer is required. Default is True.

        Raises:
            OptimizerNotSetError: If required and optimizer is None.
            BackendMismatchError: If optimizer and model backends differ.
        """
        
        if self._optimizer is None:
            if required:
                raise OptimizerNotSetError(model_stage_name=self.label)
        
        else:
            if not self._optimizer.backend == self.backend:
                raise BackendMismatchError(
                    expected=self.backend, 
                    received=self._optimizer.backend, 
                    message=f"Optimizer backend does not match model backend: {self._optimizer.backend} != {self.backend}"
                )
    
    def _validate_source(
        self, 
        batch_input:Dict[str,  Batch], 
        losses:List[AppliedLoss] = None
    ):
        """
        Validates that the required sources for input and losses are present in batch_input.

        Args:
            batch_input (Dict[str, Batch]): Mapping of source labels to Batches.
            losses (List[AppliedLoss], optional): Losses to compute.

        Raises:
            ValueError: If any expected input or loss role is missing.
        """
        
        # Check that self.input exists in batch_input
        if not self.input in batch_input.keys():
            raise ValueError(
                f"batch_input missing required key: {self.input}"
            )
            
        # Check that all required loss roles exist in batch_input
        if losses is not None:
            for loss in losses:
                # node: FeatureSet or ModelStage label
                # attribute: "features", "targets", or "output"
                # role: default or a custom string (eg, 'anchor')
                for node, attribute, role in loss.parsed_inputs.values():
                    if not attribute == 'output':
                        # Check that node exists in batch_input
                        if node not in batch_input:
                            raise ValueError(f"batch_input missing required key: {node}")
                        # Check that role exists 
                        if role not in batch_input[node].available_roles:
                            raise ValueError(
                                f"batch_input missing required `{role}` role for key: {node}"
                            )
                    
        
    # ==========================================
    # Train (forward + loss + opt) Methods
    # ==========================================
    def _get_input_batch(self, all_batch_data: Dict[str, Batch]) -> Batch:
        return all_batch_data[self.input]
     
    def _train_step_torch(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> StepResult:
        """
        Runs a training step using PyTorch: forward, loss, backward, optimizer.

        Args:
            batch_input (Dict[str, Batch]): Input batches.
            losses (List[AppliedLoss]): Loss functions to apply.

        Returns:
            StepResult: Losses and model outputs.
        """
        
        # Set optimizer and train mode
        self._model.train()
        self._optimizer.zero_grad()
        
        total_loss = 0  
        loss_results: List[LossResult] = []
        outputs : Dict[str, BatchOutput] = {}
        
        # Forward pass
        outputs : Dict[str, BatchOutput] = {}
        outputs[self.label] = self.forward( self._get_input_batch(batch_input) )

        # Compute losses   
        if losses is not None:
            for loss in losses:
                loss_res = loss.compute(batch_input=batch_input, model_outputs=outputs)
                loss_results.append(loss_res)
                total_loss += loss_res.value
                    
        # Backward + opt step
        total_loss.backward()
        self._optimizer.step()
        
        total_loss = total_loss.item() if hasattr(total_loss, "item") else total_loss
        
        return StepResult(
            total_loss=total_loss,
            total_opt_loss=total_loss,
            total_non_opt_loss = 0.0,
            all_loss_results={self.label: loss_results},
            stage_outputs=outputs,
        )
        
    def _train_step_tensorflow(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> LossResult:
        """
        Runs a training step using Tensorflow: forward, loss, backward, optimizer.

        Args:
            batch_input (Dict[str, Batch]): Input batches.
            losses (List[AppliedLoss]): Loss functions to apply.

        Returns:
            StepResult: Losses and model outputs.
        """
        
        # Zero optimizer gradients
        self._optimizer.zero_grad()
        
        total_loss = 0  
        loss_results: List[LossResult] = []
        outputs : Dict[str, BatchOutput] = {}
        
        # Track gradients over forward passes & loss computation
        with tf.GradientTape() as tape:
            # Forward pass (with training=True)
            outputs : Dict[str, BatchOutput] = {}
            outputs[self.label] = self.forward( self._get_input_batch(batch_input), training=True)

        # Compute losses   
        if losses is not None:
            for loss in losses:
                loss_res = loss.compute(batch_input=batch_input, model_outputs=outputs)
                loss_results.append(loss_res)
                total_loss += loss_res.value
                
        # Backward + opt step
        grads = tape.gradient(total_loss, self._model.trainable_variables)
        self._optimizer.step(grads=grads, variables=self._model.trainable_variables)

        total_loss = float(total_loss)
        
        return StepResult(
            total_loss=total_loss,
            total_opt_loss=total_loss,
            total_non_opt_loss = 0.0,
            all_loss_results={self.label: loss_results},
            stage_outputs=outputs,
        )

    def _train_step_scikit(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> StepResult:
        # TODO
        raise NotImplementedError(f"Training for scikit model not implemented yet.")
    
    def train_step(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> StepResult:
        """
        Performs a training step (forward, loss, backward, optimizer step) for this stage.

        Only callable if this stage has an optimizer and is not frozen. Otherwise, training
        must be delegated to `ModelGraph`.

        Args:
            batch_input (Dict[str, Batch]): Mapping from upstream sources to input batches.
            losses (List[AppliedLoss]): List of loss functions to compute and backpropagate.

        Returns:
            StepResult: Contains loss values and model outputs.

        Raises:
            RuntimeError: If stage is frozen or optimizer is missing.
        """
        
        # If stage is frozen, raise error
        if self.freeze:
            raise RuntimeError(f"`train_step` called with `freeze=True`. Use `eval_step` instead.")
        
        # Ensure batch_input matches expected data in losses
        self._validate_source(batch_input=batch_input, losses=losses)
        
        # Ensure optimizer is set and matches model backend
        self._check_valid_optimizer(required=True)
        
        if self.backend == Backend.TORCH:
            return self._train_step_torch(batch_input=batch_input, losses=losses)
        
        elif self.backend == Backend.TENSORFLOW:
            return self._train_step_tensorflow(batch_input=batch_input, losses=losses)
        
        elif self.backend == Backend.SCIKIT:
            return self._train_step_scikit(batch_input=batch_input, losses=losses)
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
         
    # ==========================================
    # Eval (forward + loss) Methods
    # ========================================== 
    def _eval_step_torch(
        self,
        batch_input: Dict[str, Batch], 
        losses: Optional[List[AppliedLoss]]
    ) -> StepResult:
        """
        Runs an evaluation step using PyTorch: forward + loss (no gradients).

        Args:
            batch_input (Dict[str, Batch]): Input batches.
            losses (List[AppliedLoss]): Losses to apply.

        Returns:
            StepResult: Loss values and outputs.
        """
        
        self._model.eval()
        
        total_loss = 0  
        loss_results: List[LossResult] = []
        outputs : Dict[str, BatchOutput] = {}
        
        with torch.no_grad():
            # Forward pass
            outputs[self.label] = self.forward( self._get_input_batch(batch_input) )

            # Compute losses
            loss_results: List[LossResult] = [] 
            if losses is not None:
                for loss in losses:
                    loss_res = loss.compute(batch_input=batch_input, model_outputs=outputs)
                    loss_results.append(loss_res)
                    total_loss += loss_res.value    
                        
        total_loss = total_loss.item() if hasattr(total_loss, "item") else total_loss
        
        return StepResult(
            total_loss=total_loss,
            total_opt_loss=0.0,
            total_non_opt_loss=total_loss,
            all_loss_results={self.label: loss_results},
            stage_outputs=outputs,
        )
    
    def _eval_step_tensorflow(
        self,
        batch_input: Dict[str, Batch], 
        losses: Optional[List[AppliedLoss]]
    ) -> StepResult:
        """
        Runs an evaluation step using Tensorflow: forward + loss (no gradients).

        Args:
            batch_input (Dict[str, Batch]): Input batches.
            losses (List[AppliedLoss]): Losses to apply.

        Returns:
            StepResult: Loss values and outputs.
        """
        
        self._model.eval()
        
        total_loss = 0  
        loss_results: List[LossResult] = []
        outputs : Dict[str, BatchOutput] = {}
        
        # Forward pass (with training=True)
        outputs[self.label] = self.forward( self._get_input_batch(batch_input), training=True)
        
        # Compute losses
        loss_results: List[LossResult] = [] 
        if losses is not None:
            for loss in losses:
                loss_res = loss.compute(batch_input=batch_input, model_outputs=outputs)
                loss_results.append(loss_res)
                total_loss += loss_res.value    

        total_loss = float(total_loss)
        
        return StepResult(
            total_loss=total_loss,
            total_opt_loss=0.0,
            total_non_opt_loss=total_loss,
            all_loss_results={self.label: loss_results},
            stage_outputs=outputs,
        )
        
    def _eval_step_scikit(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> StepResult:
        # TODO
        raise NotImplementedError(f"Evaluation for scikit model not implemented yet.")
    
    def eval_step(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> StepResult:
        """
        Performs an evaluation step (forward pass and loss computation) for this stage.

        Only callable if this stage is frozen. No gradient tracking is performed.

        Args:
            batch_input (Dict[str, Batch]): Mapping from upstream sources to input batches.
            losses (List[AppliedLoss]): List of loss functions to compute.

        Returns:
            StepResult: Contains loss values and model outputs.

        Raises:
            RuntimeError: If stage is not frozen.
        """
        
        # If stage is not frozen, raise error
        if not self.freeze:
            raise RuntimeError("`eval_step` called with `freeze=False`. Use `train_step` for training.")
        
        self._validate_source(batch_input=batch_input, losses=losses)

        if self.backend == Backend.TORCH:
            return self._eval_step_torch(batch_input=batch_input, losses=losses)
        
        elif self.backend == Backend.TENSORFLOW:
            return self._eval_step_tensorflow(batch_input=batch_input, losses=losses)
        
        elif self.backend == Backend.SCIKIT:
            return self._eval_step_scikit(batch_input=batch_input, losses=losses)
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
 

        
        