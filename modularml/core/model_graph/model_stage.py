
from typing import Any, Dict, List, Optional, Tuple, Union, overload
import torch
import tensorflow as tf

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.core.data_structures.data import Data
from modularml.core.model_graph.loss import AppliedLoss, LossResult
from modularml.core.model_graph.optimizer import Optimizer
from modularml.models.base import BaseModel
from modularml.utils.backend import Backend
from modularml.utils.data_format import DataFormat, convert_to_format, get_data_format_for_backend
from modularml.utils.exceptions import BackendMismatchError, BackendNotSupportedError, OptimizerNotSetError


class ModelStage:
    def __init__(
        self,
        model: BaseModel,
        label: str,
        inputs: Union[str, List[str]],
        optimizer: Optimizer,
    ):
        self.label = label
        self.model = model
        self._freeze = False        # make stage trainable as default
        
        self.inputs: List[str] = [inputs, ] if isinstance(inputs, str) else inputs
        
        # Error checking on optimzier
        self.optimizer = optimizer
        self._check_valid_optimizer(required=False)
            
    def __repr__(self):
        return self.description_long()
        
    def description_short(self) -> str:
        return (
            f"ModelStage (label='{self.label}', "
            f"model={self.model.__class__.__name__}, "
            f"inputs={self.input_shape}, "
            f"optimizer={self.optimizer}, "
            f"backend={self.backend}, "
        )
        
    def description_long(self) -> str:
        return (
            f"ModelStage: `{self.label}`\n"
            f" + Model: {self.model.__class__.__name__} ({self.backend})\n"
            f" + Inputs: {self.inputs}\n"
            f" + Optimizer: {self.optimizer}"
        )
        
    
    
    @property
    def backend(self) -> Backend:
        return self.model.backend
    
    @property
    def freeze(self) -> bool:
        return self._freeze
    
    @freeze.setter
    def freeze(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError(f"Freeze must be a boolean, got {type(value)}")
        self._freeze = value
    
    @property
    def is_built(self) -> bool:
        return self.model.is_built()
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        inp_shape = convert_to_format(self.model.input_shape, format=DataFormat.LIST)
        if inp_shape is None:
            if self.is_built:
                raise RuntimeError(f"Input shape is None after model building.")
            else:
                raise RuntimeError(f"Model must be built before accessing input_shape")
        return inp_shape  
    
    @property
    def output_shape(self) -> Tuple[int, ...]:
        out_shape = convert_to_format(self.model.output_shape, format=DataFormat.LIST)
        if out_shape is None:
            if self.is_built:
                raise RuntimeError(f"Output shape is None after model building.")
            else:
                raise RuntimeError(f"Model must be built before accessing output_shape")
        return out_shape    
    
    def build(self, input_shape: Optional[Tuple[int]] = None, output_shape: Optional[Tuple[int]] = None):
        # Build model
        self.model.build(input_shape=input_shape, output_shape=output_shape)
        
        # Build optimizer if defined
        if self.optimizer is not None:
            if self.backend == Backend.TORCH:
                self.optimizer.build(parameters=self.model.parameters())
            elif self.backend == Backend.TENSORFLOW:
                self.optimizer.build()
            elif self.backend == Backend.SCIKIT:
                # Scikit-learn optimizers are typically fit internally
                pass
            else:
                raise BackendNotSupportedError(backend=self.backend, message="Unknown backend for optimizer building")
    
    
    @overload
    def __call__(self, batch:Batch, **kwargs) -> Batch: ...
    @overload
    def __call__(self, batch:BatchOutput, **kwargs) -> Batch: ...
    @overload
    def __call__(self, data:Data, **kwargs) -> Data: ...
    def __call__(self, x: Union[Data, Batch], **kwargs) -> Union[Data, Batch]:
        """Performs a forward pass the model and collects the outputs"""    
        return self.forward(x=x, **kwargs)
    
    
    @overload
    def forward(self, batch:Batch, **kwargs) -> BatchOutput: ...
    @overload
    def forward(self, batch:BatchOutput, **kwargs) -> BatchOutput: ...
    @overload
    def forward(self, data:Data, **kwargs) -> Data: ...
    def forward(self, x: Union[Data, Batch, BatchOutput], **kwargs) -> Union[Data, BatchOutput]:
        """Performs a forward pass the model and collects the outputs"""    
                
        if isinstance(x, Batch):
            all_outputs = {}
            sample_uuids = {}
            for role, samples in x.role_samples.items():
                # Format features for this backend
                features = samples.get_all_features(format=get_data_format_for_backend(self.backend))
                all_outputs[role] = self.model(features, **kwargs)
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
                # Format any-format to this backend
                features = convert_to_format(x.features[role], format=get_data_format_for_backend(self.backend))
                all_outputs[role] = self.model(features, **kwargs)
                sample_uuids[role] = x.sample_uuids[role]
            
            # In order to preserve auto-grad for pytorch, we cannot modify underlying
            # data format until after loss computation and optimizer stepping
            return BatchOutput(
                features=all_outputs,       # Preserves tensors
                sample_uuids=sample_uuids
            )  
                
        elif isinstance(x, Data):
            x = x.to_backend(target=self.backend)
            self.model(x)
            return Data(self.model(x))
        else:
            raise TypeError(f"Input must be of type Data or Batch. Received: {type(x)}")
    
    
    # ==========================================
    # Error Checking Methods
    # ==========================================
    def _check_valid_optimizer(self, required:bool=True):
        """
        Error checking on `self.optimizer` attribute. 

        Args:
            required (bool, optional): Whether Optimizer must be non-None. Defaults to True.
        """
        if required and self.optimizer is None:
            raise OptimizerNotSetError(model_stage_name=self.label)
        
        if not self.optimizer.backend == self.backend:
            raise BackendMismatchError(
                expected=self.backend, 
                received=self.optimizer.backend, 
                message=f"Optimizer backend does not match model backend: {self.optimizer.backend} != {self.backend}"
            )
    
    def _validate_inputs(
        self, 
        batch_input:Dict[str,  Batch], 
        losses:List[AppliedLoss] = None
    ):
        """
        batch_input: contains Batch data (role: SampleCollection) keyed by 
            FeatureSet.label or ModelStage.label 
        """
        
        # Check that all required input sources exist in batch_input
        missing = set([
            source for source in self.inputs
            if source not in batch_input.keys()
        ])
        if missing:
            raise ValueError(
                f"batch_input missing required key: {missing}"
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
    def _train_step_torch(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> Dict[str, Any]:
        
        # Set optimizer and train mode
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0  
        loss_results: List[LossResult] = []
        outputs : Dict[str, BatchOutput] = {}
        
        # Forward pass
        outputs : Dict[str, BatchOutput] = {}
        if len(self.inputs) > 1:
            raise NotImplementedError(
                f"Multiple inputs for a base ModelStage is not supported. "
                f"Use a subclass of MergeStage to join multiple input streams."
            )    
        outputs[self.label] = self.forward(batch_input[self.inputs[0]])

        # Compute losses   
        if losses is not None:
            for loss in losses:
                loss_res = loss.compute(batch_input=batch_input, model_outputs=outputs)
                loss_results.append(loss_res)
                total_loss += loss_res.value
                    
        # Backward + opt step
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item() if hasattr(total_loss, "item") else total_loss,
            "loss_results": loss_results,
            "stage_outputs": outputs
        }
        
    def _train_step_tensorflow(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> Dict[str, Any]:
        
        # Zero optimizer gradients
        self.optimizer.zero_grad()
        
        total_loss = 0  
        loss_results: List[LossResult] = []
        outputs : Dict[str, BatchOutput] = {}
        
        # Track gradients over forward passes & loss computation
        with tf.GradientTape() as tape:
            # Forward pass (with training=True)
            outputs : Dict[str, BatchOutput] = {}
            if len(self.inputs) > 1:
                raise NotImplementedError(
                    f"Multiple inputs for a base ModelStage is not supported. "
                    f"Use a subclass of MergeStage to join multiple input streams."
                )    
            outputs[self.label] = self.forward(batch_input[self.inputs[0]], training=True)

            # Compute losses   
        if losses is not None:
            for loss in losses:
                loss_res = loss.compute(batch_input=batch_input, model_outputs=outputs)
                loss_results.append(loss_res)
                total_loss += loss_res.value
                
        # Backward + opt step
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.step(grads=grads, variables=self.model.trainable_variables)

        return {
            "total_loss": float(total_loss),
            "loss_results": loss_results,
            "stage_outputs": outputs
        }
        
    def _train_step_scikit(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"Training for scikit model not implemented yet.")
    
    def train_step(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> Dict[str, Any]:
        """
        Performs forward, loss computation, and optimizer step for this stage.

        Args:
            batch_input (Dict[str, Batch]): Combined outputs and input batches for loss inputs.
            losses (List[AppliedLoss]): Loss functions to optimize for this stage.
            
        Returns:
            Dict[str, Any] including:
            - "total_loss": float
            - "loss_results": List[LossResult]
            - "stage_outputs": Dict[str, BatchOutput]
        """
        
        # If stage is frozen, raise error
        if self.freeze:
            raise RuntimeError(f"`train_step` called with `freeze=True`. Use `eval_step` instead.")
        
        # Ensure batch_input matches expected data in losses
        self._validate_inputs(batch_input=batch_input, losses=losses)
        
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
    ) -> Dict[str, Any]:
        
        self.model.eval()
        
        total_loss = 0  
        loss_results: List[LossResult] = []
        outputs : Dict[str, BatchOutput] = {}
        
        with torch.no_grad():
            # Forward pass
            if len(self.inputs) > 1:
                raise NotImplementedError(
                    f"Multiple inputs for a base ModelStage is not supported. "
                    f"Use a subclass of MergeStage to join multiple input streams."
                )    
            outputs[self.label] = self.forward(batch_input[self.inputs[0]])

            # Compute losses
            loss_results: List[LossResult] = [] 
            if losses is not None:
                for loss in losses:
                    loss_res = loss.compute(batch_input=batch_input, model_outputs=outputs)
                    loss_results.append(loss_res)
                    total_loss += loss_res.value    
                        
        return {
            "total_loss": total_loss.item() if hasattr(total_loss, "item") else total_loss,
            "loss_results": loss_results,
            "stage_outputs": outputs
        }
    
    def _eval_step_tensorflow(
        self,
        batch_input: Dict[str, Batch], 
        losses: Optional[List[AppliedLoss]]
    ) -> Dict[str, Any]:
        self.model.eval()
        
        total_loss = 0  
        loss_results: List[LossResult] = []
        outputs : Dict[str, BatchOutput] = {}
        
        # Forward pass (with training=True)
        if len(self.inputs) > 1:
            raise NotImplementedError(
                f"Multiple inputs for a base ModelStage is not supported. "
                f"Use a subclass of MergeStage to join multiple input streams."
            )    
        outputs[self.label] = self.forward(batch_input[self.inputs[0]], training=True)
        
        
        # Compute losses
        loss_results: List[LossResult] = [] 
        if losses is not None:
            for loss in losses:
                loss_res = loss.compute(batch_input=batch_input, model_outputs=outputs)
                loss_results.append(loss_res)
                total_loss += loss_res.value    
                        
        return {
            "total_loss": total_loss.item() if hasattr(total_loss, "item") else total_loss,
            "loss_results": loss_results,
            "stage_outputs": outputs
        }
        
    def _eval_step_scikit(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"Evaluation for scikit model not implemented yet.")
    
    def eval_step(
        self,
        batch_input: Dict[str, Batch], 
        losses: List[AppliedLoss]
    ) -> Dict[str, Any]:
        """
        Performs forward pass and computes loss (no gradient updates).

        Args:
            batch_input (Dict[str, Batch]): Inputs to this stage and any loss dependencies.
            losses (List[AppliedLoss]): Loss functions to evaluate.
            
        Returns:
            Dict[str, Any] including:
            - "total_loss": float
            - "loss_results": List[LossResult]
            - "stage_outputs": Dict[str, BatchOutput]
        """
        
        # If stage is not frozen, raise error
        if not self.freeze:
            raise RuntimeError("`eval_step` called with `freeze=False`. Use `train_step` for training.")
        
        self._validate_inputs(batch_input=batch_input, losses=losses)

        if self.backend == Backend.TORCH:
            return self._eval_step_torch(batch_input=batch_input, losses=losses)
        
        elif self.backend == Backend.TENSORFLOW:
            return self._eval_step_tensorflow(batch_input=batch_input, losses=losses)
        
        elif self.backend == Backend.SCIKIT:
            return self._eval_step_scikit(batch_input=batch_input, losses=losses)
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
 
        
        