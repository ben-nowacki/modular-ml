
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from matplotlib import pyplot as plt
import torch
import tensorflow as tf
import numpy as np

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.sample import Sample, SampleCollection
from modularml.core.loss import AppliedLoss
from modularml.core.optimizer import Optimizer
from modularml.core.samplers.feature_sampler import FeatureSampler
from modularml.models.base import BaseModel
from modularml.utils.backend import Backend
from modularml.utils.data_format import get_format_type_from_backend
from modularml.utils.exceptions import BackendMismatchError, BackendNotSupportedError, LossError, LossNotSetError, ModelStageInputError, OptimizerNotSetError


if TYPE_CHECKING:
    from modularml.core.model_stage import ModelStage
    from modularml.core.data_structures.feature_set import FeatureSet
    
    
    
# @dataclass
# class BatchData:
#     """
#     Structures data within a single training batch.
    
#     Attributes:
#         sample_keys (List[str]): Names of different samples per batch element. \
#             For example, `['anchor', 'positive', 'negative']`
#         features (Dict[str, Any]): Tensors or similar keyed by the sample type. \
#             For example, `{'anchor': tensor, 'positive': tensor, ...}`
#         targets (Dict[str, Any]): Tensors or similar keyed by the sample type. \
#             For example, `{'anchor': tensor, 'positive': tensor, ...}`
#         tags (Dict[str, Any], optional): Any optional set of tags or meta-data \
#             organized by sample type.
#         weights (any, optional): Optional weighted applied to each batch element. \
#             Should have a shape of [batch_size, ] or [batch_size, num_samples].
#     """
#     sample_keys: List[str]
#     features: Dict[str, Any]
#     targets: Dict[str, Any]
#     tags: Optional[Dict[str, Any]] = None
#     weights: Optional[Any] = None



class ModelStage:
    def __init__(
        self,
        model: BaseModel,
        label: str,
        sampler: FeatureSampler,
        optimizer: Optimizer,
        loss: Union[AppliedLoss, List[AppliedLoss]] = None,
    ):
        self.label = label
        self.model = model
        self.sampler = sampler
        
        self.optimizer = optimizer
        self._check_valid_optimizer(required=False)
        
        self.losses = loss if isinstance(loss, list) else [loss, ] if not loss is None else None
        self._check_valid_losses(required=False)
        
        
        self.input_from = [input_from, ] if isinstance(input_from, StageInput) else input_from
        self._resolved_input_names = [s.resolved_label for s in self.input_from]
        self._freeze = False        # make stage trainable as default
        
        # Build model if config dict
        self.model: BaseModel = (
            model if isinstance(model, BaseModel) else BaseModel.from_config(model)
        )
        
        # Build optimizer if config dict
        self.optimizer: Optimizer = (
            optimizer if isinstance(optimizer, Optimizer) else Optimizer.from_config(optimizer)
        )

        # Build losses
        self.losses: List[AppliedLoss] = []
        for l in loss if isinstance(loss, list) else [loss, ] if loss is not None else []:
            l = l if isinstance(l, AppliedLoss) else AppliedLoss.from_config(l)
            self.losses.append(l)
            
    
    @property
    def backend(self) -> Backend:
        return self.model.backend
    
    @property
    def freeze(self) -> bool:
        return self._freeze
    
    @freeze.setter
    def freeze(self, value: bool):
        if not isinstance(value, bool):
            raise ModelStageInputError(f"Freeze must be a boolean, got {type(value)}")
        self._freeze = value
    
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
    
    # TODO: 
    def _check_valid_losses(self, required:bool=False):
        """
        Error checking on `self.losses` attribute. 

        Args:
            required (bool, optional): Whether losses must be non-None. Defaults to False.
        """
        
        if required and self.losses is None:
            raise LossNotSetError(model_stage_name=self.label)
        
        if isinstance(self.losses, AppliedLoss):
            self.losses = [self.losses, ]
        
        for app_loss in self.losses:
            # check backend
            if not app_loss.backend == self.backend:
                raise BackendMismatchError(
                    expected=self.backend, 
                    received=app_loss.backend, 
                    message=f"AppliedLoss backend does not match model backend: {app_loss.backend} != {self.backend}"
                )
        
            # check that loss roles match sampler roles
            set(app_loss.between)
    
    
    
    def __call__(self, batch: Batch) -> Batch:
        return self.forward(batch=batch)
    
    def forward(self, batch: Batch) -> Batch:
        """
        Performs a forward pass of batch data through the model \
        and collects the outputs
        """
        
        all_outputs = {}
        for role, sample_list in batch.formatted_samples.items():
            # Get features in required backend data format
            temp = SampleCollection(samples=sample_list)
            features = temp.get_all_features(format=get_format_type_from_backend(self.backend))
            
            # Forward model pass
            outputs = self.model(features)
            
            # Model outputs get stored into a new Batch where
            # features becomes the model output
            all_outputs[role] = [
                Sample(
                    features={'embedding': outputs[i]},
                    targets=s.targets,
                    tags=s.tags,
                    sample_id=s.sample_id,  # retain original sample UUID
                    index=s.index
                )
                for i,s in enumerate(sample_list)
            ]
        
        return Batch(
            _samples=all_outputs,
            _sample_weights=batch._sample_weights,
            index=batch.index,
            batch_id=batch.batch_id         # retain input batch UUID
        )
    
    
    def train_step(
        self,
        batch:Batch,
    ):
        """
        Performs a single training step (forward + loss + backward/fit) for this stage. 
        Optimizer stepping is skipped if `ModelStage.freeze` is True.
        """
        
        
        
        
        
        # TODO: Check that input (and target) matches ModelStage.input_from definitions
        self._validate_stage_input(stage_input=stage_input)
        if stage_target is not None:
            self._validate_stage_input(stage_input=stage_target)
        
        # TORCH training logic
        if self.backend == Backend.TORCH:
            self._check_valid_optimizer(required=True)
            
            # Set torch model to eval mode if ModelStage frozen
            if self.freeze: 
                self.model.eval()
            # Otherwise set to train mode and zero optimizer
            else:
                self.model.train()
                self.optimizer.zero_grad()
                
            # Forward model pass 
            stage_output = {}       # keyed by batch.sample_keys
            for sample_key in batch.sample_keys:
                args = _build_model_args(stage_input=stage_input, sample_key=sample_key)
                stage_output[sample_key] = self.model(*args)
                
            # Compute losses
            total_loss, loss_mapping = self._compute_losses(
                stage_output=stage_output,
                batch=batch,
            )
            
            # Backpropagation if ModelStage not frozen
            if not self.freeze and total_loss is not None:
                total_loss.backward()
                self.optimizer.step()
                
            # Return losses
            return (
                total_loss.item() if total_loss is not None else None, 
                loss_mapping, 
                stage_output
            )
                  
        # TENSORFLOW training logic
        elif self.backend == Backend.TENSORFLOW:
            import tensorflow as tf
            self._check_valid_optimizer(required=True)
    
            # Forward pass & loss computation
            with tf.GradientTape() as tape:
                stage_output = {}       # keyed by batch.sample_keys
                for sample_key in batch.sample_keys:
                    args = _build_model_args(stage_input=stage_input, sample_key=sample_key)
                    stage_output[sample_key] = self.model(*args, training=True)

                total_loss, loss_mapping = self._compute_losses(
                    stage_output=stage_output,
                    batch=batch,
                )

            # Backpropogration if ModelStage not frozen
            if not self.freeze and total_loss is not None:
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.step(grads=grads, variables=self.model.trainable_variables)

            # Return losses
            return (
                float(total_loss) if total_loss is not None else None, 
                loss_mapping, 
                stage_output
            )
        
        # SCIKIT training logic
        elif self.backend == Backend.SCIKIT:
            import numpy as np
            self._check_valid_optimizer(required=False)
            
            # scikit estimators require fit on full data, cannot backprop step-by-step
            if stage_target is None:
                raise ModelStageInputError(
                    f"`stage_target` must be provided for a backend of {self.backend}"
                )
                
            if not self.freeze:
                
                # TODO: add scikit training (fit) logic
                raise NotImplementedError(
                    f"ModelStage.train_step() not implemented for the {self.backend} backend"
                )
                # # Concatenate on the last axis (num_features)
                # X_concat = np.concatenate(model_input, axis=-1)  # Shape: (batch_size, feature_len, total_features)
                # y_concat = np.concatenate(model_target, axis=-1)
                
                # # Flatten feature_len and total_features into a single axis
                # X_flat = X_concat.reshape(X_concat.shape[0], -1)  # Shape: (batch_size, feature_len * total_features)
                # y_flat = y_concat.reshape(y_concat.shape[0], -1)

                # # Fit to all samples in this batch
                # self.model.fit(X_flat, y_flat)
            
            # Forward model pass 
            stage_output = {}       # keyed by batch.sample_keys
            for sample_key in batch.sample_keys:
                if sample_key not in stage_target:
                    raise ModelStageInputError(f"Missing sample key in `stage_target`: {sample_key}")
    
                # Input stream merging is handled by the BaseModel class
                args = []
                for s in self._resolved_input_names:
                    inp_raw = stage_input[s][sample_key]
                    inp_flat = inp_raw.reshape(inp_raw.shape[0], -1)
                    args.append( inp_flat )
                    
                # Ensure output matches the shape of target
                out_raw = self.model(*args)
                out_flat = out_raw.reshape(stage_target[sample_key].shape)
                stage_output[sample_key] = out_flat
                
            # Compute losses
            total_loss, loss_mapping = self._compute_losses(
                stage_output=stage_output,
                batch=batch,
            )
            
            # Return losses
            return (
                float(total_loss) if total_loss is not None else None, 
                loss_mapping,
                stage_output
            )

        else:
            raise BackendNotSupportedError(backend=self.backend, method="ModelStage.train_step()")

        
        
        
    
    def train_step(
        self, 
        batch:Batch, 
        stage_input:Dict[str, Dict[str, Any]], 
        stage_target:Dict[str, Any] = None
    ) -> Tuple[float, Dict[str, float], Any]:
        """
        Performs a single training step (forward + loss + backward/fit) for this stage. 
        Optimizer stepping is skipped if `ModelStage.freeze` is True.
        
        Args:
            batch (BatchData): The batch data for this pass of the full ModelGraph.
            stage_input (Dict[str, Dict[str, any]]): The input data specifically for this ModelStage. \
                The outer dict keys must match the `input_from.resolved_name` values. The inner dict \
                must have keys matching `batch.sample_keys`.
            stage_target (Dict[str, any]): The expected output data for this ModelStage. \
                The structure should match the ***inner*** dict of `stage_input`. This is only required for \
                scikit-based ModelStages (the model is fit to this data). Otherwise, it is ignored.
        Returns:
            Tuple[float, Dict[str, Any], Dict[str, Any]]: total_loss, loss_mapping, stage_output.
                * total_loss (float): total loss for this training step
                * loss_mapping (Dict[str, Any]): loss values tracked by each AppliedLoss
                * stage_output (Dict[str, Any]): the ModelStage outputs keyed by `batch.sample_keys`
        """
        
        def _build_model_args(stage_input, sample_key:str) -> List[Any]:
            # Input stream merging is handled by the BaseModel class
            # Here we just combine them into a list of arguments
            args = []
            for s in self._resolved_input_names:
                args.append( stage_input[s][sample_key] )
            
            return args
        
        # Check that input (and target) matches ModelStage.input_from definitions
        self._validate_stage_input(stage_input=stage_input)
        if stage_target is not None:
            self._validate_stage_input(stage_input=stage_target)
        
        # TORCH training logic
        if self.backend == Backend.TORCH:
            self._check_valid_optimizer(required=True)
            
            # Set torch model to eval mode if ModelStage frozen
            if self.freeze: 
                self.model.eval()
            # Otherwise set to train mode and zero optimizer
            else:
                self.model.train()
                self.optimizer.zero_grad()
                
            # Forward model pass 
            stage_output = {}       # keyed by batch.sample_keys
            for sample_key in batch.sample_keys:
                args = _build_model_args(stage_input=stage_input, sample_key=sample_key)
                stage_output[sample_key] = self.model(*args)
                
            # Compute losses
            total_loss, loss_mapping = self._compute_losses(
                stage_output=stage_output,
                batch=batch,
            )
            
            # Backpropagation if ModelStage not frozen
            if not self.freeze and total_loss is not None:
                total_loss.backward()
                self.optimizer.step()
                
            # Return losses
            return (
                total_loss.item() if total_loss is not None else None, 
                loss_mapping, 
                stage_output
            )
                  
        # TENSORFLOW training logic
        elif self.backend == Backend.TENSORFLOW:
            import tensorflow as tf
            self._check_valid_optimizer(required=True)
    
            # Forward pass & loss computation
            with tf.GradientTape() as tape:
                stage_output = {}       # keyed by batch.sample_keys
                for sample_key in batch.sample_keys:
                    args = _build_model_args(stage_input=stage_input, sample_key=sample_key)
                    stage_output[sample_key] = self.model(*args, training=True)

                total_loss, loss_mapping = self._compute_losses(
                    stage_output=stage_output,
                    batch=batch,
                )

            # Backpropogration if ModelStage not frozen
            if not self.freeze and total_loss is not None:
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.step(grads=grads, variables=self.model.trainable_variables)

            # Return losses
            return (
                float(total_loss) if total_loss is not None else None, 
                loss_mapping, 
                stage_output
            )
        
        # SCIKIT training logic
        elif self.backend == Backend.SCIKIT:
            import numpy as np
            self._check_valid_optimizer(required=False)
            
            # scikit estimators require fit on full data, cannot backprop step-by-step
            if stage_target is None:
                raise ModelStageInputError(
                    f"`stage_target` must be provided for a backend of {self.backend}"
                )
                
            if not self.freeze:
                
                # TODO: add scikit training (fit) logic
                raise NotImplementedError(
                    f"ModelStage.train_step() not implemented for the {self.backend} backend"
                )
                # # Concatenate on the last axis (num_features)
                # X_concat = np.concatenate(model_input, axis=-1)  # Shape: (batch_size, feature_len, total_features)
                # y_concat = np.concatenate(model_target, axis=-1)
                
                # # Flatten feature_len and total_features into a single axis
                # X_flat = X_concat.reshape(X_concat.shape[0], -1)  # Shape: (batch_size, feature_len * total_features)
                # y_flat = y_concat.reshape(y_concat.shape[0], -1)

                # # Fit to all samples in this batch
                # self.model.fit(X_flat, y_flat)
            
            # Forward model pass 
            stage_output = {}       # keyed by batch.sample_keys
            for sample_key in batch.sample_keys:
                if sample_key not in stage_target:
                    raise ModelStageInputError(f"Missing sample key in `stage_target`: {sample_key}")
    
                # Input stream merging is handled by the BaseModel class
                args = []
                for s in self._resolved_input_names:
                    inp_raw = stage_input[s][sample_key]
                    inp_flat = inp_raw.reshape(inp_raw.shape[0], -1)
                    args.append( inp_flat )
                    
                # Ensure output matches the shape of target
                out_raw = self.model(*args)
                out_flat = out_raw.reshape(stage_target[sample_key].shape)
                stage_output[sample_key] = out_flat
                
            # Compute losses
            total_loss, loss_mapping = self._compute_losses(
                stage_output=stage_output,
                batch=batch,
            )
            
            # Return losses
            return (
                float(total_loss) if total_loss is not None else None, 
                loss_mapping,
                stage_output
            )

        else:
            raise BackendNotSupportedError(backend=self.backend, method="ModelStage.train_step()")




    
    
    
    
    
    def _validate_stage_input(self, stage_input:Dict[str, Dict[str, Any]], batch:BatchData):
        """
        Verifies that `stage_input` contains the keys required by `self.input_from` and `batch`.
        """
        if not isinstance(stage_input, dict):
            raise ModelStageInputError(
                message=f"Expected input to be a dictionary. Recevied: {type(stage_input)}"
            )
            
        missing_inp_keys = []
        for s in self._resolved_input_names:
            if not s in stage_input.keys():
                missing_inp_keys.append(s)
        if missing_inp_keys:
            raise ModelStageInputError(
                f"`stage_input` is missing declared StageInput keys: {missing_inp_keys}"
            )
            
        sample_keys = batch.sample_keys
        for s in self._resolved_input_names:
            if not isinstance(stage_input[s], dict):
                raise ModelStageInputError(
                    f"`stage_input` must be a nest dictionary with the inner dict having key matching "
                    f"those defined by `batch.sample_keys`. "
                )
            for s_key in sample_keys:
                if not s_key in stage_input[s].keys():
                    raise ModelStageInputError(
                        f"`stage_input['{s}']` is missing a required sample key: '{s_key}'."
                    )
        
    
    
    
    
    
    
    
    
    # ==========================================
    # Training/Evaluation Methods
    # ==========================================
    def set_train_mode(self, freeze:bool=False):
        """Set the stages trainability

        Args:
            freeze (bool, optional): Whether to freeze training. Defaults to False.
        """
        self._freeze = freeze

    def _compute_losses(self, stage_output:Dict[str, Any], batch:BatchData) -> tuple[Any, Dict[str, Any]]:
        """Computes the total loss with given model output.

        Args:
            stage_output (Dict[str, Any]): The model output. The keys must match \
                the `batch.sample_keys` values.
            batch (BatchData): The batch data for this pass of the full ModelGraph.

        Returns:
            tuple[Any, Dict[str, Any]]: total_loss:Any, loss_mapping:Dict[str, Any]
        """
        
        total_loss = 0.0
        loss_mapping = {}
        for loss in self.losses:
            loss_res = None
            if loss.between == 'target':
                if loss.sample_key not in stage_output:
                    raise ModelStageInputError(f"Sample key '{loss.sample_key}' not found in stage output.")
                loss_res = loss.compute(
                    outputs=stage_output[loss.sample_key],
                    targets=batch.targets[loss.sample_key],
                )
                
            elif loss.between == 'feature':
                if loss.sample_key not in stage_output:
                    raise ModelStageInputError(f"Sample key '{loss.sample_key}' not found in stage output.")
                loss_res = loss.compute(
                    outputs=stage_output[loss.sample_key],
                    features=batch.features[loss.sample_key],
                )

            elif loss.between == 'sample':
                loss_res = loss.compute(
                    outputs=stage_output
                )
            
            total_loss += loss_res['loss']
            if loss_res['label'] in loss_mapping:
                raise LossError(f"Multiple losses found with the same label: {loss_res['label']}.")
            loss_mapping[loss_res['label']] = loss_res['loss']
                
        return total_loss, loss_mapping


    def eval_step(
        self, 
        batch:BatchData,
        stage_input:Dict[str, Dict[str, Any]], 
    ) -> Tuple[float, Dict[str, float], Any]:
        """
        Performs a single forward pass without loss computation or optimization.
        Intended for evaluation/inference only. 
        
        Args:
            batch (BatchData): The batch data for this pass of the full ModelGraph.
            stage_input (Dict[str, Dict[str, any]]): The input data specifically for this ModelStage. \
                The outer dict keys must match the `input_from.resolved_name` values. The inner dict \
                must have keys matching `batch.sample_keys`.
            
        Returns:
            Dict[str, Any]: the model outputs with the same structure as the inner dict of `stage_input`.
        """
        
        def _build_model_args(stage_input, sample_key:str) -> List[Any]:
            # Input stream merging is handled by the BaseModel class
            # Here we just combine them into a list of arguments
            args = []
            for s in self._resolved_input_names:
                args.append( stage_input[s][sample_key] )
            
            return args
        
        # Check that input (and target) matches ModelStage.input_from definitions
        self._validate_stage_input(stage_input=stage_input)

        # TORCH evaluation logic
        if self.backend == Backend.TORCH:
            # Set torch model to eval mode
            self.model.eval()
            
            # Forward model pass 
            stage_output = {}       # keyed by batch.sample_keys
            with torch.no_grad():
                for sample_key in batch.sample_keys:
                    args = _build_model_args(stage_input=stage_input, sample_key=sample_key)
                    stage_output[sample_key] = self.model(*args)
            
            return stage_output
                
        # TENSORFLOW evaluation logic
        elif self.backend == Backend.TENSORFLOW:
            # Forward model pass 
            stage_output = {}       # keyed by batch.sample_keys
            for sample_key in batch.sample_keys:
                args = _build_model_args(stage_input=stage_input, sample_key=sample_key)
                stage_output[sample_key] = self.model(*args, training=False)

            return stage_output
                  
        # SCIKIT evaluation logic
        elif self.backend == Backend.SCIKIT:
            # Forward model pass 
            stage_output = {}       # keyed by batch.sample_keys
            for sample_key in batch.sample_keys:
                # Input stream merging is handled by the BaseModel class
                args = []
                for s in self._resolved_input_names:
                    inp_raw = stage_input[s][sample_key]
                    inp_flat = inp_raw.reshape(inp_raw.shape[0], -1)
                    args.append( inp_flat )
                    
                stage_output[sample_key] = self.model(*args)

            return stage_output

        else:
            raise BackendNotSupportedError(backend=self.backend, method="ModelStage.train_step()")

    
    # ==========================================
    # State/Config Management Methods
    # ==========================================	
    def get_config(self) -> Dict[str, Any]:
        return {
            "model": self.model.get_config(),
            "name": self.name,
            "input_from": self.input_from,
            "loss": [l.get_config() for l in self.losses],
            "optimizer": self.optimizer.get_config(),
        }
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelStage":
        return cls(
            model=BaseModel.from_config(config['model']),
            name=config["name"],
            input_from=config["input_from"],
            loss=config.get("loss"),
            optimizer=config.get("optimizer"),
        )
        
        
    # ==========================================
    # Descriptors / User-Convience Methods
    # ==========================================
    def __repr__(self):
        return (
            f"ModelStage: `{self.name}`\n"
            f"  Model: {self.model.__class__.__name__} ({self.backend})\n"
            f"  Inputs: {self.input_from}\n"
            f"  Losses: {self.losses}\n"
            f"  Optimizer: {self.optimizer}"
        )
        
    def visualize(
        self, 
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize this ModelStage: inputs, stage node, and any attached LossSpecs.

        Args:
            save_path (str, optional): If provided, saves the figure to this path.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        def get_nodes_by_attr_value(all_nodes:dict, node_attr_key:str, value):
            res = {}
            for n_idx, node_attr in all_nodes.items():
                if node_attr[node_attr_key] == value:
                    res[n_idx] = node_attr
            return res

        graph = nx.DiGraph()
        fig, ax = plt.subplots(figsize=(6,4))

        # Construct nodes
        all_nodes = {}
        node_idx = 0
        xy_ratio = 2		# x is spaced 2 times further apart than y
        for x, (type, items) in enumerate([
            ("Feature", set(self._resolved_input_names)), 
            ("Model", [self.name,]), 
            ("Loss", self.losses if not self.losses is None else [])
        ]):
            for y, item in enumerate(items):
                label = None
                in_edges = []
                if isinstance(item, str):
                    label = item
                    if type == 'Model':
                        in_edges = self._resolved_input_names
                    
                elif isinstance(item, AppliedLoss):
                    label = item.label
                    in_edges.append(self.name)
                    # if item.between in ['feature', 'target']:
                    # 	in_edges.append(item.featureset_name)
                    # 	description = f"Loss `({item.name})` between {item.featureset_name}.{item.between}"
                    # 	
                    # elif item.between == 'sample':
                    # 	description = f"Loss `({item.name})` between samples"

                all_nodes[node_idx] = {
                    "label": label,
                    "x_pos": x * xy_ratio, 						# left aligned
                    "y_pos": y - (len(items) / 2) + 0.5,	 	# center about y=0
                    "type": type,
                    "in_edges": in_edges,
                }
                graph.add_node(node_idx, **all_nodes[node_idx])
                node_idx += 1

        # Construct edges
        for n_idx, node in all_nodes.items():
            for v in node['in_edges']:
                # v is the name of the node to attach
                # Find the node_idx with that value
                v_nodes = list(get_nodes_by_attr_value(all_nodes, node_attr_key='label', value=v).keys()) 
                graph.add_edge(v_nodes[0], n_idx,)

    
        node_plot_params = {
            'Feature':{
                'node_shape': 's',
                'node_color': 'lightgray',
                'edgecolors': 'black', 
                'linewidths':  5*(1/xy_ratio),
                'node_size':   3000*(1/xy_ratio),
            },
            'Model':{
                'node_shape': 'o',
                'node_color': 'skyblue',
                'edgecolors': 'black', 
                'linewidths':  5*(1/xy_ratio),
                'node_size':   5000*(1/xy_ratio),
            },
            'Loss':{
                'node_shape': 'D',
                'node_color': 'salmon',
                'edgecolors': 'black', 
                'linewidths':  5*(1/xy_ratio),
                'node_size':   1000*(1/xy_ratio),
            }
        }
        edge_plot_params = {
            'Feature': {
                'arrows': True,
                'arrowstyle': '-|>',
                'arrowsize': 30*(1/xy_ratio),
                'width': 4*(1/xy_ratio),
                'style': 'solid',
                'node_size': node_plot_params['Model']['node_size']*1.1
            },
            'Model': {
                'arrows': True,
                'arrowstyle': '-|>',
                'arrowsize': 30*(1/xy_ratio),
                'width': 4*(1/xy_ratio),
                'style': 'solid',
                'node_size': node_plot_params['Model']['node_size']*1.15
            },
            'Loss': {
                'arrows': True,
                'arrowstyle': '-|>',
                'arrowsize': 30*(1/xy_ratio),
                'width': 4*(1/xy_ratio),
                'style': 'solid',
                'node_size': node_plot_params['Loss']['node_size']*2
            },
        }
        legend_elements = []
        for t in ['Feature', 'Model', 'Loss']:
            node_subset = get_nodes_by_attr_value(
                all_nodes = all_nodes,
                node_attr_key = 'type',
                value = t
            )
            nx.draw_networkx_nodes(
                G = graph,
                pos = {
                    n_idx: (graph.nodes[n_idx]['x_pos'], graph.nodes[n_idx]['y_pos'])
                    for n_idx in graph.nodes
                },
                ax = ax,
                nodelist = list(node_subset.keys()),
                **node_plot_params[t]
            )
            for n_idx, node in node_subset.items():
                legend_elements.append(
                    Line2D(
                        [0], [0], 
                        marker=node_plot_params[t]['node_shape'],
                        color='white',
                        label=f"{n_idx}: <{t}> `{node['label']}`",
                        markerfacecolor=node_plot_params[t]['node_color'], 
                        markersize=10, 
                        markeredgecolor='black')
                )

            edgelist = [
                edge
                for n in list(node_subset.keys())
                for edge in graph.in_edges(n)
            ]
            if edgelist:
                nx.draw_networkx_edges(
                    G = graph,
                    pos = {
                        n_idx: (graph.nodes[n_idx]['x_pos'], graph.nodes[n_idx]['y_pos'])
                        for n_idx in graph.nodes
                    },
                    ax = ax,
                    edgelist = edgelist,
                    **edge_plot_params[t]
                )

        
        ys = [n['y_pos'] for n in all_nodes.values()]
        ax.legend(handles=legend_elements, title="ModelStage Legend", bbox_to_anchor=(1, 0.5), loc='center left')
        ax.axis('off')
        ax.set_xlim(-xy_ratio/2, xy_ratio*2+xy_ratio/2)
        ax.set_ylim(np.min(ys)-0.5, np.max(ys)+0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.margins(0.25)
        if save_path:
            fig.savefig(save_path)
        return fig, ax
    
      
      