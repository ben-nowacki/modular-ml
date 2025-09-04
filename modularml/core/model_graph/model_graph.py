
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque
import warnings
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.core.data_structures.data import Data
from modularml.core.data_structures.feature_set import FeatureSet
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.data_structures.step_result import StepResult
from modularml.core.model_graph.loss import AppliedLoss, LossResult
from modularml.core.model_graph.model_stage import ModelStage
from modularml.core.model_graph.optimizer import Optimizer
from modularml.utils.backend import Backend, backend_requires_optimizer
from modularml.utils.data_format import DataFormat, convert_to_format, to_python
from modularml.utils.exceptions import BackendNotSupportedError



def make_dummy_data(shape: Tuple[int, ...]) -> Data:
    """
    Creates a dummy Data object
    """
    # Create dummy data
    d = Data(np.ones(shape=shape))
    
    return d

def make_dummy_batch(feature_shape: Tuple[int, ...], target_shape: Tuple[int, ...] = (1,1), batch_size:int=8) -> Batch:
    sample_coll = SampleCollection([
        Sample(
            features={
                f'features_{x}': make_dummy_data(shape=feature_shape[1:])
                for x in range(feature_shape[0])
            },
            targets={
                f'targets_{x}': make_dummy_data(shape=target_shape[1:])
                for x in range(target_shape[0])
            },
            tags={'tags_1': make_dummy_data(shape=(1,)), 'tags_2': make_dummy_data(shape=(1,))},
        )
        for i in range(batch_size)
    ])
    return Batch(
        role_samples = {'default': sample_coll}, 
        label='dummy', 
    )    
    


class ModelGraph:
    def __init__(
        self,
        nodes: List[Union[FeatureSet, ModelStage]],
        optimizer: Optional[Optimizer] = None
    ):
        """_summary_

        Args:
            nodes (List[Union[FeatureSet, ModelStage]]): A list of FeatureSets and \
                ModelStages comprising this ModelGraph.
            optimizer (Optional[Optimizer], optional): If provided, this single optimizer \
                will be used across all child ModelStages. This requires that all ModelStages \
                have the same backend. Defaults to None.
        """
        self.all_nodes : Dict[str, Union[FeatureSet, ModelStage]] = {
            node.label: node 
            for node in nodes
        }
        
        # Separate nodes into ModelStages and FeatureSets (set in _validate_graph)
        self._model_stages : Dict[str, ModelStage] = {}
        self._feature_sets : Dict[str, FeatureSet] = {}
        
        # Validate graph connections of provided nodes & sort
        self._validate_graph_connections()
        self._sorted_stage_labels = self._topological_sort()
        
        # If an optimizer is provided, check that:
        # 1. all optimizer-requiring stages have same backend
        # 2. warn if stages have their own optimizer (will be overwritten)
        self._optimizer = optimizer
        self._stages_req_opt : Optional[Dict[str, ModelStage]]  = None
        self._validate_optimizer()
        
        self._built = False
               
    @property
    def feature_set_labels(self) -> List[str]:
        return list(self._feature_sets.keys())
    
    @property
    def model_stage_labels(self) -> List[str]:
        return list(self._model_stages.keys())
    
    @property
    def is_built(self) -> bool:
        return self._built
    
    def __repr__(self):
        return self.summary()
    
    def summary(self) -> str:
        """Returns a compact summary of all stages in the graph."""
        
        msg = f"ModelGraph(n_nodes={len(self.all_nodes)})"
        for label, fs in self._feature_sets.items():
            msg += f"\n  + {repr(fs)}"
        for label, stage in self._model_stages.items():
            msg += f"\n  + {stage.description_short()}"
        
        return msg
    
 
    def _validate_graph_connections(self):
        """Check for backend consistency and graph connectivity."""
        for label, node in self.all_nodes.items():
            if isinstance(node, FeatureSet): 
                self._feature_sets[label] = node
            elif isinstance(node, ModelStage): 
                self._model_stages[label] = node
            else:
                raise TypeError(
                    f"Unknown node type. Must be of type ModelStage or FeatureSet. "
                    f"Received: {type(node)}"
                )
                
        # Warn if using mixed backend: not thoroughly tested
        used_backends = set(stage.backend for stage in self._model_stages.values())
        if len(used_backends) > 1:
            warnings.warn(
                "Mixed backends detected in ModelGraph. Though allowed, minimal testing has been "
                "conducted. Gradient flow may break during training.",
                category=UserWarning,
                stacklevel=2
            )
        
        # Check for unreachable stages
        reachable = set()
        frontier = [
            inp
            for label, stage in self._model_stages.items()
            for inp in stage.inputs
            if inp not in self._model_stages
        ]
        # BFS traversal
        while frontier:
            current = frontier.pop()			# Current stage in search
            if current in reachable: continue	# Ignore if already seen
            reachable.add(current)	
            for label, stage in self._model_stages.items():
                inputs: List[str] = stage.inputs if isinstance(stage.inputs, list) else [stage.inputs, ]
                if current in inputs:
                    frontier.append(label)		# Add next connected stage

        unreachable = set(self._model_stages) - reachable
        if unreachable:
            warnings.warn(
                f"Unreachable ModelStages detected in ModelGraph: {sorted(unreachable)}. ",
                category=UserWarning,
                stacklevel=2
            )
    
    def _topological_sort(self) -> List[str]:
        """Topological sort of model graph using Kahn's algorithm."""
        in_degree = defaultdict(int)        # Number of dependencies (value) for each stage (key)
        children = defaultdict(list)        # List of child stage names (value) for each stage (key)
        all_stage_names = set(self._model_stages.keys())

        # Initialize in-degree of all nodes to 0
        for stage_name in all_stage_names:
            in_degree[stage_name] = 0

        for stage_name, stage in self._model_stages.items():
            # Get all parent stages for this current stage
            parents = stage.inputs if isinstance(stage.inputs, list) else [stage.inputs]
            for p in parents:
                # If p is a base node (ie, it's name is FeatureSet.label), continue
                if p in self._feature_sets:
                    continue
                # Otherwise, increment the in_degree (how many parents this stage has)
                if p not in self.all_nodes:
                    raise ValueError(
                        f"Invalid input source '{p}' for stage '{stage_name}'.")
                in_degree[stage_name] += 1
                children[p].append(stage_name)

        sorted_stage_names = []
        queue = deque([k for k in all_stage_names if in_degree[k] == 0])

        while queue:
            stage_name = queue.popleft()
            sorted_stage_names.append(stage_name)
            for child in children[stage_name]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_stage_names) != len(all_stage_names):
            unresolved = all_stage_names - set(sorted_stage_names)
            raise ValueError(f"Cyclic dependency detected in ModelGraph: {unresolved}")

        return sorted_stage_names
    
    def _validate_optimizer(self):
        # Get stages that require optimizer
        self._stages_req_opt = {
            label: stage 
            for label, stage in self._model_stages.items()
            if backend_requires_optimizer(stage.backend)
        }
        
        # Ensure all stages have their own optimizer if global one not provided
        if self._optimizer is None: 
            for label, stage in self._stages_req_opt.items():
                if stage._optimizer is None:
                    raise RuntimeError(
                        f"ModelStage (`{label}`) is missing an optimizer. "
                        f"Provide one at the stage level or to ModelGraph."
                    )
        
        # Ensure all stages have the same backend
        else:
            for label, stage in self._stages_req_opt.items():
                # Overwrite existing optimizers at stage-level (and warn)
                if stage._optimizer is not None:
                    warnings.warn(
                        (
                            f"Optimizer were provided to ModelGraph and an underlying ModelStage (`{label}`). "
                            f"The stage-level optimizer will be overwritten."
                        ),
                        category=UserWarning,
                        stacklevel=2,
                    )
                    stage._optimizer = None
                
                # Check for matching backend
                if not stage.backend == self._optimizer.backend:
                    raise RuntimeError(
                        f"Optimizer backend (`{self._optimizer.backend.value}`) doesn't match the backend for `{label}`. "
                        f"A global optimizer can only be provided to ModelGraph when "
                        f"all underlying ModelStages have a matching backend. "
                    )

    def build_all(self, reset:bool = False):
        """Build all ModelStages contain in this ModelGraph"""
        
        if reset:
            # Separate nodes into ModelStages and FeatureSets (set in _validate_graph)
            self._model_stages : Dict[str, ModelStage] = {}
            self._feature_sets : Dict[str, FeatureSet] = {}
            
            # Validate graph connections of provided nodes & sort
            self._validate_graph_connections()
            self._sorted_stage_labels = self._topological_sort()
            
            self._stages_req_opt : Optional[Dict[str, ModelStage]]  = None
            self._validate_optimizer()
            
            self._built = False
        
        
        # Build ModelStages, infer shapes if needed
        for stage_label in self._sorted_stage_labels:
            node : ModelStage = self._model_stages[stage_label]
        
            if not node.is_built:
                # Try building without input shapes
                try: node.build()
                except: pass
                
                # Try inferring input and output shapes
                try:
                    input_shape = self._infer_input_shape(node.inputs)
                    output_shape = self._infer_output_shape(node, input_shape)

                    node.build(
                        input_shape=input_shape,
                        output_shape=output_shape
                    )
                    
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to build node: {node.label}. {e}"
                    )
            
            print(f"Inferred shapes for `{node.label}`: ", node.input_shape, "->", node.output_shape)
    
        # Build global optimizer if defined
        if self._optimizer is not None:
            
            if self._optimizer.backend == Backend.TORCH:
                # Collect model parameters from all stages
                all_model_params = []
                for stage in self._stages_req_opt.values():
                    all_model_params.extend( list(stage.model.parameters()) )
                    
                # Build optimizer will all parameters
                self._optimizer.build(
                    force_rebuild=True,
                    parameters=all_model_params
                )
                
            elif self._optimizer.backend == Backend.TENSORFLOW:
                self._optimizer.build(force_rebuild=False)
                
            elif self._optimizer.backend == Backend.SCIKIT:
                # Scikit-learn optimizers are typically fit internally
                pass
            
            else:
                raise BackendNotSupportedError(backend=self._optimizer.backend, message="Unknown backend for optimizer building")
   
        self._built = True
            
    def _infer_input_shape(self, inputs: List[str]) -> Tuple[int, ...]:
        """Attempts to infer the input shape given the input specs"""
        
        input_shapes = []
        for inp in inputs:
            # Get previous node
            prev_node = self.all_nodes[inp]
            
            if isinstance(prev_node, FeatureSet):
                input_shapes.append(tuple(int(d) for d in prev_node.feature_shape))
                continue
            
            elif isinstance(prev_node, ModelStage):
                if prev_node.output_shape is None:
                    raise ValueError(
                        f"Previous ModelStage has no output shape. "
                        f"Run .build() to perform model input/output shape inference."
                    )
                input_shapes.append(tuple(int(d) for d in prev_node.output_shape))
                continue
            
            else:
                raise TypeError(f"Unknown node type: {prev_node}")
            
        if len(input_shapes) == 1:
            return input_shapes[0]
        else:
            # TODO: how to concantenate
            raise NotImplementedError(
                f"Output shape determination for multiple input sources is not "
                f"implemented yet. Received shapes: {input_shapes}"
            )
              
    def _infer_output_shape(self, node: ModelStage, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Attempts to infer the output shape of a ModelStage
        Runs a Dummy input with `input_shape` and check outputs size.
        """
        # Make dummy input for ModelStage (add batch_dim of 1)
        X: Data = make_dummy_data(shape=(1, *input_shape))
    
        # Collect model output
        y = node.forward(X)
        
        # Drop batch dimension
        return tuple(int(dim) for dim in y.shape[1:])


    
    def dummy_foward(self, batch_size:int = 8) -> Batch:
        """
        A foward pass through the entire ModelGraph with a dummy input to test connections.
        """
        if len(self._feature_sets.keys()) > 1:
            raise NotImplementedError(
                f"`dummy_forward` doesn't currently support ModelGraphs with multiple FeatureSets."
            )
        fs = self._feature_sets[self.feature_set_labels[0]]
        batch = make_dummy_batch(feature_shape=fs.feature_shape, batch_size=batch_size)
        multi_batch = {self.feature_set_labels[0]: batch}
        
        res = self.forward(multi_batch)
        output : BatchOutput = res[self._sorted_stage_labels[-1]]
        return convert_to_format(output.features, format=DataFormat.NUMPY).tolist()
        
    def forward(self, batches: Dict[str, Batch]) -> Dict[str, Batch]:
        
        missing_featuresets = []
        for fs in self._feature_sets.keys():
            if fs not in batches.keys(): missing_featuresets.append(fs)
        if missing_featuresets:
            raise ValueError(
                f"The batches provided to ModelGraph is missing data from required "
                f"FeatureSets. Missing: {missing_featuresets}"
            )
        
        # Stores output from each node in ModelGraph
        cache : Dict[str, Batch] = {}
        
        # Add FeatureSet data
        for fs_label in self.feature_set_labels:
            cache[fs_label] = batches[fs_label]
            
        # Topological forward pass through ModelStages
        for label in self._sorted_stage_labels:
            stage = self._model_stages[label]
            inputs: List[Batch] = []
            for inp in stage.inputs:
                if inp not in cache:
                    raise ValueError(f"Missing input `{inp}` for stage `{label}`")
                inputs.append(cache[inp])
            
            # TODO: Combine multiple inputs into one input (e.g., tuple, dict, or concat)
            stage_input = inputs[0] if len(inputs) == 1 else tuple(inputs)
            
            # Model forward pass
            output: BatchOutput = stage.forward(stage_input)
            cache[label] = output
            
        return cache
   
   
    
    def _stagewise_train_step(
        self,
        batch_input: Dict[str, Batch],
        losses: Dict[str, List[AppliedLoss]],
    ) -> StepResult:
        
        # Cache all stage outputs
        cache: Dict[str, Union[Batch, BatchOutput]] = dict(batch_input)
        loss_cache: Dict[str, List[LossResult]] = defaultdict(list)
        total_loss = 0.0
        total_opt_loss = 0.0
        total_non_opt_loss = 0.0
        
        for stage_label in self._sorted_stage_labels:
            stage = self._model_stages[stage_label]
            
            # Get losses for this stage
            stage_losses = losses.get(stage_label, None)
            
            # Forward pass + loss (+ opt if not frozen)
            res : StepResult = None
            if stage.freeze:
                if stage_losses:
                    raise ValueError(
                        f"`{stage_label}` ModelStage has losses applied but is frozen."
                    )
                else:
                    res = stage.eval_step(
                        batch_input=cache,
                        losses=stage_losses
                    )
            else:
                if stage_losses is None:
                    raise ValueError(
                        f"`{stage_label}` ModelStage is set to trainable but has no applied losses."
                    )
                else:
                    res = stage.train_step(
                        batch_input=cache,
                        losses=stage_losses
                    )
            
            # Cache stage outputs
            cache[stage_label] = res.stage_outputs[stage_label]

            # Record losses
            loss_cache[stage_label] = res.all_loss_results[stage_label]
            total_loss += res.total_loss
            total_opt_loss += res.total_opt_loss
            total_non_opt_loss += res.total_non_opt_loss
            
        return StepResult(
            total_loss=total_loss,            
            total_non_opt_loss=total_non_opt_loss,
            total_opt_loss=total_opt_loss,
            
            all_loss_results=loss_cache,
            stage_outputs={k: v for k, v in cache.items() if k in self._model_stages}
        )
   
    def _train_step_torch(
        self,
        batch_input: Dict[str, Batch], 
        losses: Dict[str, List[AppliedLoss]],
    ) -> StepResult:
        
        import torch
        
        # Cache all stage outputs
        cache: Dict[str, Union[Batch, BatchOutput]] = dict(batch_input)
        # Cache all loss results (keyed by ModelStage.label)
        loss_cache: Dict[str, List[LossResult]] = defaultdict(list)
            
        # Set optimizer
        self._optimizer.zero_grad()
        
        # There may be cases were the ModelGraph consits of PyTorch & non-optimizable stages (eg, scikit)
        # To optimizate the PyTorch stages, we need to separate out losses for optimization and not
        opt_losses : List[LossResult] = []
        non_opt_losses : List[LossResult] = []
        
        # Go through all stages in sorted order and collect outputs
        for stage_label in self._sorted_stage_labels:
            
            # Get stage and attached losses
            stage = self._model_stages[stage_label]
            stage_losses = losses.get(stage_label, None)
            
            # If stage doesn't need optimzer, delegate training
            if not backend_requires_optimizer(stage.backend):
                step_res = stage.train_step(batch_input=batch_input, losses=stage_losses)
                cache[stage_label] = step_res.stage_outputs[stage_label]
                loss_cache[stage_label].extend(step_res.all_loss_results[stage_label])
                non_opt_losses.extend(step_res.all_loss_results[stage_label])
                continue
            
            # Perform train or eval depending on freeze state
            if stage.freeze: 
                stage.model.eval()
                if stage_losses is not None:
                    raise ValueError(
                        f"`{stage_label}` ModelStage has losses applied but is frozen."
                    )
                step_res = stage.eval_step(batch_input=batch_input, losses=stage_losses)
                cache[stage_label] = step_res.stage_outputs[stage_label]
                loss_cache[stage_label].extend(step_res.all_loss_results[stage_label])
                non_opt_losses.extend(step_res.all_loss_results[stage_label])
                continue
                
            stage.model.train()
            model_output : BatchOutput = stage.forward(stage.get_input_batch(cache))
            cache[stage_label] = model_output
            
            # Compute stage loss (store entire LossResult for later computation)
            if stage_losses is not None:
                for loss in stage_losses:
                    loss_res = loss.compute(batch_input=batch_input, model_outputs={stage_label: model_output})
                    loss_cache[stage_label].append(loss_res)
                    opt_losses.append(loss_res)
            
        # Perform optimization stepping
        total_opt_loss = torch.stack([lr.value for lr in opt_losses]).sum()
        total_opt_loss.backward()
        self._optimizer.step()
        
        # Convert total loss to float now that gradient tracking is done
        total_opt_loss = total_opt_loss.item()
        
        # Aggregate all losses for logging
        total_non_opt_loss = 0
        for lr in non_opt_losses: 
            total_non_opt_loss += to_python(lr.value)
        
        return StepResult(
            total_loss=float(total_non_opt_loss) + float(total_opt_loss),
            total_opt_loss=float(total_opt_loss),
            total_non_opt_loss=float(total_non_opt_loss),
            all_loss_results=loss_cache,
            stage_outputs={k: v for k, v in cache.items() if k in self._model_stages}
        )
    
    def _train_step_tensorflow(
        self,
        batch_input: Dict[str, Batch], 
        losses: Dict[str, List[AppliedLoss]],
    ) -> StepResult:
        
        import tensorflow as tf
        
        # Cache all stage outputs
        cache: Dict[str, Union[Batch, BatchOutput]] = dict(batch_input)
        # Cache all loss results (keyed by ModelStage.label)
        loss_cache: Dict[str, List[LossResult]] = defaultdict(list)
        # Collect trainable variables for optimization
        trainable_vars = []
        
        # There may be cases were the ModelGraph consits of TF & non-optimizable stages (eg, scikit)
        # To optimizate the TF stages, we need to separate out losses for optimization and not
        opt_losses : List[LossResult] = []
        non_opt_losses : List[LossResult] = []
        
        with tf.GradientTape(persistent=True) as tape:
            # Go through all stages in sorted order and collect outputs
            for stage_label in self._sorted_stage_labels:
                # Get stage and attached losses
                stage = self._model_stages[stage_label]
                stage_losses = losses.get(stage_label, None)
                
                # If stage doesn't need optimzer, delegate training
                if not backend_requires_optimizer(stage.backend):
                    step_res = stage.train_step(batch_input=batch_input, losses=stage_losses)
                    cache[stage_label] = step_res.stage_outputs[stage_label]
                    loss_cache[stage_label].extend(step_res.all_loss_results[stage_label])
                    non_opt_losses.extend(step_res.all_loss_results[stage_label])
                    continue
                
                # Perform train or eval depending on freeze state
                if stage.freeze:
                    if stage_losses is not None:
                        raise ValueError(
                            f"`{stage_label}` ModelStage has losses applied but is frozen."
                        )
                    step_res = stage.eval_step(batch_input=batch_input, losses=stage_losses)
                    cache[stage_label] = step_res.stage_outputs[stage_label]
                    loss_cache[stage_label].extend(step_res.all_loss_results[stage_label])
                    non_opt_losses.extend(step_res.all_loss_results[stage_label])
                    continue
                    
                # Forward requires training=True for training in tensorflow
                model_output : BatchOutput = stage.forward(
                    stage.get_input_batch(cache), 
                    training=True
                )
                cache[stage_label] = model_output
                
                # Track trainable variables for this stage
                trainable_vars.extend( stage.model.trainable_variables )
                
                # Compute stage loss (store entire LossResult for later computation)
                if stage_losses is not None:
                    for loss in stage_losses:
                        loss_res = loss.compute(batch_input=batch_input, model_outputs={stage_label: model_output})
                        loss_cache[stage_label].append(loss_res)
                        opt_losses.append(loss_res)
            
        # Perform optimization stepping
        total_opt_loss = tf.add_n([lr.value for lr in opt_losses])
        grads = tape.gradient(total_opt_loss, trainable_vars)
        self._optimizer.step(grads=grads, variables=trainable_vars)
        del tape    # Clean up persistent tape
        
        # Convert total loss to float now that gradient tracking is done
        total_opt_loss = float(total_opt_loss.numpy())
        
        # Aggregate all losses for logging
        total_non_opt_loss = 0
        for lr in non_opt_losses: 
            total_non_opt_loss += to_python(lr.value)
        
        return StepResult(
            total_loss=float(total_non_opt_loss) + float(total_opt_loss),
            total_opt_loss=float(total_opt_loss),
            total_non_opt_loss=float(total_non_opt_loss),
            all_loss_results=loss_cache,
            stage_outputs={k: v for k, v in cache.items() if k in self._model_stages}
        )
    
    def _graphwise_train_step(
        self,
        batch_input: Dict[str, Batch],
        losses: Dict[str, List[AppliedLoss]],
    ) -> StepResult:
        
        if self._optimizer.backend == Backend.TORCH:
            return self._train_step_torch(batch_input=batch_input, losses=losses)

        elif self._optimizer.backend == Backend.TENSORFLOW:
            return self._train_step_tensorflow(batch_input=batch_input, losses=losses)
        
        else:
            raise ValueError(f"Unknown backend for optimization: {self._optimizer.backend}")

    def train_step(
        self,
        batch_input: Dict[str, Batch],
        losses: Dict[str, List[AppliedLoss]],
        trainable_stages: List[str],
    ) -> StepResult:
        """
        Performs a full forward and training step over the entire model graph.

        If `self._optimizer` is set, performs joint training (end-to-end graph optimization).
        Otherwise, delegates to individual `ModelStage.train_step()` calls.

        Args:
            batch_input (Dict[str, Batch]): Input batches keyed by FeatureSet label.
            losses (Dict[str, List[AppliedLoss]]): List of AppliedLosses keyed by ModelStage label.
            trainable_stages (List[str], optional): Which stages to train. Others will be frozen.

        Returns:
            StepResult
        """
        
        # Freeze/unfreeze stages
        for stage_label, stage in self._model_stages.items():
            stage.freeze = stage_label not in trainable_stages
        
        # Mode 1: Graph-level training
        if self._optimizer is not None:
            return self._graphwise_train_step(
                batch_input=batch_input,
                losses=losses,
            )
        
        # Mode 2: Stage-level training
        else:
            return self._stagewise_train_step(
                batch_input=batch_input,
                losses=losses,
            )
            
    def eval_step(
        self,
        batch_input: Dict[str, Batch],
        losses: Dict[str, List[AppliedLoss]],
    ) -> StepResult:
        """
        Performs a full forward and evaluation step (no optimization).

        Args:
            batch_input (Dict[str, Batch]): Input batches keyed by FeatureSet label.
            losses (Dict[str, List[AppliedLoss]],): A List of AppliedLosses keyed by the ModelStage \
                on which they should be applied.

        Returns:
            StepResult
        """
        
        # Freeze all stages
        for stage in self._model_stages.values():
            stage.freeze = True
                
        # Cache all stage outputs
        cache: Dict[str, Union[Batch, BatchOutput]] = dict(batch_input)
        loss_cache: Dict[str, List[LossResult]] = defaultdict(list)
        total_loss = 0.0
        total_opt_loss = 0.0
        total_non_opt_loss = 0.0
        
        # Go through stages in sorted order        
        for stage_label in self._sorted_stage_labels:
            stage = self._model_stages[stage_label]
            
            # Get losses for this stage
            stage_losses = losses.get(stage_label, None)
            input_data = {k:v for k,v in cache.items()}
        
            # Forward pass + loss
            step_res = stage.eval_step(
                batch_input=input_data,
                losses=stage_losses
            )
            
            # Cache stage outputs
            cache[stage_label] = step_res.stage_outputs[stage_label]
            loss_cache[stage_label] = step_res.all_loss_results[stage_label]
            
            total_loss += step_res.total_loss
            total_opt_loss += step_res.total_opt_loss
            total_non_opt_loss += step_res.total_non_opt_loss

        return StepResult(
            total_loss=total_loss,
            total_opt_loss=total_opt_loss,
            total_non_opt_loss=total_non_opt_loss,
            all_loss_results=loss_cache,
            stage_outputs={k: v for k, v in cache.items() if k in self._model_stages}
        )
        

 
    def visualize(
        self, 
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize the structure of the model graph.

        Args:
            save_path (str, optional): If provided, saves the figure to this path.

        Returns:
            Tuple[plt.Figure, plt.Axes]: matplotlib.pyplot Figure and Axes.
        """
        import networkx as nx
        
        graph = nx.DiGraph()

        for label, stage in self._model_stages.items():
            inputs : List[str] = stage.inputs if isinstance(stage.inputs, list) else [stage.inputs, ]
            for inp in inputs:
                graph.add_edge(inp, label,)

        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                graph.nodes[node]["layer"] = layer

        pos = nx.multipartite_layout(graph, subset_key="layer")

        # Assign numeric IDs and build label map
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        id_to_node = {v: k for k, v in node_to_id.items()}
        labels = {node: str(idx) for node, idx in node_to_id.items()}

        # Identify feature nodes vs stage nodes
        feature_nodes = [n for n in graph.nodes if n not in self._model_stages]
        stage_nodes = [n for n in graph.nodes if n in self._model_stages]


        fig, ax = plt.subplots(figsize=(6, 3))

        # Draw features as squares
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=feature_nodes,
            node_shape='s', node_color='lightgray',
            edgecolors='black', linewidths=2, node_size=2000
        )

        # Draw stages as circles
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=stage_nodes,
            node_shape='o', node_color='skyblue',
            edgecolors='black', linewidths=2, node_size=2000
        )

        # Draw edges
        nx.draw_networkx_edges(
            graph, pos, ax=ax, 
            width=2, 
            style='solid',
            arrows=True,             # Enable arrows
            arrowstyle='-|>',         # Style of the arrow
            arrowsize=20,            # Size of the arrow head
            node_size=2200
            )

        # Draw labels (all)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=12)

        # Add legend mapping node ID to stage name
        legend_elements = [
            Line2D(
                [0], [0], 
                marker='s' if name in feature_nodes else 'o', 
                color='w', 
                label=f"{idx}: {'FeatureSet' if name in feature_nodes else 'ModelStage'} `{name}`",
                markerfacecolor='lightgray' if name in feature_nodes else 'skyblue', 
                markersize=10, 
                markeredgecolor='black')
            for idx, name in id_to_node.items()
        ]
        ax.legend(handles=legend_elements, title="ModelGraph Legend", bbox_to_anchor=(1, 0.5), loc='center left')

        ax.margins(0.20)
        ax.axis('off')
        if save_path:
            fig.savefig(save_path)
        return fig, ax

    
    
    