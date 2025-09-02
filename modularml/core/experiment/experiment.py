



from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from modularml.core.model_graph import ModelGraph
from modularml.core.samplers.feature_sampler import FeatureSampler
from modularml.core.experiment.eval_phase import EvaluationPhase
from modularml.core.experiment.training_phase import TrainingPhase
from modularml.utils.data_format import DataFormat, convert_to_format


'''
Top-level orchestrator

Responsibilities:
    Contains the entire training/evaluation configuration.
    Owns:
        ModelGraph: complete DAG of FeatureSet + ModelStage nodes.
        One or more TrainingPhase instances (can include pretraining, finetuning, etc.).
        TrackingManager: logs losses, metrics, configs, checkpoints.
    Manages:
        .run(): executes all TrainingPhases in order.
        .evaluate(): runs evaluations on designated stages.
        
Example:
    Experiment.run()
    │
    ├── TrainingPhase 1 (pretrain encoder+decoder)
    │   ├── Sampler → Batches
    │   ├── ModelGraph.forward(Batch)
    │   ├── AppliedLoss.compute(outputs, batch)
    │   └── Optimizer step(s)
    │
    ├── TrainingPhase 2 (fine-tune regressor)
    │   ├── Sampler → Batches
    │   ├── ModelGraph.forward(Batch)
    │   ├── AppliedLoss.compute(...)
    │   └── Optimizer step(s)
    │
    └── Done!
'''


# TODO: all callbacks in BasePhase to run custom scipts after each phase


class Experiment:
    """
    The `Experiment` class coordinates the training and evaluation of a modular model graph
    across one or more `TrainingPhase` and `EvaluationPhase` configurations.

    It handles:
    - Building the model graph (if not already built)
    - Resolving losses and role mappings for each phase
    - Executing each phase in order
    - Optionally logging via a `TrackingManager`

    Args:
        graph (ModelGraph): The model graph containing all stages and data sources.
        phases (List[Union[TrainingPhase, EvaluationPhase]]): List of phases to run (can include pretraining, fine-tuning, evaluation, etc).
        tracking (TrackingManager, optional): Optional tracker for logging metrics, artifacts, or experiment metadata.
    """
    
    def __init__(
        self,
        graph: ModelGraph,
        phases: List[Union[TrainingPhase, EvaluationPhase]],
        tracking: Optional["TrackingManager"] = None,                   # type: ignore
    ):
        """
        Initialize an Experiment with a model graph and a list of training and/or evaluation phases.
        
        Args:
            graph (ModelGraph): The modular computation graph containing model stages and feature sets.
            phases (List[Union[TrainingPhase, EvaluationPhase]]): Ordered list of training and evaluation phases
                that define the experiment procedure.
            tracking (Optional[TrackingManager]): Optional tracking manager for logging metrics, parameters,
                and outputs. Can be used to connect with MLflow or local logging tools.
        """
        
        self.graph = graph
        self.phases = (
            [phases] if isinstance(phases, TrainingPhase) else phases
        )
        self.tracking = tracking

        # Build graph (required before .run())
        if not self.graph.is_built:
            self.graph.build_all()

        # Resolve losses for each training phase
        for phase in self.phases:
            phase.resolve_loss_inputs_and_roles(self.graph)
    
        
    def run(self):
        """
        Run all training and evaluation phases in order.

        Calls `run_training_phase` for each `TrainingPhase`, and `run_evaluation_phase`
        for each `EvaluationPhase`. Raises `TypeError` if an unrecognized phase type is passed.
        """
        for phase in self.phases:
            if isinstance(phase, TrainingPhase):
                self.run_training_phase(phase)
            elif isinstance(phase, EvaluationPhase):
                self.run_evaluation_phase(phase)
            else:
                raise TypeError(f"Unknown phase type: {phase}")
    
    def run_training_phase(self, phase: TrainingPhase):
        """
        Run a single training phase over a fixed number of epochs.

        For each epoch:
        - Batches are sampled from the associated `FeatureSampler`s.
        - Losses are computed and backpropagated for the trainable stages.
        - Average sample loss is reported every 10 epochs.

        Args:
            phase (TrainingPhase): Training phase object that defines samplers, loss functions,
            number of epochs, and which model stages are trainable.
        """
        
        if not phase.is_resolved:
            _ = phase.resolve_loss_inputs_and_roles(self.graph)
        
        print(f"Executing TrainingPhase: {phase.label}")
        print(f"  > Training")
        for epoch in range(phase.n_epochs):
            all_batches = phase.get_batches(self.graph._feature_sets)
            num_batches = min([len(b) for b in all_batches.values()])

            total_loss = 0.0
            n_samples = 0
            for i in range(num_batches):
                n_samples += all_batches[ list(all_batches.keys())[0] ][i].n_samples
                batch = {k: v[i] for k,v in all_batches.items()}
                result = self.graph.train_step(
                    batch_input=batch,
                    losses=phase._loss_mapping,
                    trainable_stages=phase.get_trainable_stages(),
                )
                total_loss += result["total_loss"]
                
            avg_sample_loss = total_loss / n_samples
            if epoch % 10 == 0:
                print(f"    - Epoch {epoch}: Avg. sample loss = {avg_sample_loss:.4f}")

    def run_evaluation_phase(self, phase: EvaluationPhase) -> Dict[str, list]:
        """
        Run a single evaluation phase and collect model outputs for analysis.

        For each batch:
        - Performs forward pass through the graph
        - Computes losses (if specified)
        - Records model outputs and associates them with input sample UUIDs and roles

        Args:
            phase (EvaluationPhase): EvaluationPhase to execute.

        Returns:
            Dict[str, list]: A dict containing the following keys:
                - "sample_uuid": Unique identifier of the input sample
                - "role": Role assigned by the sampler (e.g., 'default', 'anchor', etc.)
                - "{ModelStage.label}.output": Output from the model stage for this sample

        This output can be used for downstream visualization, metric computation,
        t-SNE plots, scatter comparisons, and more.
        """

        if not phase.is_resolved:
            phase.resolve_loss_inputs_and_roles(self.graph)
            
        print(f"Executing EvaluationPhase: {phase.label}")
        print(f"  > Evaluating")
        
        all_batches = phase.get_batches(self.graph._feature_sets)
        num_batches = min([len(b) for b in all_batches.values()])
        total_loss = 0.0
        n_samples = 0
        model_outputs = defaultdict(list)
        for i in range(num_batches):
            n_samples += all_batches[ list(all_batches.keys())[0] ][i].n_samples
            
            batch = {k: v[i] for k,v in all_batches.items()}
            result = self.graph.eval_step(
                batch_input=batch,
                losses=phase._loss_mapping,
            )
            
            total_loss += result["total_loss"]
            for k in result['stage_outputs'].keys():
                model_outputs[k].append( result['stage_outputs'][k] )
            
        avg_sample_loss = total_loss / n_samples
        print(f"    - Avg. sample loss = {avg_sample_loss:.4f}")


        # Convert all output data to single dict
        data : Dict[str, list] = defaultdict(list)
        for stage_label, batches in model_outputs.items():
            for b in batches:
                for role in b.available_roles:
                    # Record model outputs
                    outputs = convert_to_format(b.features[role], format=DataFormat.LIST)
                    data[f"{stage_label}.output"].extend(outputs)
                    
                    # Record role labels
                    data['role'].extend( [role, ] * len(outputs) )
                    
                    # Record sample uuids
                    s_uuids = convert_to_format(b.sample_uuids[role], format=DataFormat.LIST)
                    data['sample_uuid'].extend( s_uuids )
        
        return convert_to_format(data, format=DataFormat.DICT_LIST) 
    

