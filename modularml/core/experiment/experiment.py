



from typing import List, Optional

from modularml.core.model_graph import ModelGraph
from modularml.core.training.training_phase import TrainingPhase


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


class Experiment:
    def __init__(
        self,
        graph: ModelGraph,
        phases: List[TrainingPhase],
        tracking: Optional["TrackingManager"] = None    # type: ignore
    ):
        pass
    
    
    
    def run_phase(phase: TrainingPhase):
        pass
    
    def run():
        pass
    
    def evaluate():
        pass