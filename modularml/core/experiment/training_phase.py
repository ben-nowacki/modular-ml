

from typing import Any, Dict, List, Optional

from modularml.core.model_graph.loss import AppliedLoss
from modularml.core.samplers.feature_sampler import FeatureSampler
from modularml.core.experiment.base_phase import BasePhase



class TrainingPhase(BasePhase):
    """
    Encapsulates a single stage of training for a ModelGraph.

    Responsibilities:
        - Trains a specified subset of the ModelGraph (e.g., encoder only, or encoder + head).
        - Defines the loss functions (`AppliedLoss`) to optimize.
        - Defines sampling logic through one or more `FeatureSampler`s.
        - Provides logic for merging multi-role batches if needed (e.g., in contrastive learning).
        - Computes the available input sources at each model stage for loss computation.

    Design:
        - Sampling is defined via `samplers`, a dictionary where keys are strings like 
          "FeatureSet" or "FeatureSet.subset", and values are FeatureSampler instances.
        - Each FeatureSampler is dynamically bound to its source during `get_batches()`.
        - All samplers in a phase must use the same batch size to avoid shape mismatch.

    Notes:
        - Use `get_batches()` to retrieve aligned batches from each sampler.
        - Use `get_available_loss_inputs()` to analyze valid data sources for loss computation.
    """

    def __init__(
        self,
        label: str,
        losses : List[AppliedLoss],
        
        samplers: Dict[str, FeatureSampler],
        batch_size: int,
        n_epochs: int,
        
        merge_policy: Optional[str] = None,
        merge_mapping: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes a TrainingPhase.

        Args:
            label (str): Name of this training phase (e.g., "pretrain_encoder").
            losses (List[AppliedLoss]): List of loss functions and their input mappings.
            samplers (Dict[str, FeatureSampler]): Mapping from source string to FeatureSampler. 
                Keys must be of the form "FeatureSet" or "FeatureSet.subset".
            batch_size (int): Batch size to enforce across all samplers (overrides existing sampler batch sizes).
            n_epochs (int): Number of training epochs.
        
            merge_policy (str, optional): Strategy to merge roles if needed. Not yet implemented. TODO
            merge_mapping (Dict[str, Any], optional): Custom mapping for role merges. Not yet implemented. TODO
        """
        
        super().__init__(
            label=label,
            losses=losses,
            samplers=samplers,
            batch_size=batch_size,
            merge_policy=merge_policy,
            merge_mapping=merge_mapping
        )
        
        self.n_epochs = n_epochs
        
    def get_trainable_stages(self) -> List[str]:
        return list(self._loss_mapping.keys())
        
