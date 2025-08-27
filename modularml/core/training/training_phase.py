

from typing import List, Union

from modularml.core.model_graph.loss import AppliedLoss
from modularml.core.model_graph.model_stage import ModelStage
from modularml.core.samplers.feature_sampler import FeatureSampler

'''
Encapsulates one stage of training

Responsibilities:
    Trains a subset of the ModelGraph (e.g. encoder only, or encoder + decoder)
    Defines:
        included_stages: which ModelStages to train
        losses: list of AppliedLoss with selectors
        sampler: a FeatureSampler that returns Batch objects
        optimizer_config or per-stage overrides
    Enforces:
        Compatibility of Sampler roles with AppliedLoss role selectors
        That all needed feature keys exist in each role's Sample.features
'''


class TrainingPhase:
    def __init__(
        self,
        name: str,
        model_stages: Union[List[ModelStage], List[str]],
        losses : List[AppliedLoss],
        sampler: FeatureSampler,
        
        # training params like epochs, early stopping, etc
    ):
        pass
    
    