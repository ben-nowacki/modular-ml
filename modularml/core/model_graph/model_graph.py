

'''
DAG of FeatureSets and ModelStages

Responsibilities:
    Stores all nodes: FeatureSets and ModelStages.
    Stores all edges: how outputs of one stage feed another.
    Performs:
        Topological sorting
        Lazy build() propagation of shapes
        Forward pass through the graph
'''


from typing import Any, Dict, Union

from modularml.core.data_structures.feature_set import FeatureSet
from modularml.core.model_stage import ModelStage


class ModelGraph:
    def __init__(
        self,
        nodes: Dict[str, Union[FeatureSet, ModelStage]],
    ):
        # edges: Dict[str, List[str]] : source -> list of downstream targets
        pass
    
    def build():
        pass
    
    def forward(batch: Batch) -> Dict[str, Any]:
        """Returns all intermediate ModelStage outputs, keyed by `ModelStage.label`

        Args:
            batch (Batch): _description_

        Returns:
            Dict[str, Any]: _description_
        """
        pass
    
    
    
    