import warnings

from modularml.core.experiment.base_phase import BasePhase
from modularml.core.graph.graph_node import GraphNode
from modularml.core.loss.applied_loss import AppliedLoss
from modularml.core.samplers.feature_sampler import FeatureSampler


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
        losses: list[AppliedLoss],
        samplers: dict[str, FeatureSampler],
        batch_size: int,
        n_epochs: int,
        nodes_to_freeze: list[str | GraphNode] | None = None,
        # merge_policy: str | None = None,
        # merge_mapping: dict[str, Any] | None = None,
    ):
        """
        Initializes a TrainingPhase.

        Args:
            label (str): Name of this training phase (e.g., "pretrain_encoder").
            losses (list[AppliedLoss]): list of loss functions and their input mappings.
            samplers (dict[str, FeatureSampler]): Mapping from source string to FeatureSampler.
                Keys must be of the form "FeatureSet" or "FeatureSet.subset".
            batch_size (int): Batch size to enforce across all samplers (overrides existing sampler batch sizes).
            n_epochs (int): Number of training epochs.
            nodes_to_freeze (list[str | GraphNode] | None): The list of ModelGraph nodes to freeze during this \
                training phase. Defaults to None (ie, it will train all stages).

            # merge_policy (str, optional): Strategy to merge roles if needed. Not yet implemented. TODO
            # merge_mapping (dict[str, Any], optional): Custom mapping for role merges. Not yet implemented. TODO

        """
        super().__init__(
            label=label,
            losses=losses,
            samplers=samplers,
            batch_size=batch_size,
            # merge_policy=merge_policy,
            # merge_mapping=merge_mapping,
        )

        self.n_epochs = n_epochs

        self._frozen_nodes: list[str] = []
        if not isinstance(nodes_to_freeze, list):
            nodes_to_freeze = [nodes_to_freeze]
        for n in nodes_to_freeze:
            if isinstance(n, str):
                self._frozen_nodes.append(n)
            elif isinstance(n, GraphNode):
                self._frozen_nodes.append(n.label)

    def get_frozen_nodes(self) -> list[str]:
        # Check that nodes with losses aren't frozen
        nodes_with_loss = set(self._loss_mapping.keys())
        overlapped = set(self._frozen_nodes).intersection(nodes_with_loss)
        if len(overlapped) != 0:
            msg = (
                f"The following nodes are frozen but have an AppliedLoss tied to them: {overlapped}"
                f"The attached losses will be ignored during optimizer stepping."
            )
            warnings.warn(message=msg, category=UserWarning, stacklevel=2)

        # Return frozen nodes
        return self._frozen_nodes
