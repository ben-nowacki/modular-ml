import warnings

from modularml.core.experiment.phase_io import PhaseIO
from modularml.core.graph.graph_node import GraphNode
from modularml.core.graph.model_graph import ModelGraph
from modularml.core.loss.applied_loss import AppliedLoss
from modularml.core.samplers.feature_sampler import FeatureSampler


class TrainingPhase:
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
        train_samplers: dict[str, FeatureSampler],
        batch_size: int,
        n_epochs: int,
        nodes_to_freeze: list[str | GraphNode] | None = None,
        val_samplers: dict[str, FeatureSampler] | None = None,
        # Early stopping parameters
        early_stop_patience: int | None = None,
        early_stop_metric: str = "val_loss",
        early_stop_mode: str = "min",
        early_stop_min_delta: float = 0.0,
    ):
        """
        Initializes a TrainingPhase.

        Args:
            label (str): Name of this training phase (e.g., "pretrain_encoder").
            losses (list[AppliedLoss]): List of loss functions and their input mappings.
            train_samplers (dict[str, FeatureSampler]): Mapping from source string to FeatureSampler.
                Keys must be of the form "FeatureSet" or "FeatureSet.subset".
            batch_size (int): Batch size to enforce across all samplers.
            n_epochs (int): Number of training epochs.
            nodes_to_freeze (list[str | GraphNode] | None): The list of ModelGraph nodes to freeze during this
                training phase. Defaults to None (i.e., it will train all stages).
            val_samplers (dict[str, FeatureSampler] | None): Validation samplers, if provided.
            early_stop_patience (int | None): Number of epochs with no improvement before stopping early.
                If None, early stopping is disabled.
            early_stop_metric (str): Which loss metric to monitor (e.g., "val_loss" or "train_loss").
            early_stop_mode (str): One of {"min", "max"}.
                - "min": Training stops when the monitored metric stops decreasing.
                - "max": Training stops when the monitored metric stops increasing.
            early_stop_min_delta (float): Minimum change in the monitored metric to be considered an improvement.

        """
        self.label = label

        self.train_io = PhaseIO(
            samplers=train_samplers,
            losses=losses,
            batch_size=batch_size,
        )
        self.val_io: PhaseIO | None = (
            PhaseIO(
                samplers=val_samplers,
                losses=losses,
                batch_size=batch_size,
            )
            if val_samplers is not None
            else None
        )

        self.n_epochs = n_epochs

        # Early stopping configuration
        self.early_stop_patience = early_stop_patience
        self.early_stop_metric = early_stop_metric
        self.early_stop_mode = early_stop_mode
        self.early_stop_min_delta = early_stop_min_delta

        # Track frozen nodes
        self._frozen_nodes: list[str] = []
        if not isinstance(nodes_to_freeze, list):
            nodes_to_freeze = [nodes_to_freeze]
        for n in nodes_to_freeze:
            if isinstance(n, str):
                self._frozen_nodes.append(n)
            elif isinstance(n, GraphNode):
                self._frozen_nodes.append(n.label)

    @property
    def resolved(self) -> bool:
        return self.train_io.resolved and (self.val_io is None or self.val_io.resolved)

    @property
    def frozen_nodes(self) -> list[str]:
        if not self.resolved:
            raise RuntimeError("You must call `.resolve(...)` before accessing frozen_nodes.")
        return self._frozen_nodes

    def resolve(self, graph: ModelGraph):
        # Resolve PhaseIO instances
        self.train_io.resolve(graph=graph)
        if self.val_io is not None:
            self.val_io.resolve(graph=graph)

        # Check that user-specified frozen nodes don't have a loss applied to them
        nodes_with_loss = set(self.train_io.losses_mapped_by_node.keys())
        overlapped = set(self._frozen_nodes).intersection(nodes_with_loss)
        if len(overlapped) != 0:
            msg = (
                f"The following nodes are frozen but have an AppliedLoss tied to them: {overlapped}"
                f"The attached losses will be ignored during optimizer stepping."
            )
            warnings.warn(message=msg, category=UserWarning, stacklevel=2)
