from modularml.core.experiment.phase_io import PhaseIO
from modularml.core.graph.model_graph import ModelGraph
from modularml.core.loss.applied_loss import AppliedLoss
from modularml.core.samplers.feature_sampler import FeatureSampler


class EvaluationPhase:
    """
    Encapsulates a single stage of evaluation for a ModelGraph.

    Responsibilities:
        - Performs a full forward pass of all batch data defined by `samplers`
        - Any provided losses are computed and optionally logged

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
        samplers: dict[str, FeatureSampler],
        batch_size: int,
        losses: list[AppliedLoss] | None = None,
    ):
        """
        Initializes an EvaluationPhase.

        Args:
            label (str): Name of this evaluation phase (e.g., "evaluate_encoder").
            samplers (dict[str, FeatureSampler]): Mapping from source string to FeatureSampler.
                Keys must be of the form "FeatureSet" or "FeatureSet.subset".
            batch_size (int): Batch size to enforce across all samplers (overrides existing sampler batch sizes).
            losses (list[AppliedLoss], optional): Optional list of loss functions and their input mappings.
        """
        self.label = label

        self.eval_io = PhaseIO(
            samplers=samplers,
            batch_size=batch_size,
            losses=losses,
        )

    @property
    def resolved(self) -> bool:
        return self.eval_io.resolved

    def resolve(self, graph: ModelGraph):
        # Resolve PhaseIO
        self.eval_io.resolve(graph=graph)
