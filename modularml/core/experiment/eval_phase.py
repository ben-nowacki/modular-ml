from typing import Any

from modularml.core.experiment.base_phase import BasePhase
from modularml.core.loss.applied_loss import AppliedLoss
from modularml.core.samplers.feature_sampler import FeatureSampler


class EvaluationPhase(BasePhase):
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
        # merge_policy: str | None = None,
        # merge_mapping: dict[str, Any] | None = None,
    ):
        """
        Initializes an EvaluationPhase.

        Args:
            label (str): Name of this evaluation phase (e.g., "evaluate_encoder").
            samplers (dict[str, FeatureSampler]): Mapping from source string to FeatureSampler.
                Keys must be of the form "FeatureSet" or "FeatureSet.subset".
            batch_size (int): Batch size to enforce across all samplers (overrides existing sampler batch sizes).
            losses (list[AppliedLoss], optional): Optional list of loss functions and their input mappings.

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
