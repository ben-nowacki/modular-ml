import copy
import warnings
from collections import defaultdict

from modularml.core.data_structures.batch import Batch
from modularml.core.graph.feature_set import FeatureSet
from modularml.core.graph.graph_node import GraphNode
from modularml.core.graph.model_graph import ModelGraph
from modularml.core.loss.applied_loss import AppliedLoss
from modularml.core.samplers.feature_sampler import FeatureSampler


class PhaseIO:
    def __init__(
        self,
        samplers: dict[str, FeatureSampler],
        losses: list[AppliedLoss] | None = None,
        batch_size: int = 32,
    ):
        # Store losses and losses grouped by ModelGraph node label
        self.losses = losses or []
        self._loss_mapping = defaultdict(list)

        # Parsed samplers if provided
        self.samplers: dict[tuple[str, str | None], FeatureSampler] = {}
        if samplers:
            for k, v in samplers.items():
                v.batch_size = batch_size  # overwrite pre-existing batch_size
                self.samplers[self._parse_sampler_spec(k)] = copy.deepcopy(v)
        self.batch_size = batch_size

        # Internal state for lazy validation
        self._resolved = False

    @property
    def resolved(self) -> bool:
        return self._resolved

    @property
    def losses_mapped_by_node(self) -> dict[str, list[AppliedLoss]]:
        if not self._resolved:
            raise RuntimeError("You must call `.resolve(...)` before accessing losses.")
        return self._loss_mapping

    def _parse_sampler_spec(self, spec: str) -> tuple[str, str | None]:
        """
        Parses a string that references a FeatureSet or FeatureSubset.

        Accepted formats:
            - "FeatureSet" (use all samples in FeatureSet)
            - "FeatureSet.subset" (use only a named subset)

        Returns:
            tuple[str, Optional[str]]: (FeatureSet label, Subset label or None)

        Raises:
            ValueError: If the string format is invalid.

        """
        featureset, subset = None, None

        parts = spec.split(".")
        if len(parts) == 1:
            featureset, subset = parts[0], None
        elif len(parts) == 2:
            featureset, subset = parts
        else:
            msg = f"Invalid `samplers` key: {spec}"
            raise ValueError(msg)

        return featureset, subset

    def resolve(self, graph: ModelGraph):
        # Step 1: Sample batches from all samplers
        batches = self.get_batches({k: graph._nodes[k] for k in graph.source_node_labels})

        # Step 2: Use the first batch from each source
        batch_input = {fs: b[0] for fs, b in batches.items()}

        # TODO: merge policy needs to be check prior to forward pass. Need to:  # noqa: FIX002
        #  - For each stage in graph._sorted_stage_labels, get required inputs.
        #  - If multiple inputs, check if suitable merge policy

        # Step 3: Run forward pass
        outputs = graph.forward(batch_input)
        for source_node_lbl in graph.source_node_labels:
            outputs.pop(source_node_lbl)

        # Step 4: Collect available input keys
        available_inputs: set[str] = set()

        # Valid FeatureSet-based AppliedLoss input keys (eg, "FeatureSet.targets" or "FeatureSet.features.anchor")
        for fs_lbl, batch in batch_input.items():
            for role in batch.available_roles:
                available_inputs.add(f"{fs_lbl}.features.{role}")
                available_inputs.add(f"{fs_lbl}.targets.{role}")

        # Valid ModelStage-based AppliedLoss input keys (eg, "Encoder.output" or "Encoder.output.anchor")
        for stage_name, batch in outputs.items():
            for role in batch.available_roles:
                available_inputs.add(f"{stage_name}.output.{role}")

        # Step 5: Determine loss-required inputs
        required_inputs = set()
        self._loss_mapping.clear()
        for loss in self.losses:
            for source in loss.all_inputs.values():
                # source is of type: tuple[str, str, Optional[str]] for node, attribute, role
                required_inputs.add(f"{source[0]}.{source[1]}" + (f".{source[2]}" if source[2] is not None else ""))

                # Record mapping of each loss to any ModelStages
                if source[0] in graph.connected_node_labels:
                    self._loss_mapping[source[0]].append(loss)

        # Check is inputs required by AppliedLosses is missing from forward pass
        missing_inputs = required_inputs - available_inputs
        if missing_inputs:
            msg = f"Missing inputs required by `losses`: {missing_inputs}. Available inputs: {available_inputs}"
            raise ValueError(msg)

        self._resolved = True

    def get_losses_for_node(self, node: str | GraphNode) -> list[AppliedLoss] | None:
        if not self._resolved:
            raise RuntimeError("You must call `.resolve(...)` before accessing losses.")
        node_lbl = node if isinstance(node, str) else node.label
        return self._loss_mapping.get(node_lbl, None)

    def get_batches(self, featuresets: dict[str, FeatureSet]) -> dict[str, list[Batch]]:
        """
        Binds each FeatureSampler to its corresponding FeatureSet or Subset and builds batches for training.

        Args:
            featuresets (dict[str, FeatureSet]): dictionary of available FeatureSets keyed by \
                their labels. E.g., `{FeatureSet.label: FeatureSet, ...}`

        Returns:
            dict[str, list[Batch]]: dictionary mapping each FeatureSet label to a list of Batches.

        Raises:
            ValueError: If a required FeatureSet or FeatureSubset is missing.

        Warnings:
            If samplers yield differing numbers of batches, only the minimum number will be used.

        """
        # 1. Check that featuresets match sample keys
        for sampler_spec in self.samplers:
            if sampler_spec[0] not in featuresets:
                msg = f"Required FeatureSet (`{sampler_spec[0]}`) is missing: {featuresets.keys()}"
                raise ValueError(msg)

            if sampler_spec[1] is not None and sampler_spec[1] not in featuresets[sampler_spec[0]].available_subsets:
                msg = (
                    f"Required FeatureSubset (`{sampler_spec}`) is missing: "
                    f"{featuresets[sampler_spec[0]].available_subsets}"
                )
                raise ValueError(msg)

        # Batches are keyed by FeatureSet (drop subset label?)
        batches: dict[str, list[Batch]] = {}

        # 2. Build batches (ensure samplers are bound to sources)
        for (fs_lbl, fss_lbl), sampler in self.samplers.items():
            if not sampler.is_bound():
                # Get FeatureSet
                source = featuresets[fs_lbl]
                # Get FeatureSubset if specified
                if fss_lbl is not None:
                    source = source.get_subset(fss_lbl)

                # Bind source to sampler
                sampler.bind_source(source=source)

            batches[fs_lbl] = sampler.batches

        # 3. Check n_batches for mismatch
        batch_lens = {len(b) for b in batches.values()}
        if len(batch_lens) > 1:
            warnings.warn(
                f"`samplers` resulted in a differing number of batches: {batch_lens}. "
                f"Only the lesser will be used: {min(batch_lens)} batches.",
                stacklevel=2,
            )

        return batches
