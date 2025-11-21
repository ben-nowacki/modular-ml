from typing import TYPE_CHECKING

import numpy as np

from modularml.core.data.batch_view import BatchView
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.graph.featureset import FeatureSet
from modularml.core.references.data_reference import DataReference
from modularml.core.sampling.base_sampler import BaseSampler
from modularml.utils.data_format import ensure_list

if TYPE_CHECKING:
    from modularml.core.data.sample_collection import SampleCollection


class SimpleSampler(BaseSampler):
    def __init__(
        self,
        source: FeatureSet | FeatureSetView | None = None,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        group_by: list[str] | None = None,
        stratify_by: list[str] | None = None,
        strict_stratification: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize a sampler that splits a FeatureSet or view into batches.

        Description:
            `SimpleSampler` implements three batching strategies:

            1. **Group-based batching (`group_by`)**
                Samples sharing identical values for the specified columns are \
                placed into the same bucket. Each bucket is then partitioned \
                into batches.

            2. **Stratified batching (`stratify_by`)**
                Samples are grouped into strata defined by column values. \
                Batches are created by interleaving samples from each stratum \
                to maintain balanced representation.

                - If `strict_stratification=True`, batching stops once any \
                  stratum is exhausted (perfectly balanced but may drop samples).
                - If False, batching continues until all samples are consumed \
                  (uses all samples but later batches may be unbalanced).

            3. **Sequential batching**
                Samples are taken in order (optionally shuffled) and cut into \
                fixed-size batches.

            The sampler always returns **zero-copy BatchView objects** that \
            reference the original FeatureSet. BatchViews do *not* materialize \
            data; they only store role → row-index mappings.

        Args:
            source (FeatureSet | FeatureSetView):
                If provided, batches are built immediately; otherwise, call \
                `bind_source()` later.
            batch_size (int):
                Number of samples in each batch.
            shuffle (bool, optional):
                Whether to shuffle samples (and later the resulting batches).
            group_by (list[str], optional):
                FeatureSet key(s) defining grouping behavior. \
                Only one grouping strategy can be active at a time.
            stratify_by (list[str], optional):
                FeatureSet key(s) defining strata for stratified sampling. \
                Conflicts with `group_by`.
            strict_stratification (bool, optional):
                See description above.
            drop_last (bool, optional):
                Drop the final incomplete batch.
            seed (int, optional):
                Random seed for reproducible shuffling.

        Raises:
            ValueError:
                If both `group_by` and `stratify_by` are provided.

        """
        # Ensure that group_by & stratify_by lists
        self.group_by: list[str] = ensure_list(group_by)
        self.stratify_by: list[str] = ensure_list(stratify_by)
        self.strict_stratification = strict_stratification

        if self.stratify_by and self.group_by:
            msg = "Both `group_by` and `stratify_by` cannot be applied at the same."
            raise ValueError(msg)

        super().__init__(source=source, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, seed=seed)

    def build_batches(self) -> list[BatchView]:
        """
        Build batches from the bound source according to the configured strategy.

        Description:
            Selects one of the following strategies:
                - **Group-based** (`_build_grouped_batches`)
                - **Stratified** (`_build_stratified_batches`)
                - **Sequential** (`_build_sequential_batches`)

            Each method returns a list of **BatchView** objects containing:
                - role_indices={"default": <absolute row indices>}
                - no data copies (a view of the original FeatureSet)
                - optional role weights (None for SimpleSampler)

        Returns:
            list[BatchView]:
                Zero-copy views of the parent FeatureSet.

        Raises:
            RuntimeError: If no source is bound.
            TypeError: If the source is not a FeatureSetView.
            ValueError: If stratification is requested but `batch_size < n_strata`.

        """
        if self.source is None:
            raise RuntimeError("`bind_source` must be called before batches can be built.")
        if not isinstance(self.source, FeatureSetView):
            msg = f"`source` must be of type FeatureSetView. Received: {type(self.source)}"
            raise TypeError(msg)

        view: FeatureSetView = self.source
        abs_indices = view.indices

        # Groupby logic
        if self.group_by:
            print("grouping")
            return self._build_grouped_batches(view, abs_indices)

        # Stratify logic
        if self.stratify_by:
            print("stratifying")
            return self._build_stratified_batches(view, abs_indices)

        # Default (sequential batching)
        return self._build_sequential_batches(view, abs_indices)

    # =====================================================
    # Helpers
    # =====================================================
    def _build_grouped_batches(self, view: FeatureSetView, abs_indices: np.ndarray) -> list[BatchView]:
        """
        Build batches such that each batch contains samples only from a single group.

        Description:
            Samples are assigned to groups based on the `group_by` column values. \
            Each group is independently shuffled (optional) and split into \
            batches of size `batch_size`.

        Returns:
            list[BatchView]:
                One BatchView per constructed batch. Each batch has role "default".

        Notes:
            - No cross-group mixing occurs.
            - Groups with fewer samples than `batch_size` produce no batch \
              when `drop_last=True`.

        """
        # self.group_by contains a list of strings, each string can be any column in FeatureSet
        # DataReference infers the node, domain, key, and variant given a user-defined strings
        # E.g, "voltages" -> ("MyFS", "features", "voltages", "raw")
        group_refs: list[DataReference] = [
            DataReference.from_string(
                x,
                known_attrs={"node": view.source.label},
                required_attrs=["node", "domain", "key", "variant"],
            )
            for x in self.group_by
        ]
        # Collect data for each defined group_by key
        group_data: dict[str, np.ndarray] = {}
        coll: SampleCollection = view.to_samplecollection()
        for data_ref in group_refs:
            node, domain, key, variant = data_ref.node, data_ref.domain, data_ref.key, data_ref.variant
            if not node == view.source.label:
                raise ValueError(
                    "Error with parsing group_by terms. Inferred DataReference does not refer to `source`.",
                )
            col_data = coll.get_variant_data(domain=domain, key=key, variant=variant, fmt="numpy")
            k = data_ref.to_string()
            if k in group_data:
                msg = f"DataReference.to_string() already exists in `group_data`: {k}"
                raise ValueError(msg)
            group_data[k] = col_data

        # Convert `group_data` to rows of tuples, each tuple becomes the unique grouping key
        # These row_tuples are used to construct unique group buckets
        # Each bucket holds absolute indices of the rows of `view` belonging to each bucket
        buckets: dict[tuple, list[int]] = {}
        for i, abs_idx in enumerate(abs_indices):
            row_vals = tuple(group_data[k][i] for k in group_data)
            buckets.setdefault(row_vals, []).append(abs_idx)

        # Build batches
        batches: list[BatchView] = []
        for abs_idxs in buckets.values():
            # Shuffle if specified
            idxs = np.asarray(abs_idxs)
            if self.shuffle:
                self.rng.shuffle(idxs)

            # Build batches from this group
            for start in range(0, len(idxs), self.batch_size):
                batch_idxs = idxs[start : start + self.batch_size]
                if len(batch_idxs) < self.batch_size and self.drop_last:
                    continue

                batches.append(
                    BatchView(
                        source=view.source,
                        role_indices={"default": batch_idxs},
                        role_indice_weights=None,
                    ),
                )

        # Shuffle again, if specified, and return
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def _build_stratified_batches(self, view: FeatureSetView, abs_indices: np.ndarray) -> list[BatchView]:
        """
        Build batches ensuring balanced representation across strata.

        Description:
            Samples are partitioned into strata defined by `stratify_by` columns. \
            Batches are created by interleaving samples from each stratum.

            - If `strict_stratification=True`, interleaving stops when the first \
              stratum is exhausted (perfectly balanced).
            - Otherwise, interleaving continues until all strata are empty.

        Returns:
            list[BatchView]:
                BatchViews where each batch contains samples from all strata.

        Raises:
            ValueError:
                If the number of strata exceeds `batch_size`.

        """
        # self.stratify_by contains a list of strings, each string can be any column in FeatureSet
        # DataReference infers the node, domain, key, and variant given a user-defined strings
        # E.g, "voltages" -> ("MyFS", "features", "voltages", "raw")
        strata_refs: list[DataReference] = [
            DataReference.from_string(
                x,
                known_attrs={"node": view.source.label},
                required_attrs=["node", "domain", "key", "variant"],
            )
            for x in self.stratify_by
        ]
        # Collect data for each defined stratify_by key
        strata_data: dict[str, np.ndarray] = {}
        coll: SampleCollection = view.to_samplecollection()
        for data_ref in strata_refs:
            node, domain, key, variant = data_ref.node, data_ref.domain, data_ref.key, data_ref.variant
            if not node == view.source.label:
                raise ValueError(
                    "Error with parsing stratify_by terms. Inferred DataReference does not refer to `source`.",
                )
            col_data = coll.get_variant_data(domain=domain, key=key, variant=variant, fmt="numpy")
            k = data_ref.to_string()
            if k in strata_data:
                msg = f"DataReference.to_string() already exists in `strata_data`: {k}"
                raise ValueError(msg)
            strata_data[k] = col_data

        # Construct strata buckets
        # Each bucket defines a unique strata class
        # Each bucket holds absolute indices of the rows of `view` belonging to each strata
        buckets: dict[tuple, list[int]] = {}
        for i, abs_idx in enumerate(abs_indices):
            row_vals = tuple(strata_data[k][i] for k in strata_data)
            buckets.setdefault(row_vals, []).append(abs_idx)

        # Number of buckets must be <= batch size
        if len(buckets) > self.batch_size:
            msg = f"The batch size must be larger than the number of strata: {self.batch_size} < {len(buckets)}"
            raise ValueError(msg)

        # Shuffle absolute indices in each bucket
        for k, abs_idxs in buckets.items():
            buckets[k] = np.asarray(abs_idxs)
            if self.shuffle:
                self.rng.shuffle(buckets[k])

        # Interleave (cast each bucket to an iterator)
        iterators = {k: iter(v) for k, v in buckets.items()}
        keys = list(iterators)

        # Holds all abs. indices sorted such each each strata is evenly distributed
        # along the length of the list
        interleaved_idxs = []

        # There are 2 ways to performed stratification:
        #   - interleave until all samples are used up
        #       - after first group runs out, no longer balanced strata
        #   - interleave until first group runs out
        #       - all batches balanced, but may not use all samples
        exhausted = set()
        while (len(exhausted) == 0) if self.strict_stratification else (len(exhausted) < len(keys)):
            # Add next element from each strata into interleaved array
            for k in keys:
                if k in exhausted:
                    continue
                try:
                    interleaved_idxs.append(next(iterators[k]))
                except StopIteration:
                    exhausted.add(k)

        # Construct batches from interleaved sample indices
        interleaved_idxs = np.array(interleaved_idxs)

        batches: list[BatchView] = []
        for start in range(0, len(interleaved_idxs), self.batch_size):
            batch_idxs = interleaved_idxs[start : start + self.batch_size]
            if len(batch_idxs) < self.batch_size and self.drop_last:
                continue

            batches.append(
                BatchView(
                    source=view.source,
                    role_indices={"default": batch_idxs},
                    role_indice_weights=None,
                ),
            )

        # Shuffle again, if specified, and return
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def _build_sequential_batches(self, view: FeatureSetView, abs_indices: np.ndarray) -> list[BatchView]:
        """
        Build batches by sequentially slicing through the sample indices.

        Description:
            Samples are optionally shuffled and then partitioned into batches \
            of size `batch_size`.

        Returns:
            list[BatchView]:
                Simple sequential batches represented as BatchViews.

        Notes:
            - Incomplete batches are dropped when `drop_last=True`.
            - No grouping or balancing is applied.

        """
        # Shuffle indices if specified
        if self.shuffle:
            self.rng.shuffle(abs_indices)

        # Build batches from indices
        batches: list[BatchView] = []
        for start in range(0, len(abs_indices), self.batch_size):
            batch_idxs = abs_indices[start : start + self.batch_size]
            if len(batch_idxs) < self.batch_size and self.drop_last:
                continue

            batches.append(
                BatchView(
                    source=view.source,
                    role_indices={"default": batch_idxs},
                    role_indice_weights=None,
                ),
            )

        # Shuffle again, if specified, and return
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches
