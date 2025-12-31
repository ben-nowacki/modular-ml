from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from modularml.core.data.batch import Batch, SampleData, SampleShapes
from modularml.core.data.featureset import FeatureSet
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.schema_constants import DOMAIN_FEATURES, DOMAIN_SAMPLE_ID, DOMAIN_TAGS, DOMAIN_TARGETS
from modularml.utils.data.conversion import to_numpy
from modularml.utils.data.data_format import DataFormat, format_is_tensorlike
from modularml.utils.data.pyarrow_data import resolve_column_selectors
from modularml.utils.representation.summary import Summarizable


@dataclass(frozen=True)
class BatchView(Summarizable):
    """
    A multi-role, zero-copy grouped view over a parent FeatureSet.

    Roles:
        A batch can contain one or more roles, each defined by a set of \
        row indices into the parent FeatureSet.
    """

    source: FeatureSet
    role_indices: dict[str, NDArray[np.int64]]
    role_indice_weights: dict[str, NDArray[np.float32]] | None = None

    def __post_init__(self):
        # Validate source
        if not isinstance(self.source, FeatureSet):
            msg = f"`source` must be of type FeatureSet. Received: {type(self.source)}"
            raise TypeError(msg)

        # Validate indices
        if not isinstance(self.role_indices, dict):
            raise TypeError("role_indices must be a dict[str, np.ndarray]")
        np_idxs = {}
        for r, idxs in self.role_indices.items():
            np_idxs[r] = np.asarray(idxs, dtype=np.int64)
        # Override frozen attributes
        object.__setattr__(self, "role_indices", np_idxs)

        # Validate weights
        if self.role_indice_weights is not None:
            idx_keys = set(self.role_indices.keys())
            weight_keys = set(self.role_indice_weights.keys())
            if idx_keys != weight_keys:
                msg = f"Indices and weights do not have the same role keys: {idx_keys} != {weight_keys}"
                raise ValueError(msg)

            np_weights = {}
            for k, wghts in self.role_indice_weights.items():
                np_weights[k] = np.asarray(wghts, dtype=np.float32)

            # Override frozen attributes
            object.__setattr__(self, "role_indice_weights", np_weights)

    # ==========================================
    # Properties
    # ==========================================
    @property
    def roles(self) -> list[str]:
        return list(self.role_indices.keys())

    @property
    def n_samples(self) -> int:
        """Batch size is defined as the number of samples in each role."""
        # all roles have the same number of samples
        first_role = next(iter(self.role_indices))
        return len(self.role_indices[first_role])

    # ==========================================
    # Data Accessors
    # ==========================================
    def get_role_view(self, role: str) -> FeatureSetView:
        """
        Construct a FeatureSetView for a specific role.

        Description:
            Creates a :class:`FeatureSetView` that includes only the rows
            corresponding to the given role and restricts its columns according
            to this BatchView's selected feature/target/tag keys.

        Args:
            role (str):
                Role identifier to extract (e.g., "anchor", "pair", "positive").

        Returns:
            FeatureSetView:
                A lightweight view over the parent FeatureSet restricted to
                the specified role and shared column filters.

        Raises:
            KeyError:
                If the specified role does not exist in `role_indices`.

        """
        if role not in self.role_indices:
            msg = f"Role '{role}' not found in BatchView (available: {list(self.role_indices.keys())})."
            raise KeyError(msg)

        return FeatureSetView(
            source=self.source,
            indices=self.role_indices[role],
            columns=self.source.get_all_keys(include_domain_prefix=True, include_rep_suffix=True),
            label=role,
        )

    def materialize_batch(
        self,
        fmt: DataFormat = DataFormat.NUMPY,
        *,
        columns: str | list[str] | None = None,
        features: str | list[str] | None = None,
        targets: str | list[str] | None = None,
        tags: str | list[str] | None = None,
        rep: str | None = None,
    ) -> Batch:
        """
        Convert this BatchView into a fully materialized in-memory Batch.

        Description:
            Builds a runtime-ready :class:`Batch` object from the current BatchView
            by converting all role-based subsets into backend-specific tensor formats.
            Each role (e.g., "anchor", "pair") is materializedindependently based
            on its assigned sample indices and selected feature/target/tag columns.

            By default, all columns (and all representations) are used in batch creation.

            This method enforces that the leading dimension of all tensors matches
            the expected batch size (`self.n_samples`).

        Args:
            fmt (DataFormat):
                Desired output data format, typically one of
                :attr:`DataFormat.NUMPY`, :attr:`DataFormat.TORCH`,
                or :attr:`DataFormat.TENSORFLOW`. Defaults to `NUMPY`.

            columns (str | list[str] | None):
                Fully-qualified column names to include in the materialized tensor.

            features (str | list[str] | None):
                Feature-domain selectors. Accepts bare keys, key/rep pairs,
                or wildcards.

            targets (str | list[str] | None):
                Target-domain selectors.

            tags (str | list[str] | None):
                Tag-domain selectors.

            rep (str | None):
                Default representation suffix applied when omitted in selectors.


        Returns:
            Batch:
                A fully materialized :class:`Batch` instance containing:
                - `outputs`: node- and role-specific :class:`SampleData` objects \
                    with "features", "targets", and "tags".
                - `role_sample_weights`: optional array-like weight vectors.
                - `shapes`: per-node :class:`SampleShapes` objects describing \
                    per-sample tensor shapes (without batch dimension).

        Example:
            ```python
            >>> batch = batch_view.materialize_batch(fmt=DataFormat.TORCH)
            >>> batch.outputs["MyFeatureSet"]["anchor"].features.shape
            torch.Size([1, 101])
            ```

        Raises:
            TypeError:
                If the requested `fmt` is not a tensor-compatible DataFormat.
            ValueError:
                If any tensor does not have the batch size as its leading dimension,
                or if sample weight shapes do not match `n_samples`.

        """
        # Format must be tensor-like:
        if not format_is_tensorlike(fmt=fmt):
            msg = f"DataFormat must be tensor-like. Received: {fmt}"
            raise TypeError(msg)

        # Get column filters
        # Each domain defaults to all columns, unless subset specified
        all_cols = self.source.get_all_keys(include_domain_prefix=True, include_rep_suffix=True)
        all_cols.remove(DOMAIN_SAMPLE_ID)

        # Fill any empty domain keys with all columns
        selected: dict[str, set[str]] = resolve_column_selectors(
            all_columns=all_cols,
            columns=columns,
            features=features,
            targets=targets,
            tags=tags,
            rep=rep,
            include_all_if_empty=True,
        )

        # Construct tensors for each role
        role_data: dict[str, SampleData] = {}
        shapes: SampleShapes | None = None
        role_sample_weights: dict[str, np.ndarray] = {}
        for role in self.role_indices:
            # Get tensor-like data for each domain
            fsv = self.get_role_view(role).select(
                features=list(selected[DOMAIN_FEATURES]),
                targets=list(selected[DOMAIN_TARGETS]),
                tags=list(selected[DOMAIN_TAGS]),
            )

            # Extract feature/target/tag values in chosen format
            features = fsv.get_features(
                fmt=fmt,
                rep=None,
                include_domain_prefix=True,
                include_rep_suffix=True,
            )
            targets = fsv.get_targets(
                fmt=fmt,
                rep=None,
                include_domain_prefix=True,
                include_rep_suffix=True,
            )
            tags = fsv.get_tags(
                fmt=fmt,
                rep=None,
                include_domain_prefix=True,
                include_rep_suffix=True,
            )
            uuids = fsv.get_sample_uuids(fmt=fmt)

            # Build SampleData for this role
            role_data[role] = SampleData(
                sample_uuids=uuids,
                features=features,
                targets=targets,
                tags=tags,
            )

            # Ensure shapes are as expected (batch_size is first dim)
            all_shapes = {
                DOMAIN_FEATURES: features.shape,
                DOMAIN_TARGETS: targets.shape,
                DOMAIN_TAGS: tags.shape,
            }
            for k, v in all_shapes.items():
                if v[0] != self.n_samples:
                    msg = (
                        f"{k}.shape data does not have batch_size as leading dimension: {v}. "
                        f"Expected: ({self.n_samples}, ...)."
                    )
                    raise ValueError(msg)
            # Drop batch_size
            shapes = SampleShapes(shapes={k: v[1:] for k, v in all_shapes.items()})

            # Add role_sample_weights (if defined)
            if self.role_indice_weights and role in self.role_indice_weights:
                # Ensure weights are 1D arrays with shape equal to batch_size
                weights = to_numpy(self.role_indice_weights[role]).reshape(-1)
                if weights.shape[0] != self.n_samples:
                    msg = (
                        f"Shape of sample weights does not match batch_size: {weights.shape}. "
                        f"Expected: ({self.n_samples},)."
                    )
                    raise ValueError(msg)
                role_sample_weights[role] = weights
            # Otherwise, default to weights of 1
            else:
                role_sample_weights[role] = np.ones(shape=(self.n_samples))

        # Construct Batch
        return Batch(
            batch_size=self.n_samples,
            role_data=role_data,
            shapes=shapes,
            role_weights=role_sample_weights,
        )

    def __repr__(self) -> str:
        return f"BatchView(n_samples={self.n_samples}, roles={self.roles})"

    def __str__(self):
        return self.__repr__()

    def _summary_rows(self) -> list[tuple]:
        return [
            ("source", self.source.label),
            ("n_samples", self.n_samples),
            ("roles", [(r, "") for r in self.roles]),
        ]
