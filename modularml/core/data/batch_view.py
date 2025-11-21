import numpy as np

from modularml.core.data.batch import Batch, NodeShapes, RoleData
from modularml.core.data.featureset_view import FeatureSetView
from modularml.core.data.sample_schema import FEATURES_COLUMN, TAGS_COLUMN, TARGETS_COLUMN
from modularml.core.graph.featureset import FeatureSet
from modularml.utils.data_conversion import to_numpy
from modularml.utils.data_format import DataFormat, format_is_tensorlike


class BatchView:
    """
    A multi-role, zero-copy grouped view over a parent FeatureSet.

    Roles:
        A batch can contain one or more roles, each defined by a set of \
        row indices into the parent FeatureSet.
    """

    def __init__(
        self,
        source: FeatureSet,
        role_indices: dict[str, np.ndarray],
        role_indice_weights: dict[str, np.ndarray] | None = None,
    ):
        self.source = source
        self.role_indices = role_indices
        self.role_indice_weights = role_indice_weights

        # validation
        if not isinstance(self.source, FeatureSet):
            raise TypeError("BatchView.source must be a FeatureSet")
        if not isinstance(self.role_indices, dict):
            raise TypeError("role_indices must be a dict[str, np.ndarray]")

        # ensure arrays
        for r, idx in self.role_indices.items():
            self.role_indices[r] = np.asarray(idx, dtype=int)

    @property
    def roles(self) -> list[str]:
        return list(self.role_indices.keys())

    @property
    def n_samples(self) -> int:
        """Batch size is defined as the number of samples in each role."""
        # all roles have the same number of samples
        first_role = next(iter(self.role_indices))
        return len(self.role_indices[first_role])

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
            label=role,
        )

    def materialize_batch(self, *, fmt: DataFormat = DataFormat.NUMPY) -> Batch:
        """
        Convert this BatchView into a fully materialized in-memory Batch.

        Description:
            Builds a runtime-ready :class:`Batch` object from the current \
            BatchView by converting all role-based subsets into backend-specific \
            tensor formats. Each role (e.g., "anchor", "pair") is materialized \
            independently based on its assigned sample indices and selected \
            feature/target/tag columns.

            The resulting Batch contains:
            - A mapping of node_label → role → :class:`RoleData`
            - Optional per-sample weights for each role
            - Recorded output shapes (excluding batch dimension) for each domain

            This method enforces that the leading dimension of all tensors \
            matches the expected batch size (`self.n_samples`).

        Args:
            fmt (DataFormat):
                Desired output data format, typically one of \
                :attr:`DataFormat.NUMPY`, :attr:`DataFormat.TORCH`, \
                or :attr:`DataFormat.TENSORFLOW`. Defaults to ``NUMPY``.

        Returns:
            Batch:
                A fully materialized :class:`Batch` instance containing:
                - ``outputs``: node- and role-specific :class:`RoleData` objects \
                    with "features", "targets", and "tags".
                - ``role_sample_weights``: optional array-like weight vectors.
                - ``shapes``: per-node :class:`NodeShapes` objects describing \
                    per-sample tensor shapes (without batch dimension).

        Example:
            ```python
            >>> batch = batch_view.materialize_batch(fmt=DataFormat.TORCH)
            >>> batch.outputs["MyFeatureSet"]["anchor"].features.shape
            torch.Size([1, 101])
            ```

        Raises:
            TypeError:
                If the requested ``fmt`` is not a tensor-compatible DataFormat.
            ValueError:
                If any tensor does not have the batch size as its leading dimension,
                or if sample weight shapes do not match ``n_samples``.

        """
        # Format must be tensor-like:
        if not format_is_tensorlike(fmt=fmt):
            msg = f"DataFormat must be tensor-like. Received: {fmt}"
            raise TypeError(msg)

        role_data: dict[str, RoleData] = {}
        shapes: NodeShapes | None = None
        role_sample_weights: dict[str, np.ndarray] = {}

        for role in self.role_indices:
            # Get tensor-like data for each domain
            coll = self.get_role_view(role).to_samplecollection()
            features = coll.get_features(fmt=fmt)
            targets = coll.get_targets(fmt=fmt)
            tags = coll.get_tags(fmt=fmt)
            uuids = coll.get_sample_uuids(fmt=fmt)

            role_data[role] = RoleData(
                sample_uuids=uuids,
                features=features,
                targets=targets,
                tags=tags,
            )

            # Ensure shapes are as expected (batch_size is first dim)
            all_shapes = {
                FEATURES_COLUMN: features.shape,
                TARGETS_COLUMN: targets.shape,
                TAGS_COLUMN: tags.shape,
            }
            for k, v in all_shapes.items():
                if v[0] != self.n_samples:
                    msg = (
                        f"{k}.shape data does not have batch_size as leading dimension: {v}. "
                        f"Expected: ({self.n_samples}, ...)."
                    )
                    raise ValueError(msg)
            # Drop batch_size
            shapes = NodeShapes(shapes={k: v[1:] for k, v in all_shapes.items()})

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

        # Get GraphNode label of originating FeatureSet
        node_label = f"{self.source.label}"

        # Construct Batch
        return Batch(
            outputs={node_label: role_data},
            role_sample_weights=role_sample_weights,
            shapes={node_label: shapes},
        )

    def __repr__(self) -> str:
        return f"BatchView(n_samples={self.n_samples}, roles={self.roles})"

    def __str__(self):
        return self.__repr__()
