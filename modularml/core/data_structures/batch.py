import uuid
from dataclasses import dataclass, field
from typing import Any

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.utils.exceptions import ShapeSpecError


@dataclass
class Batch:
    """
    A container representing a batch of samples grouped by roles (e.g., 'anchor', 'positive').

    Attributes:
        role_samples (dict[str, SampleCollection]):
            A mapping from string-based roles to SampleCollection objects.
            For example, a triplet batch may use:
            {'anchor': SampleCollection, 'positive': SampleCollection, 'negative': SampleCollection}.

        role_sample_weights (dict[str, Data], optional):
            Optional mapping of roles to per-sample weight arrays.
            If not provided, weights are assumed to be uniform (1.0 for all samples).

        label (str, optional):
            Optional label or tag associated with the batch (e.g., for tracking or logging).

        uuid (str):
            A unique identifier automatically assigned to each batch instance.

    """

    role_samples: dict[str, SampleCollection]
    role_sample_weights: dict[str, Data] = None
    label: str | None = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """
        Perform post-initialization checks.

        Checks performed include:
            - Ensures all role sample collections have the same feature/target shape.
            - Initializes sample weights to 1.0 if not provided.
            - Validates consistency between sample and weight lengths.

        """
        # Enforce consistent shapes
        try:
            f_shapes = list({c.feature_shape_spec.merged_shape for c in self.role_samples.values()})
            if len(f_shapes) != 1:
                msg = f"Inconsistent feature shapes across Batch roles: {f_shapes}."
                raise ValueError(msg)
            self._feature_shape = f_shapes[0]
        except ShapeSpecError as e:
            msg = f"Batch features have incompatible shapes. {e}"
            raise RuntimeError(msg) from e

        try:
            t_shapes = list({c.target_shape_spec.merged_shape for c in self.role_samples.values()})
            if len(t_shapes) != 1:
                msg = f"Inconsistent target shapes across Batch roles: {t_shapes}."
                raise ValueError(msg)
            self._target_shape = t_shapes[0]
        except ShapeSpecError as e:
            msg = f"Batch features have incompatible shapes. {e}"
            raise RuntimeError(msg) from e

        # Check weight shapes
        if self.role_sample_weights is None:
            self.role_sample_weights = {r: Data([1] * len(c)) for r, c in self.role_samples.items()}
        else:
            # Check that each sample weight key matches sample length
            for r, c in self.role_samples.items():
                if r not in self.role_sample_weights:
                    msg = f"Batch `role_sample_weights` is missing required role: `{r}`"
                    raise KeyError(msg)
                if len(self.role_sample_weights[r]) != len(c):
                    msg = (
                        f"Length of batch sample weights does not match length of samples "
                        f"for role `{r}`: {len(self.role_sample_weights[r])} != {len(c)}."
                    )
                    raise ValueError(msg)

    @property
    def available_roles(self) -> list[str]:
        """
        List of role names (e.g., ['anchor', 'positive']).

        Returns:
            list[str]: The list of available role keys.

        """
        return list(self.role_samples.keys())

    @property
    def feature_shape(self) -> tuple[int, ...]:
        """
        Feature shape shared across all roles.

        Returns:
            tuple[int, ...]: Shape of features.

        """
        return self._feature_shape

    @property
    def target_shape(self) -> tuple[int, ...]:
        """
        Target shape shared across all roles.

        Returns:
            tuple[int, ...]: Shape of targets.

        """
        return self._target_shape

    @property
    def n_samples(self) -> int:
        """
        Number of samples in each role (equal for all roles).

        Returns:
            int: Sample count per role.

        """
        if not hasattr(self, "_n_samples") or self._n_samples is None:
            self._n_samples = len(self.role_samples[self.available_roles[0]])
        return self._n_samples

    def __len__(self):
        """
        Return number of samples per role.

        Returns:
            int: Sample count.

        """
        return self.n_samples

    def get_samples(self, role: str) -> SampleCollection:
        """
        Retrieve the sample collection for a given role.

        Args:
            role (str): The role name to retrieve.

        Returns:
            SampleCollection: The samples associated with that role.

        """
        return self.role_samples[role]


@dataclass
class BatchOutput:
    """
    Output container for ComputationNodes, representing the raw output tensor(s) for each role.

    Attributes:
        features (dict[str, Any]):
            Mapping from role names to output feature tensors.

        sample_uuids (dict[str, Any]):
            Mapping from role names to sample UUID lists.

        targets (dict[str, Any], optional):
            Optional mapping from role names to target tensors.

        tags (dict[str, Any], optional):
            Optional mapping from role names to auxiliary tag data.

    """

    features: dict[str, Any]
    sample_uuids: dict[str, Any]
    targets: dict[str, Any] | None = None
    tags: dict[str, Any] | None = None
    sample_weights: dict[str, Any] = None

    def __post_init__(self):
        """
        Perform validation and shape enforcement.

        Checks performed include:
            - Checks for consistency of feature, target, and tag shapes.
            - Ensures keys match between features and sample_uuids.

        """
        # Enforce consistent shapes
        f_shapes = list({self.features[role].shape for role in self.features})
        if len(f_shapes) != 1:
            msg = f"Inconsistent feature shapes across BatchOutput roles: {f_shapes}."
            raise ValueError(msg)
        self._feature_shape = f_shapes[0]

        if self.targets is not None:
            t_shapes = list({self.targets[role].shape for role in self.targets})
            if len(t_shapes) != 1:
                msg = f"Inconsistent target shapes across Batch roles: {t_shapes}."
                raise ValueError(msg)
            self._target_shape = t_shapes[0]
        else:
            self._target_shape = None

        # Ensure consistent roles across attributes
        keys = [
            tuple(x.keys())
            for x in [self.sample_uuids, self.targets, self.features, self.tags, self.sample_weights]
            if x is not None
        ]
        if len(set(keys)) > 1:
            msg = "Role names do not match across BatchOutput attributes."
            raise KeyError(msg)

    @property
    def available_roles(self) -> list[str]:
        """
        List of all roles present in the output.

        Returns:
            list[str]: The keys in the `features` dictionary.

        """
        return list(self.features.keys())

    @property
    def feature_shape(self) -> tuple[int, ...]:
        """
        Shape of the output features.

        Returns:
            tuple[int, ...]: The shape tuple of feature tensors.

        """
        return self._feature_shape

    @property
    def target_shape(self) -> tuple[int, ...]:
        """
        Shape of the output targets (if available).

        Returns:
            tuple[int, ...]: The shape tuple of target tensors.

        """
        return self._target_shape
