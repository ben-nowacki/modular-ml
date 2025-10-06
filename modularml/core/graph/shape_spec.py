from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShapeSpec:
    """Describes per-feature input/output shapes for a node."""

    shapes: dict[str, tuple[int, ...]]

    # ==================================================
    # Basic properties
    # ==================================================
    @property
    def unique_shapes(self) -> set[tuple[int, ...]]:
        """Return the unique set of shapes across all feature entries."""
        return set(self.shapes.values())

    @property
    def merged_axis(self) -> int | None:
        """
        Infer the axis along which this ShapeSpec's shapes could be merged.

        Returns:
            int | None: Axis index to concatenate along, or None if ambiguous.

        """
        all_shapes = list(self.shapes.values())
        if len(all_shapes) <= 1:
            return -1  # Trivial or single-shape case

        rank = len(all_shapes[0])
        if not all(len(s) == rank for s in all_shapes):
            return None  # Different ranks, cannot infer axis

        differing_axes = set()
        for axis in range(rank):
            dims = {s[axis] for s in all_shapes}
            if len(dims) > 1:
                differing_axes.add(axis)

        if len(differing_axes) == 0:
            return -1  # identical shapes
        if len(differing_axes) == 1:
            return differing_axes.pop()
        return None  # ambiguous (multiple differing axes)

    @property
    def merged_shape(self) -> tuple[int, ...]:
        """
        Compute the merged shape of all contained feature shapes.

        Returns:
            tuple[int, ...]: The merged tensor shape.

        Raises:
            ValueError: If shapes are incompatible or cannot be merged.

        Examples:
            >>> ShapeSpec({"a": (1, 32), "b": (1, 16)}).merged_shape
            (1, 48)
            >>> ShapeSpec({"a": (8,), "b": (12,)}).merged_shape
            (20,)

        """
        all_shapes = list(self.shapes.values())

        # 1. Handle trivial case
        if len(all_shapes) == 0:
            raise ValueError("Cannot compute merged shape for an empty ShapeSpec.")
        if len(all_shapes) == 1:
            return all_shapes[0]

        # 2. Check rank consistency
        rank = len(all_shapes[0])
        if not all(len(s) == rank for s in all_shapes):
            msg = f"Inconsistent ranks among shapes: {all_shapes}"
            raise ValueError(msg)

        # 3. Determine merge axis
        merge_axis = self.merged_axis
        if merge_axis is None:
            msg = f"Cannot determine a unique merge axis for shapes: {all_shapes}"
            raise ValueError(msg)

        # 4. Verify compatibility along all other axes
        for axis in range(rank):
            if axis == merge_axis:
                continue
            dim = all_shapes[0][axis]
            if not all(s[axis] == dim for s in all_shapes):
                msg = f"Incompatible dimensions on axis {axis} for shapes {all_shapes}"
                raise ValueError(msg)

        # 5. Merge along the chosen axis
        merged_dims = list(all_shapes[0])
        merged_dims[merge_axis] = sum(s[merge_axis] for s in all_shapes)
        return tuple(merged_dims)

    # ==================================================
    # Shape access
    # ==================================================
    def get(self, key: str) -> tuple[int, ...]:
        """Safely get shape for a given feature key."""
        return self.__getitem__(key)

    def __getitem__(self, key: str) -> tuple[int, ...]:
        if key not in self.shapes:
            msg = f"Key `{key}` does not exist in `shapes`: {list(self.shapes.keys())}"
            raise KeyError(msg)
        return self.shapes[key]

    def __eq__(self, other: ShapeSpec) -> bool:
        return self.shapes == other.shapes

    def __hash__(self):
        # Dicts are unhashable; convert to frozenset of items
        return hash(frozenset(self.shapes.items()))

    def __repr__(self) -> str:
        return f"ShapeSpec({self.shapes!r})"

    # -------------------------------------------------------------------------
    # Multi-Spec comparison
    # -------------------------------------------------------------------------
    def compatible_merge_with(self, other: ShapeSpec) -> bool:
        """
        Check whether this ShapeSpec can be merged with another.

        Two ShapeSpecs are compatible if:
          - They have identical feature keys, and
          - Each corresponding shape is either identical or matches
            in all but one dimension (mergeable along one axis).

        Returns:
            bool: True if compatible, False otherwise.

        """
        # Same object or identical specs
        if self == other:
            return True

        # Require identical keys
        if self.shapes.keys() != other.shapes.keys():
            return False

        for k in self.shapes:
            s1, s2 = self.shapes[k], other.shapes[k]
            if len(s1) != len(s2):
                return False
            diff_axes = [i for i, (a, b) in enumerate(zip(s1, s2, strict=True)) if a != b]
            if len(diff_axes) > 1:
                return False
        return True

    def infer_merge_axis_with(self, other: ShapeSpec) -> int | None:
        """
        Infer the merge axis between two ShapeSpecs if they are compatible.

        Returns:
            int | None: Axis index if mergeable, else None.

        """
        if not self.compatible_merge_with(other):
            return None

        axes = set()
        for k in self.shapes:
            s1, s2 = self.shapes[k], other.shapes[k]
            diff_axes = [i for i, (a, b) in enumerate(zip(s1, s2, strict=True)) if a != b]
            axes.update(diff_axes)

        if len(axes) == 0:
            return -1  # identical
        if len(axes) == 1:
            return axes.pop()
        return None

    def merged_shape_with(self, other: ShapeSpec) -> tuple[int, ...]:
        """
        Return the merged shape if two ShapeSpecs are compatible.

        Raises ValueError if not mergeable.
        """
        axis = self.infer_merge_axis_with(other)
        if axis is None:
            msg = f"Cannot merge incompatible ShapeSpecs: {self.shapes} vs {other.shapes}"
            raise ValueError(msg)

        first_shape = next(iter(self.shapes.values()))
        other_shape = next(iter(other.shapes.values()))

        merged_dims = list(first_shape)
        merged_dims[axis] = first_shape[axis] + other_shape[axis]
        return tuple(merged_dims)
