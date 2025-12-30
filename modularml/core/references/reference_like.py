from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modularml.core.references.data_reference_group import DataReferenceGroup


@runtime_checkable
class ReferenceLike(Protocol):
    """
    Structural interface for reference objects used in a Experiment.

    Any object implementing this protocol can be resolved into a
    DataReferenceGroup.
    """

    @property
    def node_id(self) -> str:
        """ID of the ExperimentNode this reference points to."""
        ...

    @property
    def node_label(self) -> str:
        """Label of the ExperimentNode this reference points to."""
        ...

    def resolve(self) -> DataReferenceGroup:
        """
        Resolve this reference into a DataReferenceGroup.

        Args:
            default_node (str | None):
                Optional fallback node label to bind this reference to
                if it does not explicitly specify one.

        Returns:
            DataReferenceGroup

        """
        ...
