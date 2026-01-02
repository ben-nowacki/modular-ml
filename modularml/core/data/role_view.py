from __future__ import annotations

from typing import TYPE_CHECKING

from modularml.utils.representation.summary import Summarizable

if TYPE_CHECKING:
    from modularml.core.data.batch import Batch
    from modularml.core.data.sample_data import SampleData
    from modularml.core.data.sample_shapes import SampleShapes


class RoleView(Summarizable):
    """Lightweight, read-only view over a single role in a Batch."""

    __slots__ = ("_batch", "_role")

    def __init__(self, batch: Batch, role: str):
        self._batch = batch
        self._role = role

    # ===============================================
    # Core accessors
    # ===============================================
    @property
    def role(self) -> str:
        return self._role

    @property
    def data(self) -> SampleData:
        return self._batch.role_data[self._role]

    @property
    def shapes(self) -> SampleShapes:
        return self._batch.shapes

    @property
    def weights(self):
        return self._batch.role_weights[self._role]

    # ===============================================
    # Convenience passthroughs
    # ===============================================
    @property
    def sample_uuids(self):
        return self.data.sample_uuids

    @property
    def features(self):
        return self.data.features

    @property
    def targets(self):
        return self.data.targets

    @property
    def tags(self):
        return self.data.tags

    def __repr__(self) -> str:
        return f"RoleView(role='{self._role}', batch_size={self._batch.batch_size})"

    def _summary_rows(self) -> list[tuple]:
        return [
            ("role", self.role),
            ("batch_size", self._batch.batch_size),
            ("domains", [(k, "") for k in self.data.data]),
            ("shapes", [(k, str(v)) for k, v in self.shapes.shapes.items()]),
        ]
