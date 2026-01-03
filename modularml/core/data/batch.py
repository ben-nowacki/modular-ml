import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from modularml.core.data.role_view import RoleView
from modularml.core.data.sample_data import RoleData, SampleShapes
from modularml.utils.representation.summary import Summarizable


@dataclass(frozen=True)
class Batch(Summarizable):
    """
    Immutable, role-structured tensor container produced from a single BatchView.

    A Batch represents one sampler's worth of data for a single
    execution step. It contains no graph or node semantics.
    """

    batch_size: int

    # Core role-based storage
    role_data: RoleData
    shapes: SampleShapes
    role_weights: Mapping[str, NDArray[np.float32]]
    role_masks: dict[str, NDArray[np.int8]]

    # Tracking
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ================================================
    # Validation
    # ================================================
    def __post_init__(self):
        roles = set(self.role_data)

        if not roles:
            raise ValueError("MaterializedBatch must contain at least one role.")

        if set(self.role_weights) != roles:
            raise ValueError("role_weights keys must match role_data keys.")

        for role in roles:
            data = self.role_data[role]
            weights = self.role_weights[role]
            mask = self.role_masks[role]

            if weights.shape != (self.batch_size,):
                msg = f"role_weights['{role}'] has shape {weights.shape}, expected ({self.batch_size},)"
                raise ValueError(msg)
            if mask.shape != (self.batch_size,):
                msg = f"role_masks['{role}'] has shape {mask.shape}, expected ({self.batch_size},)"
                raise ValueError(msg)

            # Validate batch dimension consistency
            for domain, tensor in data.data.items():
                if tensor is not None and tensor.shape[0] != self.batch_size:
                    msg = (
                        f"{role}.{domain} has leading dimension {tensor.shape[0]}, "
                        f"expected batch_size={self.batch_size}"
                    )
                    raise ValueError(msg)

    # ================================================
    # Role access
    # ================================================
    @property
    def roles(self) -> list[str]:
        return list(self.role_data.keys())

    def get_role(self, role: str) -> RoleView:
        if role not in self.role_data:
            msg = f"Role '{role}' not found. Available roles: {self.roles}"
            raise KeyError(msg)
        return RoleView(self, role)

    # ================================================
    # Pseudo-attribute access
    # ================================================
    def __getattr__(self, name: str):
        # Called only if attribute not found normally
        if name in self.role_data:
            return self.get_role(name)
        msg = f"{self.__class__.__name__} has no attribute '{name}' (available roles: {self.roles})"
        raise AttributeError(msg)

    # ================================================
    # Representation
    # ================================================
    def _summary_rows(self) -> list[tuple]:
        return [
            ("batch_size", self.batch_size),
            ("roles", [(role, "") for role in self.roles]),
            ("shapes", [(k, str(v)) for k, v in self.shapes.shapes.items()]),
            ("uuid", self.uuid),
        ]

    def __repr__(self) -> str:
        return f"Batch(batch_size={self.batch_size}, roles={self.roles}, uuid='{self.uuid}')"
