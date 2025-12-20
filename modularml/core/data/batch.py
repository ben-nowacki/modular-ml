import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from modularml.core.data.role_view import RoleView
from modularml.core.data.sample_data import SampleData
from modularml.core.data.sample_shapes import SampleShapes
from modularml.utils.representation.summary import format_summary_box

# @dataclass
# class Batch:
#     """
#     Materialized data container for one forward or training pass through the graph.

#     Description:
#         A Batch represents a collection of role-specific data derived from one or \
#         more FeatureSets, organized per model-graph node. Each node's output data \
#         is stored as a mapping of roles to SampleData objects, containing tensors \
#         for the standardized schema domains (features, targets, tags).

#         The Batch also records optional sample weights and per-node output shapes.

#     Args:
#         batch_size (int):
#             Batch size. Corresponds to first dimension of output tensors.

#         outputs:
#             Dictionary mapping node labels to role mappings, where each role maps \
#             to a SampleData object. Structure: \
#                 node_label -> role -> SampleData(domain -> tensor-like data)

#         role_sample_weights:
#             Optional mapping of role names to per-sample weights \
#             (array-like of shape (batch_size,)).

#         shapes:
#             Optional mapping of node labels to SampleShapes objects, describing the \
#             tensor shapes for each schema domain (excluding batch dimension).

#         uuid:
#             Unique identifier automatically generated for tracking/logging this batch.

#     """

#     batch_size: int

#     # Outputs of each node
#     # Initiallized with FeatureSet sample during batch creation
#     # Keyed as follows: node_label -> role -> SampleData(feature/target/tag -> tensor-like data)
#     outputs: dict[str, dict[str, SampleData]]

#     # Weights of each sample in each role
#     # Initiallize during batch creation (and then immutable?)
#     # Keyed as follows: role -> np.ndarray with shape (batch_size,)
#     role_sample_weights: dict[str, np.ndarray]

#     # Shape of outputs
#     # Keyed as follows: node_label -> SampleShapes(feature/target/tag -> tuple[int, ...])
#     shapes: dict[str, SampleShapes]

#     # Unique ID for logging/tracking batches
#     uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

#     def __repr__(self) -> str:
#         """
#         Return a concise summary of the batch contents.

#         Examples:
#             Batch(nodes=["enc", "clf"], roles={"enc":['anchor','pair'], ...}, shapes={"enc":(32,1,16), ...}, uuid=...)

#         """
#         # Collect summary information
#         node_labels = list(self.outputs.keys())
#         node_roles = {node: list(self.outputs[node].keys()) for node in node_labels}

#         return f"Batch(nodes={len(node_labels)}, roles={node_roles}, shapes={self.shapes}, uuid='{self.uuid}')"

#     def __str__(self) -> str:
#         return self.__repr__()

#     def summary(self) -> str:
#         rows = []

#         rows.append(f"batch_size  : {self.batch_size}")
#         rows.append("nodes       : " + ", ".join(self.outputs.keys()))

#         roles_fmt = {node: list(roles.keys()) for node, roles in self.outputs.items()}
#         rows.append(f"roles       : {roles_fmt}")

#         shapes_fmt = {}
#         for node, node_shapes in self.shapes.items():
#             shapes_fmt[node] = dict(node_shapes.shapes.items())
#         rows.append(f"shapes      : {shapes_fmt}")

#         # weight_fmt = {role: list(w.shape) for role, w in self.role_sample_weights.items()}
#         # rows.append(f"weights     : {weight_fmt}")

#         rows.append(f"uuid        : {self.uuid}")

#         # Formatting
#         width = max(len(r) for r in rows) if rows else 0
#         title = f"{self.__class__.__name__}"
#         border = "─" * width
#         body = "\n".join(rows)
#         return f"┌─ {title} {'─' * (width - len(title) - 3)}┐\n{body}\n└{border}┘"


@dataclass(frozen=True)
class Batch:
    """
    Immutable, role-structured tensor container produced from a single BatchView.

    A Batch represents one sampler's worth of data for a single
    execution step. It contains no graph or node semantics.
    """

    batch_size: int

    # Core role-based storage
    role_data: Mapping[str, SampleData]
    shapes: SampleShapes
    role_weights: Mapping[str, NDArray[np.float32]]

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

            if weights.shape != (self.batch_size,):
                msg = f"role_weights['{role}'] has shape {weights.shape}, expected ({self.batch_size},)"
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
    # Introspection
    # ================================================
    def summary(self, max_width: int = 88) -> str:
        rows = [
            ("batch_size", self.batch_size),
            ("roles", [(role, "") for role in self.roles]),
            ("shapes", [(k, str(v)) for k, v in self.shapes.shapes.items()]),
            ("uuid", self.uuid),
        ]

        return format_summary_box(
            title=self.__class__.__name__,
            rows=rows,
            max_width=max_width,
        )

    def __repr__(self) -> str:
        return f"Batch(batch_size={self.batch_size}, roles={self.roles}, uuid='{self.uuid}')"
