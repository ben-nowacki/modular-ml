from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from modularml.core.io.protocols import Configurable
from modularml.core.references.reference_like import ReferenceLike

if TYPE_CHECKING:
    from modularml.context.experiment_context import ExperimentContext
    from modularml.context.resolution_context import ResolutionContext
    from modularml.core.experiment.experiment_node import ExperimentNode


class ResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentReference(ReferenceLike, Configurable):
    """Base class for references resolvable at the Experiment scope."""

    def resolve(self, ctx: ResolutionContext):
        if ctx.experiment is None:
            msg = f"{type(self).__name__} requires an ExperimentContext"
            raise ResolutionError(msg)
        return self._resolve_experiment(ctx.experiment)

    def _resolve_experiment(self, experiment: ExperimentContext):
        raise NotImplementedError

    def to_string(
        self,
        *,
        separator: str = ".",
        include_node_id: bool = False,
    ) -> str:
        """
        Joins all non-null fields into a single string.

        Example:
        ``` python
            ref = DataReference(node='PulseFeatures', domain='features', key='voltage')
            ref.to_string()
            # 'PulseFeatures.features.voltage'
        ```

        """
        attrs = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None and (f.name != "node_id" or include_node_id)
        }
        return separator.join(v for v in attrs.values())

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperimentReference:
        """Reconstructs the reference from config."""
        raise NotImplementedError


@dataclass(frozen=True)
class ExperimentNodeReference(ExperimentReference):
    """Reference to an ExperimentNode by label or id."""

    node_label: str | None = None
    node_id: str | None = None

    def resolve(self, ctx: ResolutionContext) -> ExperimentNode:
        """Resolves this reference to a ExperimentNode instance."""
        return super().resolve(ctx=ctx)

    def _resolve_experiment(self, experiment: ExperimentContext) -> ExperimentNode:
        # Prefer node_id resolution if given
        if self.node_id is not None:
            if not experiment.has_node(node_id=self.node_id):
                msg = f"No node exists with ID='{self.node_id}' in the given ExperimentContext."
                raise ResolutionError(msg)
            return experiment.get_node(node_id=self.node_id)

        # Fallback to node label
        if self.node_label is not None:
            if not experiment.has_node(label=self.node_label):
                msg = f"No node exists with label='{self.node_label}' in the given ExperimentContext."
                raise ResolutionError(msg)
            return experiment.get_node(label=self.node_label)

        raise ResolutionError("Both node_label and node_id cannot be None.")

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperimentReference:
        """Reconstructs the reference from config."""
        return cls(**config)


# TODO

# class PhaseReference(ReferenceLike):
#     def resolve(self, ctx: ResolutionContext):
#         if ctx.phase is None:
#             raise ResolutionError("Phase reference requires ExperimentPhase")
#         return self._resolve_phase(ctx.phase)


# # SamplerStreamReference
# # PhaseInputReference
# # RoleReference


# class ExecutionReference(ReferenceLike):
#     def resolve(self, ctx: ResolutionContext):
#         if ctx.execution is None:
#             raise ResolutionError("Execution reference requires ExecutionContext")
#         return self._resolve_execution(ctx.execution)


# # ModelOutputReference
# # BatchRoleOutputReference
# # LossInputReference


# @dataclass(frozen=True)
# class ModelOutputReference(ExecutionReference):
#     node: GlobalReference
#     role: PhaseReference

#     def _resolve_execution(self, exec_ctx):
#         node = self.node.resolve(exec_ctx.ctx)
#         role = self.role.resolve(exec_ctx.ctx)
#         return exec_ctx.outputs[node][role]
