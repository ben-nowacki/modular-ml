from dataclasses import dataclass
from typing import Literal

from modularml.context.resolution_context import ResolutionContext
from modularml.core.data.schema_constants import DOMAIN_FEATURES
from modularml.core.references.execution_reference import ExecutionReference
from modularml.core.references.experiment_reference import ResolutionError


@dataclass(frozen=True)
class ModelOutputReference(ExecutionReference):
    # ModeNode-specifiers
    node_label: str
    node_id: str

    # IO-specifiers
    role: str | None = None
    domain: Literal["outputs"] = "outputs"

    def _resolve_execution(self, ctx: ResolutionContext):
        exec_ctx = ctx.execution

        # Access RoleData outputs by this ModeNode
        batch_output = exec_ctx.model_outputs[self.node_id]

        role = self.role
        if role is None:
            if len(batch_output.available_roles) != 1:
                msg = (
                    "ModelOutputReference must specify a `role` when multiple "
                    f"roles exist in the output data. Available roles: {batch_output.available_roles}."
                )
                raise ResolutionError(msg)
            role = batch_output.available_roles[0]

        # Get output data (domain=outputs)
        return batch_output.get_data(
            role=role,
            domain=self.domain,
        )


@dataclass(frozen=True)
class ModelInputReference(ExecutionReference):
    # ModeNode-specifiers
    node_label: str
    node_id: str

    # IO-specifiers
    role: str | None = None

    def _resolve_execution(self, ctx: ResolutionContext):
        exec_ctx = ctx.execution

        # Access Batch input for this ModeNode
        batch_input = exec_ctx.input_batches[self.node_id]

        role = self.role
        if role is None:
            if len(batch_input.available_roles) != 1:
                msg = (
                    "ModelInputReference must specify a `role` when multiple "
                    f"roles exist in the input data. Available roles: {batch_input.available_roles}."
                )
                raise ResolutionError(msg)
            role = batch_input.available_roles[0]

        # Get input data (domain=features)
        return batch_input.get_data(
            role=role,
            domain=DOMAIN_FEATURES,
        )
