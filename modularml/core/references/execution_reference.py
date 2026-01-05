from typing import Any, TypeAlias, overload

from modularml.context.execution_context import ExecutionContext
from modularml.context.experiment_context import ExperimentContext
from modularml.context.resolution_context import ResolutionContext
from modularml.core.references.experiment_reference import ExperimentReference, ResolutionError

TensorLike: TypeAlias = Any


class ExecutionReference(ExperimentReference):
    """Reference that resolves against a single execution step (forward/backward pass)."""

    @overload
    def resolve(self, ctx: ResolutionContext) -> TensorLike: ...
    @overload
    def resolve(self, ctx: ExecutionContext) -> TensorLike: ...

    def resolve(self, ctx: ResolutionContext | ExecutionContext) -> TensorLike:
        if isinstance(ctx, ResolutionContext):
            if ctx.experiment is None:
                msg = f"{type(self).__name__} requires an ExperimentContext"
                raise ResolutionError(msg)

            if ctx.execution is None:
                msg = f"{type(self).__name__} requires an ExecutionContext"
                raise ResolutionError(msg)

            return self._resolve_execution(ctx=ctx)

        if isinstance(ctx, ExecutionContext):
            return self._resolve_execution(
                ctx=ResolutionContext(
                    experiment=ExperimentContext.get_active(),
                    execution=ctx,
                ),
            )

        msg = f"Context must be either a ResolutionContext or ExecutionContext. Received: {type(ctx)}."
        raise TypeError(msg)

    def _resolve_execution(self, ctx: ResolutionContext):
        raise NotImplementedError
