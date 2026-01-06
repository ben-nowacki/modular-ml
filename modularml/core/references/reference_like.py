from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modularml.context.resolution_context import ResolutionContext


@runtime_checkable
class ReferenceLike(Protocol):
    """
    Structural interface for reference objects.

    A ReferenceLike resolves into a concrete object given
    a ResolutionContext.
    """

    def resolve(self, ctx: ResolutionContext) -> Any: ...
