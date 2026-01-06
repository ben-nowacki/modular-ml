from dataclasses import dataclass

from modularml.context.execution_context import ExecutionContext
from modularml.context.experiment_context import _ACTIVE_EXPERIMENT_CONTEXT, ExperimentContext
from modularml.core.experiment.phase import ExperimentPhase


@dataclass
class ResolutionContext:
    experiment: ExperimentContext | None = _ACTIVE_EXPERIMENT_CONTEXT
    phase: ExperimentPhase | None = None
    execution: ExecutionContext | None = None
