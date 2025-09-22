from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modularml.models.base import BaseModel

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.step_result import StepResult
from modularml.core.loss.applied_loss import AppliedLoss
from modularml.utils.backend import Backend


class TrainableMixin:
    @property
    def model(self) -> "BaseModel":
        pass

    @property
    def freeze(self):
        raise NotImplementedError

    @freeze.setter
    def freeze(self, value: bool):
        raise NotImplementedError

    @property
    def backend(self) -> Backend:
        raise NotImplementedError

    def train_step(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss],
    ) -> StepResult:
        raise NotImplementedError


class EvaluableMixin:
    def eval_step(
        self,
        batch_input: dict[str, Batch],
        losses: list[AppliedLoss],
    ) -> StepResult:
        raise NotImplementedError
