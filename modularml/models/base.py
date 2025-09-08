from abc import ABC, abstractmethod

from modularml.utils.backend import Backend


class BaseModel(ABC):
    def __init__(self, backend: Backend):
        super().__init__()
        self._backend = backend
        self._built = False

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...] | None:
        """
        Shape of the expected input to this mode.

        Returns:
            tuple[int, ...] | None: Input shape tuple (e.g., (32, 64)) or None if unknown.

        """

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...] | None:
        """
        Shape of the output produced by the model.

        Returns:
            tuple[int, ...] | None: Output shape tuple (e.g., (32, 10)) or None if unknown.

        """

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def is_built(self) -> bool:
        return self._built

    @abstractmethod
    def build(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
    ):
        """Build the internal model layers given an input shape."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Run a forward pass."""
