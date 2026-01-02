from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modularml.utils.data.comparators import deep_equal
from modularml.utils.environment.optional_imports import ensure_tensorflow, ensure_torch
from modularml.utils.errors.exceptions import BackendNotSupportedError, LossError
from modularml.utils.nn.backend import Backend, infer_backend, normalize_backend

if TYPE_CHECKING:
    from collections.abc import Callable


def _safe_infer_backend(obj_or_cls: Any) -> Backend:
    backend = infer_backend(obj_or_cls=obj_or_cls)

    if backend == Backend.NONE:
        raise ValueError("Could not infer backend from loss class. Specify backend explicitly.")

    return backend


class Loss:
    """
    A backend-agnostic wrapper around loss functions used in model training.

    This class allows the use of built-in loss functions from supported backends (PyTorch, TensorFlow)
    or custom-defined loss functions (e.g., scikit-learn, numpy-based) and ensures compatibility with
    the modular training workflow.
    """

    def __init__(
        self,
        loss: str | type | Callable | None = None,
        *,
        loss_kwargs: dict[str, Any] | None = None,
        backend: Backend | None = None,
        reduction: str = "none",
        factory: Callable | None = None,
    ):
        # Supported initialization modes:
        # 1. Name + backend
        # 2. Loss class
        # 3. Callable loss function
        # 4. Callable factory
        if loss is not None and factory is not None:
            raise ValueError("Provide either a loss fnc/cls/name (`loss`) or a factory `factory`, not both.")

        # Runtime attributes
        self.name: str | None = None  # name of importable class (eg, torch "MSELoss")
        self.cls: type | None = None  # loss class (eg, torch.nn.MSELoss)
        self.fn: Callable | None = None  # loss function (callable)
        self._factory: Callable | None = factory  # factory to generate a loss class during __call__

        self.reduction = reduction  # reduction argument to pass during class construction
        self.kwargs = loss_kwargs  # other kwargs to pass to class construction
        self._backend: Backend | None = normalize_backend(backend) if backend else None
        self._callable: Callable | None = None  # built callable -> this is used during __call__

        # Case 1: loss name
        if isinstance(loss, str):
            self.name = loss.lower()
            if backend is None:
                raise ValueError("Backend must be specified when initializing a loss with a string-name.")
            self._backend = normalize_backend(backend)
            self.cls = self._resolve()
            self.kwargs = loss_kwargs or {}

            self.fn = None
            self._factory = None
            self._callable = None

        # Case 2: loss class
        elif isinstance(loss, type):
            self.cls = loss
            self.name = loss.__name__.lower()
            self._backend = _safe_infer_backend(loss)

            self.kwargs = loss_kwargs or {}
            self.fn = None
            self._factory = None
            self._callable = None

        # Case 3: callable loss function/instance
        elif callable(loss):
            self.fn = loss
            self.name = loss.__name__
            self._backend = _safe_infer_backend(loss)

            self.kwargs = loss_kwargs or {}
            self._factory = None
            self._callable = None

        # Case 4: factory
        elif factory is not None:
            if backend is None:
                raise ValueError("Backend must be specified when using loss factory.")
            self._backend = normalize_backend(backend)

        else:
            msg = "Loss must be initialized with name, class, callable, or factory."
            raise LossError(msg)

    def __eq__(self, other):
        if not isinstance(other, Loss):
            msg = f"Cannot compare equality between Loss and {type(other)}"
            raise TypeError(msg)

        # Compare config
        if not deep_equal(self.get_config(), other.get_config()):
            return False

        if hasattr(self, "get_state"):
            return deep_equal(self.get_state(), other.get_state())

        return True

    __hash__ = None

    # ================================================
    # Internal helpers
    # ================================================
    def _resolve(self) -> Callable:
        """
        Resolve a named loss to its backend-specific class by introspection.

        Strategy:
            1. Determine backend-specific loss module path
            2. Inspect all classes defined in that module
            3. Match class name case-insensitively against `self.name`
        """
        if not isinstance(self.name, str):
            raise LossError("Loss name must be a string to resolve dynamically.")

        name_lc = self.name.lower()

        # Resolve backend optimizer module
        if self.backend == Backend.TORCH:
            torch = ensure_torch()
            module = torch.nn
            keywords = ["loss"]
        elif self.backend == Backend.TENSORFLOW:
            tf = ensure_tensorflow()
            module = tf.keras.losses
            keywords = []
        else:
            raise BackendNotSupportedError(backend=self.backend, method="Loss._resolve()")

        # Inspect available classes
        candidates: dict[str, type] = {}
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
            except Exception:  # noqa: BLE001, S112
                continue
            if not isinstance(attr, type):
                continue

            # Match class names ignoring case
            attr_key = attr_name.lower()
            candidates[attr_key] = attr
            for _k in keywords:
                if _k in attr_key:
                    _attr_key = attr_key.replace(_k, "")
                    candidates[_attr_key] = attr

        # Resolve loss
        loss_cls = candidates.get(name_lc)
        if loss_cls is None:
            available = sorted(candidates.keys())
            msg = f"Unknown loss name '{self.name}' for backend '{self.backend}'. Available losses: {available}"
            raise LossError(msg)

        return loss_cls

    # ================================================
    # Properties
    # ================================================
    @property
    def allowed_keywords(self) -> list[str]:
        """
        Returns the list of valid keyword arguments for the current loss function.

        Returns:
            List[str]: A list of argument names accepted by the loss function.

        """
        if self._callable is None:
            msg = "Loss callable has not been built yet."
            raise RuntimeError(msg)

        # Get the signature object
        sig = inspect.signature(self._callable)
        # Iterate through the parameters in the signature
        arg_names = [param.name for param in sig.parameters.values()]
        return arg_names

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            raise LossError("Loss backend has not been resolved.")
        return self._backend

    @property
    def is_built(self) -> bool:
        return self._callable is not None

    # ================================================
    # Build
    # ================================================
    def build(
        self,
        *,
        backend: Backend | None = None,
        force_rebuild: bool = False,
        **kwargs,
    ):
        """
        Instantiate the loss callable if not already provided.

        Args:
            backend (Backend | None):
                Backend to use for loss, if not already set.
            force_rebuild (bool, optional):
                Whether to force rebuild loss.
            kwargs:
                Additional keyword-arguments to use in loss class
                instantiation.

        """
        if self.is_built and not force_rebuild:
            raise LossError(
                message=(
                    "Loss.built() is being called on an already instantiated loss. "
                    "If you want to rebuild the loss, set `force_rebuild=True`."
                ),
            )

        # Set/validate backend
        if backend is not None:
            if self.backend is not None and backend != self.backend:
                msg = f"Backend passed to Loss.build differs from backend at init: {backend} != {self.backend}"
                raise ValueError(msg)
            self.backend = backend
        if self.backend is None:
            raise ValueError("Backend must be set before building loss.")

        # Update kwargs
        self.kwargs |= kwargs

        # Cases 1-2: resolved class
        if self.cls is not None:
            if self.backend == Backend.TORCH:  # noqa: SIM114
                self._callable = self.cls(reduction=self.reduction, **self.kwargs)
            elif self.backend == Backend.TENSORFLOW:
                self._callable = self.cls(reduction=self.reduction, **self.kwargs)
            else:
                raise BackendNotSupportedError(self.backend, "Loss.build")
            return

        # Case 2: callable loss function
        if self.fn is not None:
            self._callable = self.fn
            return

        # Case 3: factory
        if self._factory is not None:
            self._callable = self._factory(**self.kwargs)
            return

    # ================================================
    # Callable
    # ================================================
    def __call__(self, *args, **kwargs):
        """
        Executes the underlying loss callable with given arguments.

        Returns:
            Any: Output of the loss callable.

        Raises:
            LossError: If the loss callable fails during execution.

        """
        if not self.is_built:
            self.build()

        try:
            return self._callable(*args, **kwargs)
        except Exception as e:
            msg = f"Loss execution failed: {e}"
            raise LossError(msg) from e

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """Get config of this loss."""
        name = None if self.name is None else str(self.name).lower()
        return {
            "loss": self.fn or name or None,
            "loss_kwargs": self.kwargs,
            "backend": None if self.backend is None else str(self.backend.value).lower(),
            "reduction": self.reduction,
            "factory": self._factory,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Loss:
        return cls(**config)

    # ================================================
    # Representation
    # ================================================
    def __repr__(self):
        return f"Loss(name={self.name}, backend={self._backend})"

    # ================================================
    # Serialization
    # ================================================
    def save(self, filepath: Path, *, overwrite: bool = False) -> Path:
        """
        Serializes this Loss to the specified filepath.

        Args:
            filepath (Path):
                File location to save to. Note that the suffix may be overwritten
                to enforce the ModularML file extension schema.
            overwrite (bool, optional):
                Whether to overwrite any existing file at the save location.
                Defaults to False.

        Returns:
            Path: The actual filepath to write the Loss is saved.

        """
        from modularml.core.io.serialization_policy import SerializationPolicy
        from modularml.core.io.serializer import serializer

        return serializer.save(
            self,
            filepath,
            policy=SerializationPolicy.BUILTIN,
            overwrite=overwrite,
        )

    @classmethod
    def load(cls, filepath: Path, *, allow_packaged_code: bool = False) -> Loss:
        """
        Load a Loss from file.

        Args:
            filepath (Path):
                File location of a previously saved Loss.
            allow_packaged_code : bool
                Whether bundled code execution is allowed.

        Returns:
            Loss: The reloaded optimizer.

        """
        from modularml.core.io.serializer import _enforce_file_suffix, serializer

        # Append proper sufficx only if no suffix is given
        if Path(filepath).suffix == "":
            filepath = _enforce_file_suffix(path=filepath, cls=cls)

        return serializer.load(filepath, allow_packaged_code=allow_packaged_code)
