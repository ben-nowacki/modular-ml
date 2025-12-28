from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Configurable(Protocol):
    """
    Protocol for objects that can be reconstructed from a JSON-serializable config.

    The config must contain only JSON-safe values (dict/list/str/int/float/bool/None).
    """

    def get_config(self) -> dict[str, Any]:
        """
        Return a JSON-serializable configuration for reconstructing the object.

        Returns:
            dict[str, Any]: Configuration used to reconstruct the object.

        """
        ...

    @classmethod
    def from_config(cls, config: dict):
        """
        Construct an object from a configuration dict.

        Args:
            config (dict[str, Any]): Configuration used to construct the object.

        Returns:
            Self: Constructed object.

        """
        ...


@runtime_checkable
class Stateful(Protocol):
    """
    Protocol for objects that can export and restore runtime state.

    State may be JSON-safe OR backend-native; handlers are responsible for encoding.
    """

    def get_state(self) -> Any:
        """
        Return runtime/learned state used to fully reproduce the object.

        Returns:
            Any: Runtime state (handler must know how to encode/decode).

        """
        ...

    def set_state(self, state: Any) -> None:
        """
        Restore runtime/learned state.

        Args:
            state (Any): State object produced by get_state().

        """
        ...
