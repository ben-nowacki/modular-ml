from __future__ import annotations

from dataclasses import dataclass

from modularml.core.transforms.scaler import Scaler


@dataclass
class ScalerRecord:
    """
    Metadata describing a single applied scaling transform.

    Stores the minimal information required to:
      - reconstruct transform configuration
      - check dependency ordering
      - undo or invert the transform
      - replay transform history

    Args:
        order : int
            Monotonically increasing identifier indicating transform order.
        domain : str
            One of {"features", "targets", "tags"}.
        keys : tuple[str]
            Column names within the domain that were transformed.
        variant_in : str
            Input variant name (e.g., "raw" or "transformed").
        variant_out : str
            Output variant name produced by this transform.
        fit_split : str | None
            Name of the split used for fitting, or None if fit on all samples.
        merged_axes : tuple[int] | None
            Axes merged during flattening prior to scaling (if any).
        flatten_meta : dict
            Metadata required to reverse flattening (e.g., original_shape).
        scaler_object : Scaler
            Fitted scaler instance supporting transform() and inverse_transform().

    """

    order: int
    domain: str
    keys: tuple[str]
    variant_in: str
    variant_out: str
    fit_split: str | None
    merged_axes: tuple[int] | None
    flatten_meta: dict
    scaler_object: Scaler

    # ==========================================
    # SerializableMixin
    # ==========================================
    def get_state(self) -> dict:
        """
        Full serializable state including fitted parameters.

        Include scaler's fitted state, flattening metadata, and applied \
        specification, and ordering information.
        """
        return {
            "version": "1.0",
            "order": self.order,
            "domain": self.domain,
            "keys": self.keys,
            "variant_in": self.variant_in,
            "variant_out": self.variant_out,
            "fit_split": self.fit_split,
            "merged_axes": self.merged_axes,
            "flatten_meta": self.flatten_meta,
            "scaler_state": self.scaler_object.get_state(),
        }

    def set_state(self, state: dict) -> None:
        """Restore the full ScalerRecord state (including Scaler state)."""
        if state.get("version") != "1.0":
            msg = f"Unsupported version: {state.get('version')}"
            raise NotImplementedError(msg)

        self.order = state["order"]
        self.domain = state["domain"]
        self.keys = tuple(state["keys"])
        self.variant_in = state["variant_in"]
        self.variant_out = state["variant_out"]
        self.fit_split = state["fit_split"]
        self.merged_axes = state["merged_axes"]
        self.flatten_meta = state["flatten_meta"]

        # scaler reconstruction
        self.scaler_object = Scaler.from_state(state["scaler_state"])
