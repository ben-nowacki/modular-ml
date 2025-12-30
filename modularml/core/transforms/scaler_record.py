from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modularml.core.io.protocols import Configurable
from modularml.core.transforms.scaler import Scaler


@dataclass(frozen=True)
class ScalerRecord(Configurable):
    """
    Metadata describing a single applied scaling transform.

    Stores the minimal information required to:
      - reconstruct transform configuration
      - check dependency ordering
      - undo or invert the transform
      - replay transform history

    Args:
        order (int):
            Monotonically increasing identifier indicating transform order.
        domain (str):
            One of {"features", "targets", "tags"}.
        keys (tuple[str]):
            Column names within the domain that were transformed.
        rep_in (str):
            Input representation name (e.g., "raw" or "transformed").
        rep_out (str):
            Output representation name produced by this transform.
        fit_split (str | None):
            Name of the split used for fitting, or None if fit on all samples.
        merged_axes (tuple[int] | None):
            Axes merged during flattening prior to scaling (if any).
        flatten_meta (dict):
            Metadata required to reverse flattening (e.g., original_shape).
        scaler_obj (Scaler):
            Fitted scaler instance supporting transform() and inverse_transform().
        scaler_artifact_path (str):
            Relative path to fully serialized scaler artifact (relative to FeatureSet artifact).
            E.g., "scalers/scaler_000.sc.mml".

    """

    order: int
    domain: str
    keys: tuple[str]
    rep_in: str
    rep_out: str
    fit_split: str | None
    merged_axes: tuple[int] | None
    flatten_meta: dict

    scaler_obj: Scaler | None = None

    def __eq__(self, other):
        if not isinstance(other, ScalerRecord):
            msg = f"Cannot compare equality between ScalerRecord and {type(other)}"
            raise TypeError(msg)

        return (
            self.order == other.order
            and self.domain == other.domain
            and self.keys == other.keys
            and self.rep_in == other.rep_in
            and self.rep_out == other.rep_out
            and self.fit_split == other.fit_split
            and self.merged_axes == other.merged_axes
            and self.flatten_meta == other.flatten_meta
        )

    __hash__ = None

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return a JSON-serializable configuration.

        Note:
            This config does not include the scaler instance, only the
            `scaler.get_config()` dict.

        Returns:
            dict[str, Any]: Configuration used to reconstruct this record.

        """
        return {
            "order": self.order,
            "domain": self.domain,
            "keys": self.keys,
            "rep_in": self.rep_in,
            "rep_out": self.rep_out,
            "fit_split": self.fit_split,
            "merged_axes": self.merged_axes,
            "flatten_meta": self.flatten_meta,
            "scaler_config": None if self.scaler_obj is None else self.scaler_obj.get_config(),
        }

    @classmethod
    def from_config(cls, config: dict) -> ScalerRecord:
        """
        Reconstructs the record from config.

        This *does not* rebuild the scaler state, only its config.
        """
        scaler_cfg = config.get("scaler_config")
        return cls(
            order=config["order"],
            domain=config["domain"],
            keys=config["keys"],
            rep_in=config["rep_in"],
            rep_out=config["rep_out"],
            fit_split=config["fit_split"],
            merged_axes=config["merged_axes"],
            flatten_meta=config["flatten_meta"],
            scaler_obj=None if scaler_cfg is None else Scaler.from_config(scaler_cfg),
        )
