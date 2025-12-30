from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any, ClassVar, get_args

from modularml.core.data.schema_constants import (
    DOMAIN_SAMPLE_ID,
    INVALID_LABEL_CHARACTERS,
    REP_RAW,
    T_ALL_DOMAINS,
    T_ALL_REPS,
)
from modularml.core.experiment.experiment_context import ExperimentContext
from modularml.core.io.protocols import Configurable


@dataclass(frozen=True)
class DataReference(Configurable):
    """
    Canonical reference to any data or computation output in a ModularML Experiment.

    Description:
        A `DataReference` uniquely identifies a data source or tensor-like output \
        within an experiment. This unified reference type can describe:
          - FeatureSet data (`node`, `collection`, `domain`, `split`, `key`)
          - Training data views or folds (`stage`, `split`, `fold`, `role`)
          - Sampler batches (`batch`)
          - Computation node outputs (`domain='output'`, `stage`)

        All references are hierarchical and serialized as a compact, human-readable \
        path string.
        Each component is optional, so references remain flexible but structured.

    Notes:
        - Field values cannot contain the global separator (default '.') or any of \
          `INVALID_LABEL_CHARACTERS`.
        - All components are optional, but certain combinations are meaningful only \
          for specific data types (e.g., `collection` applies only to FeatureSets).

    """

    separator: ClassVar[str] = "."

    # Experiment fields
    stage: str | None = None

    # ModelGraph fields
    node: str | None = None
    node_id: str | None = None

    # FeatureSet fields
    domain: T_ALL_DOMAINS | None = None
    key: str | None = None
    rep: T_ALL_REPS | None = None

    # Splitter field
    split: str | None = None
    fold: str | None = None

    # Sampler fields
    role: str | None = None
    batch: str | None = None

    # ==========================================
    # Class Methods
    # ==========================================
    @classmethod
    def from_string(
        cls,
        x: str,
        known_attrs: dict[str, str] | None = None,
        required_attrs: list[str] | None = None,
    ) -> DataReference:
        """
        Attempts to construct a DataReference from a string.

        Description:
            Given a string representation, and context of the current active Experiment,
            a DataReference instance is constructed.

        Args:
            x (str): The string to parse.
            known_attrs (dict[str, str], optional): Known attributes can be provided. They will be \
                used to supplement any attributes parsed from `x`.
            required_attrs (list[str], optional): Can specify the minimum attributes that must be \
                parseable. The order does not matter.

        """
        # Get global ExperimentContext and active Experiment (if exists)
        ctx = ExperimentContext
        # exp = ctx.get_active()

        # Split by separator
        parts = x.split(cls.separator)

        # Load known fields (nodes, stages, etc) from ExperimentContext
        known_nodes = set(ctx.available_nodes())
        known_stages = set(ctx.available_stages())
        known_domains = set(get_args(T_ALL_DOMAINS))  # {"features","targets","tags","sample_id","output"}
        known_reps = set(get_args(T_ALL_REPS))  # {"raw","transformed"}

        # Storage for final parsing
        all_attrs = ["stage", "node", "node_id", "domain", "key", "rep", "split", "fold", "role", "batch"]
        parsed = dict.fromkeys(all_attrs)

        # Load with known attributes (if any)
        if known_attrs is not None:
            for k, v in known_attrs.items():
                parsed[k] = v

        # First pass: search for known values (node, stage, domain)
        unmatched = []
        for p in parts:
            if p in known_nodes:
                if parsed["node"] is not None and parsed["node"] != p:
                    msg = f"Multiple node matches found in '{x}'."
                    raise ValueError(msg)
                parsed["node"] = p
            elif p in known_stages:
                if parsed["stage"] is not None and parsed["stage"] != p:
                    msg = f"Multiple stage matches found in '{x}'."
                    raise ValueError(msg)
                parsed["stage"] = p
            elif p in known_domains:
                if parsed["domain"] is not None and parsed["domain"] != p:
                    msg = f"Multiple domain matches found in '{x}'."
                    raise ValueError(msg)
                parsed["domain"] = p
            elif p in known_reps:
                if parsed["rep"] is not None and parsed["rep"] != p:
                    msg = f"Multiple representation matches found in '{x}'."
                    raise ValueError(msg)
                parsed["rep"] = p
            else:
                unmatched.append(p)

        # If no matches, cannot parse
        if all(v is None for v in parsed.values()):
            raise RuntimeError("Failed to parse string.")

        # Before trying a static alignment with the highest priority match,
        # check if the node if provided, and corresponds to a FeatureSet
        # If so, we can extract extra information from `ctx`
        if parsed["node"] is not None and ctx.node_is_featureset(parsed["node"]):
            # Get all FeatureSet keys (eg, ["sample_id", "features.voltage.raw", ....])
            # Each element (except "sample_id") is formatted as domain.key.rep
            all_fs_keys: list[str] = ctx.get_all_featureset_keys(parsed["node"])
            all_fs_keys.remove(DOMAIN_SAMPLE_ID)
            # Split into parts -> list[tuple[domain, key, rep]]
            all_fs_keys: list[tuple[str, str, str]] = [x.split(".") for x in all_fs_keys]

            # Check if an unmatch part could be a key
            for unm in unmatched:
                if unm in [x[1] for x in all_fs_keys]:
                    parsed["key"] = unm
                    unmatched.remove(unm)
                    break

            # Check if we know the key
            if parsed["key"] is not None:
                # Find candidates (index 1 of all_fs_keys)
                candidates: list[tuple[str, str, str]] = [x for x in all_fs_keys if x[1] == parsed["key"]]

                # Infer `domain` if needed
                if parsed["domain"] is None:
                    # Ensure all domains of candidates match
                    cand_domains: list[str] = [x[0] for x in candidates]
                    if len(set(cand_domains)) == 1:
                        parsed["domain"] = cand_domains[0]

                # Infer `rep` if needed
                if parsed["rep"] is None:
                    # Take match if only 1 rep exists
                    cand_reps: list[str] = [x[2] for x in candidates]
                    if len(set(cand_reps)) == 1:
                        parsed["rep"] = cand_reps[0]
                    # Otherwise, we will assume the default rep (ie, "raw")
                    elif REP_RAW in cand_reps:
                        parsed["rep"] = REP_RAW
                        k = parsed["key"]
                        msg = (
                            f"The DataReference `key` ('{k}') has several possible `reps`: {cand_reps}. "
                            f"The default representation ('{REP_RAW}') is being selected."
                        )
                        warnings.warn(msg, category=UserWarning, stacklevel=2)

        # All `required_attrs` must be defined
        if parsed["node"] is not None:
            parsed["node_id"] = ctx.get_node(label=parsed["node"]).node_id
        if required_attrs:
            for k in required_attrs:
                if parsed[k] is None:
                    msg = f"Failed to parse string. Could not determine required attribute '{k}'"
                    raise ValueError(msg)

        # Create ref with known/parsed fields (rest are set to None)
        ref = DataReference(**parsed)

        # Validate that the parsed fields make sense in the current context
        ctx.validate_data_ref(ref, check_featuresets=True)

        return ref

    @classmethod
    def set_separator(cls, sep: str) -> None:
        """Globally change the separator for all DataReference subclasses."""
        if len(sep) != 1:
            raise ValueError("Separator must be a single character.")
        cls.separator = sep

    @classmethod
    def get_separator(cls) -> str:
        """Return the global separator character."""
        return cls.separator

    # ==========================================
    # Validation
    # ==========================================
    def validate_fields(self):
        """Validate that all fields contain legal characters and values."""
        # Ensure `domain` is an allowed string
        if self.domain is not None and self.domain not in get_args(T_ALL_DOMAINS):
            msg = f"`{self.domain}` must be one of {get_args(T_ALL_DOMAINS)}"
            raise ValueError(msg)

        # Ensure no fields contains the separator character
        invalid_chars = INVALID_LABEL_CHARACTERS | set(type(self).separator)
        for fld in fields(DataReference):
            val = getattr(self, fld.name)
            if any((isinstance(val, str) and x in val) for x in invalid_chars):
                msg = (
                    f"Invalid character in field '{fld.name}' value '{val}'. "
                    f"Fields cannot contain any of the following: {invalid_chars}"
                )
                raise ValueError(msg)

    def __post_init__(self):
        self.validate_fields()

    # ==========================================
    # Representation
    # ==========================================
    def to_string(self, separator: str = ".") -> str:
        """
        Joins all non-null fields into a single string.

        Example:
        ``` python
            ref = DataReference(node='PulseFeatures', domain='features', key='voltage')
            ref.to_string()
            # 'PulseFeatures.features.voltage'
        ```

        """
        attrs = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None and f.name != "node_id"
        }
        return separator.join(v for v in attrs.values())

    def __str__(self) -> str:
        """
        Return a readable string representation for console or logs.

        Example:
            DataReference(node='PulseFeatures', split='train', domain='features', key='voltage')

        """
        attrs = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None and f.name != "node_id"
        }
        attr_str = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{self.__class__.__name__}({attr_str})"

    __repr__ = __str__

    def summary(self, *, include_none: bool = False) -> str:
        """
        Generate a multi-line summary for use in Jupyter or debugging.

        Example:
            >>> ref.summary()
            ┌─ DataReference ─────────────────────────────┐
            node        : PulseFeatures
            split       : train
            domain      : features
            key         : voltage
            └───────────────────────────────────────────┘

        """
        rows = []
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None or include_none:
                rows.append(f"{f.name:<12}: {val}")
        width = max(len(r) for r in rows) if rows else 0
        border = "─" * width
        body = "\n".join(rows)
        return f"┌─ {self.__class__.__name__} {'─' * (width - len(self.__class__.__name__) - 4)}┐\n{body}\n└{border}┘"

    # ================================================
    # Configurable
    # ================================================
    def get_config(self) -> dict[str, Any]:
        """
        Return a JSON-serializable configuration.

        Returns:
            dict[str, Any]: Configuration used to reconstruct this reference.

        """
        config = self.__dict__.copy()

        # Ensure node_id and node label references are accurate at time of state save
        if self.node_id is not None:
            config["node"] = ExperimentContext.get_node(node_id=self.node_id).label

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DataReference:
        """Reconstructs the reference from config."""
        return cls(**config)
