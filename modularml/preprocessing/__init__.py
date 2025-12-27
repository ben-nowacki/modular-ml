# Import registry first
from modularml.core.transforms.scaler_registry import SCALER_REGISTRY

# Import all scaler modules
from .absolute import Absolute
from .negate import Negate
from .per_sample_min_max import PerSampleMinMaxScaler
from .per_sample_zero import PerSampleZeroStart
from .segmented_scaler import SegmentedScaler

# Optionally import sklearn scalers
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)

__all__ = [  # noqa: RUF022
    # classes
    "Absolute",
    "Negate",
    "PerSampleMinMaxScaler",
    "PerSampleZeroStart",
    "SegmentedScaler",
    # sklearn
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "Normalizer",
    "QuantileTransformer",
    "PowerTransformer",
    # registry
    "SCALER_REGISTRY",
]

# Register sklearn scalers
SCALER_REGISTRY.update(
    {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "MaxAbsScaler": MaxAbsScaler,
        "RobustScaler": RobustScaler,
        "Normalizer": Normalizer,
        "QuantileTransformer": QuantileTransformer,
        "PowerTransformer": PowerTransformer,
        "PerSampleMinMaxScaler": PerSampleMinMaxScaler,
        "PerSampleZeroStart": PerSampleZeroStart,
        "Negate": Negate,
        "Absolute": Absolute,
        "SegmentedScaler": SegmentedScaler,
    },
)
