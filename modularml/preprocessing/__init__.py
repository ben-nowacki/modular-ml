
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)

from .absolute import Absolute
from .negate import Negate
from .per_sample_min_max import PerSampleMinMaxScaler
from .per_sample_zero import PerSampleZeroStart
from .segmented_scaler import SegmentedScaler

__all__ = [
    "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", 
    "Normalizer", "QuantileTransformer", "PowerTransformer",
    
    "Absolute", "Negate", "PerSampleMinMaxScaler", "PerSampleZeroStart",
    "SegmentedScaler",
]


SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
    "normalize": Normalizer,
    "quantile": QuantileTransformer,
    "power": PowerTransformer,
    
    "sample_minmax": PerSampleMinMaxScaler,
    "sample_zero": PerSampleZeroStart,
    "negate": Negate,
    "absolute": Absolute,
    "segmented": SegmentedScaler,
}