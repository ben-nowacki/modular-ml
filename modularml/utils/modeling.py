from enum import Enum

import numpy as np

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.utils.backend import Backend


class PadMode(str, Enum):
    """
    Enum representing universal padding modes.

    Description:
        This enum standardizes padding mode names across different backends
        (e.g., PyTorch, TensorFlow, NumPy). Use this to ensure consistent
        behavior regardless of the chosen backend.

    Values:
        CONSTANT: Pad with a constant value.
        REFLECT: Reflect without including edge (mirror).
        REPLICATE: Repeat edge values.
        CIRCULAR: Wrap around (circular padding).
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


def map_pad_mode_to_backend(mode: PadMode, backend: str | Backend) -> str:  # noqa: PLR0911
    """
    Map a universal PadMode to backend-specific string.

    Args:
        mode (PadMode): Universal padding mode.
        backend (str): One of ['torch', 'tensorflow', 'scikit'].

    Returns:
        str: Backend-specific padding mode string.

    Raises:
        ValueError: If mode or backend is unsupported.

    """
    if isinstance(backend, str):
        backend = Backend(backend.lower())

    match (mode, backend):
        case (PadMode.CONSTANT, _):
            return "constant"
        case (PadMode.REFLECT, "torch") | (PadMode.REFLECT, "tensorflow") | (PadMode.REFLECT, "scikit"):
            return "reflect"
        case (PadMode.REPLICATE, "torch"):
            return "replicate"
        case (PadMode.REPLICATE, "tensorflow"):
            return "SYMMETRIC"
        case (PadMode.REPLICATE, "scikit"):
            return "edge"
        case (PadMode.CIRCULAR, "torch"):
            return "circular"
        case (PadMode.CIRCULAR, "scikit"):
            return "wrap"
        case _:
            msg = f"Pad mode {mode} not supported for backend '{backend}'"
            raise ValueError(msg)


def make_dummy_data(shape: tuple[int, ...]) -> Data:
    """
    Creates a dummy `Data` object filled with ones for testing or placeholder use.

    Args:
        shape (tuple[int, ...]): The shape of the data tensor to create.

    Returns:
        Data: A `Data` object containing a tensor of ones with the specified shape.

    """
    # Create dummy data
    d = Data(np.ones(shape=shape))

    return d


def make_dummy_batch(
    feature_shape: tuple[int, ...],
    target_shape: tuple[int, ...] = (1, 1),
    batch_size: int = 8,
) -> Batch:
    """
    Creates a dummy `Batch` object with synthetic samples for testing model components.

    Each sample contains multiple named feature and target entries, and dummy tags.

    Args:
        feature_shape (tuple[int, ...]): Shape of the feature tensor as (n_features, feature_dim).
        target_shape (tuple[int, ...], optional): Shape of the target tensor as (n_targets, target_dim). Defaults to (1, 1).
        batch_size (int, optional): Number of samples in the batch. Defaults to 8.

    Returns:
        Batch: A `Batch` object with randomly generated dummy features, targets, and tags.

    """
    sample_coll = SampleCollection(
        [
            Sample(
                features={f"features_{x}": make_dummy_data(shape=feature_shape[1:]) for x in range(feature_shape[0])},
                targets={f"targets_{x}": make_dummy_data(shape=target_shape[1:]) for x in range(target_shape[0])},
                tags={"tags_1": make_dummy_data(shape=(1,)), "tags_2": make_dummy_data(shape=(1,))},
            )
            for i in range(batch_size)
        ]  # noqa: COM812
    )
    return Batch(
        role_samples={"default": sample_coll},
        label="dummy",
    )
