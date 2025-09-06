import numpy as np

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.data import Data
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection


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
    feature_shape: tuple[int, ...], target_shape: tuple[int, ...] = (1, 1), batch_size: int = 8
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
        ]
    )
    return Batch(
        role_samples={"default": sample_coll},
        label="dummy",
    )
