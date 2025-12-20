import uuid

import numpy as np

from modularml.core.data.batch import Batch
from modularml.core.data.sample_data import SampleData
from modularml.core.data.sample_shapes import SampleShapes
from modularml.utils.data.formatting import ensure_list


def make_dummy_sample_data(
    batch_size: int,
    feature_shape: tuple[int, ...],
    target_shape: tuple[int, ...] | None = None,
    tag_shape: tuple[int, ...] | None = None,
    seed: int = 1,
):
    """Feature/target/tag shapes must not include batch_dim."""
    rng = np.random.default_rng(seed=seed)
    return SampleData(
        sample_uuids=np.asarray([str(uuid.uuid4()) for _ in range(batch_size)]),
        features=rng.random(batch_size, *feature_shape),
        targets=rng.random(batch_size, *target_shape) if target_shape is not None else None,
        tags=rng.random(batch_size, *tag_shape) if tag_shape is not None else None,
    )


def make_dummy_batch(
    source_label: str,
    feature_shape: tuple[int, ...],
    target_shape: tuple[int, ...] | None = None,
    tag_shape: tuple[int, ...] | None = None,
    batch_size: int = 8,
    role_labels: str | list[str] = "default",
    seed: int = 1,
) -> Batch:
    """
    Creates a dummy `Batch` object with synthetic samples for testing model components.

    Each sample contains multiple named feature and target entries, and dummy tags.

    Args:
        source_label (str):
            Label of source FeatureSet, from which this batch is constructed.
        feature_shape (tuple[int, ..]):
            Tensor shape of feature data, excluding batch dimension.
        target_shape (tuple[int, ..], optional):
            Tensor shape of target data, excluding batch dimension.
        tag_shape (tuple[int, ..], optional):
            Tensor shape of tag data, excluding batch dimension.
        batch_size (int, optional):
            Number of samples in the batch. Must match first dimension of features,
            targets, and tags shapes. Defaults to 8.
        role_labels (str | list[str], optional):
            Number of roles to simulate in the returned Batch. Defaults to only a
            single "default" role.
        seed (int):
            Random generator seed.

    Returns:
        Batch: A `Batch` object with randomly generated dummy features, targets, and tags.

    """
    role_labels = ensure_list(role_labels)
    role_data: dict[str, SampleData] = {
        r: make_dummy_sample_data(
            batch_size=batch_size,
            feature_shape=feature_shape,
            target_shape=target_shape,
            tag_shape=tag_shape,
            seed=seed,
        )
        for r in role_labels
    }
    return Batch(
        batch_size=batch_size,
        outputs={source_label: role_data},
        role_sample_weights={k: np.ones_like(role_data[k].features) for k in role_data},
        shapes={
            source_label: SampleShapes(
                features_shape=(batch_size, *feature_shape),
                targets_shape=(batch_size, *target_shape),
                tags_shape=(batch_size, *tag_shape),
            ),
        },
    )
