from collections.abc import Sequence
from typing import Any

import numpy as np

from modularml.core.data.featureset import FeatureSet

rng = np.random.default_rng(seed=13)


def generate_dummy_data(
    shape: tuple[int, ...] = (),
    dtype: str = "float",
    *,
    min_val: Any = None,
    max_val: Any = None,
    choices: Sequence[Any] | None = None,
) -> float | int | str | np.ndarray:
    """
    Generate dummy scalar or array data of specified type.

    Args:
        shape (tuple): Shape of the output.
            - () or (1,) -> scalar
            - (n,), (m,n) -> ndarray
        dtype (str): One of {"float", "int", "str"}.
        min_val (Any, optional): Minimum value (for numeric ranges).
        max_val (Any, optional): Maximum value (for numeric ranges).
        choices (Sequence, optional): Literal list of values to sample from.

    Returns:
        Union[float, int, str, np.ndarray]: Generated value(s).

    """
    size = int(np.prod(shape)) if shape else 1

    # Handle string case separately
    if dtype == "str":
        if not choices:
            raise ValueError("For dtype='str', please provide `choices` list.")
        values = rng.choice(choices, size=size)
    elif choices is not None:
        # pick from literal numeric choices
        values = rng.choice(choices, size=size)

    elif dtype == "float":
        lo = min_val if min_val is not None else 0.0
        hi = max_val if max_val is not None else 1.0
        values = rng.uniform(lo, hi, size)
    elif dtype == "int":
        lo = min_val if min_val is not None else 0
        hi = max_val if max_val is not None else 10
        values = rng.integers(lo, hi + 1, size)
    else:
        msg = f"Unsupported dtype: {dtype}"
        raise ValueError(msg)

    # Scalar vs array formatting
    if not shape or shape == (1,):
        return values[0]
    return np.array(values).reshape(shape)


def generate_dummy_featureset(
    feature_shape_map: dict[str, tuple[int, ...]] | None = None,
    target_shape_map: dict[str, tuple[int, ...]] | None = None,
    tag_type_map: dict[str, str] | None = None,
    target_type: str = "numeric",
    n_samples: int = 1000,
    label: str = "TestFeatureSet",
) -> FeatureSet:
    """
    Generate a synthetic :class:`FeatureSet` populated with dummy samples.

    This function wraps :func:`generate_dummy_data` to create a collection of
    random samples for testing higher-level abstractions (e.g., samplers,
    splitters, training pipelines) without requiring real datasets.

    Args:
        feature_shape_map (dict[str, tuple[int, ...]], optional):
            Mapping of feature names to shapes.
            Defaults to {"X1": (1, 100), "X2": (1, 100)}.
        target_shape_map (dict[str, tuple[int, ...]], optional):
            Mapping of target names to shapes.
            Defaults to {"Y1": (1, 1), "Y2": (1, 10)}.
        tag_type_map (dict[str, str], optional):
            Mapping of tag names to data types (e.g., {"T_FLOAT": "float", "T_STR": "str"}).
            Defaults to one float tag and one string tag.
        target_type (str, optional):
            Type of targets to generate:
            - `"numeric"` → continuous float values (default).
            - `"categorical"` → categorical string labels from {"A", "B", "C"}.
        n_samples (int, optional):
            Number of samples to generate. Defaults to 1000.
        label (str, optional): FeatureSet name. Defaults to "TestFeatureSet".

    Returns:
        FeatureSet:
            A dummy FeatureSet containing `n_samples` randomly generated
            :class:`Sample` objects.

    Examples:
        >>> fs = generate_dummy_featureset(n_samples=10)
        >>> len(fs)
        10
        >>> fs.label
        'TestFeatureSet'
        >>> list(fs.features.keys())[:2]
        ['X1', 'X2']

    """
    # Get column data shapes (this doesn't not include number of samples)
    feature_shape_map = feature_shape_map or {
        "X1": (1, 100),
        "X2": (1, 100),
    }
    target_shape_map = target_shape_map or {
        "Y1": (1, 1),
        "Y2": (1, 10),
    }
    tag_type_map = tag_type_map or {
        "T_FLOAT": "float",
        "T_STR": "str",
    }

    # Add sample_len dimension
    for k in feature_shape_map:
        feature_shape_map[k] = (n_samples, *feature_shape_map[k])
    for k in target_shape_map:
        target_shape_map[k] = (n_samples, *target_shape_map[k])

    # Generate feature data
    features = {k: generate_dummy_data(shape=v, dtype="float") for k, v in feature_shape_map.items()}

    # Generate target data (numeric or categorical)
    if target_type == "numeric":
        targets = {k: generate_dummy_data(shape=v, dtype="float") for k, v in target_shape_map.items()}
    elif target_type == "categorical":
        # categorical → pick from finite string/int labels
        targets = {
            k: generate_dummy_data(shape=v, dtype="str", choices=["A", "B", "C"]) for k, v in target_shape_map.items()
        }
    else:
        msg = f"Unsupported target_type: {target_type}"
        raise ValueError(msg)

    # Generate tag data
    tags = {
        k: generate_dummy_data(shape=(n_samples,), dtype=v, choices=(["red", "blue", "green"] if v == "str" else None))
        for k, v in tag_type_map.items()
    }

    # Create FeatureSet
    return FeatureSet.from_dict(
        label=label,
        data=features | targets | tags,
        feature_keys=list(feature_shape_map.keys()),
        target_keys=list(target_shape_map.keys()),
        tag_keys=list(tag_type_map.keys()),
    )


if __name__ == "__main__":
    fs = generate_dummy_featureset()
    print(fs)
    sc = fs.get_collection("original")
    print(sc.feature_keys, sc.feature_shapes, sc.feature_dtypes)
    print(sc.target_keys, sc.target_shapes, sc.target_dtypes)
    print(sc.tag_keys, sc.tag_shapes, sc.tag_dtypes)
