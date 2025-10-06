from collections.abc import Sequence
from typing import Any

import numpy as np

from modularml.core.api import Data, FeatureSet, Sample

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


def generate_dummy_sample(
    feature_shape_map: dict[str, tuple[int, ...]] | None = None,
    target_shape_map: dict[str, tuple[int, ...]] | None = None,
    tag_type_map: dict[str, str] | None = None,
    target_type: str = "numeric",  # "numeric" or "categorical"
) -> Sample:
    """
    Generate a synthetic :class:`Sample` object for testing.

    This utility creates dummy features, targets, and tags with configurable
    shapes and data types. Useful for unit tests of FeatureSets, samplers, and
    splitters without needing real datasets.

    Args:
        feature_shape_map (dict[str, tuple[int, ...]], optional):
            Mapping of feature names to their shapes.
            Defaults to {"X1": (1, 100), "X2": (1, 100)}.
        target_shape_map (dict[str, tuple[int, ...]], optional):
            Mapping of target names to their shapes.
            Defaults to {"Y1": (1, 1), "Y2": (1, 10)}.
        tag_type_map (dict[str, str], optional):
            Mapping of tag names to data types (e.g., {"T_FLOAT": "float", "T_STR": "str"}).
            Defaults to one float tag and one string tag.
        target_type (str, optional):
            Type of targets to generate:
            - `"numeric"` → continuous float values (default).
            - `"categorical"` → categorical string labels from {"A", "B", "C"}.

    Returns:
        Sample:
            A dummy sample containing randomly generated features, targets, and tags.

    Raises:
        ValueError:
            If `target_type` is not one of {"numeric", "categorical"}.

    Examples:
        >>> s = generate_dummy_sample()
        >>> list(s.features.keys())
        ['X1', 'X2']
        >>> list(s.targets.keys())
        ['Y1', 'Y2']
        >>> s.tags
        {'T_FLOAT': Data(...), 'T_STR': Data(...)}

        >>> s_cat = generate_dummy_sample(target_type="categorical")
        >>> s_cat.targets["Y1"]  # contains string labels like "A", "B", "C"

    """
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

    # generate features
    features = {k: Data(generate_dummy_data(shape=v, dtype="float")) for k, v in feature_shape_map.items()}

    # generate targets (numeric or categorical)
    if target_type == "numeric":
        targets = {k: Data(generate_dummy_data(shape=v, dtype="float")) for k, v in target_shape_map.items()}
    elif target_type == "categorical":
        # categorical → pick from finite string/int labels
        targets = {
            k: Data(generate_dummy_data(shape=v, dtype="str", choices=["A", "B", "C"]))
            for k, v in target_shape_map.items()
        }
    else:
        msg = f"Unsupported target_type: {target_type}"
        raise ValueError(msg)

    # generate tags
    tags = {
        k: Data(generate_dummy_data(shape=(1,), dtype=v, choices=(["red", "blue", "green"] if v == "str" else None)))
        for k, v in tag_type_map.items()
    }

    return Sample(features=features, targets=targets, tags=tags)


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

    This function wraps :func:`generate_dummy_sample` to create a collection of
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
    samples = []
    for _ in range(n_samples):
        samples.append(
            generate_dummy_sample(
                feature_shape_map=feature_shape_map,
                target_shape_map=target_shape_map,
                tag_type_map=tag_type_map,
                target_type=target_type,
            ),
        )
    return FeatureSet(label=label, samples=samples)
