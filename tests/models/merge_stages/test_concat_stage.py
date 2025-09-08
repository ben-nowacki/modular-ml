import numpy as np
import pytest
import tensorflow as tf
import torch

from modularml.models.merge_stages import ConcatStage
from modularml.utils.backend import Backend
from modularml.utils.data_format import to_python
from modularml.utils.modeling import PadMode


# ----------------------------
# Fixtures for backends and data
# ----------------------------
@pytest.fixture(params=[Backend.TORCH, Backend.TENSORFLOW, Backend.SCIKIT])
def backend(request):
    return request.param


@pytest.fixture
def example_inputs(backend):
    if backend == Backend.TORCH:
        return [
            torch.randn(2, 3),
            torch.randn(2, 5),
        ]
    if backend == Backend.TENSORFLOW:
        return [
            tf.random.normal((2, 3)),
            tf.random.normal((2, 5)),
        ]
    return [
        np.random.randn(2, 3),
        np.random.randn(2, 5),
    ]


@pytest.fixture
def example_inputs_with_mismatch(backend):
    if backend == Backend.TORCH:
        return [
            torch.randn(2, 3),
            torch.randn(3, 5),
        ]
    if backend == Backend.TENSORFLOW:
        return [
            tf.random.normal((2, 3)),
            tf.random.normal((3, 5)),
        ]
    return [
        np.random.randn(2, 3),
        np.random.randn(3, 5),
    ]


@pytest.fixture
def example_inputs_with_mismatch_v2(backend):
    if backend == Backend.TORCH:
        return [
            torch.randn(1, 101),
            torch.randn(1, 1),
        ]
    if backend == Backend.TENSORFLOW:
        return [
            tf.random.normal((1, 101)),
            tf.random.normal((1, 1)),
        ]
    return [
        np.random.randn(1, 101),
        np.random.randn(1, 1),
    ]


# ----------------------------
# Basic test: concat without padding
# ----------------------------
def test_concat_no_padding(backend, example_inputs):
    stage = ConcatStage(label="concat", axis=1)
    stage._backend = backend

    out = stage.apply_merge(example_inputs)

    assert out.shape == (2, 8)


def test_concat_no_padding_mismatch_inputs(backend, example_inputs_with_mismatch):
    stage = ConcatStage(label="concat", axis=1)
    stage._backend = backend

    with pytest.raises(ValueError, match="Mismatch in non-concat dimension"):
        out = stage.apply_merge(example_inputs_with_mismatch)


# ----------------------------
# Test: invalid backend raises
# ----------------------------
def test_invalid_backend_raises():
    stage = ConcatStage(label="bad", axis=1)
    stage._backend = "invalid-backend"

    x = np.ones((2, 3))
    with pytest.raises(ValueError, match="not a valid Backend"):
        stage.apply_merge([x, x])


# ----------------------------
# Test: concat with padding
# ----------------------------
def test_concat_axis1_with_constant_padding(backend, example_inputs_with_mismatch):
    stage = ConcatStage(
        label="concat",
        axis=1,
        pad_inputs=True,
        pad_mode=PadMode.CONSTANT,
        pad_value=0.0,
    )
    stage._backend = backend

    pad_dim = max([x.shape[0] for x in example_inputs_with_mismatch])
    concat_dim = sum([x.shape[1] for x in example_inputs_with_mismatch])

    result = stage.apply_merge(example_inputs_with_mismatch)
    assert result.shape == (pad_dim, concat_dim)


def test_concat_axis0_with_constant_padding(backend, example_inputs_with_mismatch_v2):
    # tests that [(1, 101), (1,1)] gets padded and concats to (2,101)

    stage = ConcatStage(
        label="concat",
        axis=0,
        pad_inputs=True,
        pad_mode=PadMode.CONSTANT,
        pad_value=0.0,
    )
    stage._backend = backend

    pad_dim = max([x.shape[1] for x in example_inputs_with_mismatch_v2])
    concat_dim = sum([x.shape[0] for x in example_inputs_with_mismatch_v2])

    result = stage.apply_merge(example_inputs_with_mismatch_v2)
    assert result.shape == (concat_dim, pad_dim)
