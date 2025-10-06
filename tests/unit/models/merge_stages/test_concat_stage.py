import numpy as np
import pytest
import tensorflow as tf
import torch

from modularml.core.graph.merge_stages import ConcatStage
from modularml.utils.backend import Backend
from modularml.utils.modeling import PadMode


# ----------------------------
# Fixtures for backends and data
# ----------------------------
@pytest.fixture(params=[Backend.TORCH, Backend.TENSORFLOW, Backend.SCIKIT])
def backend(request):
    return request.param


@pytest.fixture
def example_inputs(rng, backend):
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
        rng.normal(size=(2, 3)),
        rng.normal(size=(2, 5)),
    ]


@pytest.fixture
def example_inputs_with_mismatch(rng, backend):
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
        rng.normal(size=(2, 3)),
        rng.normal(size=(3, 5)),
    ]


@pytest.fixture
def example_inputs_with_mismatch_v2(rng, backend):
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
        rng.normal(size=(1, 101)),
        rng.normal(size=(1, 1)),
    ]


# ----------------------------
# Basic test: concat without padding
# ----------------------------
def test_concat_no_padding(backend, example_inputs):
    stage = ConcatStage(label="concat", axis=1, upstream_nodes=["dummy1", "dummy2"])
    stage._backend = backend

    out = stage.apply_merge(example_inputs, includes_batch_dim=False)

    assert out.shape == (2, 8)


def test_concat_no_padding_mismatch_inputs(backend, example_inputs_with_mismatch):
    stage = ConcatStage(label="concat", axis=1, upstream_nodes=["dummy1", "dummy2"])
    stage._backend = backend

    with pytest.raises(ValueError, match="Mismatch in non-concat dimension"):
        out = stage.apply_merge(example_inputs_with_mismatch, includes_batch_dim=False)


# ----------------------------
# Test: invalid backend raises
# ----------------------------
def test_invalid_backend_raises():
    stage = ConcatStage(label="bad", axis=1, upstream_nodes=["dummy1", "dummy2"])
    stage._backend = "invalid-backend"

    x = np.ones((2, 3))
    with pytest.raises(ValueError, match="not a valid Backend"):
        stage.apply_merge([x, x], includes_batch_dim=False)


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
        upstream_nodes=["dummy1", "dummy2"],
    )
    stage._backend = backend

    pad_dim = max([x.shape[0] for x in example_inputs_with_mismatch])
    concat_dim = sum([x.shape[1] for x in example_inputs_with_mismatch])

    result = stage.apply_merge(example_inputs_with_mismatch, includes_batch_dim=False)
    assert result.shape == (pad_dim, concat_dim)


# def test_concat_axis0_with_constant_padding(backend, example_inputs_with_mismatch_v2):
#     # tests that [(1, 101), (1,1)] gets padded and concats to (2,101)

#     stage = ConcatStage(
#         label="concat",
#         axis=0,
#         pad_inputs=True,
#         pad_mode=PadMode.CONSTANT,
#         pad_value=0.0,
#         upstream_nodes=["dummy1", "dummy2"],
#     )
#     stage._backend = backend

#     pad_dim = max([x.shape[1] for x in example_inputs_with_mismatch_v2])
#     concat_dim = sum([x.shape[0] for x in example_inputs_with_mismatch_v2])

#     result = stage.apply_merge(example_inputs_with_mismatch_v2)
#     assert result.shape == (concat_dim, pad_dim)
