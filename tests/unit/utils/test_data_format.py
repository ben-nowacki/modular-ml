import pytest

from modularml.utils.backend import Backend
from modularml.utils.data_format import (
    DataFormat,
    format_requires_compatible_shapes,
    get_data_format_for_backend,
    normalize_format,
)


# ---------- normalize_format ----------
@pytest.mark.unit
def test_normalize_format_aliases():
    assert normalize_format("pd") == DataFormat.PANDAS
    assert normalize_format("np") == DataFormat.NUMPY
    assert normalize_format("torch.tensor") == DataFormat.TORCH
    assert normalize_format(DataFormat.LIST) == DataFormat.LIST


@pytest.mark.unit
def test_normalize_format_invalid():
    with pytest.raises(ValueError, match="Unknown data format: unknown"):
        normalize_format("unknown")


# ---------- format_requires_compatible_shapes ----------
@pytest.mark.unit
def test_format_requires_compatible_shapes():
    assert format_requires_compatible_shapes(DataFormat.NUMPY)
    assert format_requires_compatible_shapes(DataFormat.TORCH)
    assert format_requires_compatible_shapes(DataFormat.TENSORFLOW)
    assert not format_requires_compatible_shapes(DataFormat.DICT)


# ---------- get_data_format_for_backend ----------
@pytest.mark.unit
def test_get_data_format_for_backend_valid():
    assert get_data_format_for_backend(Backend.TORCH) == DataFormat.TORCH
    assert get_data_format_for_backend("torch") == DataFormat.TORCH
    assert get_data_format_for_backend(Backend.TENSORFLOW) == DataFormat.TENSORFLOW
    assert get_data_format_for_backend("tensorflow") == DataFormat.TENSORFLOW
    assert get_data_format_for_backend(Backend.SCIKIT) == DataFormat.NUMPY
    assert get_data_format_for_backend(Backend.NONE) == DataFormat.NUMPY


@pytest.mark.unit
def test_get_data_format_for_backend_invalid():
    with pytest.raises(ValueError, match=r"'some_other_backend' is not a valid Backend"):
        get_data_format_for_backend("some_other_backend")
