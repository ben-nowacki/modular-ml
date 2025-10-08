from typing import Any

import numpy as np
import pandas as pd

from modularml.utils.backend import Backend, infer_backend
from modularml.utils.data_format import DataFormat, format_requires_compatible_shapes, normalize_format

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

from modularml.core.data_structures.data import Data
from modularml.core.graph.shape_spec import ShapeSpec
from modularml.utils.error_handling import ErrorMode


def merge_dict_of_arrays_to_numpy(
    data: dict[str, Any],
    *,
    errors: ErrorMode = ErrorMode.RAISE,
) -> np.ndarray:
    """
    Merges a dict of arrays (possibly with different shapes) into a single 2D NumPy array.

    The merge axis is inferred from the ShapeSpec of the data. Shapes must be compatible
    along all but one dimension.

    Args:
        data (dict[str, Any]): Mapping of feature name → array-like values.
        errors (ErrorMode): Error-handling behavior.

    Returns:
        np.ndarray: Concatenated NumPy array.

    Raises:
        ValueError: If shapes are incompatible or merging fails.

    """
    # Convert all values to numpy arrays first
    np_dict = {k: to_numpy(v, errors=errors) for k, v in data.items()}

    # Collect shapes and check consistency
    shapes = {k: v.shape for k, v in np_dict.items()}
    shape_spec = ShapeSpec(shapes=shapes)

    # Try merging along inferred axis
    try:
        _ = shape_spec.merged_shape
        merge_axis = shape_spec.merged_axis
    except Exception as e:
        msg = f"Cannot merge features with shapes {shapes}: {e}"
        raise ValueError(msg) from e

    # Default: concatenate along last axis if identical shapes
    if merge_axis is None:
        merge_axis = -1

    try:
        return np.concatenate(list(np_dict.values()), axis=merge_axis)
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Failed to concatenate feature arrays along axis {merge_axis}: {e}"
            raise ValueError(msg) from e
        if errors == ErrorMode.IGNORE:
            return np_dict
        if errors == ErrorMode.COERCE:
            return np.column_stack(list(np_dict.values()))


def convert_dict_to_format(  # noqa: PLR0911
    data: dict[str, Any],
    fmt: str | DataFormat,
    errors: ErrorMode = ErrorMode.RAISE,
) -> Any:
    """
    Converts a dictionary of data arrays into the specified format.

    Args:
        data: Dict of arrays, lists, scalars, or tensors.
        fmt: Target data format to convert into.
        errors: How to handle incompatible types:
            - ErrorMode.RAISE: Raise an error when conversion fails.
            - ErrorMode.COERCE: Force conversion where possible.
            - ErrorMode.IGNORE: Leave unconvertible objects unchanged.

    Returns:
        Converted object.

    """
    fmt = normalize_format(fmt)

    # Dictionary-like formats
    if fmt == DataFormat.DICT:
        return to_python(data)
    if fmt == DataFormat.DICT_LIST:
        return {k: to_list(v, errors=errors) for k, v in data.items()}
    if fmt == DataFormat.DICT_NUMPY:
        return {k: to_numpy(v, errors=errors) for k, v in data.items()}
    if fmt == DataFormat.DICT_TORCH:
        return {k: to_torch(v, errors=errors) for k, v in data.items()}
    if fmt == DataFormat.DICT_TENSORFLOW:
        return {k: to_tensorflow(v, errors=errors) for k, v in data.items()}

    # DataFrame
    if fmt == DataFormat.PANDAS:
        return pd.DataFrame({k: to_list(v, errors=errors) for k, v in data.items()})

    # Unified array formats (NUMPY, TORCH, TENSORFLOW)
    if format_requires_compatible_shapes(fmt):
        np_merged = merge_dict_of_arrays_to_numpy(data, errors=errors)

        if fmt == DataFormat.NUMPY:
            return np_merged
        if fmt == DataFormat.TORCH:
            if torch is None:
                raise ImportError("PyTorch is not installed.")
            return torch.tensor(np_merged, dtype=torch.float32)
        if fmt == DataFormat.TENSORFLOW:
            if tf is None:
                raise ImportError("TensorFlow is not installed.")
            return tf.convert_to_tensor(np_merged, dtype=tf.float32)

    # List
    if fmt == DataFormat.LIST:
        return [list(row) for row in zip(*[to_list(v, errors=errors) for v in data.values()], strict=True)]

    msg = f"Unsupported data format: {fmt}"
    raise ValueError(msg)


def to_python(obj):  # noqa: PLR0911
    """
    Recursively converts an object into its native Python equivalent.

    Supported conversions:
    - NumPy scalars    -> Python scalars
    - NumPy arrays     -> Python lists
    - PyTorch tensors  -> Python scalars or lists
    - TensorFlow tensors -> Python scalars or lists
    - Dicts, tuples, and lists -> Recursively converted

    Args:
        obj: Any object to convert.

    Returns:
        Python-native object.

    """
    # NumPy
    if isinstance(obj, np.generic):  # np.int64, np.float64, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Pandas
    if isinstance(obj, pd.Series):
        return to_python(obj.values)

    # PyTorch
    if torch is not None and isinstance(obj, torch.Tensor):
        # Move to CPU, detach from graph if needed, convert to list or scalar
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    # TensorFlow
    if tf is not None and isinstance(obj, tf.Tensor):
        # Use .numpy() safely, then convert like numpy arrays
        np_obj = obj.numpy()
        if np_obj.ndim == 0:
            return np_obj.item()
        return np_obj.tolist()

    # Containers
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(to_python(v) for v in obj)
    if isinstance(obj, Data):
        return to_python(obj.value)

    # Base case
    return obj


def to_list(obj: Any, errors: ErrorMode = ErrorMode.RAISE):  # noqa: PLR0911
    """
    Converts any object into a Python list.

    Args:
        obj: Any object to convert.
        errors: How to handle non-listable objects.
            - "raise": Raise TypeError if the object cannot be converted.
            - "coerce": Force conversion where possible (wrap scalars, arrays, tensors, etc.).
            - "ignore": Leave incompatible objects unchanged.

    Returns:
        list or object (if errors="ignore" and incompatible).

    """
    # If we're ignoring incompatible types, leave dicts unchanged directly
    if errors == ErrorMode.IGNORE and isinstance(obj, dict):
        return obj

    py_obj = to_python(obj)

    # If it's already a list or tuple, convert directly
    if isinstance(py_obj, list | tuple | np.ndarray):
        return list(py_obj)

    # If it's a scalar, decide based on `errors`
    if np.isscalar(py_obj):
        return [py_obj]

    # Dicts aren't naturally convertible to lists
    if isinstance(py_obj, dict):
        if errors == ErrorMode.RAISE:
            raise TypeError("Cannot convert dict to list. Use DICT format instead.")
        if errors == ErrorMode.COERCE:
            # Convert dict values into a list of values
            return list(py_obj.values())
        if errors == ErrorMode.IGNORE:
            return py_obj

    # Fallback: try NumPy coercion if possible
    try:
        return np.asarray(py_obj).tolist()
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to list."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return [py_obj]


def to_numpy(  # noqa: PLR0911
    obj: Any,
    errors: ErrorMode = ErrorMode.RAISE,
    *,
    _top_level: bool = True,
) -> np.ndarray:
    """
    Recursively converts any object into a NumPy array.

    Nested lists/tuples are traversed so every sub-sequence
    is converted to np.ndarray where appropriate. Scalars are left
    as plain Python types inside the structure; only the *outermost*
    call wraps the final scalar into a 0-D array.

    Args:
        obj: Object to convert.
        errors: Error handling mode (RAISE, COERCE, IGNORE).
        _top_level: Internal flag to track recursion depth.

    Returns:
        np.ndarray or object (if IGNORE and conversion is not possible).

    """
    # If it's already a numpy array, just return
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, Data) and isinstance(obj.value, np.ndarray):
        return obj.value

    py_obj = to_python(obj)

    # Dicts must use DICT_NUMPY format unless coerced
    if isinstance(py_obj, dict):
        if errors == ErrorMode.RAISE:
            raise TypeError("Cannot convert dict directly to NumPy array. Use DICT_NUMPY instead.")
        if errors == ErrorMode.COERCE:
            return np.array(list(py_obj.values()))
        if errors == ErrorMode.IGNORE:
            return py_obj

    # Sequences (lists, tuples) -> convert directly
    if isinstance(py_obj, list | tuple):
        try:
            # Recursively convert every element to numpy before stacking
            converted = [to_numpy(item, errors=errors, _top_level=False) for item in py_obj]
            return np.array(
                converted,
                dtype=object if any(isinstance(c, np.ndarray) and c.ndim == 0 for c in converted) else None,
            )
        except Exception as e:
            if errors == ErrorMode.RAISE:
                msg = f"Cannot convert nested sequence of type {type(py_obj)} to NumPy array."
                raise TypeError(msg) from e
            if errors == ErrorMode.IGNORE:
                return py_obj
            if errors == ErrorMode.COERCE:
                return np.array([py_obj])

    # Scalars -> wrap into a 0-D array
    if np.isscalar(py_obj):
        if _top_level:
            return np.asarray(py_obj)
        return py_obj  # leave scalars within a nest object unchanged

    # Unsupported type
    if errors == ErrorMode.RAISE:
        msg = f"Cannot convert object of type {type(py_obj)} to NumPy array."
        raise TypeError(msg)
    if errors == ErrorMode.IGNORE:
        return py_obj
    if errors == ErrorMode.COERCE:
        return np.array([py_obj])
    return None


def to_torch(obj: Any, errors: ErrorMode = ErrorMode.RAISE) -> "torch.Tensor":  # pyright: ignore[reportInvalidTypeForm]
    """Converts any object into a PyTorch tensor."""
    if torch is None:
        raise ImportError("PyTorch is not installed.")

    # If it's already a Torch Tensor, just return
    if isinstance(obj, torch.Tensor):
        return obj

    py_obj = to_python(obj)
    try:
        return torch.as_tensor(np.asarray(py_obj), dtype=torch.float32)
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to Torch tensor."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return torch.as_tensor(np.asarray([py_obj]), dtype=torch.float32)


def to_tensorflow(obj: Any, errors: ErrorMode = ErrorMode.RAISE) -> "tf.Tensor":  # pyright: ignore[reportInvalidTypeForm]
    """Converts any object into a TensorFlow tensor."""
    if tf is None:
        raise ImportError("TensorFlow is not installed.")

    # If it's already a Tensforflow Tensor, just return
    if isinstance(obj, tf.Tensor):
        return obj

    py_obj = to_python(obj)
    try:
        return tf.convert_to_tensor(np.asarray(py_obj), dtype=tf.float32)
    except Exception as e:
        if errors == ErrorMode.RAISE:
            msg = f"Cannot convert object of type {type(py_obj)} to TensorFlow tensor."
            raise TypeError(msg) from e
        if errors == ErrorMode.IGNORE:
            return py_obj
        if errors == ErrorMode.COERCE:
            return tf.convert_to_tensor(np.asarray([py_obj]), dtype=tf.float32)


def enforce_numpy_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape != target_shape:
        arr = arr.reshape(target_shape)
    return arr


def convert_to_format(
    data: Any,
    fmt: str | DataFormat,
    errors: ErrorMode = ErrorMode.RAISE,
) -> Any:
    """
    Converts a data object into the specified format.

    Args:
        data: Dicts, arrays, lists, scalars, or tensors.
        fmt: Target data format to convert into.
        errors: How to handle incompatible types:
            - ErrorMode.RAISE: Raise an error when conversion fails.
            - ErrorMode.COERCE: Force conversion where possible.
            - ErrorMode.IGNORE: Leave unconvertible objects unchanged.

    Returns:
        Converted object.

    """
    fmt = normalize_format(fmt)
    if isinstance(data, dict):
        return convert_dict_to_format(data=data, fmt=fmt, errors=errors)

    if fmt == DataFormat.NUMPY:
        return to_numpy(data, errors=errors)

    if fmt == DataFormat.LIST:
        return to_list(data, errors=errors)

    if fmt == DataFormat.TORCH:
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        return to_torch(data, errors=errors)

    if fmt == DataFormat.TENSORFLOW:
        if tf is None:
            raise ImportError("TensorFlow is not installed.")
        return to_tensorflow(data, errors=errors)

    msg = f"Unsupported data format: {fmt}"
    raise ValueError(msg)


def shapes_similar_except_singleton(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> bool:
    """
    Returns True if two shapes are equal except for singleton (size-1) dimensions.

    Examples:
        (32, 1, 4) ≈ (32, 4)        → True
        (1, 64, 1, 1) ≈ (64,)       → True
        (16, 8, 4) ≈ (8, 16, 4)     → False  (order mismatch)
        (32, 2, 4) ≈ (32, 4)        → False  (non-singleton mismatch)

    Args:
        shape_a (tuple[int]): First shape.
        shape_b (tuple[int]): Second shape.

    Returns:
        bool: True if the shapes are equivalent ignoring singleton dims.

    """
    # Remove singleton dimensions
    reduced_a = tuple(dim for dim in shape_a if dim != 1)
    reduced_b = tuple(dim for dim in shape_b if dim != 1)

    return reduced_a == reduced_b


def align_ranks(arr1: Any, arr2: Any, backend: Backend | None = None) -> tuple:
    """
    Align the *ranks* (number of dimensions) of two array-like objects.

    This function ensures both arrays have the same number of dimensions by inserting singleton
    axes (using backend-safe operations like `unsqueeze` or `expand_dims`). It does **not**
    permute or reshape non-singleton dimensions, and it preserves autograd in all supported
    backends.

    The operation is symmetric; the array with the smaller rank is modified so that both
    arrays end up with the same shape, as long as they are compatible under singleton
    expansion. If both arrays already have the same rank and shape, they are returned
    unchanged.

    Examples:
        >>> import numpy as np
        >>> a = np.random.random(size=(32, 1, 100))
        >>> b = np.random.random(size=(32, 100))
        >>> a.shape, b.shape
        ((32, 1, 100), (32, 100))
        >>> a2, b2 = align_ranks(a, b)
        >>> a2.shape, b2.shape
        ((32, 1, 100), (32, 1, 100))

    Constraints:
        - Only singleton (size-1) dimensions are added or removed.
        - Arrays with mismatched non-singleton sizes (e.g. (32,1,4) vs (32,1))
          will raise a ValueError.
        - The total number of elements (`prod(shape)`) must match.
        - Does not detach tensors or break computational graphs.

    Args:
        arr1 (Any):
            First array-like object (must expose `.shape`).
        arr2 (Any):
            Second array-like object (must expose `.shape`).
        backend (Backend | None):
            Backend identifier determining which framework (Torch, TensorFlow, NumPy)
            operations are used. If None, backend is inferred for each array.

    Returns:
        tuple:
            A pair `(arr1_aligned, arr2_aligned)` such that `arr1_aligned.shape == arr2_aligned.shape`.

    Raises:
        ValueError:
            - If arrays have equal rank but different shapes.
            - If shapes cannot be aligned by inserting/removing only singleton dimensions.
        TypeError:
            If an unsupported backend is provided.
        ImportError:
            If the specified backend is unavailable (e.g., PyTorch not installed).
        RuntimeError:
            If alignment fails despite matching total element counts.

    Notes:
        - This function is intended for internal use in loss computation or tensor comparison,
          where user-defined models may add trailing singleton axes (e.g., via Conv1D or Linear).
        - Only the array with the smaller rank is modified; the other is returned unchanged.
        - Autograd gradients are preserved (no `.detach()`, `.numpy()`, or graph breakage).

    """
    from math import prod  # noqa: PLC0415

    def _modify_arr(arr, singleton_idxs: list[int], arr_backend: Backend | None = None):
        """Insert singletons at each index in `singleton_idxs`."""
        if arr_backend is None:
            arr_backend = infer_backend(arr)

        if arr_backend == Backend.TORCH:
            if torch is None:
                msg = f"PyTorch is not installed but a backend of {Backend.TORCH} was used."
                raise ImportError(msg)
            arr = convert_to_format(arr, fmt=DataFormat.TORCH)
            for idx in singleton_idxs:
                arr = arr.unsqueeze(idx)
            return arr
        if arr_backend == Backend.TENSORFLOW:
            if tf is None:
                msg = f"TensorFlow is not installed but a backend of {Backend.TENSORFLOW} was used."
                raise ImportError(msg)
            arr = convert_to_format(arr, fmt=DataFormat.TENSORFLOW)
            for idx in singleton_idxs:
                arr = tf.expand_dims(arr, axis=idx)
            return arr
        if arr_backend in (Backend.SCIKIT, Backend.NONE):
            arr = convert_to_format(arr, fmt=DataFormat.NUMPY)
            for idx in singleton_idxs:
                arr = np.expand_dims(arr, axis=idx)
            return arr
        msg = f"Unsupported backend: {arr_backend}"
        raise TypeError(msg)

    shape_1, shape_2 = arr1.shape, arr2.shape

    # If already aligned, return or reshape
    if len(shape_1) == len(shape_2):
        if shape_1 != shape_2:
            raise ValueError("Arrays are the same rank but not in the same order.")
        return arr1, arr2

    # Ranks cannot be matched via singleton additions/subtractions
    if prod(shape_1) != prod(shape_2):
        msg = f"Arrays cannot be aligned via addition or removal of singletons: {shape_1} & {shape_2}"
        raise ValueError(msg)

    # Use highest rank as reference (ref), other as comparison (cmp)
    ref, cmp = shape_2, shape_1
    if len(shape_1) > len(shape_2):
        ref, cmp = shape_1, shape_2

    # For each item in ref, see if cmp matches:
    cmp_assignments = -1 * np.ones(shape=len(ref))  # each element corresponds to index in ref
    for i in range(len(cmp)):
        for j in range(i, len(ref)):
            if cmp[i] == ref[j]:
                cmp_assignments[j] = j
                break

    # The index of each -1 element is where a singleton needs to be inserted
    idxs = np.argwhere(cmp_assignments == -1).reshape(-1).tolist()
    if len(shape_1) > len(shape_2):  # arr1 is ref, modify arr2
        arr2 = _modify_arr(arr2, singleton_idxs=idxs, arr_backend=backend)
    else:  # arr2 is ref, modify arr1
        arr1 = _modify_arr(arr1, singleton_idxs=idxs, arr_backend=backend)

    if arr1.shape != arr2.shape:
        raise RuntimeError("Failed to match ranks.")

    return arr1, arr2
