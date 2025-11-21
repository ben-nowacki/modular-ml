import numpy as np


def deep_equal(a, b):
    """Recursively check deep equality of arbitrarily nested dicts/lists/tuples."""
    if a is b:
        return True

    # NumPy array
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)

    # Primitive types
    if isinstance(a, (int, float, str, bool)) or a is None:
        return a == b

    # Dictionary
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k]) for k in a)

    # List or tuple
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b, strict=True))

    # Fallback
    return a == b
