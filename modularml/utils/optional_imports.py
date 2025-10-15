def ensure_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        msg = "pandas is required. Please install pandas to continue."
        raise ImportError(msg) from exc
    return pd


def check_pandas():
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd


def ensure_torch():
    try:
        import torch
    except ImportError as exc:
        msg = "torch is required. Please install torch to continue."
        raise ImportError(msg) from exc
    return torch


def check_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def ensure_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        msg = "tensorflow is required. Please install tensorflow to continue."
        raise ImportError(msg) from exc
    return tf


def check_tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        return None
    return tf
