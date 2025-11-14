import math

import numpy as np


def normal_round(num, ndigits: int = 0):
    """
    Rounds a float to the specified number of decimal places.

    Args:
        num (any): the value to round
        ndigits: the number of digits to round to.

    """
    if ndigits == 0:
        return int(num + (np.sign(num) * 0.5))
    digit_value = 10**ndigits
    return int(num * digit_value + (np.sign(num) * 0.5)) / digit_value


def format_value_to_sig_digits(value: float, sig_digits: int = 3, *, round_integers: bool = True) -> str:
    """
    Format a value to a specified number of significant digits.

    Args:
        value (float): Input number.
        sig_digits (int): Number of significant digits to keep.
        round_integers (bool): If True, round integers to the specified
            significant digits as well. If False, only apply decimal rounding.

    Returns:
        str: String representation of the formatted number.

    Examples:
        >>> format_value_to_sig_digits(1512.345, 3)
        '1510'
        >>> format_value_to_sig_digits(12.3456, 3)
        '12.3'
        >>> format_value_to_sig_digits(0.003123, 3)
        '0.00312'
        >>> format_value_to_sig_digits(0.00001234, 3)
        '0.0000123'
        >>> format_value_to_sig_digits(1235.123, 2)
        '1200'
        >>> format_value_to_sig_digits(1235.123, 2, round_integers=False)
        '1235'

    """
    if value == 0 or math.isnan(value):
        return "0"

    order = math.floor(math.log10(abs(value)))
    if round_integers:
        # Scale number so we can round to sig_digits
        factor = 10 ** (order - sig_digits + 1)
        rounded = normal_round(value / factor) * factor
        # Decide decimals: if order >= sig_digits-1 → no decimals, else show needed
        decimals = max(0, sig_digits - order - 1)
        return f"{rounded:.{decimals}f}"

    decimals = max(0, sig_digits - order - 1)
    return f"{value:.{decimals}f}"


def flatten_dict_paths(d: dict[str, any], prefix: str = "", separator: str = ".") -> list[str]:
    """
    Recursively flatten a nested dictionary into separator-joined key paths.

    Description:
        This function traverses any nested dictionary whose:
          - Keys are strings
          - Values may be: a string, a list of strings, or another nested dictionary

        The resulting paths are returned as strings joined by the given
        separator (default: "."). Each terminal element (string or list item)
        forms the end of a full path.

    Args:
        d (dict[str, any]):
            Input nested dictionary to flatten.
        prefix (str, optional):
            Internal recursion prefix. Should generally be left empty.
        separator (str, optional):
            String used to join nested key names. Defaults to ".".

    Returns:
        list[str]:
            A list of fully flattened paths using the specified separator.

    Raises:
        TypeError:
            If a list contains non-string elements, or if an unsupported
            value type is encountered in the dictionary.

    Examples:
        ```python
        flatten_dict_paths({"a": ["b", "c"]})
        # ['a.b', 'a.c']

        flatten_dict_paths({"a": {"b": "c", "d": "e"}})
        # ['a.b.c', 'a.d.e']

        flatten_dict_paths({"a": {"b": ["c", "f"], "d": ["e", "g"]}})
        # ['a.b.c', 'a.b.f', 'a.d.e', 'a.d.g']
        ```

    """
    paths = []
    for k, v in d.items():
        full_prefix = f"{prefix}.{k}" if prefix else k

        if isinstance(v, dict):
            # Recurse deeper into nested dict
            paths.extend(flatten_dict_paths(v, prefix=full_prefix, separator=separator))
        elif isinstance(v, str):
            # Terminal string value
            paths.append(f"{full_prefix}.{v}")
        elif isinstance(v, list):
            # List of terminal strings
            for item in v:
                if not isinstance(item, str):
                    msg = f"List values must be strings, got {type(item)} in key '{k}'"
                    raise TypeError(msg)
                paths.append(f"{full_prefix}.{item}")
        else:
            msg = f"Unsupported value type {type(v)} for key '{k}'"
            raise TypeError(msg)

    return paths
