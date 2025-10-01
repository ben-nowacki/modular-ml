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
