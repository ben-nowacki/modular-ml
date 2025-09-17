import math


def format_value_to_sig_digits(value: float, sig_digits: int = 3) -> str:
    """
    Format a loss value with dynamic precision.

    Always show at least `sig_digits` significant digits after the first non-zero.

    Examples:
        1512.345 -> "1512"
        12.3456  -> "12.35"
        0.003123 -> "0.00312"
        0.00001234 -> "0.0000123"

    """
    if value == 0 or math.isnan(value):
        return "0"

    order = math.floor(math.log10(abs(value)))
    decimals = max(0, sig_digits - order - 1)
    return f"{value:.{decimals}f}"
