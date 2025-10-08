import pytest

from modularml.utils.formatting import format_value_to_sig_digits


# ---------- Basic examples from docstring ----------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "sig_digits", "expected"),
    [
        (1512.345, 3, "1510"),  # now enforces sig figs
        (12.3456, 3, "12.3"),
        (0.003123, 3, "0.00312"),
        (0.00001234, 3, "0.0000123"),
    ],
)
def test_format_value_examples(value, sig_digits, expected):
    assert format_value_to_sig_digits(value, sig_digits) == expected


# ---------- Zero and NaN ----------
@pytest.mark.unit
def test_format_value_zero_and_nan():
    assert format_value_to_sig_digits(0.0) == "0"
    assert format_value_to_sig_digits(float("nan")) == "0"


# ---------- Negative values ----------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1234.56, "-1230"),  # 3 sig figs → rounded down
        (-0.004567, "-0.00457"),
    ],
)
def test_format_value_negative_numbers(value, expected):
    assert format_value_to_sig_digits(value, sig_digits=3) == expected


# ---------- Custom sig_digits ----------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "sig_digits", "expected"),
    [
        (1234.567, 2, "1200"),  # fewer sig figs
        (0.012345, 4, "0.01235"),  # more sig figs
        (9.87654, 5, "9.8765"),
    ],
)
def test_format_value_custom_sig_digits(value, sig_digits, expected):
    assert format_value_to_sig_digits(value, sig_digits) == expected


# ---------- round_integers flag ----------
@pytest.mark.unit
def test_round_integers_false_preserves_integer_part():
    """When round_integers=False, integers are not rounded to sig figs."""
    value = 1235.123
    assert format_value_to_sig_digits(value, sig_digits=2, round_integers=False) == "1235"
    # With round_integers=True, the same call should give "1200"
    assert format_value_to_sig_digits(value, sig_digits=2, round_integers=True) == "1200"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "sig_digits", "expected_true", "expected_false"),
    [
        (98765.4321, 2, "99000", "98765"),  # big integer-like number
        (0.00098765, 2, "0.00099", "0.00099"),  # decimals unaffected
    ],
)
def test_round_integers_true_vs_false(value, sig_digits, expected_true, expected_false):
    assert format_value_to_sig_digits(value, sig_digits, round_integers=True) == expected_true
    assert format_value_to_sig_digits(value, sig_digits, round_integers=False) == expected_false
