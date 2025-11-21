import pytest

from modularml.utils.registries import CaseInsensitiveRegistry


@pytest.mark.unit
def test_basic_insertion_and_lookup():
    reg = CaseInsensitiveRegistry()
    reg["StandardScaler"] = 123

    # direct lookup
    assert reg["StandardScaler"] == 123

    # case-insensitive lookup
    assert reg["standardscaler"] == 123
    assert reg["STANDARDSCALER"] == 123

    # key membership
    assert "standardscaler" in reg
    assert "StandardScaler" in reg
    assert "unknown" not in reg


@pytest.mark.unit
def test_original_casing_preserved():
    reg = CaseInsensitiveRegistry()
    reg["MyKey"] = 1
    reg["AnotherKey"] = 2

    assert list(reg.keys()) == ["MyKey", "AnotherKey"]
    assert reg.original_keys() == ["MyKey", "AnotherKey"]


@pytest.mark.unit
def test_lowercase_collision_prevention():
    reg = CaseInsensitiveRegistry()
    reg["ABC"] = 1

    # inserting another key differing only in case → error
    with pytest.raises(KeyError):
        reg["abc"] = 2

    # inserting totally different key should work
    reg["XYZ"] = 3
    assert reg["xyz"] == 3


@pytest.mark.unit
def test_get_method():
    reg = CaseInsensitiveRegistry()
    reg["Scaler"] = 10

    # case-insensitive get
    assert reg.get("SCALER") == 10
    assert reg.get("scaler") == 10

    # missing → return default
    assert reg.get("missing", default=None) is None
    assert reg.get("missing", default=99) == 99


@pytest.mark.unit
def test_deletion():
    reg = CaseInsensitiveRegistry()
    reg["KeyOne"] = 1
    reg["KeyTwo"] = 2

    # delete using different casing
    del reg["keyone"]

    assert "KeyOne" not in reg
    assert "keyone" not in reg

    # ensure other keys remain
    assert reg["KeyTwo"] == 2


@pytest.mark.unit
def test_pop():
    reg = CaseInsensitiveRegistry()
    reg["X"] = 100

    # pop using case-insensitive key
    val = reg.pop("x")
    assert val == 100

    # key removed
    assert "X" not in reg

    # pop default for nonexistent
    assert reg.pop("missing", default=999) == 999

    with pytest.raises(KeyError):
        reg.pop("missing_no_default")


@pytest.mark.unit
def test_update():
    reg = CaseInsensitiveRegistry()

    reg.update({"A": 1, "B": 2})

    assert reg["a"] == 1
    assert reg["B"] == 2

    # collision during update
    with pytest.raises(KeyError):
        reg.update({"a": 99})  # "A" already exists as lowercase


@pytest.mark.unit
def test_mixed_case_behavior():
    reg = CaseInsensitiveRegistry()
    reg["TestKey"] = 1

    # should normalize to same entry
    assert reg["testKEY"] == 1

    # contains must be case-insensitive
    assert "TESTkey" in reg


@pytest.mark.unit
def test_invalid_key_type():
    reg = CaseInsensitiveRegistry()

    with pytest.raises(TypeError):
        reg[123] = "value"  # non-string key should raise


@pytest.mark.unit
def test_replace_same_key_exact_casing():
    """Reassigning the same key with SAME casing should succeed (normal dict behavior)."""
    reg = CaseInsensitiveRegistry()
    reg["Scaler"] = 1
    reg["Scaler"] = 2  # allowed

    assert reg["SCALER"] == 2
