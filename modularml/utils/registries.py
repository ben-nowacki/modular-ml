class CaseInsensitiveRegistry(dict):
    """
    Dictionary-like registry with case-insensitive key lookup.

    Features:
    - Keys are stored exactly as provided (preserves original casing).
    - All lookups ([], get(), in) are case-insensitive.
    - A secondary map `_lower_map` maps lowercase -> original key.
    - Lowercase collisions are forbidden to prevent silent overwrites.

    Example:
        reg = CaseInsensitiveRegistry()
        reg["StandardScaler"] = cls
        reg["MinMaxScaler"] = cls2

        assert reg["standardscaler"] is reg["StandardScaler"]
        assert "minmaxscaler" in reg

    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._lower_map: dict[str, str] = {}
        if args or kwargs:
            self.update(*args, **kwargs)

    # ==========================================
    # Internal helpers
    # ==========================================
    def _normalize(self, key: str) -> str:
        if not isinstance(key, str):
            msg = f"Registry keys must be strings, got {type(key)}"
            raise TypeError(msg)
        return key.lower()

    def get_original_key(self, key: str) -> str:
        """Return the stored original key matching this key (case-insensitive)."""
        lk = self._normalize(key)
        return self._lower_map.get(lk)

    # ==========================================
    # Core dict overrides
    # ==========================================
    def __setitem__(self, key: str, value):
        lk = self._normalize(key)

        # Enforce lowercase uniqueness
        if lk in self._lower_map and self._lower_map[lk] != key:
            msg = f"Cannot insert key '{key}' - lowercase equivalent collides with existing key '{self._lower_map[lk]}'"
            raise KeyError(msg)

        # Insert
        super().__setitem__(key, value)
        self._lower_map[lk] = key

    def __getitem__(self, key: str):
        orig = self.get_original_key(key)
        if orig is None:
            raise KeyError(key)
        return super().__getitem__(orig)

    def __delitem__(self, key: str):
        orig = self.get_original_key(key)
        if orig is None:
            raise KeyError(key)
        lk = orig.lower()
        del self._lower_map[lk]
        super().__delitem__(orig)

    def __contains__(self, key: str) -> bool:
        return self.get_original_key(key) is not None

    # ==========================================
    # Convenience methods
    # ==========================================
    def get(self, key: str, default=None):
        orig = self.get_original_key(key)
        if orig is None:
            return default
        return super().get(orig, default)

    def pop(self, key: str, default=None):
        orig = self.get_original_key(key)
        if orig is None:
            if default is None:
                raise KeyError(key)
            return default
        lk = orig.lower()
        self._lower_map.pop(lk, None)
        return super().pop(orig)

    def update(self, *args, **kwargs):
        """Ensure update respects case-insensitive uniqueness."""
        items = dict(*args, **kwargs)
        for k, v in items.items():
            self[k] = v

    # Optional: return keys case-insensitively or normally
    def original_keys(self):
        """Return keys as originally inserted."""
        return list(self.keys())
