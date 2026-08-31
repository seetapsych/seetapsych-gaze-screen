# -*- coding: utf-8 -*-


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, key by key.

    For nested dicts (TOML sections), merge recursively so that only the
    explicitly set keys in override replace those in base — other keys in
    the same section are left untouched.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
