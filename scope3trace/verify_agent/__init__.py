"""Verify Agent: deterministic evidence checks, semantic validation and the repair loop."""

import importlib

_EXPORTS = {'VerifyAgent': 'agent'}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
