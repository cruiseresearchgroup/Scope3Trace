"""Scope Agent (paper §3.3.2 stages 3-4): table reconstruction, disclosure extraction, evidence grounding and chart / image reading."""

import importlib

_EXPORTS = {'ScopeAgent': 'agent', 'ChartReader': 'chart_reader'}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
