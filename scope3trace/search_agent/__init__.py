"""Search Agent (paper §3.3.2 stages 1-2): report search (LLM with OpenRouter web search, or Google Custom Search API), download and identity check, PDF-to-text parsing / OCR and emissions content localization."""

import importlib

_EXPORTS = {'SearchAgent': 'agent'}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
