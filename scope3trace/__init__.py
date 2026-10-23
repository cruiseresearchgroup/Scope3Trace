# -*- coding: utf-8 -*-
"""
Scope3Trace multi-agent pipeline (paper §3.3, Figure 2).

Three specialized agents over shared, typed state — no free-form
conversational orchestration (Appendix F):

- SearchAgent : document acquisition + parsing/OCR + emissions-content
                localization; RETURNS a verbatim evidence list per document.
- ScopeAgent  : extraction & estimation — Asset Pool, disclosure extraction
                from the evidence list, entity linking, deterministic
                Appendix C estimation.
- VerifyAgent : verification — deterministic evidence checks (flag, never
                correct), relevance validation, consistency/logic checks,
                Evaluation Report.

Entry point: ``python -m scope3trace.run_scope3trace --help``
"""

import scope3trace.search_agent.font_repair  # noqa: F401  (repairs U+FFFD glyphs in pdfplumber)
from scope3trace.common.llm import BoundedLLM
from scope3trace.scope_agent.agent import ScopeAgent
from scope3trace.search_agent.agent import SearchAgent
from scope3trace.common.state import SharedState
from scope3trace.verify_agent.agent import VerifyAgent

__all__ = ["BoundedLLM", "ScopeAgent", "SearchAgent", "SharedState", "VerifyAgent"]
