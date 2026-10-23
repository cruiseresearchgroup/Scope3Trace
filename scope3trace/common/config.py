#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scope3Trace configuration (paper §3.3, Appendix C/D/E).

Design principles encoded here:
- All LLM calls run at temperature=0 (§3.3).
- GPT-4o is the default shared backbone; each agent's backbone can be swapped
  independently (Appendix E, heterogeneous per-agent configurations).
- Deterministic constants (unit conversions, emission factors) live here so
  that estimation (Appendix C) is fully rule-based and reproducible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# LLM backbones (Appendix E)
# ---------------------------------------------------------------------------

DEFAULT_BACKBONE = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0  # fixed by design (§3.3); do not change per-call

# Public per-token pricing (USD per 1M tokens) used for the Appendix D style
# cost accounting in the evaluation report. Edit as prices change.
PRICING_PER_MTOK: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "deepseek-chat": {"in": 0.27, "out": 1.10},
    "gemini-2.0-flash": {"in": 0.10, "out": 0.40},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "google/gemini-2.5-flash": {"in": 0.30, "out": 2.50},
}


def resolve_provider(model: str) -> Dict[str, str]:
    """Map a model name to an OpenAI-compatible endpoint + API key env var.

    Env overrides: SCOPE3TRACE_BASE_URL / SCOPE3TRACE_API_KEY apply to every
    model and take precedence (useful for proxies or self-hosted backbones).
    """
    override_base = os.environ.get("SCOPE3TRACE_BASE_URL")
    override_key = os.environ.get("SCOPE3TRACE_API_KEY")
    if override_base:
        return {"base_url": override_base.rstrip("/"),
                "api_key_env": "SCOPE3TRACE_API_KEY" if override_key else "OPENAI_API_KEY"}
    m = model.lower()
    # OpenRouter model ids carry a vendor prefix ("google/gemini-2.5-flash");
    # only those go to OpenRouter, DeepSeek / Gemini / OpenAI ids are unchanged
    if "/" in m:
        base = (os.environ.get("OPENROUTER_API_URL") or "https://openrouter.ai/api/v1").strip().strip("\"'").rstrip("/")
        base = re.sub(r"/chat/completions$", "", base)
        return {"base_url": base, "api_key_env": "OPENROUTER_API_KEY"}
    if m.startswith("deepseek"):
        return {"base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"}
    if m.startswith("gemini"):
        return {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "api_key_env": "GEMINI_API_KEY"}
    # gpt-*, o*, and anything else defaults to OpenAI
    return {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"}


@dataclass
class AgentBackbones:
    """Per-agent backbone assignment (homogeneous by default)."""
    search: str = DEFAULT_BACKBONE
    scope: str = DEFAULT_BACKBONE
    verify: str = DEFAULT_BACKBONE

    @classmethod
    def homogeneous(cls, model: str) -> "AgentBackbones":
        return cls(search=model, scope=model, verify=model)


# ---------------------------------------------------------------------------
# Deterministic constants (Appendix A.7 unit normalization, Appendix C factors)
# ---------------------------------------------------------------------------

GJ_TO_KWH = 277.778
MJ_TO_KWH = 0.277778

ENERGY_UNIT_TO_KWH: Dict[str, float] = {
    "kwh": 1.0,
    "mwh": 1_000.0,
    "gwh": 1_000_000.0,
    "gj": GJ_TO_KWH,
    "mj": MJ_TO_KWH,
    "tj": GJ_TO_KWH * 1_000.0,
}

EMISSION_UNIT_TO_TCO2E: Dict[str, float] = {
    "tco2e": 1.0,
    "tco2-e": 1.0,
    "tco2": 1.0,
    "t": 1.0,
    "tonnes": 1.0,
    "tons": 1.0,
    "kgco2e": 0.001,
    "kg": 0.001,
    "ktco2e": 1_000.0,
    "kt": 1_000.0,
    "mtco2e": 1_000_000.0,
    "mt": 1_000_000.0,
}

# Electricity grid emission factor, tonnes CO2e per kWh (Appendix C.1/C.3).
# Region- and year-specific factors can be supplied via --grid-factor-csv;
# this constant (Victoria, AU) is the documented fallback.
DEFAULT_GRID_FACTOR_T_PER_KWH = 0.85 / 1000.0
DEFAULT_GRID_FACTOR_SOURCE = (
    "default_grid_factor:highest-factor-state-fallback:VIC:0.00085 tCO2e/kWh"
)

# Embodied carbon intensity by building typology, kg CO2e per m^2
# (Appendix C.2; EPiC-style engineering life-cycle factors).
EMBODIED_FACTORS_KG_PER_SQM: Dict[str, float] = {
    "Office": 1150.0,
    "Retail": 950.0,
    "Residential Apartment": 1250.0,
    "House/Townhouse": 650.0,
    "Storage": 600.0,
    "Warehouse": 600.0,
    "Industrial": 750.0,
    "Hospital/Clinic": 1400.0,
    "Education": 1100.0,
    "Public Display Area": 1000.0,
    "Parking": 550.0,
    "Default": 900.0,
}

# ---------------------------------------------------------------------------
# Pipeline thresholds (developed on the 20-PDF dev set, §3.3.2 stage 4)
# ---------------------------------------------------------------------------

# Entity linking (§3.3.1): deterministic accept / LLM-adjudication band.
LINK_ACCEPT_SCORE = 0.85
LINK_CANDIDATE_SCORE = 0.55

# Category mapping fuzzy-match floor (§3.3.2 stage 3).
CATEGORY_FUZZY_THRESHOLD = 0.75

# Relevance validation (§3.3.3): minimum lexical overlap for "applicable"
# when no LLM adjudication is available.
RELEVANCE_RULE_THRESHOLD = 0.45

# Consistency diagnostics (§3.3.4 / A.4): tolerance for category-sum checks.
CATEGORY_SUM_PASS_TOL = 0.05
CATEGORY_SUM_WARN_TOL = 0.15

# Sanity bounds for reporting years.
YEAR_MIN, YEAR_MAX = 2000, 2035

# Localization batching: pages per LLM call and chars per keyword-centred excerpt.
LOCALIZE_PAGES_PER_CALL = 15
LOCALIZE_CHARS_PER_PAGE = 1200

# Complete table pages can contain many year/category cells. Keep the existing
# per-page extraction path, with enough output space for the retained rows.
EXTRACTION_MAX_OUTPUT_TOKENS = 16384
