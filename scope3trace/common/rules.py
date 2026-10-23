#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic rule components of Scope3Trace (§3.3, Appendix E).

Design principle: "deterministic logic such as numeric parsing, unit
normalization, category mapping, and evidence checking is delegated to
rule-based components, restricting the LLM to bounded semantic tasks."

Everything in this module is pure-Python, LLM-free, and unit-testable:
- numeric parsing (thousand separators, parentheses, EU decimal commas)
- unit normalization to tCO2e / kWh (Appendix A.7)
- GHG Protocol Scope 3 category mapping (rule-first, fuzzy fallback)
- deterministic evidence checking: a value must literally appear in its
  source snippet / page text ("Evidence Grounding Rules")
- cross-level consistency and logic checks (§3.3.4, Appendix A.4)
- string / postcode / asset-key normalization (shared with
  scripts/alignment_utils.py; local fallbacks keep the package standalone)
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from scope3trace.common.config import (
    CATEGORY_FUZZY_THRESHOLD,
    CATEGORY_SUM_PASS_TOL,
    CATEGORY_SUM_WARN_TOL,
    EMISSION_UNIT_TO_TCO2E,
    ENERGY_UNIT_TO_KWH,
    YEAR_MAX,
    YEAR_MIN,
)
from scope3trace.common.state import DiagnosticResult

# ---------------------------------------------------------------------------
# String / address normalization (shared with scripts/alignment_utils.py)
# ---------------------------------------------------------------------------

try:  # reuse the repo-wide implementations when the repo root is on sys.path
    from scripts.alignment_utils import (  # type: ignore
        build_asset_key,
        extract_postcode_from_address,
        normalize_string,
        normalize_uk_postcode,
    )
except Exception:  # standalone fallbacks (behaviour-compatible)

    def normalize_string(s: str) -> str:
        if s is None:
            return ""
        s = unicodedata.normalize("NFKC", str(s)).lower().strip()
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def normalize_uk_postcode(postcode: str) -> Optional[str]:
        if not postcode:
            return None
        p = re.sub(r"\s+", "", str(postcode).upper())
        m = re.fullmatch(r"([A-Z]{1,2}\d[A-Z\d]?)(\d[A-Z]{2})", p)
        return f"{m.group(1)} {m.group(2)}" if m else None

    def extract_postcode_from_address(address: str) -> Optional[str]:
        if not address:
            return None
        m = re.search(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", str(address).upper())
        if m:
            return normalize_uk_postcode(m.group(1))
        m = re.search(r"\b(\d{4})\b\s*$", str(address).strip())  # AU 4-digit
        return m.group(1) if m else None

    def build_asset_key(asset_name: str, postcode: Optional[str] = None) -> str:
        key = normalize_string(asset_name or "")
        return f"{key}|{postcode}" if postcode else key


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-−(]?\s*\d[\d,.\s ']*\d|\d")

# A value passed to ``parse_number`` must represent exactly one table cell.
# These forms cover the separators observed in the reports while rejecting a
# flattened sequence of several year columns such as
# ``390,746 358,580 335,624``.  Silently deleting all whitespace from that
# sequence used to create a valid-looking but catastrophically wrong number.
_SINGLE_NUMBER_FORMS = (
    re.compile(r"\d+(?:\.\d+)?"),                         # 1234 / 1234.5
    re.compile(r"\d+,\d+"),                              # decimal comma
    re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"),        # 1,234,567.8
    re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"),        # 1.234.567,8
    re.compile(r"\d{1,3}(?:[ \u00a0\u2009\u202f']\d{3})+(?:[.,]\d+)?"),  # 1 234 567
)


def _normalize_separator_spacing(raw: str) -> str:
    """Remove PDF spacing around punctuation inside a single numeric cell.

    Keep spaces between digits: those may separate different year columns.
    The caller must still validate the complete normalized token.
    """
    return re.sub(r"[ \t\u00a0]*([.,])[ \t\u00a0]*(?=\d)", r"\1", raw)


def is_single_number_token(raw: Any) -> bool:
    """Return True only when *raw* is one complete numeric cell.

    The extraction prompt requires ``value_raw`` to contain the verbatim
    number only.  Rejecting extra numeric cells here is therefore safer than
    guessing which year column the value belongs to.
    """
    if isinstance(raw, (int, float)):
        return True
    if raw is None:
        return False
    s = str(raw).strip()
    if not s:
        return False
    if s.startswith("(") or s.endswith(")"):
        if not (s.startswith("(") and s.endswith(")")):
            return False
        s = s[1:-1].strip()
    s = _normalize_separator_spacing(s.lstrip("-−").strip())
    return any(pattern.fullmatch(s) for pattern in _SINGLE_NUMBER_FORMS)


_TRAILING_MARKERS = "*†‡§✓✔#"


def strip_cell_markers(raw: Any) -> Any:
    """Drop assurance/footnote symbols printed after a numeric cell.

    Only non-alphanumeric marker glyphs are removed (``12,345*``, ``159✓``).
    Digits are never removed here: a superscript footnote digit can only be
    separated with glyph-size evidence (see table_structure).
    """
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    # an asterisk/dagger introduces footnote numbers: "298,224*7,8"
    s = re.sub(r"(?<=\d)([*\u2020\u2021\u00a7])\d{1,2}(?:,\d{1,2})*$", r"\1", s)
    s2 = s.rstrip(_TRAILING_MARKERS + " ")
    return s2 if s2 and s2 != s and re.search(r"[\d)]$", s2) else s


def infer_number_locale(texts) -> Optional[str]:
    """Infer a report's numeric convention from unambiguous tokens.

    ``eu`` = '.' groups thousands and ',' marks decimals (``5.727.100``,
    ``1.234,5``, ``12,5``); ``us`` = the reverse (``1,234,567``,
    ``1,234.5``, ``12.5``). Returns None unless one convention clearly
    dominates; single-group tokens such as ``6.022`` then keep the
    historical default interpretation.
    """
    eu = us = 0
    for text in texts:
        e, u = _locale_counts(text)
        eu, us = eu + e, us + u
    return _dominant_locale(eu, us, 5)


def _locale_counts(text):
    if not text:
        return 0, 0
    eu = len(re.findall(r"(?<![\d.,])\d{1,3}(?:\.\d{3}){2,}(?![\d.]|,\d{3})", text))
    eu += len(re.findall(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+,\d{1,3}(?![\d.,])", text))
    eu += len(re.findall(r"(?<![\d.,])\d{1,4},\d{1,2}(?![\d.,])", text))
    us = len(re.findall(r"(?<![\d.,])\d{1,3}(?:,\d{3}){2,}(?![\d,]|\.\d{3})", text))
    us += len(re.findall(r"(?<![\d.,])\d{1,3}(?:,\d{3})+\.\d{1,3}(?![\d.,])", text))
    us += len(re.findall(r"(?<![\d.,\w])\d{1,4}\.\d{1,2}(?![\d.,%])", text))
    return eu, us


def _dominant_locale(eu, us, minimum):
    if eu >= minimum and eu >= 3 * max(us, 1):
        return "eu"
    if us >= minimum and us >= 3 * max(eu, 1):
        return "us"
    return None


_PERIOD_END = re.compile(r"(?:31(?:st)?\s+(?:December|Dec\.?|March|Mar\.?)|30(?:th)?\s+(?:June|Jun\.?|September|Sept?\.?))\s*,?\s+((?:19|20)\d{2})"
                         r"|(?:financial|fiscal|reporting)\s+(?:year|period)\s+((?:19|20)\d{2})", re.I)


def infer_period_end_year(texts) -> Optional[int]:
    """The reporting-period end year most often stated in a document
    ("year ended 31 December 2024", "financial year 2025"). Used only to
    resolve relative ESRS column headers ("N", "N-1")."""
    from collections import Counter
    c = Counter()
    for t in texts:
        for m in _PERIOD_END.finditer(t or ""):
            c[int(m.group(1) or m.group(2))] += 1
    if not c:
        return None
    top = max(c.values())
    return max(y for y, n in c.items() if n == top)


def page_number_locale(text, document_locale: Optional[str] = None) -> Optional[str]:
    """A page whose own unambiguous tokens clearly follow one convention
    keeps it (an annual report may embed a sustainability statement typeset
    differently); otherwise the document convention applies."""
    e, u = _locale_counts(text)
    return _dominant_locale(e, u, 3) or document_locale


def ambiguous_separator(raw: Any, locale: Optional[str]) -> bool:
    """``128.142`` with no known convention: a single dot before exactly three
    digits may be a decimal point or a thousands separator, so binding must
    fail closed rather than guess a factor of 1000."""
    if locale is not None or raw is None:
        return False
    s = strip_cell_markers(str(raw).strip()).strip("()-\u2212 ")
    return bool(re.fullmatch(r"\d{1,3}\.\d{3}", s))


def parse_number(raw: Any, locale: Optional[str] = None) -> Optional[float]:
    """Parse a verbatim report number: '1,234,567.8', '(123)', '1 234', '1.234,5'.

    ``locale`` resolves single-separator tokens: under ``eu`` ``6.022`` is
    6022 and ``1,385`` is 1.385; otherwise ``6.022`` is 6.022 and ``1,385``
    is 1385. Multi-group tokens (``5.727.100``) are unambiguous.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = strip_cell_markers(str(raw).strip())
    if not s:
        return None
    if not is_single_number_token(s):
        return None
    negative = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.startswith("−")
    s = s.strip("()").lstrip("-−").strip()
    s = _normalize_separator_spacing(s)
    s = s.replace(" ", " ").replace("'", "")
    s = re.sub(r"\s+", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):   # EU style 1.234.567,8
            s = s.replace(".", "").replace(",", ".")
        else:                              # US style 1,234,567.8
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) > 2:                            # 1,234,567 grouping
            s = s.replace(",", "")
        elif len(parts[-1]) == 3 and locale != "eu":  # grouping
            s = s.replace(",", "")
        else:                                         # decimal comma
            s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2:                            # 5.727.100 grouping
            s = s.replace(".", "")
        elif locale == "eu" and len(parts[-1]) == 3:  # 6.022 grouping
            s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if negative else v


# ---------------------------------------------------------------------------
# Unit normalization (Appendix A.7)
# ---------------------------------------------------------------------------

def _canon_unit(unit_raw: Optional[str]) -> str:
    if not unit_raw:
        return ""
    u = str(unit_raw).lower()
    u = u.replace("co₂", "co2").replace("-", "").replace("_", "")
    return re.sub(r"[\s.]", "", u)


_UNIT_SCALE_RE = re.compile(
    r"^(?:in\s+)?(?:(?P<k>(?:['’]\s*)?000s?\)?|\(\s*000s?\s*\)|1[ ,.]?000(?![ ,.]?000)|thousands?(?:\s+of)?)"
    r"|(?P<m>1[ ,.]?000[ ,.]?000|millions?(?:\s+of)?))\s*", re.I)
_MASS_END = r"(?=co|\(|[^a-z]|$)"
_UNIT_MASS_RE = re.compile(
    r"^(?:(?P<MT>(?-i:MT|mt))" + _MASS_END +
    r"|(?P<Mt>mega\s*tonnes?|(?-i:Mtonnes?|Mt))" + _MASS_END +
    r"|(?P<kt>kilo\s*tonnes?|ktonnes?|kt|Gg)" + _MASS_END +
    r"|(?P<metric>metric\s+)?(?P<t>tonnes?|tons?|tn|t)" + _MASS_END +
    r"|(?P<kg>kilograms?|kg)" + _MASS_END +
    r"|(?P<g>grams?|g)" + _MASS_END + r")\s*", re.I)
_UNIT_GAS_RE = re.compile(
    r"^(?:of\s+)?(?:co\s*2?\s*[-‐‑–]?\s*(?:e(?:q(?:uivalents?|uiv)?)?)?\.?"
    r"|carbon\s+dioxide(?:\s+equivalents?)?|ghg|e)?\s*$", re.I)


def emission_unit_multiplier(unit_raw: Optional[str]) -> Optional[float]:
    """Return the tCO2e multiplier of an absolute emissions unit, else None.

    Case matters only for ``Mt`` (megatonne) versus ``MT``/``mt`` (metric
    tonne, common in US reports). Relative units (``/m2``, ``per FTE``) and
    unknown strings are rejected; ``/year`` and ``p.a.`` are absolute.
    """
    if not unit_raw:
        return None
    s = unicodedata.normalize("NFKC", str(unit_raw)).strip()
    s = s.replace("₂", "2")
    # Spanish "tCO2eq/000" / "tCO2e (000)": thousands of tonnes
    if re.fullmatch(r"(?i)\s*t\s*co\s*2\s*-?\s*e(?:q)?\.?\s*(?:/\s*000|\(\s*000\s*\)|x\s*1[.,]?000)\s*", s):
        s = "thousand tCO2e"
    # French order "teqCO2" / "kt éq. CO2" (tonne équivalent CO2)
    s = re.sub(r"(?i)^((?:[kM]?t|tonnes?|tons?)\b|[kM]?t(?=[ée]q))\s*[ée]q\.?\s*(co\s*2)", r"\1 \2e", s)
    # qualifiers such as "(market-based)" or "(gross)" are not part of the unit
    s = re.sub(r"\((?:[^()]*?(?:market|location|based|scope|gross|net|incl|excl|approx|estimat|restat)[^()]*)\)",
               " ", s, flags=re.I)
    s = s.strip(" ()[]:" + _TRAILING_MARKERS)
    s = re.sub(r"\s*(?:/\s*(?:yr|year|annum|a)\.?|per\s+(?:year|annum)|p\.\s?a\.?|pa)\s*$", "", s, flags=re.I)
    if "/" in s or re.search(r"\bper\b|%", s, re.I):
        return None
    # inverted order: "CO2e, tonnes", "CO2e (kt)", "CO2-e in tonnes"
    inv = re.match(r"^(?P<gas>co\s*2?\s*[-\u2010\u2011\u2013]?\s*(?:eq(?:uivalents?)?\.?|e))\s*(?:,|\(|\bin\b)\s*(?:in\s+)?(?P<mass>[^,()]+?)\s*\)?$",
                   s, re.I)
    if inv:
        s = inv.group("mass") + " " + inv.group("gas")
    scale = 1.0
    m = _UNIT_SCALE_RE.match(s)
    if m:
        scale = 1000.0 if m.group("k") else 1_000_000.0
        s = s[m.end():]
    m = _UNIT_MASS_RE.match(s)
    if not m:
        return None
    if m.group("t") or m.group("MT"):
        mass = 1.0
    elif m.group("kt"):
        mass = 1000.0
    elif m.group("Mt"):
        mass = 1_000_000.0
    elif m.group("kg"):
        mass = 0.001
    else:
        mass = 1e-6
    rest = s[m.end():].strip(" .()[]" + _TRAILING_MARKERS)
    if not _UNIT_GAS_RE.match(rest):
        return None
    return scale * mass


def normalize_emission_to_tco2e(value: Optional[float], unit_raw: Optional[str]) -> Tuple[Optional[float], bool]:
    """Return (tCO2e value, unit_assumed); reject unrecognized/relative units.

    Match the complete unit, never a prefix such as ``t`` in ``thousand``
    or ``tCO2e/m2``. The boolean is retained for API compatibility; an
    unsupported unit yields None rather than an assumed absolute amount.
    """
    if value is None:
        return None, False
    factor = emission_unit_multiplier(unit_raw)
    if factor is None:
        return None, False
    out = value * factor
    # Avoid binary artefacts such as 12.3 * 1000 = 12300.000000000002.
    if factor >= 1:
        out = float(f"{out:.10g}")
    return out, False


def normalized_emission_mass_unit(unit_raw: Optional[str]) -> Optional[str]:
    """Name the tonne-normalized unit without changing CO2 into CO2e.

    The numeric normalizer already handles the mass multiplier. Bare tonnes
    stay ``t`` when the source does not name the gas-equivalence basis.
    """
    if normalize_emission_to_tco2e(1.0, unit_raw)[0] is None:
        return None
    unit = _canon_unit(unicodedata.normalize("NFKC", str(unit_raw or ""))).strip("()[]*")
    if re.search(r"co2(?:-?e|eq|equivalent)|carbondioxideequivalent", unit) or re.search(r"co(?:e|eq)$", unit):
        return "tCO2e"
    if "co2" in unit or "carbondioxide" in unit:
        return "tCO2"
    return "t"


def table_year_check(value_raw: Optional[str], report_year: Optional[int],
                     page_text: str) -> Tuple[Optional[bool], str]:
    """Check unambiguous space-separated year columns in original PDF text.

    This deliberately abstains on split headers, merged cells, duplicate
    values, space-grouped numbers, or sparse rows. Those layouts need semantic
    validation with the full page. It never reassigns a year or edits a value.
    Only original text is used, not appended reconstructed-table fragments.
    """
    value = parse_number(value_raw)
    if value is None or report_year is None:
        return None, "year or value unavailable for column check"
    lines = page_text.split("[RECONSTRUCTED_TABLE", 1)[0].splitlines()
    # A PDF's drawing/reading order may put the header after its rows. Locate
    # candidate headers across the page, but use a column count only when it
    # identifies one distinct year sequence. Never borrow a different page's
    # header or select between competing same-width tables.
    year_token = r"20\d{2}\*{0,2}(?![\d/])"
    header_pattern = re.compile(r"(?<![\d/])" + year_token + r"(?:\s+" + year_token + r")+")
    headers_by_width: Dict[int, set] = {}
    header_lines = set()
    for i, line in enumerate(lines):
        header = header_pattern.search(line)
        if not header:
            continue
        header_lines.add(i)
        years = tuple(int(y) for y in re.findall(r"20\d{2}", header.group()))
        if len(set(years)) == len(years) and all(YEAR_MIN <= y <= YEAR_MAX for y in years):
            headers_by_width.setdefault(len(years), set()).add(years)
    matched_years = set()
    aligned_lines = set()
    for i, line in enumerate(lines):
        if i in header_lines:
            continue
        cells = []
        # A numeric suffix separated by whitespace, with placeholders retained.
        # Do not merge spaces: ambiguity means no deterministic verdict.
        for token in reversed(line.split()):
            if token in ("-", "–", "—", "/", "NA", "N/A") or token.endswith("%"):
                cells.append(token)
            elif is_single_number_token(token):
                cells.append(token)
            else:
                break
        cells.reverse()
        # Percent change and target placeholders are outside the historical run.
        change = next((i for i, c in enumerate(cells) if c.endswith("%")), len(cells))
        numeric = cells[:change]
        while numeric and numeric[-1] in ("-", "–", "—", "/", "NA", "N/A"):
            numeric.pop()
        candidates = headers_by_width.get(len(numeric), set())
        if len(candidates) != 1 or any(parse_number(c) is None for c in numeric):
            continue
        headers = next(iter(candidates))
        positions = [i for i, c in enumerate(numeric) if parse_number(c) == value]
        if len(positions) == 1:
            matched_years.add(headers[positions[0]])
            aligned_lines.add(i)
        elif positions:
            return None, "same numeric value occurs in multiple year columns"
    # A repeated amount elsewhere might support a different field/year;
    # do not reject that candidate using an unrelated table's occurrence.
    mentions = {i for i, line in enumerate(lines) if _find_number_span(str(value_raw), line)}
    if mentions - aligned_lines:
        return None, "numeric value also appears outside an unambiguous aligned row"
    if len(matched_years) == 1:
        actual = next(iter(matched_years))
        return actual == report_year, f"unambiguous table column belongs to {actual}, claimed {report_year}"
    return None, "no unambiguous original-text year column; semantic check required"


def normalize_energy_to_kwh(value: Optional[float], unit_raw: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    u = _canon_unit(unit_raw)
    for key, factor in sorted(ENERGY_UNIT_TO_KWH.items(), key=lambda kv: -len(kv[0])):
        if u.startswith(key):
            return value * factor
    return None


# ---------------------------------------------------------------------------
# GHG Protocol Scope 3 category mapping (§3.3.2 stage 3)
# ---------------------------------------------------------------------------

SCOPE3_CATEGORIES: Dict[int, Dict[str, Any]] = {
    1:  {"name": "Purchased goods and services",
         "synonyms": ["purchased goods", "goods and services", "procurement", "purchased services",
                      "purchases of products", "purchases of goods", "purchased products"]},
    2:  {"name": "Capital goods",
         "synonyms": ["capital goods", "embodied carbon", "capital equipment"]},
    3:  {"name": "Fuel- and energy-related activities",
         "synonyms": ["fuel and energy related", "fuel- and energy-related", "energy related activities",
                      "energy-related emissions not included", "upstream fuel", "well to tank", "transmission and distribution losses", "t&d losses"]},
    4:  {"name": "Upstream transportation and distribution",
         "synonyms": ["upstream transportation", "upstream transport", "inbound logistics", "upstream distribution",
                      "upstream freight", "inbound freight"]},
    5:  {"name": "Waste generated in operations",
         "synonyms": ["waste", "waste generated", "operational waste", "waste disposal", "waste to landfill"]},
    6:  {"name": "Business travel",
         "synonyms": ["business travel", "air travel", "corporate travel", "travel"]},
    7:  {"name": "Employee commuting",
         "synonyms": ["employee commuting", "commuting", "staff commuting", "employee travel to work",
                      "working from home", "homeworking"]},
    8:  {"name": "Upstream leased assets",
         "synonyms": ["upstream leased assets", "leased assets upstream", "upstream leases"]},
    9:  {"name": "Downstream transportation and distribution",
         "synonyms": ["downstream transportation", "downstream transport", "outbound logistics",
                      "downstream distribution", "downstream freight", "outbound freight"]},
    10: {"name": "Processing of sold products",
         "synonyms": ["processing of sold products", "processing sold products"]},
    11: {"name": "Use of sold products",
         "synonyms": ["use of sold products", "use phase", "product use", "sold products use",
                      "natural gas sold", "gas sold in the retail market"]},
    12: {"name": "End-of-life treatment of sold products",
         "synonyms": ["end of life", "end-of-life", "product disposal", "sold products end of life",
                      "end of product life", "product end of life", "end of products life"]},
    13: {"name": "Downstream leased assets",
         "synonyms": ["downstream leased assets", "leased assets", "tenant emissions", "tenancies",
                      "leased buildings", "downstream leases"]},
    14: {"name": "Franchises", "synonyms": ["franchise", "franchises"]},
    15: {"name": "Investments", "synonyms": ["investments", "financed emissions", "equity investments"]},
}

_CAT_NO_RE = re.compile(r"\b(?:cat(?:egory)?\.?\s*)(\d{1,2})\b", re.IGNORECASE)


def map_scope3_category(label: Optional[str]) -> Tuple[Optional[int], Optional[str], Optional[str], float]:
    """Map a raw category label to (category_no, canonical_name, method, score).

    Rule-first: explicit "Category N" → exact/synonym containment → fuzzy
    similarity (difflib) with a fixed threshold. Returns (None, None, None, 0)
    when no mapping is confident enough — never guesses.
    """
    if not label:
        return None, None, None, 0.0
    text = normalize_string(label)
    # "Transportation and distribution" without upstream/downstream names
    # either Category 4 or Category 9: no unnumbered guess
    if re.fullmatch(r"(?i)\s*(?:transport(?:ation)?(?:\s*(?:and|&)\s*distribution)?|distribution)\W*\d*\s*", str(label)):
        return None, None, None, 0.0
    m = _CAT_NO_RE.search(str(label))
    bare = None
    if not m:
        bare = re.match(r"^\s*(?:3\.)?\(?(1[0-5]|[1-9])\)?(?:[.\s–-]+)", str(label))
        m = bare
    if not m:
        # "C14 - Franchised stores", "C9: Downstream transportation"
        m = re.match(r"^\s*C\s?(1[0-5]|[1-9])\b", str(label))
    if m:
        no = int(m.group(1))
        if bare is not None and not bare.group(0).strip().startswith("3."):
            # A bare list number follows the report's own numbering (BEGES
            # items, misnumbered rows); a category name that maps exactly or
            # by synonym decides the category.
            by_name = map_scope3_category(str(label)[bare.end():])
            if by_name[0] is not None and by_name[2] in ("exact", "synonym") and by_name[0] != no:
                return by_name[0], by_name[1], "name_over_list_number", by_name[3]
        if 1 <= no <= 15:
            return no, SCOPE3_CATEGORIES[no]["name"], "regex", 1.0
    for no, spec in SCOPE3_CATEGORIES.items():
        if normalize_string(spec["name"]) == text:
            return no, spec["name"], "exact", 1.0
    # Longest contained synonym wins, so "employee travel to work" (Cat 7)
    # beats the generic "travel" (Cat 6) and "upstream leased assets" (Cat 8)
    # beats "leased assets" (Cat 13), independent of category order.
    hits = []
    for no, spec in SCOPE3_CATEGORIES.items():
        for syn in [spec["name"]] + spec["synonyms"]:
            n = normalize_string(syn)
            # word-initial match with plain inflections (travel -> travelling)
            if n and re.search(r"(?<!\w)" + re.escape(n) + r"(?:s|es|d|ed|ing|ling|led)?(?!\w)", text):
                hits.append((len(n), -no, no, spec["name"]))
    if hits:
        _, _, no, name = max(hits)
        return no, name, "synonym", 0.95
    best_no, best_score = None, 0.0
    for no, spec in SCOPE3_CATEGORIES.items():
        for cand in [spec["name"]] + spec["synonyms"]:
            score = difflib.SequenceMatcher(None, text, normalize_string(cand)).ratio()
            if score > best_score:
                best_no, best_score = no, score
    if best_no is not None and best_score >= CATEGORY_FUZZY_THRESHOLD:
        return best_no, SCOPE3_CATEGORIES[best_no]["name"], "fuzzy", best_score
    return None, None, None, best_score


# ---------------------------------------------------------------------------
# Deterministic evidence checking ("Evidence Grounding Rules", §3.3.2 stage 4)
# ---------------------------------------------------------------------------

def _number_variants(value: float) -> List[str]:
    """Render only numerically exact variants; never ground a rounded value."""
    variants: List[str] = [str(value)]
    if float(value).is_integer():
        i = int(round(value))
        variants += [f"{i}", f"{i:,}", f"{i:,}".replace(",", " "), f"{i:,}".replace(",", ".")]
    for dp in (0, 1, 2):
        variants.append(f"{value:,.{dp}f}")
        variants.append(f"{value:.{dp}f}")
    seen, out = set(), []
    for v in variants:
        if v not in seen and parse_number(v) == value:
            seen.add(v)
            out.append(v)
    return out


def _literal_number_pattern(candidate: str) -> Optional[re.Pattern]:
    """Compile a whitespace-tolerant numeric token pattern with hard boundaries."""
    literal = str(candidate or "").strip()
    if not literal or not re.search(r"\d", literal):
        return None
    pieces = [re.escape(p) for p in re.split(r"[\s ]+", literal) if p]
    body = r"[\s ]*".join(pieces)
    return re.compile(r"(?<![\d.,])" + body + r"(?![\d.,])")


def _find_number_span(candidate: str, text: str) -> Optional[Tuple[int, int]]:
    pattern = _literal_number_pattern(candidate)
    match = pattern.search(text or "") if pattern else None
    return match.span() if match else None


def verify_value_in_text(value_raw: Optional[str], value: Optional[float],
                         *texts: Optional[str]) -> Tuple[bool, str]:
    """Deterministically check that a value literally appears in its evidence.

    This is the final numeric verification of the pipeline (Appendix E):
    it matches each extracted value against the source snippet / OCR page
    text and can only flag a value — it never corrects one.
    """
    corpora = [t for t in texts if t]
    if not corpora:
        return False, "no evidence text available"
    if value_raw:
        if any(_find_number_span(str(value_raw), corpus) for corpus in corpora):
            return True, "value_raw matched as a bounded numeric span"
    if value is not None:
        for cand in _number_variants(float(value)):
            if any(_find_number_span(cand, corpus) for corpus in corpora):
                return True, f"bounded numeric variant '{cand}' matched"
    return False, "value not found in evidence text"


def ground_evidence_snippet(value_raw: Optional[str], value: Optional[float],
                            page_text: str, min_chars: int = 100,
                            max_chars: int = 400) -> Optional[str]:
    """Create a deterministic 100-400 character source span around a value.

    Reconstructed tables are line-oriented, so the matched row plus an adjacent
    row is preferred. Prose uses a centred character window. No paraphrasing is
    performed.
    """
    candidates = [str(value_raw)] if value_raw else []
    if value is not None:
        candidates.extend(_number_variants(float(value)))
    # Search the verbatim token across the entire source before considering
    # other exact formats. A rounded variant (e.g. 5 for 5.3) must never select
    # a footnote or another row ahead of the actual amount.
    for candidate in candidates:
        span = _find_number_span(candidate, page_text or "")
        if not span:
            continue
        line_start = page_text.rfind("\n", 0, span[0]) + 1
        line_end = page_text.find("\n", span[1])
        if line_end < 0:
            line_end = len(page_text)
        if min_chars <= line_end - line_start <= max_chars:
            return page_text[line_start:line_end].strip()
        center = (span[0] + span[1]) // 2
        width = min(max_chars, max(min_chars, 300))
        start = max(0, center - width // 2)
        end = min(len(page_text), start + width)
        start = max(0, end - width)
        return page_text[start:end].strip()[:max_chars]
    return None


def verify_field_context(field: str, value_raw: Optional[str], value: Optional[float],
                         page_text: str, category_no: Optional[int] = None) -> bool:
    """Check that a grounded number is near the expected semantic row label."""
    candidates = [str(value_raw)] if value_raw else []
    if value is not None:
        candidates.extend(_number_variants(float(value)))
    windows: List[str] = []
    for candidate in candidates:
        pattern = _literal_number_pattern(candidate)
        if not pattern:
            continue
        for match in pattern.finditer(page_text or ""):
            windows.append(normalize_string(
                page_text[max(0, match.start() - 320):match.end()]))
    if not windows:
        return False
    if field == "scope2_location":
        return any(("location based" in w or "locationbased" in w)
                   and "scope 2" in w for w in windows)
    if field == "scope2_market":
        return any(("market based" in w or "marketbased" in w)
                   and "scope 2" in w for w in windows)
    if field == "scope1":
        return any("scope 1" in w for w in windows)
    if field == "scope2":
        return any("scope 2" in w for w in windows)
    if field == "scope3_total":
        # Some inventories put the Scope 3 heading after the rows in PDF
        # reading order. Accept an explicitly labelled aggregate of material
        # categories only on a page containing a Scope 3 heading. Never treat
        # a purchased-electricity row or a single category as that aggregate.
        if re.search(r"\bscope 3\b", normalize_string(page_text)):
            for line in page_text.splitlines():
                if (any(_find_number_span(c, line) for c in candidates)
                        and re.search(r"\bindirect (?:ghg |greenhouse gas )?emissions "
                                      r"(?:material|all|total) categor", normalize_string(line))):
                    return True
        return any("scope 3" in w and ("total" in w or "gross indirect" in w)
                   for w in windows)
    if field.startswith("scope3_cat") and category_no is not None:
        # The category number must be written as a category marker or a list
        # number, not merely be the last digit of a year or amount.
        number_label = re.compile(rf"(?:\b(?:category|cat)\s*{category_no}\b|(?:^|\s)(?:3\s)?{category_no}\s(?=[a-z]))")
        category_name = normalize_string(SCOPE3_CATEGORIES[category_no]["name"])
        return any(number_label.search(w) or category_name in w for w in windows)
    return True


def category_hierarchy(category_no: Optional[int], value_raw: Optional[str],
                       label_raw: Optional[str], page_text: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Identify a numeric subrow in an explicitly numbered category block.

    Category mapping is not aggregation: e.g. a travel mode belongs to a
    category without being its total. Only use original text and an anchored
    data row. Abstain when the block/row is ambiguous; never sum or infer values.
    The returned fragment is verbatim and binds the child label to its amount.
    """
    if category_no not in SCOPE3_CATEGORIES or not value_raw:
        return None, "category or numeric cell unavailable", None
    text = page_text.split("[RECONSTRUCTED_TABLE", 1)[0]
    markers = list(re.finditer(r"(?im)^[ \t]*(?:category|cat\.?)\s*(\d{1,2})\b", text))
    blocks = []
    for i, marker in enumerate(markers):
        if int(marker.group(1)) != category_no:
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        block = text[marker.start():end]
        boundary = re.search(r"(?im)^\s*(?:SCOPE\s+[123]\b|TOTAL\s+(?:GHG|GREENHOUSE)|"
                             r"TABLE\s+\d|TARGETED\s+GHG|ENVIRONMENT\b)", block)
        if boundary:
            block = block[:boundary.start()]
        blocks.append(block)
    if len(blocks) != 1:
        return None, "no unique numbered category block", None
    block = blocks[0]
    # Data rows contain a run of cells or one decimal value; footnote indices
    # and isolated category numbers must not make a second data row.
    data_row = re.compile(r"\d[\d,.]*\s+[-−]?\d[\d,.]*|\d+[.,]\d+")
    data_lines = [line for line in block.splitlines() if data_row.search(line)
                  and not re.search(r"(?i)target|reduction|per head", line)]
    fragment = None
    if label_raw:
        tokens = re.findall(r"\w+", str(label_raw), re.UNICODE)
        pattern = re.compile(r"\b" + r"[\W_]+".join(re.escape(t) for t in tokens) + r"\b", re.I)
        matches = list(pattern.finditer(block)) if tokens else []
        if len(matches) == 1:
            # Stop at the first data-bearing line after this label, not an
            # arbitrary window that could borrow a sibling's identical value.
            tail = block[matches[0].start():]
            offset = 0
            for line in tail.splitlines(keepends=True):
                offset += len(line)
                if data_row.search(line):
                    fragment = tail[:offset].strip()
                    break
    if fragment is None:
        matches = [line for line in block.splitlines() if _find_number_span(str(value_raw), line)]
        if len(matches) == 1:
            fragment = matches[0]
    if fragment is None or not _find_number_span(str(value_raw), fragment):
        return None, "numeric cell is not bound to the claimed category row", None
    prefix = fragment[:_find_number_span(str(value_raw), fragment)[0]]
    prefix = re.sub(r"(?i)\b(?:category|cat\.?)\s*\d{1,2}\b", "", prefix)
    prefix = re.split(r"(?<!\w)[-−]?\d", prefix, maxsplit=1)[0]
    prefix = normalize_string(prefix)
    canonical = normalize_string(SCOPE3_CATEGORIES[category_no]["name"])
    # Do not use category synonyms here: a synonym such as 'air travel' maps
    # to Cat6 but is not proof that it aggregates the other travel modes.
    explicit_total = (prefix == canonical or prefix in ("total", "total emissions")
                      or prefix in ("total " + canonical, canonical + " total"))
    if explicit_total:
        return "category_total", "explicit category aggregate row", fragment
    if len(data_lines) > 1:
        return "component", "subrow within a multi-row category; not an aggregate", fragment
    if len(data_lines) == 1:
        return "category_total", "single disclosed amount in category block", fragment
    return None, "category hierarchy is ambiguous", fragment


# ---------------------------------------------------------------------------
# Consistency + logic checks (§3.3.4, Appendix A.4)
# ---------------------------------------------------------------------------

def org_logic_checks(org_name: str, year: Optional[int],
                     s1: Optional[float], s2_loc: Optional[float], s2_mkt: Optional[float],
                     s3_total: Optional[float], cat_values: Dict[int, float],
                     s2_total: Optional[float] = None) -> List[DiagnosticResult]:
    out: List[DiagnosticResult] = []
    tag = f"{org_name or 'org'}@{year}"

    if year is not None and not (YEAR_MIN <= year <= YEAR_MAX):
        out.append(DiagnosticResult("logic:reporting_year_range", "fail",
                                    f"{tag}: reporting_year={year} outside [{YEAR_MIN},{YEAR_MAX}]", "org-level"))
    for label, v in (("scope1", s1), ("scope2_total", s2_total), ("scope2_location", s2_loc),
                     ("scope2_market", s2_mkt), ("scope3_total", s3_total)):
        if v is not None and v < 0:
            out.append(DiagnosticResult("logic:non_negative", "fail",
                                        f"{tag}: {label}={v} is negative", "org-level"))

    # A.4: market-based Scope 2 is typically <= location-based.
    if s2_loc is not None and s2_mkt is not None:
        status = "pass" if s2_mkt <= s2_loc * 1.001 else "warn"
        out.append(DiagnosticResult("consistency:s2_market_le_location", status,
                                    f"{tag}: market={s2_mkt} vs location={s2_loc}", "org-level"))

    # Category sum vs disclosed Scope 3 total.
    if s3_total and cat_values:
        cat_sum = sum(cat_values.values())
        rel = abs(cat_sum - s3_total) / s3_total if s3_total else 0.0
        if rel <= CATEGORY_SUM_PASS_TOL:
            status = "pass"
        elif rel <= CATEGORY_SUM_WARN_TOL:
            status = "warn"
        else:
            status = "fail"
        out.append(DiagnosticResult(
            "consistency:scope3_category_sum", status,
            f"{tag}: sum(categories)={cat_sum:,.1f} vs scope3_total={s3_total:,.1f} "
            f"(rel diff {rel:.1%}; note: partial category disclosure is common)", "org-level"))
    return out


def cross_level_checks(org_name: str, year: Optional[int],
                       org_s1: Optional[float], org_s2: Optional[float],
                       bldg_s1_sum: Optional[float], bldg_s2_sum: Optional[float],
                       n_linked: int) -> List[DiagnosticResult]:
    """A.4 cross-level constraints: aggregated building S1/S2 must not exceed org totals."""
    out: List[DiagnosticResult] = []
    tag = f"{org_name or 'org'}@{year}"
    if n_linked == 0:
        out.append(DiagnosticResult("cross:linked_assets", "not_applicable",
                                    f"{tag}: no evidence-linked assets — matching requires "
                                    f"attribution evidence (§3.3.4)", "cross-level"))
        return out
    for label, org_v, bldg_v in (("scope1", org_s1, bldg_s1_sum), ("scope2", org_s2, bldg_s2_sum)):
        if org_v is None or bldg_v is None:
            out.append(DiagnosticResult(f"cross:building_{label}_le_org", "not_applicable",
                                        f"{tag}: missing {label} on one level", "cross-level"))
            continue
        status = "pass" if bldg_v <= org_v * 1.01 else "fail"
        out.append(DiagnosticResult(
            f"cross:building_{label}_le_org", status,
            f"{tag}: sum(building {label})={bldg_v:,.1f} vs org {label}={org_v:,.1f} "
            f"over {n_linked} linked assets", "cross-level"))
    return out


# ---------------------------------------------------------------------------
# Text preprocessing helpers (§3.3.2 stage 1)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "by", "with",
    "is", "are", "was", "were", "be", "as", "at", "from", "this", "that", "it",
}


def clean_page_text(text: str) -> str:
    """De-hyphenate line breaks, collapse whitespace (keeps numbers intact)."""
    if not text:
        return ""
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n", text or "") if s.strip()]


def content_tokens(text: str) -> List[str]:
    return [w for w in normalize_string(text).split() if w not in _STOPWORDS]


_SCOPE_KEYWORD_RE = re.compile(
    r"scope\s*[-–]?\s*[123]\b|tco2|t\s?co2|co2-?e|ghg|greenhouse|carbon\s+(footprint|neutral|emission)|emission",
    re.IGNORECASE)


def page_keyword_score(text: str) -> int:
    """Rule prefilter for emissions-content localization (recall-oriented)."""
    return len(_SCOPE_KEYWORD_RE.findall(text or ""))
