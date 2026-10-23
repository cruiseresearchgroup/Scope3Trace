"""Deterministic text-row binding for values outside a reconstructed table.

Paper §3.3: the Verify Agent performs "deterministic span-level checking of
every extracted value against the source OCR text". When the geometry layer
cannot rebuild a table (unruled layouts, transposed or single-year tables,
prose), a candidate is still checked against the linear PDF text:

  * the complete numeric token occurs on exactly one source line whose label
    matches the claimed row and field;
  * on a multi-value line, a year header of the same width on the page fixes
    the cell's year (otherwise the candidate is rejected, never guessed);
  * the unit multiplier is printed on that line or is the page's only
    absolute emissions unit.

Nothing is inferred, summed or corrected. Verify re-derives the binding.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from scope3trace.common.rules import (SCOPE3_CATEGORIES, emission_unit_multiplier, map_scope3_category,
                               normalize_string, parse_number, ambiguous_separator)

_NUM = re.compile(r"(?<![\w.,])[-−(]?\d[\d.,'  ]*\d\)?(?![\w.,])|(?<![\w.,])\d(?![\w.,])")
_YEAR = re.compile(r"(?<![\d/])((?:19|20)\d{2})(?:/\d{2})?(?![\d/])")
_STOP = {"the", "of", "and", "in", "from", "for", "to", "a", "an", "by", "on", "with", "total",
         "ghg", "emissions", "emission", "tco2e", "co2e", "tonnes", "t", "scope", "gross", "net"}

_FIELD_EVIDENCE = {
    "scope1": r"scope\s*1\b|\bdirect\b",
    "scope2": r"scope\s*2\b|energy indirect|indirect .*energy",
    "scope2_location": r"scope\s*2\b|energy indirect|indirect .*energy",
    "scope2_market": r"scope\s*2\b|energy indirect|indirect .*energy",
    "scope3_total": r"scope\s*3\b|value chain|other indirect",
}


def _tokens(text: str) -> List[str]:
    return [t for t in normalize_string(text or "").split() if len(t) > 1 and t not in _STOP
            and not t.isdigit()]


def _line_spans(raw: str) -> List[Tuple[int, int]]:
    spans, start = [], 0
    for line in raw.split("\n"):
        spans.append((start, start + len(line)))
        start += len(line) + 1
    return spans


def _mask_labels(line: str) -> str:
    """Hide label digits (Scope 1, Category 6, list numbers) from cell scans."""
    hide = lambda m: m.group(1) + "#" * len(m.group(2))
    line = re.sub(r"(?i)\b(scope\s*|category\s*|cat\.?\s*|3\.)(\d{1,2})\b", hide, line)
    line = re.sub(r"^(\s*)(\d{1,2})(?=[.):]?\s+[A-Za-z])", hide, line)
    return line


_CELL = re.compile(r"(?<![\w.,])[-\u2212(]?\d[\d.,'\u2009\u202f]*\)?(?![\w.,])")
_PLACEHOLDER = re.compile(r"(?<!\S)(?:[-\u2013\u2014]|n/?a|N/?A|n\.a\.|na|NA|nr|NR|NM|n\.m\.)(?!\S)")


def _cells(line: str, space_groups: bool = False) -> List[Tuple[str, int, int]]:
    """Value cells of a line in order, excluding years and percentages.

    Placeholders (``-``, ``n/a``) in the numeric tail count as cells so that
    a sparse row still aligns with its year header. Whitespace separates
    cells, except that ``space_groups`` re-joins ``1 234 567`` thousands
    when the claimed token itself uses them.
    """
    out = []
    line = _mask_labels(line)
    for m in _CELL.finditer(line):
        tok = m.group(0)
        if line[m.end():m.end() + 2].strip().startswith("%"):
            continue
        if re.fullmatch(r"\(?(?:19|20)\d{2}\)?", tok):
            continue
        # "N-1", "Y-2": period arithmetic in headers, not a numeric cell
        if re.search(r"[A-Za-z]-\s*$", line[:m.start()]):
            continue
        if parse_number(tok) is None:
            continue
        out.append((tok, m.start(), m.end()))
    if space_groups:
        merged = []
        for tok, a, b in out:
            if merged and a - merged[-1][2] == 1 and re.fullmatch(r"\d{3}(?:[.,]\d+)?", tok) \
                    and re.fullmatch(r"[-\u2212(]?\d{1,3}(?: \d{3})*", merged[-1][0]):
                pt, pa, _ = merged[-1]
                merged[-1] = (pt + " " + tok, pa, b)
            else:
                merged.append((tok, a, b))
        out = merged
    # numeric tail = after the last word containing letters (label, unit)
    words = list(re.finditer(r"\S+", line))
    alpha_end = max((w.end() for w in words if re.search(r"[A-Za-z]{2,}", w.group(0))
                     and not _PLACEHOLDER.fullmatch(w.group(0))), default=0)
    if out:
        holders = [(m.group(0), m.start(), m.end()) for m in _PLACEHOLDER.finditer(line)
                   if m.start() >= alpha_end and not line[m.end():m.end() + 2].strip().startswith("%")]
        out = sorted(out + holders, key=lambda c: c[1])
    return out


def _is_value(cell) -> bool:
    return not _PLACEHOLDER.fullmatch(cell[0])


def _header_years(lines: List[str]) -> List[Tuple[int, ...]]:
    heads = []
    for line in lines:
        line = re.sub(r"\bFY\s?(\d{2})\b", lambda m: "20" + m.group(1), line)
        if re.search(r"\bfrom\s+20\d{2}\s+to\s+20\d{2}|\bby\s+20\d{2}|\btarget", line, re.I):
            continue
        years = []
        for m in _YEAR.finditer(line):
            if line[m.end():m.end() + 1] == "%":
                continue
            y = int(m.group(1))
            if m.group(0).find("/") > 0:
                y += 1
            years.append(y)
        nums = [c for c in _cells(line) if _is_value(c)]
        if len(years) >= 2 and len(set(years)) == len(years) and not nums:
            last = list(_YEAR.finditer(line))[-1]
            change = bool(re.search(r"%|change|chg|var|\u0394|\u2206|delta|vs\.?|evolution", line[last.end():], re.I))
            heads.append((tuple(years), change))
    return heads


def _label_ok(label: str, context: str, field: str, category_no: Optional[int]) -> bool:
    ctx = normalize_string(context)
    lab = _tokens(label)
    overlap = sum(1 for t in lab if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", ctx)) / len(lab) if lab else 0.0
    if field.startswith("scope3_cat") and category_no is not None:
        marker = re.search(rf"\b(?:category|cat)\s*{category_no}\b|(?:^|\s){category_no}\s(?=[a-z])", ctx)
        name = normalize_string(SCOPE3_CATEGORIES[category_no]["name"]) in ctx
        mapped = map_scope3_category(context)[0] == category_no
        return bool(marker or name or mapped) and (overlap >= .5 or not lab)
    pat = _FIELD_EVIDENCE.get(field)
    if not pat or not re.search(pat, ctx):
        return False
    if field == "scope2_location" and not re.search(r"location|site[- ](?:specific|based)", ctx):
        return False
    if field == "scope2_market" and not re.search(r"market", ctx):
        return False
    if field in ("scope2_location", "scope2_market"):
        # the basis named nearest before the amount is the amount's basis
        last = [m.group(0) for m in re.finditer(r"location|site[- ](?:specific|based)|market", ctx)]
        if last and (last[-1] == "market") != (field == "scope2_market"):
            return False
    other = {"scope1": r"scope\s*[23]\b", "scope3_total": r"scope\s*[12]\b"}.get(field)
    if field.startswith("scope2"):
        other = r"scope\s*[13]\b"
    if other and re.search(other, ctx):
        return False
    return overlap >= .5 or not lab


def bind_text_row(value: Any, raw_text: str, locale: Optional[str] = None,
                  doc_id: Optional[str] = None, page: Optional[int] = None,
                  footnotes: Optional[Dict[str, str]] = None) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """Return (binding, ok, reason) for one candidate against linear text.

    ``footnotes`` maps tokens whose trailing digits are glyph-proven
    superscript markers to their amount (``1596`` -> ``159``).
    """
    footnotes = footnotes or {}
    token = str(value.value_raw or "").strip()
    numeric = parse_number(token, locale)
    if numeric is None or not raw_text:
        return None, False, "no numeric token"
    if ambiguous_separator(token, locale):
        return None, False, "ambiguous decimal/thousands separator: no page or document number convention"
    factor = emission_unit_multiplier(value.unit_raw)
    lines = raw_text.split("\n")
    spans = _line_spans(raw_text)
    hits = []
    space_groups = bool(re.search(r"\d\s\d{3}", token))
    for i, line in enumerate(lines):
        cells = _cells(line, space_groups)
        amount = lambda tok: footnotes.get(tok, tok)
        pos = [k for k, (tok, a, b) in enumerate(cells) if parse_number(amount(tok), locale) == numeric
               and re.sub(r"\s+", "", amount(tok)) == re.sub(r"\s+", "", token)]
        if not pos:
            continue
        first = cells[0][1] if cells else len(line)
        prefix = line[:first]
        context = prefix if prefix.strip() else ""
        # wrapped labels: up to two preceding text-only lines
        j = i - 1
        while j >= 0 and i - j <= 2 and not _cells(lines[j]) and lines[j].strip():
            context = lines[j] + " " + context
            j -= 1
        # running prose wrapped before the amount ("... market-based Scope 2
        # emissions reached / 176,859.64 tCO2e, while location-based emissions
        # were / 198,060.66 tCO2e."): the clause ending the previous line
        # names the amount that opens this line
        if not context and pos == [0] and i >= 1 and re.search(r"(?i)\b(?:were|was|is|are|reached|amounted to|totall?ed|of)\s*$", lines[i-1].strip()):
            prev_cells = _cells(lines[i-1])
            tail = lines[i-1][prev_cells[-1][2]:] if prev_cells else lines[i-1]
            context = tail
            if i >= 2 and not re.search(r"(?i)scope\s*[123]", tail):
                prev2 = _cells(lines[i-2])
                context = (lines[i-2][prev2[-1][2]:] if prev2 else lines[i-2]) + " " + context
        ok_label = _label_ok(value.label_raw or value.category_label_raw or "", context,
                             value.field, value.category_no)
        if not ok_label and value.field in ("scope2_location", "scope2_market") \
                and re.match(r"\s*(?:gross\s+)?(?:location|market)[ -]based", context, re.I):
            # "Indirect GHG emissions (scope 2)" / "Location-based 184.43 tCO2e":
            # the basis row sits under the nearest preceding Scope 2 heading.
            for j in range(i - 1, max(-1, i - 4), -1):
                if re.search(r"scope\s*[13]\b", lines[j], re.I):
                    break
                if re.search(r"scope\s*2\b|energy[ -]indirect", lines[j], re.I):
                    ok_label = _label_ok(value.label_raw or "", lines[j] + " " + context, value.field, None)
                    break
        if not ok_label:
            continue
        hits.append((i, pos, cells, context))
    if not hits:
        return None, False, "no source line carries this number with the claimed row label"
    if len({lines[i] for i, _, _, _ in hits}) > 1:
        return None, False, "number and label occur on several source lines"
    i, pos, cells, context = hits[0]
    if len(pos) != 1:
        return None, False, "same number appears in several cells of the source row"
    k = pos[0]
    year_basis = None
    values_on_row = [c for c in cells if _is_value(c)]
    if len(values_on_row) == 1 and len(cells) == 1:
        year_basis = "single value on row"
        mentioned = {int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", raw_text)}
        if value.report_year is None or (mentioned and value.report_year not in mentioned):
            return None, False, "reporting year not stated on the source page"
        heads_all = [h for h, _ in _header_years(lines)]
        if heads_all and not any(value.report_year in h for h in heads_all):
            return None, False, "page year headers do not include the claimed year"
    else:
        # exact width, or trailing variance columns announced by the header
        heads = {h for h, change in _header_years(lines)
                 if len(h) == len(cells) or (change and len(h) < len(cells) and k < len(h))}
        if len(heads) != 1:
            return None, False, "multi-value source row without one unambiguous year header"
        years = next(iter(heads))
        if years[k] != value.report_year:
            return None, False, f"source column year {years[k]} differs from claimed {value.report_year}"
        year_basis = "header " + " ".join(map(str, years))
    # unit: printed on the row / its wrapped label, else the page's only unit
    from scope3trace.scope_agent.table_structure import _raw_units
    row_units = _raw_units(context + " " + lines[i])
    page_units = {u["multiplier"] for u in _raw_units(raw_text)}
    if row_units:
        mult = {u["multiplier"] for u in row_units}
        if len(mult) != 1:
            return None, False, "conflicting units on the source row"
        unit_mult = next(iter(mult))
        unit_text = row_units[0]["text"]
        unit_basis = "row unit"
    elif len(page_units) == 1:
        unit_mult = next(iter(page_units))
        unit_text = _raw_units(raw_text)[0]["text"]
        unit_basis = "only absolute emissions unit on page"
    else:
        return None, False, "unit multiplier not determinable from source text"
    if factor is not None and factor != unit_mult:
        return None, False, "claimed unit multiplier contradicts the source unit"
    a, b = cells[k][1], cells[k][2]
    if cells[k][0] in footnotes:
        b = a + len(footnotes[cells[k][0]])
    s0 = spans[i][0]
    return ({"origin": "text_row", "status": "text_bound", "document": doc_id, "page": page,
             "field": value.field, "year": value.report_year, "parent_category": value.category_no,
             "line_index": i, "line_span": [spans[i][0], spans[i][1]],
             "source_span": [s0 + a, s0 + b], "source_fragment": raw_text[s0 + a:s0 + b],
             "cell_index": k, "n_cells": len(cells), "year_basis": year_basis,
             "unit_multiplier": unit_mult, "unit_text": unit_text, "unit_basis": unit_basis, "number_locale": locale,
             "source_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
             "quantity_type": "inventory"},
            True, "text row binding: label, year column and unit verified in source text")


def validate_text_binding(value: Any, raw_text: str, locale: Optional[str] = None,
                          footnotes: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    old = value.evidence.binding or {}
    new, ok, reason = bind_text_row(value, raw_text, locale, old.get("document"), old.get("page"), footnotes)
    if not ok:
        return False, reason
    for key in ("source_span", "source_fragment", "cell_index", "year", "field", "unit_multiplier",
                "source_sha256", "line_span"):
        if old.get(key) != new.get(key):
            return False, f"text binding changed or was tampered with: {key}"
    expected = parse_number(new["source_fragment"], locale)
    if expected is None or value.value_tco2e is None or \
            abs(value.value_tco2e - expected * new["unit_multiplier"]) > 1e-9 * max(1.0, abs(value.value_tco2e)):
        return False, "normalized amount does not equal source token times source unit"
    return True, "text-row binding independently revalidated"
