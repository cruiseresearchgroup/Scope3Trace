"""Coordinate-backed emissions tables and fail-closed value bindings.

The linear PDF text and the spatial table are independent representations.
Neither an LLM's cell id nor a matching number anywhere on a page is evidence
of the row/year/unit relationship. Bindings are re-derived during verification.
"""
from __future__ import annotations

import hashlib
import json
import contextvars
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from statistics import median

from scope3trace.common.rules import map_scope3_category, normalize_emission_to_tco2e, parse_number, ambiguous_separator

VERSION = "geometry-year-grid-v5.7"
YEAR = re.compile(r"^\(?((?:19|20)\d{2})\)?(?:[¹²³*])*$")
YEAR_RANGE = re.compile(r"(?:FY)?(?:19|20)\d{2}(?:\s*[-/\u2013]\s*(?:19|20)?\d{2})?\*?")
CATEGORY = re.compile(r"(?:^|\b)(?:category\s*|cat\.?\s*)(1[0-5]|[1-9])\b|^\s*\(?(1[0-5]|[1-9])\)?[:.\s–-]+", re.I)
UNIT = re.compile(r"(?<![A-Za-z])t\s*CO\s*[\u20822]\s*-?\s*eq?\.?\s*(?:/\s*000|\(\s*000\s*\))"
                  r"|(?<![A-Za-z])(?:[kM]?t|tonnes?)\s*[\u00e9e]q\.?\s*CO\s*[\u20822]"
                  r"|(?<![A-Za-z])(?:(?:in\s+)?(?:(?:['\u2019]\s*)?000s?\)?|\(\s*000s?\s*\)|1[,. ]000|thousands?(?:\s+of)?|millions?(?:\s+of)?)\s*)?(?:metric\s+)?(?:kilo\s*tonnes?|ktonnes|kilotonnes|mega\s*tonnes?|tonnes?|tons?|kg|Gg|[kM]?t|MT|tn)\s*(?:of\s*)?(?:CO\s*[\u20822]?\s*[-\u2010\u2011\u2013]?\s*(?:eq(?:uivalents?)?(?![a-z])\.?|e\b)|CO\s*[\u20822](?![A-Za-z0-9])|carbon\s+dioxide(?:\s+equivalents?)?)"
                  # inverted order: "CO2e, tonnes" / "CO2e (kt)"
                  r"|(?<![A-Za-z])CO\s*[\u20822]?\s*[-\u2010\u2011\u2013]?\s*(?:eq(?:uivalents?)?\.?|e)\s*(?:,|\(|\bin\b)\s*(?:in\s+)?"
                  r"(?:(?:thousands?|millions?|['\u2019]?000)\s*(?:of\s+)?)?(?:metric\s+)?(?:kilo\s*tonnes?|tonnes?|tons?|kt|Mt|MT|t)\b\)?", re.I)


def _relative_after(rest):
    """A unit followed by '/x' or 'per x' is an intensity, except per year."""
    return bool(re.match(r"\s*(?:/(?!\s*(?:yr|year|annum|a)\b)|per\s+(?!year|annum))", rest, re.I))


def _year_words(words, page):
    """Separate attached footnotes only when smaller, raised glyphs prove it."""
    result = []
    for word in words:
        w = dict(word)
        fiscal = re.fullmatch(r'((?:19|20)\d{2})[/-](\d{2})', w['text'])
        if fiscal and int(fiscal.group(2)) == (int(fiscal.group(1))+1)%100:
            w.update(text=str(int(fiscal.group(1))+1), source_text=word['text'], fiscal_period=word['text'])
        # balance-date headers: "31.12.2024", "31/12/2024"
        date_year = re.fullmatch(r'(?:[0-3]?\d[./-][01]?\d[./-])((?:19|20)\d{2})', w['text'])
        if date_year:
            w.update(text=date_year.group(1), source_text=word['text'], fiscal_period=word['text'])
        # years typeset with a thousands separator ("2,024", "2.025")
        sep_year = re.fullmatch(r'(2)[,.](0[0-4]\d)', w['text'])
        if sep_year:
            w.update(text=sep_year.group(1) + sep_year.group(2), source_text=word['text'], sep_year=True)
        current = re.fullmatch(r'N((?:19|20)\d{2})', w['text'])
        if current:
            w.update(text=current.group(1), source_text=word['text'])
        # FY2024, FY24, FY 2024/25 and 2023/2024 headers bind to the year in
        # which the reporting period ends; the literal header is retained.
        fy = re.fullmatch(r'(?:FY|F\.Y\.)\s?((?:19|20)?\d{2})(?:\s?[/\-\u2013]\s?((?:19|20)?\d{2}))?', w['text'], re.I)
        if fy:
            first = int(fy.group(1)) + (2000 if len(fy.group(1)) == 2 else 0)
            end = first
            if fy.group(2):
                second = int(fy.group(2)) + (2000 if len(fy.group(2)) == 2 else 0)
                end = second if second == first+1 else None
            if end and 1990 <= end <= 2040:
                w.update(text=str(end), source_text=word['text'], fiscal_period=word['text'])
        span = re.fullmatch(r'((?:19|20)\d{2})\s?[/\-\u2013]\s?((?:19|20)\d{2})', w['text'])
        if span and int(span.group(2)) == int(span.group(1))+1:
            w.update(text=span.group(2), source_text=word['text'], fiscal_period=word['text'])
        # "2024(tCO2e)": a year caption carrying its unit
        year_unit = re.fullmatch(r'((?:19|20)\d{2})\s*\(([^()]{1,24})\)', w['text'])
        if year_unit and UNIT.fullmatch(year_unit.group(2).strip()):
            w.update(text=year_unit.group(1), source_text=word['text'], header_unit=year_unit.group(2).strip())
        paren_ref = re.fullmatch(r'\(((?:19|20)\d{2})\)\d{1,2}\*?', w['text'])
        if paren_ref:  # "(2024)1": a bracketed year followed by its footnote number
            w.update(text=paren_ref.group(1), source_text=word['text'])
        paren_note = re.fullmatch(r'((?:19|20)\d{2})\(\d{1,2}\)\*?', w['text'])
        if paren_note:
            w.update(text=paren_note.group(1), source_text=word['text'])
        # Comma-separated footnote references (20231,2,3) cannot be a year.
        notes = re.fullmatch(r'((?:19|20)\d{2})(\d{1,2}(?:,\d{1,2})+)', w['text'])
        if notes:
            w.update(text=notes.group(1), source_text=word['text'], footnote_refs=notes.group(2))
        if re.fullmatch(r'(?:19|20)\d{2}\d{1,2}', w['text']):
            chars = sorted([c for c in page.chars
                            if w['x0']-.1 <= c['x0'] < w['x1']+.1
                            and w['top']-.1 <= c['top'] <= w['bottom']+.1], key=lambda c:c['x0'])
            if ''.join(c['text'] for c in chars) == w['text'] and len(chars) >= 5:
                base, foot = chars[:4], chars[4:]
                bsz = median(b.get('size',b['bottom']-b['top']) for b in base)
                bbot = median(b['bottom'] for b in base)
                if all((c.get('size',c['bottom']-c['top']) < .85*bsz
                        or (c.get('size',c['bottom']-c['top']) <= bsz and c['bottom'] < bbot - 1.5))
                       and c['top'] < median(b['top'] for b in base)-.3 for c in foot):
                    w.update(text=w['text'][:4], source_text=word['text'],
                             footnote_bbox=_bbox(foot), x1=max(c['x1'] for c in base),
                             top=min(c['top'] for c in base), bottom=max(c['bottom'] for c in base))
        # Small glyphs inside a year token (stray separator, footnote marker:
        # "20,25" + raised "2") are not part of the year.
        if not YEAR.fullmatch(w['text']) and re.fullmatch(r"[12][\d,.']{4,8}", w['text']):
            chars = [c for c in page.chars if w['x0']-.1 <= c['x0'] < w['x1']+.1
                     and w['top']-1 <= c['top'] <= w['bottom']+.1 and c['text'].strip()]
            if chars and ''.join(c['text'] for c in sorted(chars, key=lambda c: c['x0'])) == w['text']:
                big = max(c.get('size', c['bottom']-c['top']) for c in chars)
                kept = ''.join(c['text'] for c in sorted(chars, key=lambda c: c['x0'])
                               if c.get('size', c['bottom']-c['top']) >= .8*big)
                if re.fullmatch(r"(?:19|20)\d{2}", kept) and kept != w['text']:
                    w.update(text=kept, source_text=word['text'])
        # "Scope 1⁴,⁷": a scope/category number with raised footnote glyphs
        prev = result[-1] if result else None
        if (prev and re.fullmatch(r"(?i)scopes?|category|cat\.?", prev['text']) and abs(prev['top'] - w['top']) < 3
                and 0 <= w['x0'] - prev['x1'] < 8 and re.fullmatch(r"(?:1[0-5]|[1-9])[\d,*]{1,5}", w['text'])):
            chars = sorted([c for c in page.chars if w['x0']-.1 <= c['x0'] < w['x1']+.1
                            and w['top']-1 <= c['top'] <= w['bottom']+.1 and c['text'].strip()], key=lambda c: c['x0'])
            if chars and ''.join(c['text'] for c in chars) == w['text']:
                big = chars[0].get('size', chars[0]['bottom'] - chars[0]['top'])
                base = [c for c in chars if c.get('size', c['bottom'] - c['top']) >= .85*big]
                foot = chars[len(base):]
                if foot and base == chars[:len(base)] and all(c['top'] < base[0]['top'] - .3 for c in foot):
                    w.update(text=''.join(c['text'] for c in base), source_text=word['text'],
                             footnote_bbox=_bbox(foot), x1=max(c['x1'] for c in base))
        result.append(w)
    # "2,024 | 2,025" header captions: a separator-typed year is a year only
    # beside another one on its line; a lone "2,000" is an amount
    for w in result:
        if w.pop('sep_year', False) and not any(x is not w and x.get('source_text', '') != x['text']
                                               and re.fullmatch(r'2[,.]0[0-4]\d', x.get('source_text', ''))
                                               and abs(x['top'] - w['top']) < 3 for x in result):
            w['text'] = w.pop('source_text')
    period = getattr(page, '_s3t_period_year', None) or _PERIOD_YEAR.get()
    # "2,045 | 2,028 | 2,025" after "(thousands of tons of CO2e)" is a row of
    # amounts: separator years count only on a line that reads as a header.
    sep = [w for w in result if w.get('source_text') and re.fullmatch(r'2[,.]0\d\d', w['source_text'])]
    for w in sep:
        line = [x for x in result if abs(x['top'] - w['top']) < 3]
        seps = [x for x in line if x.get('source_text') and re.fullmatch(r'2[,.]0\d\d', x['source_text'])]
        implausible = any(int(x['text']) > (period or 2026) + 1 and int(x['text']) not in (2030, 2035, 2040, 2045, 2050)
                          for x in seps) or any(int(x['text']) > 2050 for x in seps)
        row_text = any(re.search(r"(?i)co2|^tons?\b|^tonnes?\b|^\(?thousands?\b", x['text']) for x in line if x not in seps) \
            or sum(1 for x in line if re.fullmatch(r"[a-z]{2,}", x['text'])) >= 2
        # other grouped amounts on the line ("1,200 1,900 2,000 2,000 1,939")
        amounts = any(re.fullmatch(r"[-(]?\d{1,3}(?:[,.]\d{3})+(?:[.,]\d+)?\)?", x['text']) for x in line if x not in seps)
        if implausible or row_text or amounts:
            for x in seps:
                x['text'] = x.pop('source_text')
    return _resolve_relative_years(result, period)


_PERIOD_YEAR = contextvars.ContextVar('s3t_period_year', default=None)


def _resolve_relative_years(words, document_period_year=None):
    """ESRS templates label columns "N" / "N-1". They are resolved only when
    the page itself states the reporting-period end ("... to 30 June 2025",
    "financial year 2024"); otherwise they stay unresolved and unbound."""
    rel = [w for w in words if re.fullmatch(r"N(?:\s?[-\u2013\u2212]\s?[1-3])?", w['text'])]
    if not rel:
        return words
    from scope3trace.common.rules import infer_period_end_year
    end = infer_period_end_year([" ".join(x['text'] for x in words)]) or document_period_year
    if not end:
        return words
    for w in rel:
        same_line = [x for x in words if x is not w and abs(x['top'] - w['top']) < 3]
        # only a standalone column caption; "(total N-1 - total N-2)" is a formula
        touching = [x for x in same_line if (0 <= w['x0'] - x['x1'] < 6 or 0 <= x['x0'] - w['x1'] < 6)]
        if any(re.search(r"[A-Za-z]{2,}", x['text']) or re.search(r"[()]", x['text'])
               or re.fullmatch(r"\(?(?:19|20)\d{2}\)?\S*", x['text']) for x in touching):
            continue  # a formula, or "N 2024" where the year is printed anyway
        if not any(re.match(r"\(?(?:N(?:\b|\s?[-/\u2013\u2212])|%|/|Base\b|Comparative|(?:19|20)\d{2}\b)", x['text']) for x in same_line):
            continue
        m = re.fullmatch(r"N(?:\s?[-\u2013\u2212]\s?([1-3]))?", w['text'])
        k = int(m.group(1) or 0)
        w.update(text=str(end - k), source_text=w.get('source_text', w['text']), relative_year=k)
    return words


def _boundary(text):
    s = text.lower()
    s = re.sub(r"\bgroup\s+(?:ambition|strategy|targets?|programme|program|initiative)\b", " ", s)
    if re.search(r'non.financial indicators of|parent.company|standalone|company.only',s):
        return 'individual_entity'
    if re.search(r'group\s*(?:&|and|plus)\s*companies|group.*companies with operational control',s):
        return 'group_and_operational_control'
    if 'operational control' in s and 'group' not in s:
        return 'additional_operational_control'
    if re.search(r'\bgroup\b|consolidated',s):
        return 'consolidated_group'
    return 'reported'


def _text(words):
    # Bin boundaries must not move a word with an attached raised footnote to
    # the start of its row. Use the same baseline grouping as reconstruction.
    return " ".join(w['text'] for line in _lines(words) for w in line)


_NUMERIC_WORD = re.compile(r"[\d.,' \u2212\u2013\u2014()\-]+[*\u2020\u2021\u00a7\u2713\u2714#]*(?:(?<=[*†‡§])\d{1,2}(?:,\d{1,2})*)?|n\.?s\.?|N/A", re.I)
_CHANGE_NEIGHBOUR = re.compile(r"^(?:%|/|vs\.?|v\.|\u0394|\u2206|delta|change|chg|var\.?|variation|variaci\u00f3n|evolution|yoy|between|since|versus|compared|difference)$", re.I)


def _strip_superscript(word, page):
    """Separate a trailing footnote digit run printed smaller and raised.

    ``159`` followed by a superscript ``6`` is extracted by the PDF layer as
    ``1596``. Only glyph-size and baseline evidence may split it; the full
    source token is retained for raw-text verification.
    """
    text = word['text']
    if not re.fullmatch(r"\d[\d.,]*\d", text):
        return word
    chars = sorted([c for c in page.chars if word['x0']-.1 <= c['x0'] < word['x1']+.1
                    and word['top']-1 <= c['top'] <= word['bottom']+.1 and c['text'].strip()],
                   key=lambda c: c['x0'])
    if ''.join(c['text'] for c in chars) != text or len(chars) < 2:
        return word
    size = lambda c: c.get('size', c['bottom']-c['top'])
    k = 0
    while k < len(chars)-1 and chars[-1-k]['text'].isdigit():
        k += 1
    for n in range(k, 0, -1):
        base, foot = chars[:-n], chars[-n:]
        if not base or not base[-1]['text'].isdigit():
            continue
        bs = median(size(b) for b in base)
        bbottom = median(b['bottom'] for b in base)
        # superscript: clearly smaller glyphs sitting on a raised baseline
        if all((size(c) < .8*bs and c['bottom'] < bbottom - .8) or (size(c) < .97*bs and c['bottom'] < bbottom - 1.5)
               for c in foot):
            w = dict(word)
            w.update(text=''.join(c['text'] for c in base), source_token=text,
                     footnote_marker=''.join(c['text'] for c in foot),
                     x1=max(c['x1'] for c in base))
            return w
    return word


def _numeric_units(line, cut):
    """Numeric words right of the label cut, adjacent fragments merged.

    Grouped thousands may be split into several words (``1 234 567``); those
    are separated by a single space, much narrower than a column gutter.
    """
    def in_prose(w):
        # "Included in Scopes 1 and 2": a digit glued to words is text, not a cell
        for x in line:
            if x is w or not re.search(r"[A-Za-z]", x['text']):
                continue
            if 0 <= w['x0'] - x['x1'] < 4.5 and re.fullmatch(r"[A-Za-z][A-Za-z.,;:'-]*", x['text']):
                return True
            if 0 <= x['x0'] - w['x1'] < 4.5 and re.fullmatch(r"[a-z]{2,}[.,;:]?", x['text']) \
                    and not UNIT.match(x['text']):
                return True
        return False
    def percent(w):
        # "-3 %": a percentage/variance printed with a detached sign
        return any(x['text'] in ('%', 'pp', 'p.p.') and 0 <= x['x0']-w['x1'] < 6 for x in line)
    ws = sorted([w for w in line if w['x1'] >= cut and _NUMERIC_WORD.fullmatch(w['text'])
                 and not in_prose(w) and not percent(w)], key=lambda w: w['x0'])
    units = []
    for w in ws:
        if (units and w['x0']-units[-1][-1]['x1'] < 4.5
                and re.fullmatch(r"\d{3}(?:[.,]\d+)?[*\u2020\u2021\u00a7\u2713\u2714#]*", w['text'])
                and re.fullmatch(r"[\-\u2212(]?\d{1,3}", units[-1][-1]['text'])):
            units[-1].append(w)
        else:
            units.append([w])
    return units


def _merge_left_aligned(clusters):
    """Merge right-edge clusters that are one left-aligned physical column.

    A left-aligned column has a common left edge and ragged right edges, so
    its numbers split into several right-edge clusters. Clusters are merged
    when at least three numbers share one left edge (within 2.5pt) and that
    edge holds the majority of every merged cluster's numbers.
    """
    if len(clusters) < 2:
        return clusters
    pts = sorted(((p[1], k) for k, c in enumerate(clusters) for p in c))
    groups = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - groups[-1][-1][0] > 2.5:
            groups.append([p])
        else:
            groups[-1].append(p)
    parent = list(range(len(clusters)))

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k
    for g in groups:
        if len(g) < 3:
            continue
        ks = sorted({k for _, k in g})
        ks = [k for k in ks if sum(kk == k for _, kk in g) * 2 > len(clusters[k])]
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    merged = defaultdict(list)
    for k, c in enumerate(clusters):
        merged[find(k)].extend(c)
    return [sorted(merged[k]) for k in sorted(merged, key=lambda r: min(p[0] for p in merged[r]))]


def _rotated_words(page, words):
    """Column captions set in vertical type ("LP | LTR | OA | Group").

    pdfplumber returns rotated glyphs as one-letter fragments on different
    baselines. Each vertical run of rotated letters becomes one caption word,
    placed at the run's vertical centre so that captions of one header band
    share a line. Only short alphabetic runs are converted.
    """
    chars = [c for c in getattr(page, 'chars', []) or []
             if c.get('matrix') and abs(c['matrix'][1]) > .9 and abs(c['matrix'][0]) < .1 and c['text'].strip()]
    if not chars:
        return words
    cols = []
    for c in sorted(chars, key=lambda c: (c['x0'], c['top'])):
        col = next((k for k in cols if abs(k[0]['x0'] - c['x0']) < 1.5), None)
        if col is None:
            cols.append([c])
        else:
            col.append(c)
    runs = []
    for col in cols:
        col.sort(key=lambda c: c['top'])
        run = [col[0]]
        for c in col[1:]:
            if c['top'] - run[-1]['bottom'] > 2:
                runs.append(run); run = [c]
            else:
                run.append(c)
        runs.append(run)
    out, boxes = [], []
    for run in runs:
        seq = run if run[0]['matrix'][1] < 0 else run[::-1]
        text = ''.join(c['text'] for c in seq)
        if not re.fullmatch(r"[A-Za-z]{2,12}", text):
            continue
        x0, x1 = min(c['x0'] for c in run), max(c['x1'] for c in run)
        t0, t1 = min(c['top'] for c in run), max(c['bottom'] for c in run)
        mid = (t0 + t1) / 2
        boxes.append((x0, x1, t0, t1))
        out.append({'text': text, 'x0': x0, 'x1': x1, 'top': mid - 3, 'bottom': mid + 3,
                    'doctop': mid - 3, 'upright': False, 'rotated': True,
                    'rotated_bbox': [round(x0, 2), round(t0, 2), round(x1, 2), round(t1, 2)]})
    # captions of one header band share their vertical centre; a lone
    # rotated run (a page-margin running title) is not a column caption
    pairs = [(o, b) for o, b in zip(out, boxes)
             if sum(1 for q in out if q is not o and abs((q['top'] + 3) - (o['top'] + 3)) < 3) >= 2]
    out, boxes = [o for o, _ in pairs], [b for _, b in pairs]
    if not out:
        return words
    keep = [w for w in words if not any(a - 1 <= w['x0'] and w['x1'] <= b + 1 and c - 1 <= w['top'] and w['bottom'] <= d + 1
                                         for a, b, c, d in boxes)]
    return keep + out


def _drop_detached_scripts(words):
    """Remove sub/superscript digits extracted as their own word on a
    slightly offset baseline ("tCO" + lowered "2" + "eq"). They touch a
    letter-ending word horizontally and are never table amounts."""
    out = []
    glued = {}
    for w in words:
        if re.fullmatch(r"\d{1,2}|[24]\)", w['text']):
            h = w['bottom'] - w['top']
            touching = [x for x in words if x is not w and 0.8 < abs(x['top'] - w['top']) < 7
                        and re.search(r"[A-Za-z]$", x['text'])
                        and -1.5 <= w['x0'] - x['x1'] <= 1.5
                        and h <= (x['bottom'] - x['top']) * .85]
            if touching:
                # the CO2 subscript completes its chemical symbol ("tCO" -> "tCO2")
                if re.search(r"(?:CO|CH|N)$", touching[0]['text']) and w['text'] in ('2', '4', '2)', '4)'):
                    glued[id(touching[0])] = w
                continue
        out.append(w)
    # the glued word spans its subscript glyph, so the unit's glyph check sees it
    return [dict(w, text=w['text'] + glued[id(w)]['text'], x1=max(w['x1'], glued[id(w)]['x1']))
            if id(w) in glued else w for w in out]


def _caption_right(year_word, words):
    """Right edge of a year caption: the year plus directly following words
    on its baseline ("2022 (baseline)", "2025 emissions"), stopping at the
    next year or a variance word."""
    x1 = year_word['x1']
    same = sorted((x for x in words if x is not year_word and abs(x['top'] - year_word['top']) < 3
                   and x['x0'] >= year_word['x1'] - .5), key=lambda x: x['x0'])
    for x in same:
        if x['x0'] - x1 >= 6:
            break
        if YEAR.fullmatch(x['text']) or _CHANGE_NEIGHBOUR.match(x['text']) or len(x['text']) > 14:
            break
        x1 = x['x1']
    return x1


def _caption_runs(words):
    """Split caption words into horizontally contiguous runs per baseline."""
    runs = []
    for line in _lines(words):
        cur = [line[0]]
        for w in line[1:]:
            if w['x0'] - cur[-1]['x1'] < 8:
                cur.append(w)
            else:
                runs.append(cur)
                cur = [w]
        runs.append(cur)
    return runs


_STATUS_WORD = re.compile(r"(?i)not|n/?a|n\.a\.|na|nil|none|nm|available|applicable|reported|relevant|material|calculated|disclosed|estimated")


def _is_base_year_caption(w, words):
    """"BASE YEAR 2024" / "Base year (2019)": a baseline column, not the
    reporting-year column, when the same year is also a reported column."""
    left = sorted([x for x in words if abs(x['top']-w['top']) < 3 and 0 <= w['x0']-x['x1'] < 40],
                  key=lambda x: -x['x1'])
    if len(left) >= 2 and w['x0'] - left[0]['x1'] < 6:
        pair = {left[0]['text'].lower().strip(':'), left[1]['text'].lower().strip(':')}
        if 'base' in pair and pair & {'year', 'año', 'ano', 'année', 'jahr', 'anno', 'jaar', 'år'}:
            return True
    # "Base year" printed directly above the year
    above = [x for x in words if 0 < w['top'] - x['top'] < 14 and x['x0'] < w['x1'] + 4 and x['x1'] > w['x0'] - 4]
    texts = [x['text'].lower() for x in sorted(above, key=lambda x: x['x0'])]
    return 'base' in texts and any(t.startswith('year') for t in texts)


def _is_change_year(w, words):
    """A year inside a variance heading ("% 2025/2024", "vs 2024",
    "Change / 2024-2025", "% 2024/ over 2023") is not a reporting column."""
    if re.search(r'[/]', w.get('source_text', w['text'])) and not w.get('fiscal_period'):
        return True
    neighbours = [x for x in words if x is not w and abs(x['top']-w['top']) < 3
                  and (0 <= w['x0']-x['x1'] < 6 or 0 <= x['x0']-w['x1'] < 6)]
    def _owned_by_next(x):
        # "N | % N/N-1": a "%" right of the year that opens the next caption
        return (x['x0'] > w['x1'] and x['text'] == '%' and any(
            0 <= y['x0'] - x['x1'] < 6 and abs(y['top'] - x['top']) < 3
            and (YEAR.fullmatch(y['text']) or re.fullmatch(r"N(?:-\d)?", y.get('source_text', y['text'])))
            for y in words if y is not x and y is not w))
    if any(_CHANGE_NEIGHBOUR.match(x['text']) and not _owned_by_next(x) for x in neighbours):
        return True
    # a stacked heading directly above the year
    for x in words:
        if not (0 < w['top']-x['top'] < 18 and x['x0'] < w['x1'] and x['x1'] > w['x0']):
            continue
        if any(YEAR.fullmatch(y['text']) for y in words if abs(y['top']-w['top']) < 3 and y is not w
               and y['x0'] < x['x1'] and y['x1'] > x['x0']):
            continue
        if _CHANGE_NEIGHBOUR.match(x['text']) or re.fullmatch(r"(?:FY)?(?:19|20)\d{2}\s*/|%", x['text']):
            return True
    return False


def _merge_offset_cell_lines(body_lines, cut):
    """Columns typeset on slightly different baselines split one row into
    several lines (amounts 2-8pt above or below the labelled amounts). A line
    holding only amounts is merged into an adjacent line when their amounts
    occupy different columns."""
    def amounts(line):
        return [w for u in _numeric_units(line, cut) for w in u]
    def label_words(line):
        return [w for w in line if w['x1'] < cut and re.search('[A-Za-z]', w['text'])]
    lines = [list(l) for l in body_lines]
    k = 0
    while k < len(lines):
        L = lines[k]
        if amounts(L) and not label_words(L) and all(w['x0'] >= cut - 2 for w in L):
            yl = min(w['top'] for w in L)
            best = None
            for j in (k-1, k+1):
                if 0 <= j < len(lines):
                    M = lines[j]
                    ym = min(w['top'] for w in M)
                    if abs(ym - yl) <= 8 and amounts(M) and not any(
                            a['x0'] < b['x1'] + 3 and b['x0'] < a['x1'] + 3 for a in amounts(L) for b in amounts(M)):
                        if best is None or abs(ym - yl) < best[0]:
                            best = (abs(ym - yl), j)
            # amounts centred between two labelled rows that both admit them
            # ("1,810,000 ... 4,398,128" spanning a block of category rows)
            # belong to a merged cell, not to either row: they bind to none
            near = []
            for j in (k-1, k+1):
                if 0 <= j < len(lines):
                    M = lines[j]
                    ym = min(w['top'] for w in M)
                    if abs(ym - yl) <= 8 and amounts(M) and label_words(M) and not any(
                            a['x0'] < b['x1'] + 3 and b['x0'] < a['x1'] + 3 for a in amounts(L) for b in amounts(M)):
                        near.append(abs(ym - yl))
            if len(near) == 2 and abs(near[0] - near[1]) < 2.5:
                del lines[k]
                continue
            if best:
                j = best[1]
                lines[j] = sorted(lines[j] + L, key=lambda w: w['x0'])
                del lines[k]
                continue
        k += 1
    return lines


def _segment_header(lines, header_y, years):
    """A line of segment captions directly under the year header that ends
    with a consolidated column ("France | Rest of Europe | ... | Group")."""
    first = min(w['x0'] for w in years)
    for line in lines:
        top = min(w['top'] for w in line)
        if not header_y + 3 < top <= header_y + 60:
            continue
        alpha = [w for w in line if re.search(r"[A-Za-z]{2,}", w['text'])]
        if any(re.fullmatch(r"[-(]?\d{1,3}(?:[,.' ]\d{3})+(?:[.,]\d+)?\)?|[-(]?\d+[.,]\d+\)?", w['text']) for w in line):
            return None  # body amounts reached
        tot = re.compile(r"(?i)group|total|consolidated|gesamt|groupe|grupo|konzern")
        captions = {w['text'].lower() for w in alpha if not tot.fullmatch(w['text'])
                    and not re.fullmatch(r"(?i)units?|with|and|of|the|companies|control|operational|share|%|in", w['text'])}
        if (len(alpha) >= 4 and any(tot.fullmatch(w['text']) for w in alpha) and len(captions) >= 3
                and min(w['x0'] for w in alpha) < first - 40):
            # every year spans its own captions ending in a total column
            ys = sorted(years, key=lambda w: w['x0'])
            ok, rich = True, False
            for k, y in enumerate(ys):
                lo = ys[k-1]['x1'] if k else first - 250
                hi = ys[k+1]['x0'] if k + 1 < len(ys) else y['x1'] + 250
                span = [w for w in alpha if lo <= (w['x0'] + w['x1']) / 2 <= hi]
                if not any(tot.fullmatch(w['text']) for w in span):
                    ok = False
                if sum(1 for w in span if w['text'].lower() in captions) >= 2:
                    rich = True
            if ok and rich:
                return alpha
    # captions wrapped over several lines ("Consolidat-/ed Group | Subjects
    # over which operational control is exercised | Total")
    band = []
    for line in lines:
        top = min(w['top'] for w in line)
        if not header_y + 3 < top <= header_y + 70:
            continue
        if any(re.fullmatch(r"[-(]?\d{1,3}(?:[,.' ]\d{3})+(?:[.,]\d+)?\)?|[-(]?\d+[.,]\d+\)?", w['text']) for w in line):
            break
        band += [w for w in line if w['x0'] >= first - 60 and re.search(r"[A-Za-z]{2,}", w['text'])]
    tot = re.compile(r"(?i)group|total|consolidated|gesamt|groupe|grupo|konzern")
    if len(band) >= 5 and sum(1 for w in band if tot.fullmatch(w['text'])) >= len(years):
        ys = sorted(years, key=lambda w: w['x0'])
        ok, rich = True, False
        for k, y in enumerate(ys):
            lo = ys[k-1]['x1'] if k else first - 60
            hi = ys[k+1]['x0'] if k + 1 < len(ys) else y['x1'] + 120
            span = [w for w in band if lo <= (w['x0'] + w['x1']) / 2 <= hi]
            if not any(tot.fullmatch(w['text']) for w in span):
                ok = False
            # a year centred over two or more captioned sub-columns
            if (any(w['x1'] < y['x0'] - 10 for w in span) and any(w['x0'] > y['x1'] + 10 for w in span)
                    and len({round(w['x0'] / 40) for w in span}) >= 3):
                rich = True
        if ok and rich:
            return band
    return None


def _hoist_centred_group_labels(body_lines, cut):
    """A group label vertically centred on its sub-rows ("Scope 2" between
    "Energy (market-based)" and "Energy (location-based)") is printed after
    the first sub-row. Move it above that row so it heads every sub-row."""
    def numeric(line):
        return bool(_numeric_units(line, cut))
    lines = list(body_lines)
    k = 1
    while k < len(lines) - 1:
        L, A, B = lines[k], lines[k-1], lines[k+1]
        yl = min(w['top'] for w in L)
        text = _text(L)
        if (not numeric(L) and numeric(A) and numeric(B) and len(text.split()) <= 4
                and all(w['x1'] < cut for w in L)
                and 0 < yl - min(w['top'] for w in A) < 12 and 0 < min(w['top'] for w in B) - yl < 12):
            la = [w for w in A if w['x1'] < cut and re.search('[A-Za-z]', w['text'])]
            lb = [w for w in B if w['x1'] < cut and re.search('[A-Za-z]', w['text'])]
            lx1 = max(w['x1'] for w in L)
            if la and lb and min(w['x0'] for w in la) > lx1 and min(w['x0'] for w in lb) > lx1:
                lines[k-1], lines[k] = L, A
                k += 2
                continue
        k += 1
    return lines


def _assign_data_anchors(cols, body_lines, cut, next_y, gap):
    """Locate each year column from the right edges of its printed numbers.

    Header words may be centred while numbers are right-aligned, so a header
    edge is only a hint. Right edges of numeric cells cluster tightly per
    physical column; a header claims a cluster only when the printed cells
    overlap the header horizontally or end within 18pt of it, nearest first
    and in left-to-right order. Columns without a header (comparatives,
    variance) stay unassigned and their cells are not bound to any year.
    """
    pts = []
    for line in body_lines:
        if min(w['top'] for w in line) >= next_y:
            break
        for u in _numeric_units(line, cut):
            if all(parse_number(w['text']) is None for w in u):
                continue  # placeholders (-, n/a) do not locate a column
            pts.append((u[-1]['x1'], u[0]['x0']))
    if not pts:
        return
    pts.sort()
    clusters = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - clusters[-1][-1][0] > 7:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    clusters = _merge_left_aligned(clusters)
    min_count = 2 if len(pts) >= 6 else 1
    clusters = [c for c in clusters if len(c) >= min_count]
    pairs = []
    for j, col in enumerate(cols):
        hx0, hx1 = col['bbox'][0], col.get('header_x1', col['bbox'][2])
        for k, c in enumerate(clusters):
            cx1 = median(p[0] for p in c)
            cx0 = median(p[1] for p in c)
            dist = min(abs(cx1 - col['anchor']), abs(cx1 - hx1))
            overlap = cx0 <= hx1 + 3 and cx1 >= hx0 - 3
            # wide columns: a left-set year caption over amounts right-aligned
            # a little further right (caption 422-440, amounts ending at 463)
            right_set = cx1 > hx1 and cx0 - hx1 < 8 and cx1 - hx1 <= min(30, gap * .45)
            if overlap or dist <= 18 or right_set:
                pairs.append((dist, j, k))
    # a single year heading over a single column of amounts set further right
    # ("TOTAL SCOPE 3 EMISSIONS (tCO2e)   2024" / amounts ending 50pt right)
    if not pairs and len(cols) == 1:
        right = [(k, c) for k, c in enumerate(clusters)
                 if 0 < median(p[0] for p in c) - cols[0]['bbox'][2] <= 80 and len(c) >= 2]
        if len(right) == 1:
            pairs.append((median(p[0] for p in right[0][1]) - cols[0]['bbox'][2], 0, right[0][0]))
    pairs.sort()
    used_c, used_k = {}, set()
    for dist, j, k in pairs:
        if j in used_c or k in used_k:
            continue
        if any((jj < j and kk > k) or (jj > j and kk < k) for jj, kk in used_c.items()):
            continue
        used_c[j] = k
        used_k.add(k)
    owner = {k: j for j, k in used_c.items()}
    # Centred columns: wide amounts end further right than narrow ones and
    # form a second cluster. It joins the column when that header is its only
    # admissible header and no other column's cells lie between them.
    eligible = defaultdict(set)
    for dist, j, k in pairs:
        eligible[k].add(j)
    for k, c in enumerate(clusters):
        if k in owner or len(eligible[k]) != 1:
            continue
        j = next(iter(eligible[k]))
        mine = [kk for kk, jj in owner.items() if jj == j]
        if not mine:
            continue
        lo_k, hi_k = min(p[0] for p in c), max(p[0] for p in c)
        between = [kk for kk, jj in owner.items() if jj != j and
                   any(min(lo_k, min(q[0] for q in clusters[m])) < min(q[0] for q in clusters[kk]) <
                       max(hi_k, max(q[0] for q in clusters[m])) for m in mine)]
        if not between:
            owner[k] = j
    for j in {j for j in owner.values()}:
        ks = [k for k, jj in owner.items() if jj == j]
        main = used_c[j]
        xs = [p[0] for k in ks for p in clusters[k]]
        cols[j].update(data_anchor=round(median(p[0] for p in clusters[main]), 3),
                       data_lo=round(min(xs), 3), data_hi=round(max(xs), 3),
                       cell_x0=round(min(p[1] for k in ks for p in clusters[k]), 3))
    return [(min(p[0] for p in c), max(p[0] for p in c), owner.get(k), len(c)) for k, c in enumerate(clusters)]


def _slot_numeric_words(line, cols, cut, gap, page, clusters=None):
    slots = defaultdict(list)
    have_data = any('data_anchor' in c for c in cols)
    tolerance = min(15, gap*.4)
    for unit in _numeric_units(line, cut):
        unit = [_strip_superscript(w, page) if len(unit) == 1 or w is unit[-1] else w for w in unit]
        x1 = unit[-1]['x1']
        if have_data:
            cand = [j for j, c in enumerate(cols) if 'data_anchor' in c
                    and c['data_lo']-4 <= x1 <= c['data_hi']+4]
            if len(cand) == 1:
                slots[cand[0]].extend(unit)
                continue
            if cand:
                continue
        # header-edge rule (centred numbers of varying width), unless the
        # number lies inside another physical column of printed cells
        j = min(range(len(cols)), key=lambda k: abs(x1-cols[k]["anchor"]))
        if abs(x1-cols[j]["anchor"]) <= tolerance:
            # another header's column, or an unlabelled column of >= 2 cells
            # (e.g. "Comparative"); a lone wide number is not a column
            inside_other = any(lo-4 <= x1 <= hi+4 and owner != j and (owner is not None or n >= 2)
                               for lo, hi, owner, n in (clusters or []))
            if not inside_other:
                slots[j].extend(unit)
                continue
        # a short centred cell ("0", "-") inside one column's printed extent
        if have_data and re.fullmatch(r"0|[-\u2010\u2011\u2013\u2014]", _text(unit)):
            mid = (unit[0]['x0'] + unit[-1]['x1']) / 2
            inside = [j for j, c in enumerate(cols) if 'cell_x0' in c and c['cell_x0'] - 1 <= mid <= c['data_hi'] + 1]
            c_ = cols[inside[0]] if len(inside) == 1 else None
            rivals = [u for u in _numeric_units(line, cut) if abs(u[0]['x0'] - unit[0]['x0']) > .5 and c_ is not None
                      and c_['cell_x0'] - 1 <= (u[0]['x0'] + u[-1]['x1']) / 2 <= c_['data_hi'] + 1]
            if len(inside) == 1 and not slots.get(inside[0]) and not rivals:
                slots[inside[0]].extend(unit)
    return slots


def _bbox(words):
    return [round(min(w["x0"] for w in words), 3), round(min(w["top"] for w in words), 3),
            round(max(w["x1"] for w in words), 3), round(max(w["bottom"] for w in words), 3)]


def _lines(words):
    rows = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        # Top alignment avoids attaching superscripts to the following row.
        if not rows or abs(w["top"] - median(x["top"] for x in rows[-1])) > 3:
            rows.append([w])
        else:
            rows[-1].append(w)
    return [sorted(row, key=lambda w: w["x0"]) for row in rows]


_HYPHEN_JOIN = re.compile(r"(?i)(?:thousands?|millions?|tonnes?|tons?|equivalents?|metric|kilo\w*)")


def _dehyphenate_units(words):
    """'thou-' + 'sands' (a hyphenated line break inside a unit) -> 'thousands'."""
    ordered = [w for line in _lines(words) for w in line]
    out, skip = [], set()
    for k, w in enumerate(ordered):
        if k in skip:
            continue
        if w['text'].endswith('-') and k + 1 < len(ordered):
            nxt = ordered[k + 1]
            joined = w['text'][:-1] + nxt['text']
            if re.match(r"[a-z]", nxt['text']) and _HYPHEN_JOIN.fullmatch(re.sub(r"[^A-Za-z]", "", joined)):
                out.append(dict(w, text=joined, x1=max(w['x1'], nxt['x1']), bottom=max(w['bottom'], nxt['bottom']), span_x0=min(w['x0'], nxt['x0'])))
                skip.add(k + 1)
                continue
        out.append(w)
    return out


def _unit(words):
    if any(w['text'].endswith('-') for w in words):
        words = _dehyphenate_units(words)
    text = _text(words)
    # CO2's subscript often has its own baseline, after 'e' in reading order.
    for m in UNIT.finditer(text):
        raw = re.sub(r"CO\s+(eq\b|e\b)", r"CO2\1", m.group(0))
        if _relative_after(text[m.end():]):
            continue
        if normalize_emission_to_tco2e(1, raw)[0] is not None:
            unit_words, offset = [], 0
            for w in [w for line in _lines(words) for w in line]:
                end = offset+len(w['text'])
                if end > m.start() and offset < m.end():
                    unit_words.append(w)
                offset = end+1
            if not unit_words:
                continue
            bb = _bbox(unit_words)
            bb[0] = min([bb[0]] + [w['span_x0'] for w in unit_words if 'span_x0' in w])
            return {"text": raw, "source_text": _text(unit_words), "bbox": bb,
                    "multiplier": normalize_emission_to_tco2e(1, raw)[0]}
    return None


_ENERGY_ROW = re.compile(r"\b(?:fuel|energy|electricity|heat|steam|gas)\s+consumption\b|\bconsumption\s+(?:from|of)\b|\bMWh\b|\bGJ\b", re.I)


_SCOPE12_SOURCE = re.compile(r"\b(?:stationary|mobile|combustion|fugitive|refrigerants?|purchased (?:electricity|heat|steam|cooling)|electricity consumption|district heating|process emissions|company (?:cars?|vehicles)|own fleet)\b", re.I)


def _explicit_category_marker(label):
    return bool(re.match(r"\s*3\.(?:1[0-5]|[1-9])\b", label) or CATEGORY.search(label)) and not re.match(r"\s*[12]\.\d", label)


def _category(label):
    dotted = re.match(r"\s*3\.(1[0-5]|[1-9])\b", label)
    if dotted:
        return int(dotted.group(1))
    # GHG Protocol numbering 1.x / 2.x lists Scope 1 and Scope 2 sources
    if re.match(r"\s*[12]\.\d", label):
        return None
    if re.match(r"\s*[123](?:\.\d{1,2})?\.?\s+(?:gross\s+)?(?:direct|indirect|energy[ -]indirect|other indirect)\b", label, re.I) \
            and not re.search(r"\b(?:category|cat\.?)\s*\d", label, re.I):
        return None
    if _ENERGY_ROW.search(label) and not re.search(r"\b(?:category|cat\.?)\s*\d|related\s+activit", label, re.I):
        return None
    if re.search(r'excl(?:ud(?:ing|ed|es)|\.)?\s.*(?:category|cat\.?)\s*\d',label,re.I):
        return None
    # "Total emissions without use of sold products": everything but the category
    # "Business travel (excluding business trips ... within scope 1)": a
    # parenthesis carving a part out of the category keeps the category
    label = re.sub(r"(?i)\(\s*(?:excluding|excl\.?|except|not including)\b[^()]*(?:\)|$)", " ", label).strip()
    if re.search(r'(?i)\b(?:without|excluding|except)\b', label) and not re.match(r"\s*\(?(?:category\s*)?(?:1[0-5]|[1-9])\b", label, re.I):
        return None
    m = CATEGORY.search(label)
    no, _, _, score = map_scope3_category(label)
    if m:
        n = int(m.group(1) or m.group(2))
        if m.group(1):
            return n  # "Category 3", "Cat. 3"
        # A bare list number ("1. Stationary combustion", "2. Purchased
        # electricity") numbers Scope 1/2 sources as often as categories; it
        # is a category only when the words agree or name no Scope 1/2 source.
        if _SCOPE12_SOURCE.search(label):
            return None
        # "10 Processing of sold products (including Category 11: ...)": a
        # parenthetical scope note does not rename the numbered row
        if '(' in label:
            no, _, _, score = map_scope3_category(re.sub(r"\([^)]*\)?", " ", label))
        if score >= .9 and no is not None:
            return no  # the category name decides over the report's own numbering
        return n
    return no if score >= .9 else None


def _noninventory_quantity(label):
    """Explicit emissions impacts are distinct from inventory emissions.

    Do not treat a generic word such as savings (e.g. savings deposits) or a
    percentage annotation inside an absolute-emissions label as a unit.
    Relative units are identified separately at the physical unit column.
    """
    return bool(re.search(r"(?:erased|avoided)\s+(?:(?:ghg|greenhouse\s+gas|carbon)\s+)?emissions|(?:ghg|carbon)?\s*emissions?\s+(?:reductions?|savings?)\b",label,re.I))


def _declared_other_unit(left, anchor, label, row_unit):
    """Explicit activity units override an inherited carbon unit.

    Bare mass symbols are ambiguous in a GHG table; only a plain activity
    volume/weight/mass label establishes the noncarbon interpretation.
    """
    if not anchor:
        return None
    words=[w for w in left if anchor['bbox'][0]-5 <= (w['x0']+w['x1'])/2 <= anchor['bbox'][2]+5]
    text=_text(words)
    matched=[w for w in words if w['text']=='%']
    dimension='relative' if matched else None
    if not matched:
        matched=[w for w in words if re.fullmatch(r'(?:[kMGT]?Wh|m[³3²2]|ML|EUR|USD|GBP|€|\$)',w['text'],re.I)]
        if matched:dimension='noncarbon_activity'
    if not matched and not row_unit and re.search(r'\b(?:g|mg|µg|lb|lbs|grams?|kilograms?)\s*CO\s*[₂2]?\s*(?:eq|e)',text,re.I):
        matched=words
        dimension='unresolved_emission_unit'
    if not matched and not row_unit and not re.search(r'ghg|emissions?|CO[₂2]',label,re.I):
        activity=bool(re.match(r'\s*(?:volume|weight|mass)\s+(?:of\s+)?(?:treated\s+)?(?:waste|water|materials?|paper)\b',label,re.I))
        if activity:
            matched=[w for w in words if re.fullmatch(r'(?:Mt|kt|kg|t|tonnes?|tons?)',w['text'])]
            if matched:dimension='activity_mass'
    return {'text':_text(matched),'bbox':_bbox(matched),'dimension':dimension} if matched else None


# Qualifiers after "Scope N -" that describe the whole scope, not a part of it.
_WHOLE_SCOPE = (r"(?:direct|indirect|energy[ -]indirect|energy[ -]related|other indirect|value[ -]chain|group|"
                r"consolidated|company|corporate|overall|all\b|organi[sz]ation)")


_SPLIT_WORD = re.compile(r"\b([A-Z]) (ross|otal|cope|arket|ocation|missions|nergy|ndirect|irect)\b")


def _field(label, section=""):
    label = _SPLIT_WORD.sub(lambda m: m.group(1) + m.group(2), label)
    s = re.sub(r"\s+", " ", label.lower()).replace('‑','-')
    s = re.sub(r"\blocated([ -]+based)", r"location\1", s)  # "Scope 2 (located-based)"
    # "scope 1 or 2) ...": the wrapped tail of the previous label's parenthesis
    if re.match(r"\s*[a-z][^()]*\)", label):
        s = re.sub(r"^\s*[^()]*\)\s*", "", s)
    # a coverage column ("100%") printed inside a wrapped label is not a percentage metric
    s = re.sub(r"(?<![\w.])\d{1,3}(?:\.\d+)?\s?%(?=\s|$)", " ", s) if re.search(r"[a-z]{4,}.*\d%|\d%.*[a-z]{4,}", s) and not re.search(r"percent|share|proportion|\(%\)", s) else s
    # EPRA sBPR performance-measure codes
    if re.search(r"\bghg-dir-abs\b", s):
        return "scope1"
    if re.search(r"\bghg-indir-abs\b", s):
        return "scope2"
    if _noninventory_quantity(s) or re.search(r"%|percent|^share of|proportion of|intensity|per (?:fte|employee|revenue)|avoided|saving|biogenic|biomass|reduction|without saf|excluding saf|water consumption|total waste|methodologies|emission factors",s):
        return None
    # "(not included in Scope 1 or 2)" describes what a category excludes
    s = re.sub(r"\(?\b(?:not\s+(?:already\s+)?(?:(?:included|covered|counted|reported)\s+)?(?:in|under|within)|excluding|outside(?:\s+of)?)"
               r"\s+(?:the\s+)?scopes?\s*1(?:\s+\d{1,2}(?=\s))?(?:\s*(?:and|or|&|,|\+|/)\s*(?:scope\s*)?2)?\)?", " ", s)
    # "(excluding business trips with company cars that are reported within
    # scope 1)": the parenthesis names what the row leaves out
    s = re.sub(r"\(\s*(?:excluding|excl\.?|not including|except)\b[^()]*\bscopes?\s*[12]\b[^()]*(?:\)|$)", " ", s)
    scopes = set(re.findall(r"scopes?\s*([123])\b", s))
    explicit_cat = re.search(r"\b(?:category|cat\.?)\s*(?:1[0-5]|[1-9])\b|^\s*3\.(?:1[0-5]|[1-9])\b", s)
    excluded_cat = re.search(r"\bexcl(?:uding|uded|udes|\.)?\s.*(?:category|cat\.?)\s*\d", s)
    # GHG Protocol scope numbering: "1 Direct emissions", "1.1 Direct emissions
    # stationary", "2 Indirect emissions", "3 Other indirect emissions"
    gp = re.match(r"\s*([123])(?:\.\d{1,2})?\.?\s+(?:gross\s+)?(direct|indirect|energy[ -]indirect|other indirect)\b", s)
    if gp and not re.search(r"\b(?:category|cat\.?)\s*\d", s):
        n = gp.group(1)
        if n == '1' and gp.group(2) == 'direct':
            return "scope1"
        if n == '2' and gp.group(2) in ('indirect', 'energy indirect', 'energy-indirect'):
            return "scope2_location" if re.search(r"location|site[- ](?:specific|based)", s) else "scope2_market" if "market" in s else "scope2"
        if n == '3' and gp.group(2) in ('indirect', 'other indirect'):
            return "scope3_total"
    # "Scope 3 category 15 - financed scope 1 and 2": the explicit category
    # marker names the row; investee scopes are part of its description.
    if explicit_cat and not excluded_cat and not re.search(r"scope\s*[12]\s*(?:and|\+|&|,)\s*(?:scope\s*)?[23]\s*(?:and|\+|&|,)?\s*(?:scope\s*)?3?\s*(?:total|emissions)?\s*$", s):
        cat = _category(label)
        if cat:
            return f"scope3_cat{cat}"
    # "Total GHG emissions / Including Scope 2 - market-based": a basis
    # variant of the combined total, not the Scope 2 amount.
    if re.match(r"\s*(?:including|incl\.)\s+scope\s*2\b", s):
        return None
    combined = re.search(r"scopes?\s*1\s*(?:and|\+|&|,)\s*(?:scope\s*)?2|scopes?\s*1\s*[–-]\s*[23]|scopes?\s*[12]\s*(?:and|\+|&|,)\s*(?:scope\s*)?3|other scope|number of employees|per employee|efficiency|/meur", s)
    if combined or len(scopes) > 1:
        return None
    # A total that names exactly one scope is that scope's total, not a
    # combined inventory total.
    if not scopes and re.search(r"total (?:corporate )?(?:ghg|greenhouse gas) emissions|total emissions", s) \
            and not re.search(r"value chain", s):
        return None
    if scopes == {"3"} and (excluded_cat or (re.search(r"\btotal\b", s) and not explicit_cat)):
        return "scope3_total"
    # An explicit Scope 1/2 label is never a Scope 3 category, whatever its
    # numbering or activity words (e.g. "Scope 1 - company cars").
    # Inside a Scope 1/2 block, activity words ("Energy procurement",
    # "business travel by car") do not make a Scope 3 category unless the row
    # is explicitly numbered.
    numbered = re.match(r"\s*\(?(?:1[0-5]|[1-9])[.):]?\s+[A-Za-z]", s)
    in_scope12 = (section or "").startswith(("scope1", "scope2")) and not scopes
    cat = None if ((scopes & {"1", "2"} or in_scope12) and not explicit_cat and not (in_scope12 and numbered)) else _category(label)
    if cat:
        return f"scope3_cat{cat}"
    if "3" in scopes and not re.search(r"flag|non.flag", s):
        return "scope3_total"
    if "2" in scopes or ("scope2" in section and re.search(r"market|location", s)):
        return "scope2_location" if re.search(r"location|site[- ](?:specific|based)", s) else "scope2_market" if "market" in s else "scope2"
    if "1" in scopes:
        return "scope1"
    # GHG Protocol / GRI 305 standard names without the word "scope".
    if re.search(r"^(?:total\s+)?other indirect (?:ghg |greenhouse gas )?emissions|value chain (?:ghg |greenhouse gas )?emissions|^total (?:ghg |greenhouse gas )?emissions (?:of|from|in) the value chain", s):
        return "scope3_total"
    # "Total indirect GHG emissions" directly under a Scope 3 heading
    if (section or "").startswith("scope3") and re.match(r"^(?:total\s+)?(?:gross\s+)?indirect\b(?!.*energy)", s):
        return "scope3_total"
    # ESRS E1-6 wording: "Total Gross indirect (Scope 3) GHG emissions"
    if re.search(r"^(?:total\s+)?gross indirect (?:ghg |greenhouse gas )?emissions", s):
        return "scope3_total"
    if re.search(r"^(?:gross\s+)?(?:total\s+)?energy indirect (?:ghg |greenhouse gas )?emissions", s):
        return "scope2_location" if re.search(r"location|site[- ](?:specific|based)", s) else "scope2_market" if "market" in s else "scope2"
    if re.search(r"^(?:gross\s+)?(?:total\s+)?direct (?:ghg |greenhouse gas )?emissions\b", s):
        return "scope1"
    return None


def _row_similarity(a, b, component=False):
    def canon(s):
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    a, b = canon(a), canon(b)
    if component:
        a=re.sub(r'^of\s+which\s+','',a)
        b=re.sub(r'^of\s+which\s+','',b)
    if not a or not b:
        return 0
    if a == b or (not component and (a in b or b in a)):
        return 1
    return SequenceMatcher(None, a, b).ratio()


def _source_unit(unit, page):
    """Check the unit's physical glyphs, including a displaced CO2 subscript.

    The text extractor's missing subscript cannot justify inserting a digit
    without checking the PDF's character layer. Relative units remain invalid.
    """
    if unit.get('glyph_checked'):
        return dict(unit)  # bare caption unit, glyphs already compared
    x0,y0,x1,y1 = unit['bbox']
    box = [x0-.2,y0-.2,x1+.2,y1+5]
    chars = [c for c in page.chars if box[0] <= (c['x0']+c['x1'])/2 <= box[2]
             and box[1] <= (c['top']+c['bottom'])/2 <= box[3]]
    literal = ''.join(c['text'] for c in sorted(chars,key=lambda c:c['x0']))
    literals = [literal]
    if chars:
        # a caption wrapped over several lines ("in thousands" / "of t CO2e"):
        # read line by line, a smaller subscript joining the line it hangs from
        big = max(c.get('size', 0) for c in chars)
        lines = []
        for c in sorted([c for c in chars if c.get('size', 0) >= .8 * big], key=lambda c: c['top']):
            if lines and abs(c['top'] - lines[-1][0]['top']) < 2:
                lines[-1].append(c)
            else:
                lines.append([c])
        if len(lines) > 1:
            for c in chars:
                if c.get('size', 0) < .8 * big:
                    mid = (c['top'] + c['bottom']) / 2
                    near = [ln for ln in lines if ln[0]['top'] - 1 <= mid <= max(x['bottom'] for x in ln) + 4]
                    if near:
                        min(near, key=lambda ln: abs(ln[0]['top'] - c['top'])).append(c)
            literals.append(' '.join(''.join(c['text'] for c in sorted(ln, key=lambda c: c['x0'])) for ln in lines))
    # a hyphenated line break inside the unit ("thou-" / "sands")
    literals += [re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", lit) for lit in literals if re.search(r"[A-Za-z]-\s+[a-z]", lit)]
    candidates = [normalize_emission_to_tco2e(1,m.group(0))[0] for lit in literals for m in UNIT.finditer(lit)
                  if not _relative_after(lit[m.end():])]
    literal = literals[-1]
    return dict(unit, source_bbox=box, source_glyph_text=literal,
                evidence_status='verified' if unit['multiplier'] in candidates else 'unresolved')


class _TableRegion:
    def __init__(self, page, bbox, heading_above=None):
        self.page, self.bbox = page, bbox
        self.width, self.height = page.width, bbox[3]
        self.chars = page.chars
        # a panel heading starting in this strip may run on across the next
        # strip ("Gross greenhouse gas emissions ..., tonnes")
        self.heading_above = heading_above

    def extract_words(self, **kwargs):
        a,b,c,d = self.bbox
        all_words = self.page.extract_words(**kwargs)
        out = [w for w in all_words
               if a-.5 <= w['x0'] and w['x1'] <= c+.5 and b <= w['top'] < d]
        if self.heading_above:
            y0, y1 = self.heading_above
            for line in _lines([w for w in all_words if y0 <= w['top'] < y1]):
                run = sorted(line, key=lambda w: w['x0'])
                segs, cur = [], [run[0]]
                for w in run[1:]:
                    if w['x0'] - cur[-1]['x1'] > 15:
                        segs.append(cur); cur = [w]
                    else:
                        cur.append(w)
                segs.append(cur)
                for seg in segs:
                    if any(a - .5 <= w['x0'] < c for w in seg):
                        out.extend(w for w in seg if not (a-.5 <= w['x0'] and w['x1'] <= c+.5))
        return out

    def find_tables(self):
        return self.page.find_tables()



def _panel_strips(page, words):
    """Vertical strips of side-by-side tables whose year headers share a line.

    A gap between two header years that holds a column of row labels (three
    or more body lines with words wholly inside the gap) starts a new panel
    at that label column. Returns [(x0, x1), ...] or None.
    """
    for line in _lines(words):
        years = sorted([w for w in line if YEAR.fullmatch(w['text'])], key=lambda w: w['x0'])
        if len(years) < 4:
            continue
        y0 = min(w['top'] for w in years)
        body = [l for l in _lines(words) if y0 + 8 < min(w['top'] for w in l) < y0 + 220]
        cuts = []
        for a, b in zip(years, years[1:]):
            if b['x0'] - a['x1'] < 40:
                continue
            lab = [min(w['x0'] for w in inside) for l in body
                   for inside in [[w for w in l if a['x1'] + 5 < w['x0'] and w['x1'] < b['x0'] - 5
                                   and re.search('[A-Za-z]{3}', w['text'])]] if inside]
            if len(lab) >= 3:
                cuts.append(min(lab) - 3)
        if cuts:
            edges = [0.0] + cuts + [float(page.width)]
            groups = [[w for w in years if a <= w['x0'] < b] for a, b in zip(edges, edges[1:])]
            groups = [g for g in groups if g]
            # real side-by-side panels each repeat a multi-year header
            if len(groups) < 2 or any(len(g) < 2 for g in groups):
                continue
            if sum(1 for g in groups if any(w['text'] == groups[0][0]['text'] for w in g)) < 2:
                continue
            return list(zip(edges, edges[1:])), y0
    return None


class _SynthHeaderPage:
    """A page view with synthetic header words carried from the previous page."""
    def __init__(self, page, extra, y_limit=None):
        self.page, self.extra, self.y_limit = page, extra, y_limit
        self.width, self.height = page.width, page.height
        self.chars = page.chars
        self.bbox = getattr(page, 'bbox', None)
        self._s3t_period_year = getattr(page, '_s3t_period_year', None)

    def extract_words(self, **kwargs):
        ws = [w for w in self.page.extract_words(**kwargs) if self.y_limit is None or w['top'] < self.y_limit]
        return ws + [dict(w) for w in self.extra]

    def find_tables(self):
        return self.page.find_tables()


def _continued_grid(page, doc_id, page_no, words, prev_tables, y_limit=None, offset=0):
    """A table continued from the previous page without repeating its header.

    The previous page's last grid must reach the lower part of its page; the
    rows here must print amounts right-aligned to its column anchors. The
    header years, roles and boundaries are carried over and recorded.
    """
    cands = [t for t in prev_tables if t['bbox'][3] > .55 * page.height]
    if not cands:
        return [], []
    prev = max(cands, key=lambda t: t['bbox'][3])
    cols = [c for c in prev['columns'] if c.get('role') == 'reported']
    if not cols:
        return [], []
    anchors = {c['id']: (c.get('data_anchor') or c.get('anchor') or c['bbox'][2]) for c in prev['columns']}
    numeric = [w for w in words if re.fullmatch(r"[-(]?\d[\d,.' ]*\d\)?|\d|[-\u2013\u2014]", w['text'])
               and any(abs(w['x1'] - a) < 6 for a in anchors.values())
               and page.height * .06 < w['top'] < page.height * .94]
    if len({round(w['top']) for w in numeric}) < 2:
        return [], []
    top = min(w['top'] for w in numeric)
    # nothing but running text above the first continued row
    if any(YEAR.fullmatch(w['text']) and w['top'] > page.height * .06 and w['top'] < top for w in words):
        return [], []
    extra = []
    for c in prev['columns']:
        x1 = anchors[c['id']]
        extra.append({'text': str(c['year']), 'x0': x1 - 18, 'x1': x1, 'top': top - 14, 'bottom': top - 6,
                      'upright': True, 'synthetic': True})
    tables, issues = _reconstruct_grid(_SynthHeaderPage(page, extra, y_limit), doc_id, page_no, offset)
    out = []
    for t in tables:
        by_anchor = {round(anchors[c['id']]): c for c in prev['columns']}
        ok = True
        for c in t['columns']:
            pc = by_anchor.get(round(c['bbox'][2]))
            if pc is None:
                ok = False
                break
            c.update(year=pc['year'], role=pc['role'], boundary=pc.get('boundary', c.get('boundary')),
                     organization_boundary=pc.get('organization_boundary', c.get('organization_boundary')),
                     header=pc.get('header'), continued_header=prev['id'])
        if ok:
            t['continued_from_table'] = prev['id']
            out.append(t)
    return out, issues + ([{"kind": "continued_header", "from": prev['id']}] if out else [])

def reconstruct_page(page, doc_id, page_no, number_locale=None):
    # cropped sub-regions do not carry the page's document context
    _PERIOD_YEAR.set(getattr(page, '_s3t_period_year', None))
    # Use a ruled region only when it contains BOTH labels and year columns.
    # Numeric-only shading must not cut the labels off a partially ruled table.
    words = _year_words(page.extract_words(x_tolerance=1,y_tolerance=3) or [], page)
    regions = []
    for table in page.find_tables() or []:
        a,b,c,d = table.bbox
        inside = [w for w in words if a-.5 <= w['x0'] and w['x1'] <= c+.5 and b <= w['top'] < d]
        years = [w for w in inside if YEAR.fullmatch(w['text'])]
        numeric_left = min((w['x0'] for w in years),default=page.width)
        labels = [line for line in _lines(inside)
                  if _field(_text([w for w in line if w['x1'] < numeric_left-5]))]
        if len(years) >= 1 and len(labels) >= 2 and min(w['x0'] for line in labels for w in line) < min(w['x0'] for w in years)-40:
            # the same column grid continued in following ruled boxes (one per
            # scope section, a section heading between them) is one region
            ruled_all = page.find_tables() or []
            while True:
                more = [t2 for t2 in ruled_all if abs(t2.bbox[0] - a) < 6 and abs(t2.bbox[2] - c) < 6
                        and 0 < t2.bbox[1] - d < 45 and len(t2.rows) >= 2]
                if not more:
                    break
                d = max(t2.bbox[3] for t2 in more)
            box = [a, max(0,b-24),c,d]
            if not any(old[0] <= a and c <= old[2] and old[1] <= b and d <= old[3] for old in regions):
                regions.append(box)
    tables, issues = [], []
    strips = _panel_strips(page, words) if not regions else None
    if strips:
        # side-by-side panels sharing one header baseline (landscape pages:
        # energy | Scope 1-2 | Scope 3), each with its own label column
        strips, hy = strips
        for x0, x1 in strips:
            ts, errors = _reconstruct_grid(_TableRegion(page, [x0, 0, x1, page.height], (hy - 40, hy - 2)),
                                           doc_id, page_no, len(tables))
            tables.extend(ts);issues.extend(errors)
    elif not regions:
        tables, issues = _reconstruct_grid(page,doc_id,page_no)
    else:
        for box in regions:
            ts, errors = _reconstruct_grid(_TableRegion(page,box),doc_id,page_no,len(tables))
            tables.extend(ts);issues.extend(errors)
        # An unruled year grid above or below a ruled table is a table of its
        # own (e.g. a location-based breakdown printed under the ruled
        # inventory). Scan the full-width bands no ruled region covers.
        spans = sorted((r[1], r[3]) for r in regions)
        merged = []
        for y0, y1 in spans:
            if merged and y0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], y1)
            else:
                merged.append([y0, y1])
        edges = [0.0] + [y for m in merged for y in m] + [float(page.height)]
        for y0, y1 in zip(edges[::2], edges[1::2]):
            if y1 - y0 < 40:
                continue
            band = [w for w in words if y0 <= w['top'] < y1]
            if sum(1 for w in band if YEAR.fullmatch(w['text'])) < 2:
                continue
            ts, errors = _reconstruct_grid(_TableRegion(page, [0, y0, page.width, y1]), doc_id, page_no, len(tables))
            tables.extend(ts);issues.extend(errors)
    prev = getattr(page, '_s3t_prev_tables', None)
    if prev:
        # rows above this page's first header continue the previous page's table
        limit = min([t['bbox'][1] for t in tables] + [page.height])
        cont, more = _continued_grid(page, doc_id, page_no, [w for w in words if w['top'] < limit - 2], prev, limit - 2,
                                     len(tables))
        if cont:
            tables = list(tables) + cont
            issues = list(issues) + more
    # one-row grids under a repeated "U.o.M. | 2021 | 2023 | 2024" header
    # whose unit cell is blank take the unit their sibling grids print, when
    # all of them print the same absolute emissions unit
    for t in tables:
        if not t.get('unit_column_caption') or any(r['unit'] for r in t['rows'] if r['field']):
            continue
        sig = [(c['year'], c['role']) for c in t['columns']]
        sib_units = [u for t2 in tables if t2 is not t and t2.get('unit_column_caption')
                     and [(c['year'], c['role']) for c in t2['columns']] == sig
                     for u in t2.get('units', []) if u.get('multiplier')]
        if sib_units and len({u['multiplier'] for u in sib_units}) == 1:
            u = dict(sib_units[0], id=f"{t['id']}:u1", source='sibling_grid_unit',
                     inherited_from=sib_units[0].get('id'))
            t['units'] = list(t.get('units', [])) + [u]
            for r in t['rows']:
                if r['field'] and not r['unit']:
                    r['unit'] = u
    from scope3trace.scope_agent.source_context import attach_page_context
    attach_page_context(page, words, tables)
    from scope3trace.common.perimeter import finalize_perimeter_headers
    finalize_perimeter_headers(tables,words)
    from scope3trace.common.rules import _locale_counts
    for table in tables:
        # two tables on one page can print numbers differently; a table's own
        # unambiguous cells ("2,495,697" / "1.225.411") decide for it
        raws_ = [str(c['raw']).strip() for r in table['rows'] for c in r['cells']]
        eu, us = _locale_counts(' '.join(raws_))
        if not eu and not us:
            # three or more "1,566"-style cells and no two-digit decimals: the
            # table groups thousands with commas, whatever stray "1,2" footnote
            # references elsewhere on the page suggest (and vice versa)
            grp_c = sum(1 for x in raws_ if re.fullmatch(r"[1-9]\d{0,2}(?:,\d{3})+", x))
            grp_d = sum(1 for x in raws_ if re.fullmatch(r"[1-9]\d{0,2}(?:\.\d{3})+", x))
            dec_c = sum(1 for x in raws_ if re.fullmatch(r"\d+,\d{1,2}", x))
            dec_d = sum(1 for x in raws_ if re.fullmatch(r"\d+\.\d{1,2}", x))
            if grp_c >= 3 and not grp_d and not dec_c:
                us = 1
            elif grp_d >= 3 and not grp_c and not dec_d:
                eu = 1
        if not eu and not us:
            # ambiguous cells: the numbers printed in the table's own band
            # (down to the next table) decide, e.g. its total row
            x1s = [c['bbox'][2] for r in table['rows'] for c in r['cells']]
            x0s = [c['bbox'][0] for r in table['rows'] for c in r['cells']]
            y0 = table['bbox'][1]
            y1 = min([t2['bbox'][1] for t2 in tables if t2['bbox'][1] > y0 + 5] + [page.height])
            if x1s:
                # only amounts printed in the table's own columns (its total
                # row), never prose or percentages around it
                eu, us = _locale_counts(' '.join(w['text'] for w in words
                                                 if y0 <= w['top'] < y1 and re.fullmatch(r"[\d.,]+", w['text'])
                                                 and min(x0s) - 5 <= w['x0']
                                                 and (w['x1'] <= max(x1s) + 8 or any(abs(w['x0'] - x) < 2 for x in x0s))))
        table['number_locale'] = ('us' if us and not eu else 'eu' if eu and not us else number_locale)
    # Glyph-proven superscript footnotes glued to amounts anywhere on the
    # page (``159`` + raised ``6`` -> "1596" in linear text). The text-row
    # path uses this map; it cannot see glyph sizes itself.
    foot = {}
    for w in words:
        if re.fullmatch(r"\d[\d.,]*\d", w['text']):
            sw = _strip_superscript(w, page)
            if sw is not w and sw.get('source_token'):
                foot[sw['source_token']] = sw['text']
    if foot:
        issues = list(issues) + [{"kind": "superscript_footnotes", "tokens": foot}]
    # A table heading printed as the page's last line ("Scope 3 GHG Emissions
    # (t CO2e)") whose table starts on the next page: keep its glyph-checked
    # unit for that continued table.
    try:
        last_bottom = max((t['bbox'][3] for t in tables), default=0)
        tail = [l for l in _lines(words) if min(w['top'] for w in l) > max(last_bottom, page.height * .55)
                and min(w['top'] for w in l) < page.height * .94]
        if tail:
            line = tail[-1]
            text = _text(line)
            if re.search(r"(?i)\b(?:scope|emissions?|ghg)\b", text) and not re.search(r"\d[\d,.]{2,}", text):
                u = _unit(line)
                if u and u.get('multiplier'):
                    u = _source_unit(dict(u), page)
                    if u.get('evidence_status') == 'verified':
                        issues = list(issues) + [{"kind": "trailing_unit", "unit": u, "heading": text}]
    except Exception:
        pass
    return tables, issues


def footnoted_tokens(parsed):
    """Map of linear-text tokens whose trailing digits are glyph footnotes."""
    out = {}
    for issue in getattr(parsed, "reconstruction_issues", None) or []:
        if isinstance(issue, dict) and issue.get("kind") == "superscript_footnotes":
            out.update(issue.get("tokens") or {})
    return out


def _label_left(lines, header_y, cut, end_y):
    """Locate the repeated row-label margin, excluding adjacent prose columns."""
    starts = []
    has_amount = lambda line: any(w['x0'] >= cut and re.search(r"\d", w['text']) for w in line)
    last_amount_y = max((min(w['top'] for w in l) for l in lines
                         if header_y+5 < min(w['top'] for w in l) < min(end_y, header_y+600) and has_amount(l)),
                        default=end_y)
    for line in lines:
        if not header_y+5 < min(w['top'] for w in line) < min(end_y,header_y+600):
            continue
        left = [w for w in line if w['x1'] < cut]
        # footnotes under the table ("2 0% of Scope 1 emissions ...") carry
        # no amounts in the value columns and do not locate the label margin
        if (min(w['top'] for w in line) > last_amount_y and not has_amount(line)
                and re.match(r"\s*(?:\(?\d{1,2}\)?|\*+)\s", _text(left) + " ")):
            continue
        for j,w in enumerate(left):
            if j and w['x0']-left[j-1]['x1'] < 10:
                continue
            label = _text(left[j:])
            if re.match(r'(?:Gross\b|Scope\s*[123]\b|Significant\s+Scope|Total.*Scope|\(?(?:1[0-5]|[1-9])\)?[:. ]+)',label,re.I) and _field(label):
                starts.append(w['x0'])
    if not starts:
        return 0
    groups = defaultdict(list)
    for x in starts: groups[round(x/5)*5].append(x)
    best = max(groups.values(),key=lambda g:len(g))
    return min(x for x in starts if min(best)-16 <= x <= max(best)) if len(best)>=2 else 0


def _physical_rows(lines, ruled, header_y, end_y, cut, max_h=45):
    """Join wrapped labels to centred values using complete ruled row bounds.

    Numeric-only shading cannot determine label row bounds. All original word
    coordinates remain unchanged; this only groups their traversal.
    """
    boxes = []
    for t in ruled:
        if t.bbox[0] >= cut-40 or t.bbox[1] > header_y+25 or t.bbox[3]<header_y+35:
            continue
        for row in t.rows:
            cells = [c for c in row.cells if c]
            if cells:
                y0=min(c[1] for c in cells);y1=max(c[3] for c in cells)
                if y1-y0<max_h and header_y+3<y1<=end_y+1:
                    boxes.append((y0,y1))
    result=[];used=set()
    for i,line in enumerate(lines):
        if i in used: continue
        y=sum(w['top']+w['bottom'] for w in line)/(2*len(line))
        box=next((b for b in boxes if b[0]<=y<b[1]),None)
        if box:
            indices=[j for j,ls in enumerate(lines) if j not in used and
                     box[0] <= sum(w['top']+w['bottom'] for w in ls)/(2*len(ls)) < box[1]]
            result.append([w for j in indices for w in lines[j]]);used.update(indices)
        else:
            result.append(line);used.add(i)
    return result



_GROUP_SCOPE = re.compile(r"(?i)scopes?\s*([123])\b\s*(.*)$")


def _generic_block_total(rows, ks):
    """"Scope 1 | GHG emissions 64,353 / Thereof from own operations 40,561":
    a block whose first row is the bare "GHG emissions" and whose other rows
    are "thereof" parts. The first row is the scope's amount."""
    if not ks or rows[ks[0]]['field'] is not None:
        return False
    if not re.fullmatch(r"(?i)\s*(?:total\s+)?(?:gross\s+)?(?:ghg\s+|greenhouse gas\s+)?emissions\s*", rows[ks[0]]['label'] or ''):
        return False
    rest = [rows[k] for k in ks[1:]]
    return all(re.match(r"(?i)\s*(?:thereof|of which|davon|\(\d{1,2}\)|incl)", r['label'] or '')
               or (r['field'] or '').startswith('scope3_cat') for r in rest)


def _apply_left_group_labels(rows, words, label_left, header_y, next_y):
    """Scope group labels in a column left of the row labels.

    ESRS inventories print "Scope 1" / "Scope 2" / "Scope 3 Upstream" in a
    first column, vertically centred on their block, with a bare "Subtotal"
    row closing each block. The subtotal of a block holding exactly one
    unqualified scope label is that scope's total; a qualified block
    ("Scope 3 Upstream") is a partial sum and stays unclassified. A Scope 2
    label printed on the same line as a single basis row ("Emissions linked
    to electricity consumption (market-based)") names that row's scope.
    """
    if not label_left or not rows:
        return
    side = sorted([w for w in words if w['x0'] < label_left - 3 and header_y + 3 < w['top'] < next_y],
                  key=lambda w: (w['top'], w['x0']))
    labels = []
    for w in side:
        # a second "Scope" word starts its own label (the "Scope" column
        # heading or the "Scope 1–3" of the next line is not part of the label above)
        if (labels and w['top'] - labels[-1]['bottom'] < 6 and w['x0'] < labels[-1]['x1'] + 40
                and not (re.fullmatch(r"(?i)scopes?", w['text'])
                         and any(re.fullmatch(r"(?i)scopes?", x['text']) for x in labels[-1]['words']))):
            g = labels[-1]
            g['words'].append(w)
            g['bottom'] = max(g['bottom'], w['bottom'])
            g['x1'] = max(g['x1'], w['x1'])
        else:
            labels.append({'words': [w], 'top': w['top'], 'bottom': w['bottom'], 'x1': w['x1']})
    groups = []
    for g in labels:
        m = _GROUP_SCOPE.match(_text(sorted(g['words'], key=lambda w: (round(w['top']), w['x0']))))
        if m and len(m.group(2).split()) <= 2:
            groups.append({'scope': m.group(1), 'qualified': bool(m.group(2).strip()), 'qualifier': m.group(2).strip(),
                           'top': g['top'], 'bottom': g['bottom'], 'mid': (g['top'] + g['bottom']) / 2})
    if os.environ.get('S3T_DEBUG_ROWS'):
        print('GROUPLABELS', label_left, [(g['scope'], g['qualified'], round(g['mid'])) for g in groups])
    if not groups:
        return
    start = 0
    for k, r in enumerate(rows):
        bare_total = re.fullmatch(r"(?i)\s*(?:sub-?\s?total|total)\s*", r['label'] or '')
        if bare_total:
            y0, y1 = rows[start]['bbox'][1] - 4, r['bbox'][3] + 2
            inside = [g for g in groups if y0 <= g['mid'] <= y1]
            if (len(inside) == 1 and not inside[0]['qualified'] and inside[0]['scope'] in '13'
                    and r['field'] in (None, f"scope3_cat{r.get('parent_category')}")):
                r.update(field='scope1' if inside[0]['scope'] == '1' else 'scope3_total',
                         value_kind='category_total', parent_category=None, quantity_type='inventory',
                         scope_aggregate=True, group_label=f"Scope {inside[0]['scope']}")
            elif inside and r['field'] and r['field'].startswith('scope3_cat'):
                # a block subtotal is never a category row
                r.update(field=None, parent_category=None, quantity_type='unclassified')
            start = k + 1
            continue
        if r['field'] is None:
            mid = (r['bbox'][1] + r['bbox'][3]) / 2
            same = [g for g in groups if g['scope'] == '2' and not g['qualified'] and abs(g['mid'] - mid) < 4]
            basis = re.search(r"(?i)\b(location|market)[- ]*based", r['label'] or '')
            if same and basis:
                r.update(field=f"scope2_{basis.group(1).lower()}", quantity_type='inventory',
                         group_label='Scope 2')
    # Each row's group: labels printed level with the first row of their
    # block ("Scope 1" beside "Gross emissions") own the rows below them;
    # vertically centred labels own the contiguous block they centre on.
    mids = [(r['bbox'][1] + r['bbox'][3]) / 2 for r in rows]
    block = {}
    if len(groups) >= 2 and all(any(abs(g['mid'] - m) < 5 for m in mids) for g in groups) \
            and min(mids) >= groups[0]['mid'] - 5:
        for k, m in enumerate(mids):
            above = [g for g in groups if g['mid'] <= m + 4]
            if above:
                block[k] = above[-1]
    elif len(groups) >= 2:
        for s0 in range(len(rows)):
            s, got = s0, {}
            for g in groups:
                e = next((e for e in range(s, len(rows)) if abs((mids[s] + mids[e]) / 2 - g['mid']) < 5), None)
                if e is None:
                    break
                for k in range(s, e + 1):
                    got[k] = g
                s = e + 1
                if s >= len(rows) and g is not groups[-1]:
                    break
            if len(set(map(id, got.values()))) == len(groups):
                block = got
                break
    by_group = defaultdict(list)
    for k, g in block.items():
        by_group[id(g)].append(k)
    for g in groups:
        ks = by_group.get(id(g), [])
        if not ks:
            continue
        if g['qualified']:
            if re.match(r"\s*[\u2010-\u2014-]\s*[23]\b", g.get('qualifier', '')):
                # "Scope 1–3 | Total GHG emissions (scope 2 location-based)": the
                # basis names the Scope 2 variant inside a combined total
                for k in ks:
                    r = rows[k]
                    if r['field'] and not _category(r['label'] or ''):
                        r.update(field=None, parent_category=None, quantity_type='unclassified',
                                 group_label=f"Scope {g['scope']}{g['qualifier']}")
                continue
            if '+' in g.get('qualifier', ''):
                # "Scope 1+2" / "Scope 1+2+3" totals are combined scopes,
                # never a component inherited from the category above
                for k in ks:
                    r = rows[k]
                    if r['field'] and r['field'].startswith('scope3_cat') and not _category(r['label'] or ''):
                        r.update(field=None, parent_category=None, quantity_type='unclassified')
            continue
        labels = [(k, rows[k]['label'] or '') for k in ks]
        usable = lambda lab: not re.search(r"%|\bnet\b|\bscopes?\s*\d", lab, re.I)
        if g['scope'] == '1':
            tot = [k for k, lab in labels if re.match(r"(?i)\s*total\b", lab) and usable(lab)
                   and rows[k]['field'] in (None, 'scope1') and k in (ks[0], ks[-1])]
            if len(tot) == 1:
                rows[tot[0]].update(field='scope1', value_kind='category_total', scope_aggregate=True,
                                    quantity_type='inventory', parent_category=None, group_label='Scope 1')
                for k in ks:
                    if k != tot[0] and rows[k]['field'] == 'scope1':
                        rows[k]['value_kind'] = 'component'
            elif _generic_block_total(rows, ks):
                rows[ks[0]].update(field='scope1', value_kind='category_total', quantity_type='inventory',
                                   parent_category=None, group_label='Scope 1')
            else:
                for k, lab in labels:
                    if rows[k]['field'] is None and usable(lab) and re.search(r"(?i)\bgross\b", lab) \
                            and not re.search(r"(?i)(?:location|market)[- ]*based", lab):
                        rows[k].update(field='scope1', quantity_type='inventory', group_label='Scope 1')
        elif g['scope'] == '2':
            for k, lab in labels:
                basis = re.search(r"(?i)\b(location|market)[- ]*based", lab)
                if rows[k]['field'] is None and basis and usable(lab) and not re.match(r"(?i)\s*total\b", lab):
                    rows[k].update(field=f"scope2_{basis.group(1).lower()}", quantity_type='inventory',
                                   group_label='Scope 2')
        elif g['scope'] == '3':
            tot = [k for k, lab in labels if re.match(r"(?i)\s*(?:sub-?\s?)?total\b", lab) and usable(lab)
                   and rows[k]['field'] is None and k in (ks[0], ks[-1])
                   and not re.search(r"(?i)(?:location|market)[- ]*based", lab)]
            bare = [k for k, lab in labels if re.fullmatch(r"(?i)\s*(?:sub-?\s?)?total\s*", lab) and rows[k]['field'] is None]
            if not tot and bare:
                tot = bare[:1]
            if not tot and _generic_block_total(rows, ks):
                tot = [ks[0]]
            if len(tot) == 1:
                rows[tot[0]].update(field='scope3_total', value_kind='category_total', parent_category=None,
                                    quantity_type='inventory', scope_aggregate=True, group_label='Scope 3')




def _demote_heading_merged_children(rows):
    """"1 Purchased goods and services" printed without amounts, then
    "Meat raw materials 1,841,672 / Other edible raw materials ... /
    Packaging materials ...": the heading was joined to the first child's
    label. When that row's own words do not name the category and its
    siblings under the same category follow as parts, it is a part too."""
    for k, r in enumerate(rows):
        f = r.get('field') or ''
        if not f.startswith('scope3_cat') or r.get('value_kind') != 'category_total':
            continue
        own = (r.get('_own_text') or '').strip()
        if not own or own == (r.get('label') or '').strip() or _category(own) or own.startswith('('):
            continue
        if not re.match(r"\s*(?:category\s*)?(?:1[0-5]|[1-9])\b", r['label'] or '', re.I):
            continue
        sibs = []
        for q in rows[k+1:]:
            if q.get('field') != f:
                break
            sibs.append(q)
        if sibs and all(q.get('value_kind') == 'component' for q in sibs) and all(
                any(parse_number(c['raw']) is not None for c in q['cells']) for q in sibs):
            r['value_kind'] = 'component'
            r['heading_merged'] = True

def _infer_categories_by_order(rows):
    """Short category names the mapping cannot place ("Products",
    "Transport", "Business trips") take their place in the GHG Protocol
    order when the rows around them are confidently mapped categories and
    exactly one category number is free between them (Fuel = 3, ?, Waste = 5
    gives 4; a first row before Capital goods = 2 gives 1)."""
    def confident(r):
        if not (r.get('field') or '').startswith('scope3_cat') or r.get('value_kind') != 'category_total':
            return None
        lab = r.get('label') or ''
        if _explicit_category_marker(lab):
            return int(r['field'][10:])
        no, _, _, score = map_scope3_category(lab)
        return no if no and score >= .9 and no == int(r['field'][10:]) else None
    conf = [confident(r) for r in rows]
    if sum(1 for c in conf if c) < 3:
        return
    for k, r in enumerate(rows):
        if conf[k]:
            continue
        lab = (r.get('label') or '').strip()
        if (not lab or len(lab.split()) > 4 or not lab[0].isupper()
                or re.search(r"(?i)total|scope|%|of which|thereof|includ|excl|other|\bper\b|intensity", lab)
                # unqualified transport is Category 4 or 9: order does not decide
                or re.search(r"(?i)transport|distribution", lab)
                or not any(parse_number(c['raw']) is not None for c in r['cells'])
                or r.get('quantity_type') == 'excluded_noninventory'
                or (r.get('field') or '').startswith(('scope1', 'scope2'))):
            continue
        prev = next(((j, conf[j]) for j in range(k - 1, -1, -1) if conf[j]), None)
        nxt = next(((j, conf[j]) for j in range(k + 1, len(rows)) if conf[j]), None)
        guess = None
        if prev and nxt and nxt[1] - prev[1] == 2 and nxt[0] - prev[0] == 2:
            guess = prev[1] + 1
        elif prev is None and nxt and nxt[1] == 2 and nxt[0] == k + 1:
            guess = 1
        if guess:
            r.update(field=f"scope3_cat{guess}", value_kind='category_total', parent_category=guess,
                     parent_scope=None, scope_aggregate=False, quantity_type='inventory',
                     category_by_order=True)


def _apply_entity_column(rows, words, label_left, header_y, next_y):
    """A "Subsidiary" / "Entity" column left of the row labels: rows under an
    entity name are that entity's figures, rows under a "Total ... Group"
    caption the consolidated ones (nearest caption by vertical distance)."""
    if not label_left or not rows:
        return
    hdr = [w for w in words if abs(w['top'] - header_y) < 6 and w['x1'] < label_left - 3]
    if not any(re.fullmatch(r"(?i)subsidiar(?:y|ies)|entity|entities|company|companies|legal", w['text']) for w in hdr):
        return
    side = sorted([w for w in words if header_y + 6 < w['top'] < next_y and w['x1'] < label_left - 3],
                  key=lambda w: (w['top'], w['x0']))
    labels = []
    for w in side:
        if labels and w['top'] - labels[-1]['bottom'] < 6:
            labels[-1]['words'].append(w); labels[-1]['bottom'] = max(labels[-1]['bottom'], w['bottom'])
        else:
            labels.append({'words': [w], 'top': w['top'], 'bottom': w['bottom']})
    if not labels:
        return
    for g in labels:
        g['text'] = _text(g['words']); g['mid'] = (g['top'] + g['bottom']) / 2
    for r in rows:
        mid = (r['bbox'][1] + r['bbox'][3]) / 2
        g = min(labels, key=lambda g: abs(g['mid'] - mid))
        group = re.search(r"(?i)\btotal\b|\bgroup\b|consolidated", g['text'])
        r['organization_boundary'] = 'consolidated_group' if group else 'individual_entity'
        r['entity_label'] = g['text']

_SUM_FIELDS = ('scope1', 'scope2', 'scope2_location', 'scope2_market', 'scope3_total')


def _apply_summation_aggregates(rows, cols):
    """A row that is the printed sum of its same-field siblings is their total.

    Entity breakdowns give
    every row the same field. When one row equals the sum of the other rows
    of that field in every reported column where all of them are printed
    (two or more non-zero addends), that row is the aggregate and the others
    are its components. Without such an arithmetic identity nothing changes.
    """
    reported = [c['id'] for c in cols if c.get('role') == 'reported']
    if not reported:
        return
    def vals(r):
        out = {}
        for c in r['cells']:
            if c['column_id'] in reported:
                v = parse_number(c['raw'])
                if v is None and re.fullmatch(r"[-\u2013\u2014]", str(c['raw']).strip()):
                    v = 0.0  # a dash prints a nil amount
                if v is not None:
                    out[c['column_id']] = v
                    m = re.search(r"[.,](\d{1,2})\D*$", c['raw'])
                    dec[c['column_id']] = max(dec.get(c['column_id'], 0), len(m.group(1)) if m else 0)
        return out
    dec = {}
    # a bare "Total" summing rows of several fields (Scope 1 + Scope 2 +
    # business travel) is a combined total, not the last field's total
    for k, t in enumerate(rows):
        if not re.fullmatch(r"(?i)\s*total\s*", t.get('label') or '') or not t.get('field'):
            continue
        prior = [r for r in rows[:k] if r.get('field') and r.get('quantity_type') == 'inventory']
        if len({r['field'] for r in prior}) < 2:
            continue
        tv, ok, checked = vals(t), True, 0
        for cid in reported:
            av = [vals(r).get(cid) for r in prior]
            if cid not in tv or any(a is None for a in av):
                continue
            if abs(sum(av) - tv[cid]) > .5 * 10 ** -dec.get(cid, 0) * (len(av) + 1) + 1e-9:
                ok = False
                break
            checked += 1
        if ok and checked:
            t.update(field=None, parent_category=None, parent_scope=None, scope_aggregate=False,
                     value_kind='category_total', quantity_type='unclassified', combined_total=True)
    by_field = defaultdict(list)
    for r in rows:
        f = r.get('field')
        if (f in _SUM_FIELDS or (f or '').startswith('scope3_cat')) and r.get('quantity_type') == 'inventory':
            by_field[f].append(r)
    for f, group in by_field.items():
        if len(group) < 3:
            continue
        mults = {(r.get('unit') or {}).get('multiplier') for r in group}
        if len(mults) > 1:
            continue
        hits = []
        for k, t in enumerate(group):
            for addends in (group[:k], group[k+1:]):
                if len(addends) < 2:
                    continue
                tv, av = vals(t), [vals(a) for a in addends]
                checked, ok = 0, True
                for cid in reported:
                    if cid not in tv or any(cid not in a for a in av):
                        continue
                    nz = [a[cid] for a in av if a[cid]]
                    total = sum(a[cid] for a in av)
                    # printed rounding: half a unit of the last digit per row
                    tol = .5 * 10 ** -dec.get(cid, 0) * (len(av) + 1) + 1e-9
                    if abs(total - tv[cid]) > tol:
                        ok = False
                        break
                    if len(nz) >= 2:
                        checked += 1
                if ok and checked:
                    hits.append((t, addends))
                    break
        if len(hits) != 1:
            continue
        t, addends = hits[0]
        scope = f in _SUM_FIELDS
        t.update(value_kind='category_total', sum_of=[a['id'] for a in addends])
        if scope:
            t.update(scope_aggregate=True, parent_scope=f)
        for a in addends:
            a['value_kind'] = 'component'
            if scope:
                a.update(scope_aggregate=False, parent_scope=f)

def _reconstruct_grid(page, doc_id, page_no, table_offset=0):
    """Recover horizontal year grids independently of detected border tables.

    Each cell retains physical bounds. Ruled merged label cells supply parent
    spans; this is essential when a category label is vertically centred. A
    candidate without aligned numeric rows is discarded with a diagnostic.
    """
    words = _year_words(page.extract_words(x_tolerance=1, y_tolerance=3) or [], page)
    words = _drop_detached_scripts(words)
    words = _rotated_words(page, words)
    lines = _lines(words)
    ruled = page.find_tables() or []
    parent_spans, descriptor_spans = [], []
    for table in ruled:
        for row in table.rows:
            for cell in row.cells:
                if not cell:
                    continue
                inside = [w for w in words if cell[0] <= (w["x0"] + w["x1"])/2 <= cell[2]
                          and cell[1] <= (w["top"] + w["bottom"])/2 <= cell[3]]
                label = _text(inside)
                if re.search(r"\bCategory\s+(?:1[0-5]|[1-9])\b", label, re.I):
                    parent_spans.append({"bbox": list(cell), "label": label, "category": _category(label)})
                elif inside and cell[3]-cell[1] > 2*median(w['bottom']-w['top'] for w in inside) and not _field(label):
                    descriptor_spans.append({'bbox':list(cell),'label':label})

    headers = []
    for i, line in enumerate(lines):
        if re.search(r'\bfrom\s+20\d{2}\s+to\s+20\d{2}',_text(line),re.I):
            continue
        # a period statement "1/1/2024 - 31/12/2024" is one period, not two columns
        if re.search(r'\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}\s*[-\u2013]\s*\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}',
                     ' '.join(w.get('source_text', w['text']) for w in line)):
            continue
        years = [w for w in line if YEAR.fullmatch(w["text"])]
        # "December 2025", "Performance 2025", "Annual Report 2025": a year
        # completing a phrase is running text, not a column heading
        def _phrase_year(y):
            prev = [x for x in line if 0 <= y['x0'] - x['x1'] < 5 and abs(x['top'] - y['top']) < 3]
            return any(re.fullmatch(r"(?i)(?:january|february|march|april|may|june|july|august|september|october|"
                                    r"november|december|performance|report|review|statement|overview|highlights)\.?,?",
                                    x['text']) for x in prev)
        phrase = [y for y in years if _phrase_year(y)]
        # "31 December 2024 | 31 December 2023" column headings repeat the phrase
        if len(phrase) < 2:
            years = [y for y in years if y not in phrase]
        # Dense prose is not a header. A single-year grid requires an explicit
        # Unit heading, whereas two-year grids must have a wide numeric region.
        # A lone year heading is accepted when the line is just that heading
        # (optionally with a unit/label word) and sits over a numeric region;
        # the table is still discarded unless aligned Scope rows follow.
        lone_heading = (len(years) == 1 and len([x for x in line if x is not years[0]]) <= 3
                        and years[0]['x0'] > page.width*.3
                        # running headers/footers ("ANNUAL REPORT 2025") are not column headings
                        and page.height*.06 < years[0]['top'] < page.height*.94
                        and not re.search(r"(?i)\breport\b|\bannual\b|\bstatement\b", _text(line)))
        # "Direct Scope 1 emissions | 2024": a section label on the left and a
        # single year heading the right-hand value column
        if (len(years) == 1 and not lone_heading and years[0]['x0'] > page.width * .5
                and page.height*.06 < years[0]['top'] < page.height*.94
                and 0 < len([x for x in line if x is not years[0]]) <= 8
                and all(x['x1'] < years[0]['x0'] - 80 for x in line if x is not years[0])
                and re.search(r"(?i)\b(?:emissions?|ghg|scope|greenhouse)\b", _text(line))
                and not re.search(r"(?i)\breport\b|\bannual\b|\bstatement\b|\bin\s+20\d\d", _text(line))):
            lone_heading = True
        if len(years) < 2 and not (len(years) == 1 and (
                re.search(r"\b(?:Unit|Base year|N)\b", _text(line)) or years[0].get('fiscal_period')
                or lone_heading)):
            continue
        # Body amounts and base-year notes do not start another year grid.
        if any(w['x0']>min(y['x0'] for y in years) and re.fullmatch(r'\d+[,.]\d[\d,.]*', w['text']) for w in line):
            continue
        if re.search(r'\b(?:for corporate|for Lending|for AuM)\b', _text(line), re.I):
            continue
        first = min(w["x0"] for w in years)
        between = [w for w in line if first <= w["x0"] <= years[-1]['x1']+40 and not YEAR.fullmatch(w["text"])]
        if len(between) > 16 or (len(years) > 1 and years[-1]["x0"] - first < 20):
            continue
        if re.search(r"\b(?:were|from|during|compared|since|as of|would|towards)\b", _text(between), re.I):
            continue
        # "As 2024 and 2025 are the ...": years running inside a sentence
        def _in_sentence(y):
            near = [x for x in line if x is not y]
            before = any(0 <= y['x0']-x['x1'] < 6 and re.fullmatch(r"[A-Za-z]{2,}[.,;]?", x['text']) for x in near)
            after = any(0 <= x['x0']-y['x1'] < 6 and re.fullmatch(r"[a-z]{2,}[.,;]?", x['text']) for x in near)
            return before and after
        if sum(_in_sentence(y) for y in years) >= max(1, len(years) // 2):
            continue
        headers.append((i, years))

    # A lone year heading inside, or just above, a multi-year grid is a group
    # caption (e.g. "Target 2030"), never an independent reporting column.
    multi = [ys for _, ys in headers if len(ys) >= 2]
    def _inside_multi(ys):
        if len(ys) != 1:
            return False
        w = ys[0]
        return any(abs(min(m['top'] for m in ms) - w['top']) < 120
                   and min(m['x0'] for m in ms) - 60 <= w['x0'] <= max(m['x1'] for m in ms) + 60
                   for ms in multi)
    # A lone year heading is only trusted on a page without multi-year grids.
    headers = [(i, ys) for i, ys in headers
               if not (len(ys) == 1 and not ys[0].get('fiscal_period') and _inside_multi(ys))]

    if os.environ.get('S3T_DEBUG_GRID'):
        print('HEADERS', [(round(ys[0]['top']), [w['text'] for w in ys]) for _, ys in headers])
    tables, issues = [], []
    for hi, (line_index, years) in enumerate(headers):
        header_y = min(w["top"] for w in years)
        # Only a header in the same horizontal numeric region can end this
        # table. Sidebar/prose years must not truncate a neighbouring grid.
        anchors = [w['x1'] for w in years]
        later = [ys for _,ys in headers[hi+1:] if any(abs(w['x1']-x)<20 for w in ys for x in anchors)]
        next_y = min(w['top'] for w in later[0]) if later else page.height
        _hts = sorted(w['bottom'] - w['top'] for w in words) or [10]
        _mh = _hts[len(_hts) // 2]
        def _amount_lines(y0, y1):
            # year captions and subscript digits ("tCO" + "2") are not amounts
            return sum(1 for l in lines if y0 < min(w['top'] for w in l) < y1
                       and any(re.fullmatch(r"[-(]?\d[\d,.' ]*\d\)?|\d", w['text']) and w['x0'] >= min(y['x0'] for y in years) - 40
                               and not YEAR.fullmatch(w['text'])
                               and (len(w['text']) > 1 or w['bottom'] - w['top'] >= .8 * _mh)
                               for w in l))
        # A ruled box ends the table only when it actually holds the amounts
        # (a ruled header band is not the table).
        bottoms = [t.bbox[3] for t in ruled if len(t.rows)>=4 and t.bbox[1] <= header_y+20 < t.bbox[3]
                   and t.bbox[3] > header_y+30
                   and any(t.bbox[0] <= w['x0'] <= t.bbox[2] for w in years)
                   # a shaded band under one value column is not the table box
                   and (sum(t.bbox[0] <= w['x0'] <= t.bbox[2] for w in years) >= 2
                        or t.bbox[0] < min(w['x0'] for w in years) - 60 or len(years) == 1)
                   and _amount_lines(header_y, t.bbox[3]) >= 2]
        if bottoms:
            bt = min(bottoms)
            box = next(t for t in ruled if t.bbox[3] == bt)
            # the same column grid continued in a following ruled box (one per
            # scope section, a heading between them), and a total row printed
            # just under the last box, belong to this table
            while True:
                more = [t for t in ruled if abs(t.bbox[0] - box.bbox[0]) < 6 and abs(t.bbox[2] - box.bbox[2]) < 6
                        and 0 < t.bbox[1] - bt < 45 and len(t.rows) >= 3]
                if not more:
                    break
                box = more[0]
                bt = box.bbox[3]
            for l in lines:
                ly = min(w['top'] for w in l)
                if bt - 2 < ly < bt + 12 and re.match(r"(?i)total\b", _text(l)) and any(
                        re.fullmatch(r"[-(]?\d[\d,.' ]*\d\)?", w['text']) and w['x0'] >= min(y['x0'] for y in years) - 40
                        for w in l):
                    bt = max(w['bottom'] for w in l)
            next_y = min(next_y, bt+1)
        # Wrapped base-year cells can sit one baseline below other headers.
        years = list(years)
        g0 = [b['x1']-a['x1'] for a, b in zip(years, years[1:])]
        span = max(2*(median(g0) if g0 else 60), .5*page.width)
        hx0, hx1 = min(w['x0'] for w in years) - span, max(w['x1'] for w in years) + span
        for nearline in lines:
            if not any(abs(w['top']-header_y)<12 for w in nearline):
                continue
            for w in nearline:
                # prose: words run through the year on both sides ("In 2024 we had")
                before = any(abs(x['top']-w['top']) < 3 and 0 <= w['x0']-x['x1'] < 6
                             and re.fullmatch(r"[A-Za-z]{2,}[.,;]?", x['text']) for x in nearline if x is not w)
                after = any(abs(x['top']-w['top']) < 3 and 0 <= x['x0']-w['x1'] < 6
                            and re.fullmatch(r"[a-z]{2,}[.,;]?", x['text']) for x in nearline if x is not w)
                prose = before and after
                if (YEAR.fullmatch(w['text']) and abs(w['top']-header_y)<12 and hx0 <= w['x0'] <= hx1
                        and not prose and all(abs(w['x1']-a['x1'])>18 for a in years)):
                    years.append(w)
        years = sorted(years, key=lambda w: w["x0"])
        gaps = [b["x1"] - a["x1"] for a,b in zip(years, years[1:])]
        gap = median(gaps) if gaps else 60
        cut = years[0]["x0"] - min(35, gap*.5)
        # "Own emissions | Base year | Comparative | 2024 | ...": value columns
        # captioned without a year still start the value region; their
        # amounts are not label text (they stay unassigned to any year).
        hdr_line = [w for l in lines if -12 <= min(x['top'] for x in l) - header_y <= 2 for w in l]
        caps = [w for w in hdr_line if re.fullmatch(r"(?i)base(?:line)?|comparative|reference|previous", w['text'])
                and years[0]['x0'] - 250 < w['x0'] < years[0]['x0'] - 10
                and (w['x0'] > page.width * .4
                     or any(x['x1'] < w['x0'] - 20 and re.search('[A-Za-z]{3}', x['text']) for x in hdr_line))]
        year_cut = cut  # the value columns proper start here
        if caps:
            cut = min(cut, min(w['x0'] for w in caps) - 4)
        # Segment sub-header under the year ("France | Rest of Europe | ... |
        # Group"): the amounts start at the first segment column.
        seg_line = _segment_header(lines, header_y, years)
        if seg_line:
            cut = min(cut, min(w['x0'] for w in seg_line) - 4)
        label_left = _label_left(lines, header_y, cut, next_y)
        if label_left:
            # "Total GHG direct emissions" set a few points left of the
            # numbered category rows: a label word joined to the label text
            ext = []
            for line in lines:
                if not header_y < min(w['top'] for w in line) < next_y or not _numeric_units(line, cut):
                    continue
                left = [w for w in line if w['x1'] < cut]
                inside = [k for k, w in enumerate(left) if w['x0'] >= label_left - 1]
                if not inside:
                    continue
                j = inside[0]
                while j > 0 and left[j-1]['x0'] >= label_left - 20 and left[j]['x0'] - left[j-1]['x1'] < 5:
                    j -= 1
                if j < inside[0]:
                    ext.append(left[j]['x0'])
            if len(ext) >= 2:
                label_left = min(ext)
        grid_lines =[[w for w in line if w['x0'] >= label_left-1] for line in lines]
        grid_lines = [line for line in grid_lines if line]
        nearby = [w for w in words if header_y-65 <= w["top"] <= header_y+12 and w['x0'] >= label_left-1]
        # Multi-level year headings can cover separate organization columns.
        # Retain both; the aggregate between them is a distinct row boundary.
        subheads = [w for w in words if header_y+5 < w['top'] < header_y+40
                    and w['text'].lower() in ('group','control') and w['x0'] > cut]
        expanded = []
        if len(subheads) == 2*len(years):
            for j,w in enumerate(years):
                lo = w['x0']-gap*.3
                hi_x = years[j+1]['x0']-gap*.3 if j+1<len(years) else page.width
                pair = [s for s in subheads if lo <= s['x0'] < hi_x]
                if {s['text'].lower() for s in pair} != {'group','control'}:
                    expanded=[];break
                for s in pair:
                    expanded.append(dict(w, x0=s['x0'],x1=s['x1'],
                                         organization_boundary='consolidated_group' if s['text'].lower()=='group' else 'additional_operational_control',
                                         organization_header=s['text'], organization_bbox=_bbox([s]),
                                         year_source_bbox=_bbox([w])))
        if expanded:
            years=expanded;gaps=[b['x1']-a['x1'] for a,b in zip(years,years[1:])];gap=median(gaps)
            cut=years[0]['x0']-min(35,gap*.5)
        targets = [w for w in nearby if w["text"].lower() in ("milestones", "target", "targets")
                   and w["x0"] > max(cut, year_cut)]
        # Only broad group headings define a target region; the words 'target
        # base year' below a historical header do not make history a target.
        group_targets = [w for w in targets if w["text"].lower() == "milestones" or
                         any(abs(x['top']-w['top'])<2 and x['text'].lower() in ('year','years')
                             and 0<x['x0']-w['x1']<8 for x in nearby)
                         # "Year for targets and sub-targets"
                         or any(abs(x['top']-w['top'])<2 and x['text'].lower() in ('year','years')
                                and 0 < w['x0']-x['x1'] < 30 for x in nearby)]
        target_x = min((w["x0"]-25 for w in group_targets), default=page.width+1)
        # ESRS "Retrospective | Milestones and target years": a year column
        # printed after the retrospective block belongs to the milestones.
        # ESRS order: Base year | Comparative | N | % N/N-1 | milestones...
        # A year column after the variance column belongs to the milestones.
        if group_targets:
            mx = min(w['x0'] for w in group_targets)
            change_x = [w['x1'] for w in nearby if w['x0'] < mx and abs(w['top'] - header_y) < 45
                        and re.search(r"^(?:change|variation|evolution|%)$|N\s*/\s*N", w['text'], re.I)]
            if change_x:
                later_years = [w['x0'] for w in years if w['x0'] > min(change_x) + 2 and not _is_change_year(w, words)]
                if later_years:
                    target_x = min(target_x, min(later_years) - 1)
        if group_targets:
            seen_years = set()
            for j,w in enumerate(years):
                year = int(YEAR.fullmatch(w['text']).group(1))
                if _is_change_year(w, words):
                    continue
                if year in seen_years and w['x0'] > target_x-gap:
                    start = j-1 if j and int(YEAR.fullmatch(years[j-1]['text']).group(1)) > year else j
                    target_x = min(target_x, years[start]['x0']-1)
                    break
                seen_years.add(year)
        sbti = [w for w in nearby if w["text"].lower() == "sbti" and w["x0"] > cut]
        sbti_x = min((w["x0"]-8 for w in sbti), default=page.width+1)
        cols = []
        for j, w in enumerate(years):
            context = _text([x for x in nearby if abs(x["x1"]-w["x1"]) < gap*.45])
            # A year inside a variance heading ("% 2025/2024", "vs 2024") is
            # not a reporting column even though it carries a year token.
            is_change = _is_change_year(w, words)
            # "2018-24 | 2023-24" captioned "%" on the unit line below: variance
            cap_below = [x for x in words if 6 < x['top'] - w['top'] < 24 and x['x0'] < w['x1'] + 4 and x['x1'] > w['x0'] - 4]
            if cap_below and all(re.fullmatch(r"\(?%\)?|var\.?|chg\.?", x['text'], re.I) for x in cap_below):
                is_change = True
            # "Target" printed directly above the year names a target column
            if any(x['text'].lower() in ('target', 'targets', 'goal', 'objective') and 0 < w['top'] - x['top'] < 14
                   and x['x0'] < w['x1'] + 2 and x['x1'] > w['x0'] - 2 for x in words):
                target_x = min(target_x, w['x0'])
            cols.append({"id": f"c{j+1}", "year": int(YEAR.fullmatch(w["text"]).group(1)),
                         "header": w["text"], "header_context": context, "bbox": _bbox([w]),
                         "role": "target" if w["x0"] >= target_x else "change" if is_change else "base" if _is_base_year_caption(w, words) else "reported",
                         "boundary": "SBTi" if w["x0"] >= sbti_x else "reported",
                         "anchor": w["x1"],
                         "year_source_bbox":w.get('year_source_bbox',_bbox([w])),
                         "source_header":w.get('source_text',w['text']),
                         "footnote_bbox":w.get('footnote_bbox'),
                         "organization_boundary":w.get('organization_boundary','reported'),
                         "organization_header":w.get('organization_header'),
                         "organization_bbox":w.get('organization_bbox'),
                         # "2022 (baseline)", "2025 emissions": right-aligned cells
                         # end under the words that complete the year caption
                         "header_x1":_caption_right(w, words)})
        # ESRS "Base year | Comparative | N | % N/N-1 | 2025 | 2030": a year
        # later than the reporting period, printed after the variance column
        # that follows the period's own column, is a milestone.
        # Column-group captions above a repeated year sequence ("Cables
        # segment | Steel pipes segment | Consolidated figures" over 2022-2024
        # three times): only the consolidated group reports the company value.
        yrs_ = [c['year'] for c in cols if c['role'] == 'reported']
        if len(yrs_) >= 4 and len(set(yrs_)) < len(yrs_):
            cap_words = sorted([w for w in words if header_y - 40 <= w['top'] < header_y - 3 and w['x0'] >= cut - 5
                                and re.search('[A-Za-z]{2}', w['text'])], key=lambda w: w['x0'])
            runs_ = []
            for w in cap_words:
                if runs_ and w['x0'] - runs_[-1][-1]['x1'] < 14:
                    runs_[-1].append(w)
                else:
                    runs_.append([w])
            # wrapped captions ("Steel pipes" / "segment") merge by overlap
            merged_ = []
            for r_ in sorted(runs_, key=lambda r: r[0]['x0']):
                if merged_ and r_[0]['x0'] < merged_[-1][-1]['x1'] + 2:
                    merged_[-1] = sorted(merged_[-1] + r_, key=lambda w: (round(w['top']), w['x0']))
                else:
                    merged_.append(r_)
            tot_re = r"(?i)\b(?:consolidated|group|total)\b"
            if len(merged_) >= 2 and sum(bool(re.search(tot_re, _text(r_))) for r_ in merged_) == 1:
                centres = [((min(w['x0'] for w in r_) + max(w['x1'] for w in r_)) / 2, r_) for r_ in merged_]
                for c in cols:
                    if c['role'] != 'reported':
                        continue
                    cx = (c['bbox'][0] + c['bbox'][2]) / 2
                    _, own = min(centres, key=lambda t: abs(t[0] - cx))
                    if re.search(tot_re, _text(own)):
                        c.update(organization_boundary='consolidated_group', organization_header=_text(own))
                    else:
                        c.update(role='segment', organization_header=_text(own))
        period = getattr(page, '_s3t_period_year', None) or _PERIOD_YEAR.get()
        pcol = next((c for c in cols if c['role'] == 'reported' and c['year'] == period), None) if period else None
        if pcol:
            pct = [w for w in words if header_y + 5 < w['top'] < next_y
                   and re.fullmatch(r"\(?[-\u2013\u2014+]?[\d.,]*\)?\s*%", w['text'])]
            for c in cols:
                if c['role'] == 'reported' and c['year'] > period and c['bbox'][0] > pcol['bbox'][2]:
                    if sum(1 for w in pct if pcol['bbox'][2] < w['x0'] and w['x1'] < c['bbox'][0]) >= 2:
                        c['role'] = 'target'
        # the label-column header may wrap its unit onto the next line:
        # "GHG emissions | 2024 | 2025" / "(tCO2eq)"
        unit_lines = [_unit(line) for line in grid_lines if header_y-65 <= min(w["top"] for w in line) <= header_y+8
                      or (header_y+8 < min(w["top"] for w in line) <= header_y+16 and not _numeric_units(line, cut))]
        from scope3trace.common.perimeter import column_perimeters
        column_perimeters(lines,header_y,cut,cols)
        table_unit = next((u for u in reversed(unit_lines) if u), None)
        if table_unit is None:
            # a table title ("GHG emissions in tCO2eq") set above a tall
            # multi-line column-group band: only caption lines in between
            for tl in reversed([l for l in grid_lines if header_y - 110 <= min(w['top'] for w in l) < header_y - 65]):
                between = [l for l in grid_lines if min(w['top'] for w in tl) < min(w['top'] for w in l) < header_y - 2]
                if any(_numeric_units(l, cut) or _field(_text(l)) for l in between):
                    break
                if all(w['x1'] < cut for w in tl) and re.search(r"(?i)emissions?|ghg", _text(tl)) and _unit(tl):
                    table_unit = dict(_unit(tl), source='table_title')
                    break
        # Per-column unit captions stacked under the year header ("tCO2eq",
        # "in thousands / of t CO2e"). They hold no amounts and no row label.
        subheader_bottom = header_y
        col_units = []
        for line in grid_lines:
            top = min(w['top'] for w in line)
            if top <= header_y + 2:
                continue
            if top > header_y + 34:
                break
            right_words = [w for w in line if w['x0'] >= cut - 2]
            if any(re.fullmatch(r"[-+(]?\d[\d,.' ]*\d\)?%?|[-+]?\d{2,}", w['text']) and not YEAR_RANGE.fullmatch(w['text'])
                   for w in right_words):
                break
            # only captions printed in the value columns count; a label-side
            # heading such as "Scope 1 GHG emissions (tCO2e)" is a body line
            if not right_words:
                continue
            if any(re.search('[A-Za-z]', w['text']) and w['x1'] < cut - 2 for w in line) \
                    and _unit(right_words) is None:
                continue
            subheader_bottom = max(subheader_bottom, max(w['bottom'] for w in right_words))
        if os.environ.get("S3T_DEBUG_ROWS"): print("SUBHDR", header_y, subheader_bottom, "next_y", next_y)
        if subheader_bottom > header_y:
            stacked = [w for line in grid_lines for w in line
                       if header_y + 2 < w['top'] <= subheader_bottom and w['x0'] >= cut - 2]
            owner = {}
            for run in _caption_runs(stacked):
                rx0, rx1 = run[0]['x0'], run[-1]['x1']
                def fit(c):
                    cx0, cx1 = c['bbox'][0], c.get('header_x1', c['bbox'][2])
                    return (max(0, min(rx1, cx1) - max(rx0, cx0)), -abs(rx1 - cx1))
                best_col = max(cols, key=fit)
                for w in run:
                    owner[id(w)] = best_col['id']
            for col in cols:
                if col['role'] != 'reported':
                    continue
                mine = [w for w in stacked if owner.get(id(w)) == col['id']]
                u = _unit(sorted(mine, key=lambda w: (round(w['top']), w['x0']))) if mine else None
                if u:
                    col_units.append(u)
            mults = {u['multiplier'] for u in col_units}
            if col_units and len(mults) == 1 and table_unit is None:
                table_unit = dict(col_units[0], source='column_caption')
        if table_unit is None:
            hu = [w for w in years if w.get('header_unit')]
            if hu:
                u = _unit([dict(hu[0], text=hu[0]['header_unit'])])
                if u:
                    table_unit = dict(u, source='column_caption')
        if table_unit is None:
            # "Gross greenhouse gas emissions within scope 1, 2 and 3, tonnes":
            # a heading naming the gas, ending in a bare mass unit
            for hl in reversed([l for l in _lines([x for x in words if header_y - 40 <= x['top'] < header_y - 2])]):
                m = re.search(r"(?i)(?:greenhouse\s+gas|ghg|co2e?)\s+emissions?\b[^.;:]{0,100}?[,(]\s*"
                              r"(tonnes|tons|t|kt|thousand\s+tonnes)\)?\s*$", _text(hl))
                if m:
                    uw = [x for x in hl if re.fullmatch(r"(?i)\(?(?:tonnes|tons|t|kt|thousand)\)?,?", x['text'])][-1:]
                    if uw:
                        u0 = uw[0]
                        glyphs = ''.join(ch['text'] for ch in sorted(
                            [ch for ch in page.chars if u0['x0'] - .2 <= (ch['x0'] + ch['x1']) / 2 <= u0['x1'] + .2
                             and u0['top'] - .5 <= (ch['top'] + ch['bottom']) / 2 <= u0['bottom'] + .5], key=lambda ch: ch['x0']))
                        if glyphs.replace(' ', '') == u0['text'].replace(' ', ''):
                            unit_word = m.group(1).lower()
                            table_unit = {"text": f"{unit_word} CO2e", "source_text": _text(hl)[-120:],
                                          "bbox": _bbox(uw), "multiplier": 1000.0 if unit_word in ('kt',) or unit_word.startswith('thousand') else 1.0,
                                          "source": "heading", "source_bbox": _bbox(uw), "source_glyph_text": u0['text'],
                                          "evidence_status": "verified", "glyph_checked": True}
                    break
        if table_unit is None:
            # "CO2 emissions - Group | 2024 (ton) | 2024 (%)": a bare mass unit
            # captioned after each reported year, the emitted gas named in the
            # header label; the caption glyphs are checked directly.
            hdr = [x for x in words if abs(x['top'] - header_y) < 4]
            names_gas = re.search(r"(?i)\bco\s*2|\bghg\b|greenhouse", _text([x for x in hdr if x['x1'] < cut]))
            caps = []
            for c in cols:
                if c['role'] != 'reported':
                    continue
                cap = next((x for x in hdr if 0 <= x['x0'] - c['bbox'][2] < 5), None)
                m = cap and re.fullmatch(r"\((t|tons?|tonnes?|kt|thousand\s+tons?)\)", cap['text'], re.I)
                if m:
                    glyphs = ''.join(ch['text'] for ch in sorted(
                        [ch for ch in page.chars if cap['x0'] - .2 <= (ch['x0'] + ch['x1']) / 2 <= cap['x1'] + .2
                         and cap['top'] - .5 <= (ch['top'] + ch['bottom']) / 2 <= cap['bottom'] + .5], key=lambda ch: ch['x0']))
                    if glyphs.replace(' ', '') == cap['text'].replace(' ', ''):
                        caps.append((cap, m.group(1).lower()))
            mults = {1000.0 if u in ('kt',) or u.startswith('thousand') else 1.0 for _, u in caps}
            if names_gas and caps and len(mults) == 1 and len(caps) >= max(1, sum(c['role'] == 'reported' for c in cols) // 2):
                cap = caps[0][0]
                table_unit = {"text": f"{caps[0][1]} CO2e", "source_text": cap['text'], "bbox": _bbox([cap]),
                              "multiplier": mults.pop(), "source": "column_caption", "source_bbox": _bbox([cap]),
                              "source_glyph_text": cap['text'], "evidence_status": "verified", "glyph_checked": True}
        rows, section, current_cat = [], "", None
        non_numeric_rows = []
        scope_parent = None
        heading_no = 0
        cat_row_x0 = None
        lettered_cat = None  # category of an open "Category 15 a-" sub-category block
        # Textual definition/risk columns are not part of the metric label.
        label_limit = min((w['x0'] for w in lines[line_index]
                           if w['text'].lower()=='definition' and w['x0']<cut), default=cut+2)
        has_definition_column = label_limit < cut
        # A ruled label column excludes unlabelled baseline/percentage columns
        # from the text, even when only the current year has a known year ID.
        label_edges = [c[2] for t in ruled for rr in t.rows for c in rr.cells if c
                       and c[0] <= label_left+3 and label_left+40<c[2]<cut
                       and t.bbox[1]<header_y+25<t.bbox[3]]
        if label_edges:
            label_limit = min(label_limit, min(label_edges))
        heading = _text([w for w in words if max(0,header_y-80) <= w['top'] < header_y-5 and label_left-1 <= w['x0'] < cut])
        table_boundary = _boundary(heading)
        heading_lines = [line for line in grid_lines if header_y-65 <= min(w['top'] for w in line) < header_y]
        section = next((_field(_text([w for w in line if w['x0']<cut])) for line in reversed(heading_lines)
                        if _field(_text([w for w in line if w['x0']<cut]))), '')
        last_numeric_y = header_y
        pending = []
        body_lines = [line for line in grid_lines if min(w['top'] for w in line) > header_y+3]
        if len(cols)==1:
            body_lines = _physical_rows(body_lines, ruled, header_y, next_y, cut)
        else:
            # a fully ruled grid whose cells are vertically centred on
            # different baselines: amounts and labels of one ruled row sit on
            # separate text lines, so the ruled row bounds group them
            inside = [l for l in body_lines if min(w['top'] for w in l) < next_y]
            num_only = [l for l in inside if _numeric_units(l, cut)
                        and not any(w['x1'] < cut and re.search('[A-Za-z]{2}', w['text']) for w in l)]
            num_all = [l for l in inside if _numeric_units(l, cut)]
            grid_box = [t for t in ruled if len(t.rows) >= 8 and t.bbox[0] < cut - 40
                        and t.bbox[1] <= header_y + 5 and t.bbox[3] > header_y + 100]
            if grid_box and len(num_only) >= 3 and len(num_only) >= .3 * len(num_all):
                grouped = _physical_rows(body_lines, ruled, header_y, next_y, cut, max_h=60)
                # accept the ruled grouping only when no group stacks two
                # amounts in one value column (two physical rows merged)
                def _stacked(g):
                    xs = sorted(u[-1]['x1'] for u in _numeric_units(g, cut))
                    return any(b - a < 8 for a, b in zip(xs, xs[1:]))
                # the signature of a centred-cell grid: one ruled row holds
                # amounts of different columns on different baselines
                def _split_baselines(g):
                    tops = sorted({round(u[0]['top']) for u in _numeric_units(g, cut)})
                    return len(tops) >= 2 and tops[-1] - tops[0] > 4
                if not any(_stacked(g) for g in grouped) and sum(_split_baselines(g) for g in grouped) >= 2:
                    body_lines = grouped
        coverage_context = 'reported'
        # Only a physically merged cell covering two or more labelled numeric
        # children is a descriptor. Wrapped text of an ordinary row is retained.
        descriptors=[]
        for span in descriptor_spans:
            a,b,c,d=span['bbox']
            if not (label_left-2 <= a < c < label_limit-10 and header_y < b < next_y):
                continue
            children=[ls for ls in body_lines if b <= min(w['top'] for w in ls)+3 < d
                      and any(c-1 <= w['x0'] and w['x1'] <= label_limit and re.search('[A-Za-z]',w['text']) for w in ls)
                      and any(parse_number(w['text']) is not None and any(abs(w['x1']-col['anchor'])<15 for col in cols) for w in ls)]
            if len(children)>=2:
                descriptors.append(span)
        data_clusters = _assign_data_anchors(cols, body_lines, cut, next_y, gap)
        if seg_line:
            # each reported year binds to its Group/Total sub-column only;
            # the segment sub-columns are parts of the consolidated amount
            totals = [w for w in seg_line if re.fullmatch(r"(?i)group|total|consolidated|gesamt|groupe|grupo|konzern", w['text'])]
            ordered = sorted(range(len(cols)), key=lambda k: cols[k]['bbox'][0])
            seg_anchor = {}
            for pos, j in enumerate(ordered):
                if cols[j]['role'] != 'reported':
                    continue
                lo_x = cols[ordered[pos-1]]['bbox'][2] if pos else cut
                hi_x = cols[ordered[pos+1]]['bbox'][0] if pos + 1 < len(ordered) else page.width
                # a caption belongs to the year whose header is nearest to it
                centre = lambda b: (b[0] + b[2]) / 2
                multi_line = max(w['top'] for w in seg_line) - min(w['top'] for w in seg_line) > 4
                inside = [w for w in totals if lo_x - 2 <= w['x1'] <= hi_x + 2
                          and (not multi_line or min(range(len(cols)), key=lambda k: abs(centre(cols[k]['bbox']) - (w['x0'] + w['x1']) / 2)) == j)]
                if inside:
                    # the consolidated total is the rightmost captioned sub-column
                    tot_words = [w for w in inside if re.fullmatch(r"(?i)total|gesamt", w['text'])] or inside
                    seg_anchor[j] = max(w['x1'] for w in tot_words)
                else:
                    cols[j]['role'] = 'segment'
            owner_of = {}
            for j, x1 in seg_anchor.items():
                # the amounts under a caption end at or a little right of it
                near = [(abs(hi - x1), k) for k, (lo, hi, _, n) in enumerate(data_clusters or [])
                        if x1 - 6 <= hi <= x1 + 20]
                if near:
                    k = min(near)[1]
                    lo, hi = data_clusters[k][0], data_clusters[k][1]
                    owner_of[k] = j
                else:
                    lo, hi = x1 - 6, x1 + 2
                cols[j].update(data_anchor=hi, data_lo=lo, data_hi=hi, anchor=hi,
                               organization_boundary='consolidated_group', segment_total=True)
            data_clusters = [(lo, hi, owner_of.get(k), n) for k, (lo, hi, _, n) in enumerate(data_clusters or [])]
        body_lines = _merge_offset_cell_lines(body_lines, cut)
        body_lines = _hoist_centred_group_labels(body_lines, cut)
        num_tops = sorted(min(w['top'] for w in l) for l in body_lines if _numeric_units(l, cut))
        pending_words = {}
        grand_total = False
        for line in body_lines:
            y = min(w["top"] for w in line)
            if os.environ.get('S3T_DEBUG_LINES'):
                print('LINE', round(y), round(last_numeric_y), _text(line)[:70])
            if y >= next_y:
                break
            if y < subheader_bottom - 1 and not [w for w in line if w['x1'] < cut - 2 and re.search('[A-Za-z]', w['text'])]:
                continue
            # Stop at a large gap / footnote block; do not extend a category to
            # a narrative 'category 1 ...' hundreds of points below its table.
            if rows and y-last_numeric_y > 100:
                # a long comment column ("Development vs. base year") or rows of
                # percentages keep the grid open: measure the gap from the last
                # line with an amount or percentage under a year column
                anchors_x = [c.get('data_anchor') or c['anchor'] for c in cols]
                def _valued(l):
                    return any(re.fullmatch(r"[-+(]?\d[\d.,]*\)?%?", w['text']) and any(abs(w['x1'] - a) < 12 for a in anchors_x)
                               for w in l)
                last_val = max([min(w['top'] for w in l) for l in body_lines
                                if last_numeric_y - 1 <= min(w['top'] for w in l) < y - 2 and _valued(l)] or [last_numeric_y])
                if y - last_val > 100:
                    break
            # Label words may extend past the nominal cut when the first year
            # column is right-aligned; alphabetic words before the first
            # amount (and before the first header year) still belong to it.
            lim = label_limit
            if not has_definition_column:
                # "(scope 3)": a scope/category number is label text, not an amount
                def _label_number(u):
                    return (len(u) == 1 and re.fullmatch(r"\d{1,2}[,:)]?", u[0]['text']) and any(
                        re.fullmatch(r"(?i:\(?(?:scopes?|category|cat\.?))", x['text']) and 0 <= u[0]['x0'] - x['x1'] < 5
                        for x in line))
                first_amount = min((u[0]['x0'] for u in _numeric_units(line, cut) if not _label_number(u)), default=None)
                edge = years[0]['x0'] - 2 if first_amount is None else min(first_amount - 2, years[0]['x0'] - 2)
                lim = max(label_limit, edge)
            # A status word ("Not") ends a label only when no label word
            # follows it ("... not included in Scopes 1 and 2" keeps "not").
            _later_alpha = [w['x0'] for w in line if label_limit < w['x1'] <= lim
                            and re.search('[A-Za-z]{2}', w['text']) and not UNIT.fullmatch(w['text'])
                            and not _STATUS_WORD.fullmatch(w['text'])]
            left = [w for w in line if w["x1"] <= label_limit
                    or (w["x1"] <= lim and re.search('[A-Za-z]', w['text']) and not UNIT.fullmatch(w['text'])
                        and (not _STATUS_WORD.fullmatch(w['text'])
                             or any(0 < x - w['x1'] < 8 for x in _later_alpha)))
                    # the number of "Scope 2" / "Category 3" belongs to the label
                    or (w["x1"] <= lim and re.fullmatch(r"\d{1,2}[,:)]?", w['text'])
                        and any(re.fullmatch(r"(?i:\(?(?:scopes?|category|cat\.?))", x['text'])
                                and 0 <= w['x0'] - x['x1'] < 5 for x in line))]
            if not has_definition_column and not _numeric_units(line, cut) and left:
                # a heading without amounts may run past the value-column cut
                # ("Scope 2 GHG emissions (located-based)"): words starting
                # before the first value column and continuing the label
                for w in sorted(line, key=lambda w: w['x0']):
                    if (w not in left and w['x0'] < years[0]['x0'] - 2 and re.search('[A-Za-z]{2}', w['text'])
                            and any(0 <= w['x0'] - lw['x1'] < 6 and abs(lw['top'] - w['top']) < 3 for lw in left)):
                        left.append(w)
                left.sort(key=lambda w: w['x0'])
            descriptor=next((p for p in descriptors if p['bbox'][1] <= y+3 < p['bbox'][3]),None)
            if descriptor:
                left=[w for w in left if w['x0'] >= descriptor['bbox'][2]-1]
            right = [w for w in line if w["x1"] >= cut]
            # A second header line ("year | 2024-2025") just below the header
            # carries years, not amounts.
            if y - header_y < 14 and right and all(
                    re.fullmatch(r"(?:FY)?(?:19|20)\d{2}(?:\s*[-/\u2013]\s*(?:19|20)?\d{2})?\*?|[A-Za-z()%/.-]+", w['text'])
                    for w in right) and any(YEAR_RANGE.fullmatch(w['text']) for w in right):
                continue
            slots = _slot_numeric_words(line, cols, cut, gap, page, data_clusters)
            cells = []
            for j, ws in slots.items():
                ws = sorted(ws, key=lambda w: w["x0"])
                raw = " ".join(w["text"] for w in ws)
                if parse_number(raw) is None and len(ws) > 1:
                    # an amount plus a neighbouring placeholder in one slot
                    nums = [w for w in ws if parse_number(w['text']) is not None]
                    if len(nums) == 1:
                        ws, raw = nums, nums[0]['text']
                if parse_number(raw) is not None or re.fullmatch(r"[-–—]|n\.?s\.?|N/A", raw, re.I):
                    cell = {"column_id": cols[j]["id"], "raw": raw, "bbox": _bbox(ws),
                            "source_words": [{k:w[k] for k in ('text','x0','x1','top','bottom')} for w in ws]}
                    if any(w.get('source_token') for w in ws):
                        cell["source_token"] = " ".join(w.get('source_token', w['text']) for w in ws)
                        cell["footnote_marker"] = "".join(w.get('footnote_marker', '') for w in ws)
                    cells.append(cell)
            label = _text(left)
            # A qualitative status in the value columns is a complete source
            # row, not a wrapped label for the next numeric row. In particular,
            # "Included in Scope 1" contains a scope number, not an amount.
            right_status = _text(right)
            status_pattern = (r"(?:Included in Scope\s*[123]|Not material|Not applicable|Not reported|Not relevant|Not available"
                              r"|No data(?: available)?|Not calculated|Not estimated|Not assessed|Not disclosed)")
            status_row = bool(label and (re.fullmatch(r"(?:"+status_pattern+r"\s*)+",right_status,re.I)
                                         or (re.search(status_pattern, right_status, re.I)
                                             and not any(parse_number(c['raw']) is not None for c in cells)
                                             and re.fullmatch(r"(?:(?:"+status_pattern+r"|[-\u2013\u2014]|n/?a)\s*)+", right_status, re.I))))
            if status_row:
                label = _text([w for w in left if w not in right]) or label
            if status_row:
                non_numeric_rows.append({"label":label,"status":right_status,
                                         "bbox":_bbox(line),"label_bbox":_bbox(left),
                                         "status_bbox":_bbox(right),
                                         "parent_category":_category(label) or current_cat})
                if _category(label):
                    current_cat=_category(label)
                pending=[]
                continue
            row_unit = _unit([w for w in line if w['x0'] < cut+12])
            if row_unit is None and cells:
                # a unit-of-measure column after the amounts ("78,715 87,806 tCO2e 12%")
                after = [w for w in line if w['x0'] >= cut+12 and not _NUMERIC_WORD.fullmatch(w['text'])
                         and not re.search(r"\d%$|^%$", w['text'])]
                uom = _unit(after) if after else None
                if uom and not _relative_after(_text(after)[len(uom['source_text']):] if uom.get('source_text') else ''):
                    row_unit = uom
            unit_anchor = row_unit or table_unit
            declared_other_unit = _declared_other_unit(left,unit_anchor,label,row_unit)
            if row_unit:
                table_unit = row_unit
                # A line-wrapped unit immediately below its numeric row.
                if (rows and not rows[-1]['unit'] and not rows[-1].get('declared_noninventory_unit')
                        and y-rows[-1]['bbox'][3] < 12):
                    ub_ = row_unit['bbox']
                    r_top = rows[-1]['bbox'][1]
                    # stacked over three lines: "Thousands" / amounts / "tCO2e"
                    above_ = [w for l in body_lines if 0 < r_top - min(x['top'] for x in l) <= 9
                              and not _numeric_units(l, cut)
                              for w in l if w['x1'] > ub_[0] - 15 and w['x0'] < ub_[2] + 15
                              and re.fullmatch(r"(?i)\(?(?:thousands?|millions?|['\u2019]?000s?|kilo)\)?,?", w['text'])]
                    if above_:
                        u2 = _unit(above_ + [w for w in line if ub_[0] - .5 <= w['x0'] and w['x1'] <= ub_[2] + .5])
                        if u2 and u2.get('multiplier', 1) != row_unit.get('multiplier', 1):
                            row_unit = u2
                            table_unit = u2
                    rows[-1]['unit'] = row_unit
            if row_unit:
                # Units occupy a column or a parenthesis after the label.
                ub = row_unit['bbox']
                label = _text([w for w in left if not (ub[0]-.1 <= (w['x0']+w['x1'])/2 <= ub[2]+.1 and ub[1]-.1 <= w['top'] <= ub[3])]).strip(" ()")
            elif declared_other_unit:
                ub=declared_other_unit['bbox']
                label=_text([w for w in left if not (ub[0]-.1 <= (w['x0']+w['x1'])/2 <= ub[2]+.1 and ub[1]-.1 <= w['top'] <= ub[3])]).strip()
            if re.search(r'scope\s*3.*exclud(?:e|ing|es).*category\s*15', label, re.I):
                coverage_context = 'excluding_category15'
            elif re.search(r'scope\s*3.*financed',label,re.I):
                coverage_context = 'financed_subset'
            if (not cells and not rows and table_unit and table_unit.get('bbox')
                    and abs(table_unit['bbox'][1] - y) < 3 and table_unit['bbox'][0] >= cut - 12):
                # "Source of emission | t CO2e": the column-caption line that
                # carries the table's unit is a header, not a row's first label line
                pending = []
                continue
            if not cells:
                # Numbered category rows left blank keep the table open.
                if (_category(label) and len(label.split()) <= 10
                        and (re.match(r"\s*(?:1[0-5]|[1-9])[.):]?\s+[A-Za-z]", label)
                             # an unnumbered category name in the label column
                             # ("Upstream leased assets", no amount reported)
                             or (rows and any((r.get('field') or '').startswith('scope3_cat') for r in rows[-3:])
                                 and all(w['x1'] < cut for w in line)))):
                    last_numeric_y = y
                if re.search(r'total.*(?:ghg|greenhouse gas|emissions)',label,re.I) and not (
                        re.search(r'\bindirect\b', label, re.I) and section == 'scope3_total'):
                    current_cat, scope_parent, section = None, None, 'excluded_combined'
                if label and not re.fullmatch(r"[\d\s]+", label):
                    if has_definition_column and rows and y-last_numeric_y < 28 and not _category(label) and not re.match(r'\s*(?:Scope\s*\d\s*(?:GHG|emissions)|Total)\b',label,re.I):
                        rows[-1]['label'] += ' '+label
                        continue
                    # The last line of a vertically centred row ("Gross Scope 1 GHG
                    # Thousands" / amounts / "emissions tCO2e") lies just below its
                    # amounts and nearer to them than to the next amounts.
                    nxt = next((t for t in num_tops if t > y + .5), None)
                    gap_prev = y - rows[-1]['bbox'][1] if rows else None
                    if (rows and 0 < gap_prev < 12 and (nxt is None or nxt - y > gap_prev + 1)
                            and rows[-1].get('_band') is not None and not rows[-1].get('_tail')
                            and not rows[-1].get('_own_label')
                            # a row whose label closed its unit parenthesis is complete;
                            # a following basis label starts the next row
                            and not (rows[-1].get('_band') and re.search(r"\([^()]*(?:co2|tons?\b|tonnes?\b)[^()]*\)\s*$",
                                                                        _text(sorted(rows[-1]['_band'], key=lambda w: (round(w['top']), w['x0']))), re.I)
                                     and re.match(r"[A-Z]", label))
                            and not (re.match(r"(?i)\s*(?:location|market)[- ]based", label)
                                     and re.search(r"(?i)(?:location|market)[- ]based", rows[-1]['label']))
                            and left and (rows[-1].get('label_bbox') or rows[-1].get('_band'))
                            and min(w['x0'] for w in left) >= (min(w['x0'] for w in rows[-1]['_band']) if rows[-1].get('_band')
                                                              else rows[-1]['label_bbox'][0]) - 10
                            and (not _category(label) or (re.match(r"\s*[123]\b", label)
                                                          and re.search(r"(?i)\bscopes?\b(?!\s*\d)", rows[-1]['label'])))
                            and (not re.match(r"\s*(?:scopes?\s*[123]\b|total\b|significant\b)", label, re.I)
                                 # "Total Gross location-based" / amounts / "Scope 2 GHG emissions (tCO2eq)"
                                 or (re.match(r"\s*scopes?\s*[123]\b", label, re.I)
                                     and not re.search(r"(?i)\bscopes?\b", rows[-1]['label'])
                                     and re.match(r"(?i)\s*(?:total\s+)?(?:gross\s+)?(?:location|market)", rows[-1]['label'])))):
                        r_ = rows[-1]
                        r_['_tail'] = True
                        band = r_['_band'] + left
                        if row_unit:
                            ub = row_unit['bbox']
                            # the unit words of this line may sit outside its label words
                            band_u = band + [w for w in line if w not in band and w['x1'] <= cut + 12]
                            # a unit stacked over three lines ("Thousands" / amounts /
                            # "tCO2e"): its first word sits just above the row
                            r_top = r_['bbox'][1]
                            band_u += [w for l in body_lines if 0 < r_top - min(x['top'] for x in l) <= 9
                                       and not _numeric_units(l, cut)
                                       for w in l if w not in band_u and w['x1'] > ub[0] - 15 and w['x0'] < ub[2] + 15]
                            colw = sorted([w for w in band_u if w['x1'] > ub[0]-15 and w['x0'] < ub[2]+15
                                           and re.fullmatch(r"(?i)\(?(?:in|of|thousands?|millions?|['\u2019]?000s?|t|tco2?e?q?|co2?e?q?|eq|e|tons?|tonnes?|kt|mt|metric)\)?[,)]*", w['text'])],
                                          key=lambda w: (w['top'], w['x0']))
                            u = _unit(colw) if colw else None
                            u = u or row_unit
                            r_['unit'] = u
                            table_unit = u
                            r_['label'] = _text([w for w in r_['_band'] if w not in colw]).strip(" ()") + ' ' + label
                        else:
                            r_['label'] = (r_['label'] + ' ' + label).strip()
                            # "Scope 1 emissions (in tons" / amounts / "CO2e)"
                            if not r_['unit']:
                                u = _unit(sorted(band, key=lambda w: (round(w['top']), w['x0'])))
                                if u:
                                    r_['unit'] = u
                                    table_unit = table_unit or u
                        r_['label'] = re.sub(r"\s+", " ", r_['label']).strip()
                        if (not r_['field'] or r_['field'] == 'scope2') and not r_.get('grand_total'):
                            nf = _field(r_['label'], section)
                            if nf:
                                r_['field'] = nf
                                r_['quantity_type'] = 'inventory'
                        elif (r_['field'] and re.match(r"\s*[a-z(]", label) and re.search(r"(?i)biogenic|intensity|\bper\s|percent|%", label)
                              and _field(r_['label'], section) is None):
                            # "Total scope 3 GHG emissions including" / "biogenic CO2":
                            # the wrapped words make the row a non-inventory quantity
                            r_.update(field=None, parent_category=None, quantity_type='excluded_noninventory',
                                      quantity_exclusion='noninventory or relative metric')
                        continue
                    pending.append((label, y))
                    pending_words[y] = left
                    # "scope 1 or 2)": the wrapped tail of an open parenthesis
                    tail = bool(re.match(r"[a-z][^()]*\)", label))
                    sf = None if tail else _field(label)
                    # a wrapped category label line ("Category 7. Employee")
                    # stays inside the Scope 3 section opened by its heading
                    if sf and not (sf.startswith('scope3_cat') and (section or '').startswith('scope3')):
                        section = sf
                    if not tail and (re.search(r"\bScope\s*[123]\b", label, re.I) or (sf and not sf.startswith('scope3_cat'))):
                        current_cat = None
                        scope_parent = sf if sf in ('scope1','scope2','scope2_location','scope2_market') else None
                        heading_no += 1
                        if not sf:
                            section = "excluded_combined_or_relative_scope"
                continue
            if os.environ.get('S3T_DEBUG_ROWS'):
                print('ROW?', round(y), len(label.split()), repr(label[:80]), [c['raw'] for c in cells])
            if len(label.split()) > 35:
                if rows:
                    break
                continue
            # A footnote sentence quoting an amount ("The figures for 2024 have
            # been adjusted for Scope 3, Category 2 (2024: 35.6 tCO2e) ...") is
            # prose under the table, not a row.
            if (len(label.split()) >= 12 and len([c for c in cells if parse_number(c['raw']) is not None]) <= 1
                    and re.search(r"(?i)\b(?:have|has|had)\s+(?:not\s+)?been\b|\b(?:were|was|will be|is no longer)\b", label)):
                continue
            # A wrapped numeric row may have its first label line above it.
            # Contiguous label lines directly above (each within 15pt of the
            # next) are the row's wrapped label, up to three lines.
            if cells and not re.search('[A-Za-z]', label or '') and label_left:
                # "TOTAL SCOPE 1 | 510 681" printed in the group column left of
                # the row labels: that caption labels the total row, and label
                # lines above belong to the previous (non-numeric) row
                side = [w for w in words if abs(w['top'] - y) < 3 and w['x1'] < label_left - 2]
                if side and re.match(r"(?i)\s*(?:sub-?)?total\b", _text(side)):
                    left = side + left
                    label = _text(side)
                    pending = []
                    # a bare TOTAL after the per-scope totals sums all scopes
                    grand_total = bool(re.fullmatch(r"(?i)\s*total\s*", label)
                                       and any(re.match(r"(?i)\s*total\s+scope", r['label']) for r in rows))
            prior, last_y, prior_ys = [], y, []
            for s_, py in reversed(pending):
                is_heading = re.match(r"\s*(?:significant\s+)?scopes?\s*[123]\s+(?:ghg\s+|greenhouse gas\s+)?emissions\b(?:\s*\([^)]*\))?\s*$|.*:\s*$", s_, re.I)
                # "1. Purchased goods and services:" / its description / amounts:
                # a numbered category name ending in a colon opens the row's label
                if (is_heading and re.match(r"\s*(?:1[0-5]|[1-9])[.)]\s+[A-Z]", s_) and _category(s_)
                        and not _category(label) and last_y - py < 15 and len(prior) < 3
                        and not any(_category(q) for q in prior)):
                    prior.insert(0, s_)
                    prior_ys.insert(0, py)
                    break
                if is_heading and (prior or re.search(r"[A-Za-z]{3,}", label or "")):
                    break  # a section heading above a labelled row is not part of its label
                # a line naming another category starts another row's label
                if prior and _explicit_category_marker(s_) and any(_explicit_category_marker(q) for q in prior):
                    break
                if last_y - py < 15 and len(prior) < 3:
                    prior.insert(0, s_)
                    prior_ys.insert(0, py)
                    last_y = py
                    # a line opening with a row-start word is the label's first line
                    if re.match(r"\s*(?:(?:significant\s+)?scopes?\s*\d|total\b|gross\b|category\b|cat\.\s*\d)", s_, re.I):
                        break
                else:
                    break
            if _category(label):
                # Numbered empty categories are complete rows, not prefixes of
                # the next numbered category (e.g. Cat9 NA then Cat10 3,761).
                prior = [s for s in prior if not _category(s) or _category(s)==_category(label)]
            elif prior and _category(prior[-1]) and re.search(r"[-\u2013\u2014:]\s*$", prior[-1]):
                # "Category 4 -" / "Upstream transportation ...": a dangling
                # category marker is the first line of this row's label.
                prior = prior[-1:]
            # A label-only line directly above that refines the row's field
            # ("Gross location-based" / "Scope 2 emissions") is its first line.
            refines = (prior and label and _field(label, section) and not any(_field(s) for s in prior)
                       and (_field(" ".join(prior + [label]), section) or "").startswith(_field(label, section))
                       and _field(" ".join(prior + [label]), section) != _field(label, section))
            # A bare "Scope N" group label always heads its sub-rows.
            scope_group = prior and re.fullmatch(r"\s*scopes?\s*[123]\s*", " ".join(prior), re.I)
            # "(location-based)" / "based)": a fragment continuing the line above
            fragment = bool(re.match(r"\s*[(a-z]", label))
            if prior and (not _field(label, section) or not label or refines or scope_group or fragment or
                          any(_category(s) for s in prior) or label.startswith('(not included')):
                label = " ".join(prior + [label]).strip()
            else:
                prior_ys = []
            pending = []
            if not label or not any(c.isalpha() for c in label):
                continue
            last_numeric_y = y
            # Footnote digits glued to a label word ("Scope 1" + superscript
            # "21" -> "Scope 121") are separated only on glyph evidence; the
            # printed label is kept for source matching.
            footnoted = {w['text']: sw['text'] for w in left if w['text'].isdigit()
                         for sw in [_strip_superscript(w, page)] if sw is not w}
            field_label = re.sub(r'\b\d+\b', lambda m: footnoted.get(m.group(0), m.group(0)), label) if footnoted else label
            # "scope 1 or 2) ...": the wrapped tail of the previous row's
            # "(not included in scope 1 or 2)" names no scope of this row
            if re.match(r"[a-z][^()]*\)", field_label):
                field_label = re.sub(r"^[^()]*\)\s*", "", field_label)
            field = _field(field_label, section)
            if re.fullmatch(r'Total',label,re.I) and section=='scope3_total':
                field='scope3_total'
            is_grand = grand_total
            if grand_total:
                field, grand_total = None, False
            cat = None if field in ('scope1','scope2','scope2_location','scope2_market') else _category(field_label)
            parent_scope, scope_aggregate = None, False
            via_heading = None
            if not field and re.search(r'total.*(?:ghg|emissions)|number of employees',label,re.I):
                scope_parent = None
            if field in ('scope1','scope2','scope2_location','scope2_market','scope3_total') and (re.search(r'\bscopes?\s*[123]\b',field_label,re.I) or label.lower()=='total' and field==section):
                scope_parent = field
                parent_scope, scope_aggregate = field, True
            elif re.search(r'\bscopes?\s*[123]\b',field_label,re.I) and not cat:
                scope_parent = None
            elif (scope_parent and scope_parent.startswith('scope2')
                  and field in ('scope2_location','scope2_market')
                  and (re.fullmatch(r'(?:total\s+)?(?:gross\s+)?(?:location|market)[ -]based(?:\s+scope\s*2)?(?:\s+(?:ghg\s+)?emissions)?(?:\s*\([^)]*\))?',
                                    _SPLIT_WORD.sub(lambda m: m.group(1) + m.group(2), label),re.I)
                       # the other accounting basis is a sibling view, not a child
                       or (scope_parent in ('scope2_location','scope2_market') and field != scope_parent))):
                # A bare alternative accounting basis is another full Scope2
                # view. Qualified electricity/heating rows remain children.
                scope_parent = field
                parent_scope, scope_aggregate = field, True
            elif (scope_parent and not cat and not current_cat and field is None
                  and re.fullmatch(r"(?i)(?:total\s+)?gross\s+(?:ghg\s+|greenhouse gas\s+)?emissions\b.*", field_label)
                  and not re.search(r"%|percent|share|from regu", field_label, re.I)):
                # "Scope 1" heading / "Gross emissions (thousands of tons)": the scope total
                field = scope_parent
                parent_scope, scope_aggregate = field, True
            elif (scope_parent and not cat and not current_cat and (field is None or field.startswith(scope_parent[:6]))
                  and not (field is None and re.search(r"%|percent|share of|from regu", field_label, re.I))):
                field = field or scope_parent
                parent_scope = scope_parent
                via_heading = heading_no
            parent = next((p for p in parent_spans if p["bbox"][1] <= y+3 < p["bbox"][3]
                           and p["bbox"][0] < cut and p["bbox"][2] < cut), None)
            if parent:
                # the ruled cell holding this row's own category label is not a
                # parent: a parent leaves child label words to its right
                ub_ = (row_unit or {}).get('bbox')
                rest = [w for w in left if w['x0'] >= parent['bbox'][2]-1 and re.search('[A-Za-z]{2}', w['text'])
                        and not UNIT.fullmatch(w['text'])
                        and not (ub_ and ub_[0]-.1 <= (w['x0']+w['x1'])/2 <= ub_[2]+.1)]
                if not rest:
                    parent = None
            if parent:
                cat = parent["category"]
                # Remove repeated parent text; the span, not text order, binds
                # all sibling rows including those above the centred heading.
                child_words = [w for w in left if w['x0'] >= parent['bbox'][2]-1]
                label = _text(child_words).strip() or label
            row_x0 = min((w['x0'] for w in left), default=None)
            if field and field.startswith("scope3_cat"):
                current_cat = int(field[10:])
                cat_row_x0 = row_x0 if not parent else None
            elif field and field.startswith("scope"):
                section, current_cat = field, None
            if re.search(r"total\s*\(?\s*(?:corporate|greenhouse|ghg|emissions|location|market)|scope\s*[123]|intensity|biogenic|employees|per employee|reduction goal", label, re.I) and not field:
                current_cat = None
            # a flat list of category rows ("Fuel- and energy-related
            # activities" / "Transportation and distribution"): an unmapped
            # row level with the previous category row is its sibling, not a part
            flat_sibling = (cat is None and current_cat and not field and not parent and row_x0 is not None
                            and cat_row_x0 is not None and abs(row_x0 - cat_row_x0) < 2 and rows
                            and rows[-1].get('field') == f"scope3_cat{current_cat}" and rows[-1]['value_kind'] == 'category_total'
                            and any(parse_number(c['raw']) is not None for c in rows[-1]['cells'])
                            and re.search(r"(?i)transport|distribution", label))
            if cat is None and current_cat and not field and not flat_sibling:
                cat = current_cat
            if cat:
                field = f"scope3_cat{cat}"
                parent_scope = None
            quantity_exclusion = None
            if declared_other_unit or _noninventory_quantity(label) or re.search(r"efficiency|intensity|\bper\s*cent\b|\bpercent\b|\bin\s*%|\bspecific\b|\bkg\s*/\s*t|per employee|per net revenue|/\s*[€$£]?\s*(?:mil|mn|m|million)\b\.?\s*(?:net|of|revenue|sales|turnover|profit)|reduction goal|percentage|biogenic|water consumption|total waste|methodologies|emission factors|\bin\s+(?:million\s+)?(?:m3|m³|tonnes)(?!\s+(?:of\s+)?CO)", label, re.I):
                quantity_exclusion = (f"declared {declared_other_unit['dimension']} unit {declared_other_unit['text']} in the physical source unit column" if declared_other_unit else
                                      'independent erased/avoided-emissions indicator' if _noninventory_quantity(label) else
                                      'noninventory or relative metric')
                field, cat, current_cat = None, None, None
                parent_scope, scope_parent = None, None
            elif re.search(r"without saf|no SAF", label, re.I):
                field, cat = None, None  # alternate scenario does not end the parent block
                quantity_exclusion = 'alternate scenario, not the reported inventory view'
            kind = "category_total"
            coverage = ('excluded_from_SBTi' if re.search(r'excluded.*SBTi',label,re.I) else
                        'SBTi' if re.search(r'\bSBTi\b',label,re.I) else
                        # "... scope 3 GHG emissions covered by targets/data"
                        'covered_subset' if re.search(r'\bcovered by\b|\bin scope of\b|\(t\.\s?r\.\)|target[- ]relevant',label,re.I) else
                        coverage_context if field=='scope3_total' else 'reported')
            if field=='scope3_total' and coverage!='reported':
                scope_aggregate=False
                parent_scope='scope3_total'
            corporate_only = False
            if field=='scope3_total' and re.search(r'corporate|financed|excluding',label,re.I):
                coverage = coverage_context if coverage_context!='reported' else 'restricted_subset'
                scope_aggregate=False;parent_scope='scope3_total'
                corporate_only = not re.search(r'financed|excluding',label,re.I)
            # "Scope 3 - Upstream", "Scope 2 GHG emissions - Transport - location
            # based", "... Scope 2 Market-based - France": a breakdown of the
            # scope total by activity, segment or country, not the total.
            if field in ('scope1','scope2','scope2_location','scope2_market','scope3_total') and (
                    re.search(r"\bscope\s*[123]\b(?:\s+(?:ghg\s+)?emissions)?\s*[-\u2013\u2014]\s*"
                              r"(?!(?:location|market)[ -]*\s*based\b|total\b|gross\b|net\b|emissions?\b|\(?t\s?co|" + _WHOLE_SCOPE + r")[A-Za-z]", field_label, re.I)
                    or re.search(r"\b(?i:scope)\s*[123]\b.*\s[-\u2013\u2014]\s+(?!(?i:(?:location|market)[ -]*\s*based\b|total\b|gross\b|net\b|emissions?\b|" + _WHOLE_SCOPE + r"))[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s*$", field_label)):
                kind = "component"
                parent_scope, scope_aggregate = field, False
            # "1A. Purchased goods, product related" / "1B. ...": lettered
            # subdivisions of a category are its parts, not its total
            # "1A. Purchased goods, product related" / "1B. ...": lettered
            # subdivisions are parts of the category. For Category 15 the
            # investees' Scope 1 and 2 ("15a ... financed scope 1 and 2") is the
            # category value itself (GHG Protocol / PCAF core boundary).
            # "1A. ...", "15a ...", "Category 15 a- ...": a letter attached to
            # the number, or a spaced letter closed by punctuation ("4 U pstream"
            # is a split word, not a part letter)
            lettered = bool(cat and re.match(r"\s*(?:(?:category|cat\.?)\s*)?(?:1[0-5]|[1-9])(?:[A-Za-z](?:[.):\-\u2013]|\s)|\s[A-Za-z][.):\-\u2013])", label, re.I))
            core15 = False
            if lettered:
                # project convention: Category 15 financed emissions are always
                # recorded as the category value
                core15 = bool(cat == 15 and (re.search(r"(?i)scope\s*1\s*(?:and|&|\+|,)\s*2", label)
                                             or (re.search(r"(?i)financed", label)
                                                 and not re.search(r"(?i)scope\s*3", label))))
                kind = "category_total" if core15 else "component"
                lettered_cat = None if core15 else cat
            elif cat and not parent and re.match(r"\s*(?:(?:category|cat\.?)\s*)?(?:1[0-5]|[1-9])\b", label, re.I):
                lettered_cat = None  # a new numbered category closes the block
            in_lettered = bool(cat and lettered_cat == cat and not core15)
            if cat and (parent or not _category(label)):
                kind = "component"
            if parent_scope and not scope_aggregate:
                kind = 'component'
            # "(thou-" / "sands of tons of CO2e)": a scale word on the label
            # line above completes the unit printed on the amount line
            if row_unit and prior_ys:
                band_words = sorted([w for py in prior_ys for w in pending_words.get(py, [])] + left,
                                    key=lambda w: (round(w['top']), w['x0']))
                u2 = _unit(band_words)
                if u2 and u2['multiplier'] != row_unit['multiplier'] and re.search(r"(?i)thou|mill|000", u2['text']):
                    row_unit = u2
            if row_unit:
                table_unit = row_unit
            unit = None if declared_other_unit else row_unit or table_unit
            if is_grand:
                field, cat, parent_scope, scope_aggregate, kind = None, None, None, False, 'category_total'
            if is_grand:
                field, cat, parent_scope, scope_aggregate, kind = None, None, None, False, 'category_total'
            rid = f"r{len(rows)+1}"
            for c in cells:
                c["id"] = rid+":"+c["column_id"]
            rows.append({"id": rid, "label": label, "bbox": _bbox(line), "field": field,
                         "label_bbox":_bbox(left) if left else None,
                         "label_parent":descriptor,
                         "parent_category": cat, "parent_label": parent["label"] if parent else None,
                         "parent_bbox": parent["bbox"] if parent else None,
                         "parent_scope":parent_scope, "scope_aggregate":scope_aggregate,
                         "value_kind": kind, "unit": unit, "cells": cells,
                         "quantity_type":'excluded_noninventory' if quantity_exclusion else 'inventory' if field else 'unclassified',
                         "quantity_exclusion":quantity_exclusion,"declared_noninventory_unit":declared_other_unit,
                         "coverage_boundary":coverage,
                         "organization_boundary":_boundary(label) if _boundary(label)!='reported' else table_boundary,
                         "_band": [w for py in prior_ys for w in pending_words.get(py, [])] + left,
                         "grand_total": is_grand,
                         "_corporate_only": corporate_only,
                         "_heading": via_heading if kind == 'component' else None,
                         "_lettered": in_lettered,
                         "_own_text": _text(left),
                         # label words printed on the amount line itself (bottom-aligned labels)
                         "_own_label": any(re.search(r"[A-Za-z]{3,}", w['text']) and not UNIT.match(w['text'])
                                           and not re.fullmatch(r"(?i)\(?(?:in|of|t|tco2?e?q?|co2?e?q?\)?|e\)|eq\)?|tons?|tonnes?)\)?", w['text'])
                                           for w in left) and bool(prior_ys)})
        _apply_left_group_labels(rows, words, label_left, header_y, next_y)
        _demote_heading_merged_children(rows)
        # "Scope 1" heading over a single "Direct emissions" row (or "Scope 2"
        # over one location- and one market-based row): with no other row of
        # that scope in its block, the row is the scope value, not a part.
        blocks = defaultdict(list)
        for r in rows:
            if r.get('_heading') is not None:
                blocks[r['_heading']].append(r)
        for hn, members in blocks.items():
            fields = [r['field'] for r in members]
            if len(set(fields)) != len(fields) or not all(f in ('scope1','scope2_location','scope2_market') for f in fields) \
                    or any(re.search(r"(?i)of which|biogenic|%|\bper\b|excl|outside", r['label'] or '') for r in members):
                continue
            if any(r['field'] in fields and r.get('scope_aggregate') for r in rows if r.get('_heading') != hn):
                continue
            # the block ends at the next row of another scope or a category
            for r in members:
                r.update(value_kind='category_total', scope_aggregate=True, parent_scope=r['field'])
        # "Scope 3 Corporate GHG emissions" is the company's whole Scope 3
        # unless the table also reports financed / portfolio emissions
        if not any(re.search(r"(?i)financed|portfolio|investee", r['label'] or '') for r in rows):
            for r in rows:
                if r.get('_corporate_only'):
                    r.update(coverage_boundary='reported', scope_aggregate=True, value_kind='category_total')
        # dual Scope 2 reporting has two methods: beside a market-based row,
        # the one other "...-based" Scope 2 total (a misprinted "lease-based")
        # is the location-based figure
        if any(r['field'] == 'scope2_market' for r in rows) and not any(r['field'] == 'scope2_location' for r in rows):
            odd = [r for r in rows if r['field'] == 'scope2'
                   and re.search(r"(?i)\b(?!market\b)[a-z]{3,}[- ]based\b", r['label'] or '')]
            if len(odd) == 1:
                odd[0].update(field='scope2_location', basis_inferred='only other Scope 2 method beside market-based')
        _infer_categories_by_order(rows)
        _apply_entity_column(rows, words, label_left, header_y, next_y)
        _apply_summation_aggregates(rows, cols)
        # "Scope 2 - Energy indirect emissions 2,811" followed by its
        # location- and market-based variants: when one variant repeats the
        # parent's amounts, the two variants are the Scope 2 totals per basis.
        for pi, prow in enumerate(rows):
            if prow['field'] != 'scope2' or prow['value_kind'] != 'category_total':
                continue
            kids = []
            for r in rows[pi+1:]:
                if r['field'] in ('scope2_location', 'scope2_market') and r.get('parent_scope') in ('scope2', None):
                    kids.append(r)
                elif r['field'] and r['field'] != 'scope2':
                    break
            by = {r['field']: r for r in kids}
            if len(kids) == 2 and len(by) == 2:
                pc = {c['column_id']: c['raw'] for c in prow['cells'] if parse_number(c['raw']) is not None}
                rep_ids = {c['id'] for c in cols if c.get('role') == 'reported'}
                # a "Scope 2 GHG emissions" heading carrying only target amounts
                # has no reported Scope 2 value of its own
                no_reported = not any(cid in rep_ids for cid in pc)
                if no_reported or any(pc and pc == {c['column_id']: c['raw'] for c in k['cells'] if parse_number(c['raw']) is not None}
                       for k in kids):
                    for k in kids:
                        k['value_kind'], k['scope_aggregate'], k['parent_scope'] = 'category_total', True, k['field']
        # A unit printed further down the grid (in a later row or a unit
        # column) is the table unit for field rows printed before it, when the
        # grid states exactly one absolute emissions unit.
        abs_units = [r['unit'] for r in rows if r['unit'] and r['unit'].get('multiplier')]
        if abs_units and len({u['multiplier'] for u in abs_units}) == 1:
            for r in rows:
                if r['field'] and not r['unit'] and not r.get('declared_noninventory_unit'):
                    r['unit'] = abs_units[-1]
        # One fully specified row (field, absolute unit, two or more reported
        # year cells aligned under the header) is a table in its own right.
        reported_ids = {c['id'] for c in cols if c['role'] == 'reported'}
        # "U.o.M." captioned but left blank for this row: the page's
        # repeated sibling grids state the unit (resolved per page below)
        uom_caption = bool(re.search(r"(?i)\bU\.?\s?o\.?\s?\.?\s?M\.?(?!\w)", _text(hdr_line)))
        single_ok = (len(rows) == 1 and rows[0]['field'] and (rows[0]['unit'] or uom_caption) and
                     sum(c['column_id'] in reported_ids and parse_number(c['raw']) is not None
                         for c in rows[0]['cells']) >= 2)
        if os.environ.get('S3T_DEBUG_GRID'):
            print('GRID', header_y, 'cut', round(cut), [(c['year'], c['role'], round(c['bbox'][0])) for c in cols],
                  'unit', repr(table_unit)[:160])
            for r in rows:
                print('   ', r['label'][:60], '|', r['field'], (r['unit'] or {}).get('text'), [c['raw'] for c in r['cells']])
        # A grid of field rows without any printed unit is kept (a table
        # continued from the previous page); its rows bind only once a unit is
        # established from the continued table.
        unitless_ok = (len(rows) >= 2 and sum(1 for r in rows if r['field']) >= 2
                       and not any(r['unit'] for r in rows) and not any(r.get('declared_noninventory_unit') for r in rows)
                       and not any(_ENERGY_ROW.search(r['label']) for r in rows)) or (
                           uom_caption and len(rows) == 1 and single_ok and not rows[0]['unit']
                           and not _ENERGY_ROW.search(rows[0]['label']))
        if (len(rows) < 2 and not single_ok) or (not any(r['unit'] and r['field'] for r in rows) and not unitless_ok):
            issues.append({"header_y": header_y, "reason": "fewer than two aligned numeric rows" if len(rows)<2 else 'no recognized field with absolute emissions unit'})
            continue
        # A sole disclosed row under a merged parent covers its stated category.
        groups = defaultdict(list)
        for row in rows:
            if row["parent_category"]:
                groups[row["parent_category"]].append(row)
        for group in groups.values():
            if len(group) == 1:
                group[0]["value_kind"] = "category_total"
            else:
                # Repeated explicit category labels with distinct qualifiers
                # are children, unless they differ only by accounting basis.
                explicit = [r for r in group if _explicit_category_marker(r['label'])]
                bases = {re.sub(r'(?:location|market)[ -]based','',r['label'].lower()).strip() for r in explicit}
                dotted = [r for r in explicit if re.match(r"\s*(?:1[0-5]|[1-9])\.\d", r['label'])]
                plain = [r for r in explicit if r not in dotted]
                if len(explicit) > 1 and dotted and len(plain) == 1 and not plain[0].get('parent_label'):
                    # "6 Business travel" over "6.1 Air Travel": the dotted
                    # sub-number is a part of the numbered category row
                    for r in group:
                        r['value_kind'] = 'category_total' if r is plain[0] else 'component'
                elif len(explicit)>1 and len(bases)>1:
                    for r in explicit:
                        r['value_kind']='component'
                elif len(explicit) == 1 and not explicit[0].get('parent_label') and not explicit[0].get('heading_merged'):
                    # one numbered category row ("7 Employee commuting") is the
                    # category; unnumbered rows mapped to it by wording are parts
                    for r in group:
                        r['value_kind'] = 'category_total' if r is explicit[0] else 'component'
                totals = [r for r in group if re.match(r'\s*Total\b',r['label'],re.I)]
                if totals:
                    for r in group:
                        r['value_kind']='category_total' if r in totals else 'component'
        # "Business travel: Air / Train / Car / Taxi / Bus / Total": a bare
        # Total closing a run of rows of one category is that category's
        # total, not the Scope 3 total; a category split into several such
        # blocks (two "Franchise hotels" sub-tables) has only sub-totals
        block_totals = defaultdict(list)
        for k, t_ in enumerate(rows):
            if not re.fullmatch(r"(?i)\s*(?:sub-?\s?)?total\s*", t_['label'] or '') or t_.get('field') not in (None, 'scope3_total'):
                continue
            run = []
            for q in reversed(rows[:k]):
                if re.fullmatch(r"(?i)\s*(?:sub-?\s?)?total\s*", q['label'] or '') or not (q.get('field') or '').startswith('scope3_cat'):
                    break
                run.append(q)
            cats_ = {q['field'] for q in run}
            if len(run) >= 2 and len(cats_) == 1:
                f_ = cats_.pop()
                t_.update(field=f_, parent_category=int(f_[10:]), value_kind='category_total', scope_aggregate=False,
                          parent_scope=None, quantity_type='inventory', category_block_total=True)
                for q in run:
                    q['value_kind'] = 'component'
                block_totals[f_].append(t_)
        for f_, ts_ in block_totals.items():
            if len(ts_) > 1:
                for t_ in ts_:
                    t_['value_kind'] = 'component'
        for r in rows:
            # rows of a lettered sub-category ("Category 15 a- Financed
            # emissions" / its asset-class totals) are parts of the category
            if r.get('_lettered') and (r.get('field') or '').startswith('scope3_cat'):
                r['value_kind'] = 'component'
        tid = f"p{page_no}:t{table_offset+len(tables)+1}"
        units, parents = {}, []
        for r in rows:
            if r['unit']:
                key = json.dumps(r['unit'],sort_keys=True)
                if key not in units:
                    units[key] = dict(_source_unit(r['unit'],page),id=f'{tid}:u{len(units)+1}')
                r['unit'] = units[key]
        for cat, group in groups.items():
            pid = f'{tid}:category{cat}'
            node = {'id':pid, 'category':cat,
                    'label':next((r['parent_label'] for r in group if r['parent_label']),group[0]['label']),
                    'bbox':next((r['parent_bbox'] for r in group if r['parent_bbox']),group[0]['bbox']),
                    'row_ids':[r['id'] for r in group],
                    'numeric_row_count':sum(any(parse_number(c['raw']) is not None for c in r['cells']) for r in group),
                    'explicit_aggregate_rows':[r['id'] for r in group if r['value_kind']=='category_total' and not r['parent_bbox']]}
            parents.append(node)
            for r in group:
                r['parent_id']=pid
                r['parent_observation']=node
        scope_groups = defaultdict(list)
        for r in rows:
            if r.get('parent_scope'):
                scope_groups[r['parent_scope']].append(r)
        for scope, group in scope_groups.items():
            aggregate = next((r for r in group if r['scope_aggregate']),None)
            anchor = aggregate or group[0]
            node={'id':f'{tid}:{scope}','scope':scope,'label':anchor['label'],
                  'bbox':anchor['bbox'],'row_ids':[r['id'] for r in group],
                  'numeric_row_count':sum(any(parse_number(c['raw']) is not None for c in r['cells']) for r in group),
                  'explicit_aggregate_rows':[aggregate['id']] if aggregate else []}
            parents.append(node)
            for r in group:
                r['parent_id'],r['parent_observation']=node['id'],node
        for r in rows:
            r.pop('_band', None)
            r.pop('_tail', None)
            r.pop('_own_label', None)
            r.pop('_own_text', None)
            r.pop('_corporate_only', None)
            r.pop('_heading', None)
            r.pop('_lettered', None)
        tables.append({"id": tid, "document": doc_id, "page": page_no, "method": VERSION,
                       "bbox": [min(r["bbox"][0] for r in rows), header_y,
                                max(r["bbox"][2] for r in rows), max(r["bbox"][3] for r in rows)],
                       "columns": cols, "rows": rows, "units":list(units.values()), "parents":parents,
                       "non_numeric_rows":non_numeric_rows,
                       "unit_column_caption":uom_caption,
                       "organization_boundary":table_boundary,"source_heading":heading,
                       "quality": {"rows": len(rows), "labelled_rows": sum(bool(r['label']) for r in rows),
                                   "unit_bound_rows": sum(bool(r['unit']) for r in rows),
                                   "unit_glyph_verified_rows": sum(bool(r['unit']) and r['unit'].get('evidence_status')=='verified' for r in rows),
                                   "field_bound_rows": sum(bool(r['field']) for r in rows)}})
    # A header split over two baselines is detected twice; the second grid
    # repeats the same physical cells and would make every cell ambiguous.
    # Grids over the same rows but different printed cells (one header line
    # per column group) are complementary and both kept.
    def strength(t):
        # the grid holding more of the printed cells first: a header line
        # that misses a column yields a subset grid, which is then dropped
        return (sum(1 for r in t['rows'] if r['field'] and r['cells']),
                sum(1 for r in t['rows'] for c in r['cells'] if parse_number(c['raw']) is not None),
                sum(1 for c in t['columns'] if c['role'] == 'reported'), -t['bbox'][1])
    kept = []
    for t in sorted(tables, key=strength, reverse=True):
        sig = {tuple(round(v, 1) for v in c['bbox']) for r in t['rows'] for c in r['cells']
               if parse_number(c['raw']) is not None}
        if sig and any(len(sig & k['_sig']) >= .8*len(sig) for k in kept):
            issues.append({"header_y": t['bbox'][1], "reason": "duplicate grid of another header line"})
            continue
        t['_sig'] = sig
        kept.append(t)
    for t in kept:
        t.pop('_sig', None)
    kept.sort(key=lambda t: tables.index(t))
    return kept, issues


def format_structures(tables):
    chunks = []
    for table in tables:
        chunks.append(f"[STRUCTURED_TABLE {table['id']}] coordinate-derived; conflicts require validation")
        chunks.append("COLUMNS " + " | ".join(f"{c['id']}={c['year']} ({c['role']}; {c['boundary']}; organization={c.get('organization_boundary','reported')}; perimeter={c.get('reporting_perimeter','unspecified')})" for c in table['columns']))
        for r in table['rows']:
            chunks.append(f"{r['id']} label={r['label']} ; field={r['field']} ; parent={r['parent_category']} ; label_parent={r.get('label_parent')} ; kind={r['value_kind']} ; quantity_type={r.get('quantity_type','inventory' if r['field'] else 'unclassified')} ; exclusion={r.get('quantity_exclusion')} ; organization={r.get('organization_boundary','reported')} ; coverage={r.get('coverage_boundary','reported')} ; unit={(r['unit'] or {}).get('text','UNKNOWN')} ; " +
                          " | ".join(f"{c['column_id']}={c['raw']}" for c in r['cells']))
            if r.get('basis_link'):
                chunks.append('SOURCE_BASIS_LINK '+json.dumps(r['basis_link'],ensure_ascii=False))
        for r in table.get('non_numeric_rows',[]):
            chunks.append(f"NON_NUMERIC_ROW label={r['label']} ; parent={r['parent_category']} ; status={r['status']} ; no numeric amount disclosed")
        chunks.append("[/STRUCTURED_TABLE]")
    return "\n".join(chunks)


def _digest(table):
    return hashlib.sha256(json.dumps(table, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _raw_units(text):
    units = []
    for m in UNIT.finditer(text):
        if _relative_after(text[m.end():]):
            continue
        raw = re.sub(r'CO\s*(eq\b|e\b)',r'CO2\1',m.group(0))
        factor = normalize_emission_to_tco2e(1,raw)[0]
        if factor is not None:
            units.append({'text':m.group(0),'multiplier':factor})
    return units


def _starts_raw_cells(suffix):
    """A qualified sibling is not an exact label; later target prose is valid.

    Only the beginning determines where the numeric cells start. A row may
    carry percentages, N/A cells or target text after its reported values.
    """
    suffix = suffix.strip(' ()[],:;')
    unit_match = UNIT.match(suffix)
    if unit_match:
        suffix = suffix[unit_match.end():].strip(' ()[],:;')
    # A footnote followed by a qualifier, e.g. "2 (SBTi)", remains part
    # of another row's label; it is not the first emissions cell.
    if re.match(r'^\d+\s*\([A-Za-z]', suffix):
        return False
    return bool(re.match(r'(?:[-−–—](?=\s|$)|[+−-]?\d[\d.,]*[*†‡§✓✔]*(?:(?<=[*†‡§])\d{1,2}(?:,\d{1,2})*)?(?=\s|$))', suffix))



def _label_regex(label):
    """Linear-text pattern of a row label. A footnote number glued to a word
    in the geometry ("Investments2") may be spaced in the text layer
    ("Investments 2"), and vice versa."""
    parts = []
    for x in (label or '').split():
        m = re.fullmatch(r"(.*[A-Za-z)])(\d{1,2})", x)
        parts.append(re.escape(m.group(1)) + r"\s?" + re.escape(m.group(2)) if m else re.escape(x))
    return r"\s+".join(parts)

def _physical_raw_row(row, tables, raw_text):
    """Align equal labels by physical order, never by the desired amount.

    Merged parent labels can occur after their first child in linear text.
    Repeated child labels can also belong to distinct Scope/total blocks.
    Require the same number of physical rows and exact numeric-row label
    occurrences; otherwise leave the existing conservative checks in charge.
    The selected line still has to contain the complete source cell token.
    """
    label = ' '.join(row['label'].split()).lower()
    if not row.get('bbox'):
        return None
    pattern = _label_regex(row['label'])
    if not pattern:
        return None
    longer_labels = {candidate['label'] for table in tables for candidate in table['rows']
                     if ' '.join(candidate['label'].split()).lower().endswith(label)
                     and ' '.join(candidate['label'].split()).lower() != label}
    hits = []
    for hit in re.finditer(pattern, raw_text, re.I):
        end = raw_text.find('\n', hit.end())
        end = len(raw_text) if end < 0 else end
        start = raw_text.rfind('\n', 0, hit.start())+1
        prefix = raw_text[start:hit.end()]
        belongs_to_longer_label = any(re.search(r'\s+'.join(re.escape(x) for x in longer.split())+r'$',prefix,re.I)
                                     for longer in longer_labels)
        if (not belongs_to_longer_label and not re.match(r"[+&/]\s*\d", raw_text[hit.end():end])
                and _starts_raw_cells(raw_text[hit.end():end])):
            hits.append(hit)
    # Occurrences starting a physical line are the row labels; the same words
    # inside another line ("Scope 2, market-based tCO2 ...") belong to a
    # different, longer label.
    anchored = [h for h in hits if not raw_text[raw_text.rfind('\n', 0, h.start())+1:h.start()].strip()]
    positions = set()
    for table in tables:
        for candidate in table['rows']:
            if candidate.get('bbox') and ' '.join(candidate['label'].split()).lower() == label:
                positions.add(tuple(candidate['bbox']))
    ordered = sorted(positions, key=lambda b: (b[1], b[0]))
    key = tuple(row['bbox'])
    all_hits = list(hits)
    if len(hits) != len(ordered) and len(anchored) == len(ordered):
        hits = anchored
    toks = [str(c['raw']) for c in row.get('cells', []) if parse_number(c['raw']) is not None]
    def carries(h):
        end = raw_text.find('\n', h.end()); end = len(raw_text) if end < 0 else end
        line, pos = raw_text[h.end():end], 0
        for t in toks:
            m = re.search(r'(?<![\d.,])' + re.escape(t) + r'(?![\d,]|\.\d)', line[pos:])
            if not m:
                return False
            pos += m.end()
        return bool(toks)
    if key in positions and hits and (len(hits) != len(ordered) or not carries(hits[ordered.index(key)])):
        # an equally labelled line of another table, or tables side by side
        # (linear text reads one column first): the source line is the one
        # printing this row's own cells in order
        own = [h for h in all_hits if carries(h)]
        def shares(h):
            end = raw_text.find('\n', h.end()); end = len(raw_text) if end < 0 else end
            return any(re.search(r'(?<![\d.,])' + re.escape(t) + r'(?![\d,]|\.\d)', raw_text[h.end():end]) for t in toks)
        # every other equally labelled line must print none of these cells
        if len(own) == 1 and not any(shares(h) for h in all_hits if h is not own[0]):
            hit = own[0]
            start = raw_text.rfind('\n', 0, hit.start())+1
            end = raw_text.find('\n', hit.end()); end = len(raw_text) if end < 0 else end
            return dict(method='raw label line carrying the row cells in order', row_index=None,
                        occurrences=len(hits), row_bbox=list(key), line_span=[start, end],
                        label_span=[hit.start(), hit.end()])
    if key not in positions or len(hits) != len(ordered):
        return None
    index = ordered.index(key)
    hit = hits[index]
    start = raw_text.rfind('\n', 0, hit.start())+1
    end = raw_text.find('\n', hit.end())
    end = len(raw_text) if end < 0 else end
    return dict(method='ordered physical rows and exact raw label occurrences',
                row_index=index, occurrences=len(hits), row_bbox=list(key),
                line_span=[start, end], label_span=[hit.start(), hit.end()])


def _attach_continued_units(tables, document_pages):
    """A table continued from the previous page repeats its year columns but
    not its unit. Rows without a unit take the single unit of the previous
    page's table with the identical column signature."""
    for t in tables:
        if any(r['unit'] for r in t['rows']) or not document_pages:
            continue
        prev = next((p for p in document_pages if getattr(p, 'page', None) == t.get('page', 0) - 1), None)
        if prev is None:
            continue
        sig = [(c['year'], c['role']) for c in t['columns']]
        units, page_no, head = [], t.get('page', 0) - 1, t
        for _hop in range(3):
            same = [pt for pt in (prev.table_structures or []) if [(c['year'], c['role']) for c in pt['columns']] == sig]
            if same:
                units = [r['unit'] for r in same[-1]['rows'] if r['unit']]
                if units:
                    break
                # the previous page's part is itself a continuation
                head = same[-1]
                page_no -= 1
                prev = next((p for p in document_pages if getattr(p, 'page', None) == page_no), None)
                if prev is None:
                    break
                continue
            # the chain starts at the top of a page under the previous page's
            # last line, its heading with the unit
            owner = next((p for p in document_pages if getattr(p, 'page', None) == head.get('page')), None)
            on_page = (owner.table_structures if owner is not None and head is not t else tables) or []
            first = all(o is head or o.get('page') != head.get('page') or o['bbox'][1] > head['bbox'][1] for o in on_page)
            if first and head['bbox'][1] < 260:
                tu = [i.get('unit') for i in (getattr(prev, 'reconstruction_issues', None) or [])
                      if isinstance(i, dict) and i.get('kind') == 'trailing_unit']
                if tu:
                    units = [tu[-1]]
            break
        if not units or len({u['multiplier'] for u in units}) != 1:
            continue
        for r in t['rows']:
            if r['field'] and not r['unit']:
                r['unit'] = dict(units[-1], continued_from_page=page_no)


def bind_value(value, tables, raw_text, requested=None, document_pages=None):
    _attach_continued_units(tables, document_pages)
    """Return binding plus tri-state result. Ambiguous/mismatched table evidence
    is a rejection, not permission to fall back to a page-wide numeric match.
    """
    numeric = parse_number(value.value_raw)
    if requested is not None and not isinstance(requested,dict):
        return {'status':'rejected','reason':'malformed table binding'}, False
    factor = normalize_emission_to_tco2e(1, value.unit_raw)[0]
    # An unparseable or missing model unit (e.g. glyph-name garbage such as
    # 'tCO/two.dnomeq', or a bare 'CO2e') may inherit the bound row's unit
    # only when that unit is glyph-verified in the physical table.
    inherit_unit = factor is None
    excluded_rows = [(t['id'],r['id']) for t in tables for r in t['rows']
                     if (not value.evidence.source_document or t['document']==value.evidence.source_document)
                     and (value.evidence.page is None or t['page']==value.evidence.page)
                     and (r.get('quantity_type')=='excluded_noninventory' or _noninventory_quantity(r['label']))
                     and _row_similarity(value.label_raw,r['label'],component=True)==1]
    if excluded_rows:
        return {'status':'rejected','reason':'source row is an independent impact or relative indicator, not an inventory emissions row','excluded_rows':excluded_rows},False
    ambiguous_hit = False
    relevant, matches = [], []
    exact_rows = {(t['id'],r['id']) for t in tables for r in t['rows']
                  if (not value.evidence.source_document or t['document']==value.evidence.source_document)
                  and (value.evidence.page is None or t['page']==value.evidence.page)
                  and r['field']==value.field and r['parent_category']==value.category_no
                  and _row_similarity(value.label_raw,r['label'],component=True)==1}
    for t in tables:
        if (value.evidence.source_document and t['document'] != value.evidence.source_document
                or value.evidence.page is not None and t['page'] != value.evidence.page):
            continue
        cols = {c['id']: c for c in t['columns']}
        for r in t['rows']:
            if r.get('quantity_type')=='excluded_noninventory' or _noninventory_quantity(r['label']):
                continue
            linked_generic=(value.field=='scope2' and r.get('source_field')=='scope2'
                            and r.get('basis_link') and r.get('scope_aggregate'))
            if r['field'] != value.field and not linked_generic:
                continue
            relevant.append((t,r))
            if exact_rows and (t['id'],r['id']) not in exact_rows:
                continue
            locale = t.get('number_locale')
            numeric_t = parse_number(value.value_raw, locale)
            for cell in r['cells']:
                col = cols[cell['column_id']]
                if col['role'] != 'reported' or col['year'] != value.report_year:
                    continue
                footnote_glued = bool(cell.get('source_token') and re.sub(r"\D", "", str(cell['source_token']))
                                      == re.sub(r"\D", "", str(value.value_raw)) and parse_number(cell['raw'], locale) is not None)
                if parse_number(cell['raw'], locale) != numeric_t and not footnote_glued:
                    # the same printed digits in this table's convention
                    # ("3,695" read from a summary, "3.695" in the eu table)
                    cv = parse_number(cell['raw'], locale)
                    if not (cv is not None and factor and value.value_tco2e is not None
                            and abs(cv * factor - value.value_tco2e) < 1e-6 * max(1, abs(value.value_tco2e))
                            and re.sub(r"\D", "", str(cell['raw'])) == re.sub(r"\D", "", str(value.value_raw))):
                        continue
                if ambiguous_separator(cell['raw'], locale):
                    ambiguous_hit = True
                    continue
                if (not r['unit'] or not inherit_unit and r['unit']['multiplier'] != factor
                        or r['unit'].get('evidence_status') == 'unresolved'):
                    continue
                if r['parent_category'] != value.category_no:
                    continue
                # Components must identify their own data row, not borrow a
                # sibling's identical value or merely name the parent category.
                sim = _row_similarity(value.label_raw, r['label'],component=r['value_kind']=='component')
                if sim < (.65 if r['value_kind'] == 'component' else .5):
                    continue
                # a model pointing at a base/target/change column of the same
                # row names the row; the reported column of the year is the cell
                req_col = cols.get((requested or {}).get('column_id')) if requested and requested.get('table_id') == t['id'] else None
                ignore_col = bool(req_col and req_col.get('role') != 'reported')
                if requested and any(requested.get(k) and requested[k] != actual for k, actual in
                                     [('table_id',t['id']),('row_id',r['id'])] + ([] if ignore_col else [('column_id',col['id'])])):
                    continue
                matches.append((t,r,col,cell,sim))
    if not matches:
        if ambiguous_hit:
            return {"status":"rejected", "reason":"ambiguous decimal/thousands separator: no page or document number convention"}, False
        table_number = any(parse_number(c['raw'], t.get('number_locale')) == parse_number(value.value_raw, t.get('number_locale'))
                           for t in tables for r in t['rows'] for c in r['cells'])
        if relevant or requested or table_number:
            return {"status":"rejected", "reason":"no source cell satisfies row, year, unit, parent and value together"}, False
        return None, None
    # Row label disambiguates repeated numbers. It cannot choose a reporting
    # boundary where two identically labelled year columns remain possible.
    top = max(x[4] for x in matches)
    matches = [x for x in matches if x[4] >= top-.05]
    same_figure_tables = False
    if len(matches) > 1 and len({id(x[0]) for x in matches}) == len(matches):
        # the same figure printed in two tables of the page (a summary and a
        # detailed breakdown): identical field, year, boundary, kind and unit
        # make the choice immaterial; the fuller table is the source
        # a consolidated-figures column restates the report-wide amount
        wide = lambda b: 'reported' if b in (None, 'reported', 'consolidated_group') else b
        keyset = {(r['field'], r['parent_category'], r['value_kind'], c.get('year'), c.get('boundary'),
                   wide(c.get('organization_boundary')), r.get('coverage_boundary'), wide(r.get('organization_boundary')),
                   (r['unit'] or {}).get('multiplier')) for t, r, c, _, _ in matches}
        if len(keyset) == 1:
            matches = [max(matches, key=lambda x: (x[2].get('organization_boundary') == 'consolidated_group', len(x[0]['rows'])))]
            same_figure_tables = True
    if len(matches) != 1:
        return {"status":"rejected", "reason":"ambiguous source cells", "candidates":
                [[t['id'],r['id'],c['id']] for t,r,c,_,_ in matches]}, False
    t,r,col,cell,_ = matches[0]
    if not same_figure_tables:
        # the bound figure is also printed by another table of the page for
        # the same field and year (a summary beside a detailed breakdown)
        for t2 in tables:
            if t2 is t or t2['document'] != t['document'] or t2['page'] != t['page']:
                continue
            cols2 = {c2['id']: c2 for c2 in t2['columns']}
            if any(r2['field'] == r['field'] and c2['raw'] == cell['raw']
                   and cols2.get(c2['column_id'], {}).get('year') == col.get('year')
                   and cols2.get(c2['column_id'], {}).get('role') == 'reported'
                   for r2 in t2['rows'] for c2 in r2['cells']):
                same_figure_tables = True
                break
    basis_link=r.get('basis_link')
    perimeter=col.get('reporting_perimeter')
    perimeter_header_alignment=None
    if perimeter:
        from scope3trace.common.perimeter import bind_perimeter_header
        perimeter_header_alignment=bind_perimeter_header(col,raw_text)
        if not perimeter_header_alignment:
            return {'status':'source_conflict','reason':'reporting-perimeter header absent from original source text'},False
    if basis_link or r.get('source_field')=='scope2' and r['field']!='scope2':
        from scope3trace.scope_agent.source_context import resolve_scope2_basis
        if not basis_link or resolve_scope2_basis(r,t,document_pages or [])!=basis_link:
            return {'status':'source_conflict','reason':'cross-page accounting basis link missing, changed or contradicted by original source'},False
    if inherit_unit:
        factor = r['unit']['multiplier']
    # Compare complete cell tokens, never substrings (42M inside 342M). Spatial
    # words are the source for reconstruction. A raw-text disagreement is held
    # for review; it cannot be waived by a positive semantic model response.
    token = str(cell['raw'])
    source_region, source_offset = raw_text, 0
    raw_row_alignment = _physical_raw_row(r, [q for q in tables if q['document']==t['document'] and q['page']==t['page']], raw_text)
    if raw_row_alignment:
        start,end = raw_row_alignment['line_span']
        source_region,source_offset = raw_text[start:end],start
    # Repeated child labels (e.g. Commercial Vehicles) belong to different
    # categories. Restrict the independent linear evidence to the explicit
    # parent block before matching the child, retaining absolute offsets.
    cat=r.get('parent_category')
    parent_hits=[]
    if cat and not raw_row_alignment:
        parent_hits=list(re.finditer(r'(?im)^\s*(?:category|cat\.?)\s+'+str(cat)+r'\b[^\n]*',raw_text))
    elif not raw_row_alignment and (r.get('parent_observation') or {}).get('label','').lower()=='total':
        parent_hits=list(re.finditer(r'(?im)^\s*Total\s+(?=\d)',raw_text))
    if len(parent_hits)==1:
        start=parent_hits[0].start()
        next_parent=re.search(r'(?im)^\s*(?:category|cat\.?)\s+(?:1[0-5]|[1-9])\b',raw_text[parent_hits[0].end():])
        end=parent_hits[0].end()+next_parent.start() if next_parent else len(raw_text)
        source_region,source_offset=raw_text[start:end],start
    row_search_text,row_search_offset=source_region,source_offset
    label_pattern = _label_regex(r['label'])
    # An exact unqualified label can be a prefix of a DIFFERENT qualified row.
    # Prefer hits whose numeric suffix is not preceded by extra label words.
    # This preserves raw-row conflict detection while avoiding the SBTi row
    # when binding the separate unqualified Scope 3 aggregate.
    hits=list(re.finditer(label_pattern,row_search_text,re.I)) if label_pattern else []
    # A shorter child name may be the suffix of a different child's label
    # (Commercial Vehicles vs Passenger Cars and Light Commercial Vehicles).
    # Prefer whole row starts, while retaining the single-hit fallback for
    # flattened source text whose row label does not start a physical line.
    anchored_hits=[hit for hit in hits if not row_search_text[
        row_search_text.rfind('\n',0,hit.start())+1:hit.start()].strip()]
    if anchored_hits:
        hits=anchored_hits
    exact_hits=[]
    for hit in hits:
        end=row_search_text.find('\n',hit.end());end=len(row_search_text) if end<0 else end
        suffix=row_search_text[hit.end():end]
        # "Scope 1" is not the row "Scope 1+2": a sign glued to the label continues it
        if re.match(r"[+&/]\s*\d", suffix):
            continue
        if _starts_raw_cells(suffix):
            exact_hits.append(hit)
    if len(exact_hits) > 1:
        # a repeated text layer prints the identical row twice; identical
        # source lines are one row, not a conflict
        line_of = lambda h: row_search_text[row_search_text.rfind('\n', 0, h.start())+1:
                                            (row_search_text.find('\n', h.end()) + 1 or len(row_search_text)+1) - 1].strip()
        if len({line_of(h) for h in exact_hits}) == 1:
            exact_hits = exact_hits[:1]
        else:
            # Same label in a different table (an intensity table repeating the
            # category names): only one of the lines holds the bound cell's
            # complete token, and the coordinate cell already fixes the row.
            tok_re = r'(?<![\d.,])' + re.escape(token) + r'(?![\d]|[.,]\d)'
            holding = [h for h in exact_hits if re.search(tok_re, line_of(h))]
            if len(holding) > 1:
                # the same label in two tables both printing the amount: the
                # line carrying the reconstructed row's own cells in order
                own = [str(c['raw']) for c in r.get('cells', []) if parse_number(str(c['raw'])) is not None]
                def carries(h):
                    toks = re.findall(r"[-(]?\d[\d.,']*\d\)?|\d", line_of(h)[h.end() - h.start():])
                    toks = [x.strip() for x in toks]
                    it = iter(toks)
                    return len(own) >= 2 and all(any(t == o for t in it) for o in own)
                def exact(h):
                    toks = [x.strip() for x in re.findall(r"[-(]?\d[\d.,']*\d\)?|\d", line_of(h)[h.end() - h.start():])]
                    toks = [t for t in toks if parse_number(t) is not None]
                    return len(own) >= 2 and toks[:len(own)] == own and all(
                        not re.fullmatch(r"\d[\d.,']*", t) or False for t in toks[len(own):len(own)])
                fit = [h for h in holding if exact(h)] or [h for h in holding if carries(h)]
                if len(fit) == 1:
                    holding = fit
            if len(holding) == 1:
                exact_hits = holding
    # the same figure printed by two tables of the page: either labelled line
    # is a source of the bound amount
    if len(hits)>1 and len(exact_hits)!=1 and not (same_figure_tables and not exact_hits):
        return {'status':'source_conflict','reason':'ambiguous raw row labels; no unique unqualified numeric row'}, False
    label_hit=exact_hits[0] if len(exact_hits)==1 else hits[0] if len(hits)==1 else None
    if label_hit:
        line_end = row_search_text.find('\n',label_hit.end())
        line_end = len(row_search_text) if line_end < 0 else line_end
        suffix = row_search_text[label_hit.end():line_end]
        amounts_suffix = re.sub(r"[-+(]?\d+(?:[.,]\d+)?\s*\)?\s*%", " ", suffix)
        if not re.search(r"(?<![a-zA-Z])\d+(?:[.,]\d+)?", amounts_suffix):
            # "3. Fuel ... (not included in 16.3%" / "4,932.9 5,735.4": a
            # wrapped label prints its amounts on the next, numbers-only line
            nxt_end = row_search_text.find('\n', line_end + 1)
            nxt_end = len(row_search_text) if nxt_end < 0 else nxt_end
            nxt = row_search_text[line_end + 1:nxt_end]
            if line_end < len(row_search_text) and re.fullmatch(r"[\s\d.,()%*\-\u2013\u2014/NA]+", nxt or "x") \
                    and re.search(r"\d", nxt):
                suffix = row_search_text[label_hit.end():nxt_end]
        # If the linear source supplies numeric cells on this same row, an
        # amount elsewhere on the page cannot conceal its disagreement.
        if re.search(r"(?<![a-zA-Z])\d+(?:[.,]\d+)?", suffix):
            source_region, source_offset = suffix, row_search_offset+label_hit.end()
    row_units = _raw_units(source_region) if source_offset else []
    raw_units = row_units or _raw_units(raw_text)
    unit_conflict = bool(raw_units) and factor not in [u['multiplier'] for u in raw_units]
    raw_year_present = bool(re.search(r'(?<!\d)'+r'[ \t]*'.join(str(col['year']))+r'(?!\d)',raw_text))
    header = col.get('source_header','')
    if not raw_year_present and header and header != str(col['year']):
        raw_year_present = bool(re.search(r'\s*'.join(map(re.escape, re.sub(r'\s+', '', header))), raw_text))
    fiscal = re.fullmatch(r'((?:19|20)\d{2})[/-](\d{2})',header)
    if fiscal and int(fiscal.group(1))+1==col['year'] and int(fiscal.group(2))==col['year']%100:
        raw_year_present = header in raw_text
    pattern = r"(?<![\d.,])" + re.escape(token).replace(r"\ ", r"\s+") + r"(?![\d.,])"
    original = re.search(pattern, source_region)
    resolution = None
    if not original:
        # Only spacing can be reconciled automatically, and only for the full
        # coordinate-backed cell. Digits, signs and separators cannot change.
        compact = re.sub(r"\s+", "", token)
        spaced = r"(?<![\d.,])" + r"[ \t\u00a0]*".join(map(re.escape, compact)) + r"(?![\d.,])"
        original = re.search(spaced, source_region)
        if original:
            resolution = "spacing-only discrepancy resolved against physical source cell; all digits and punctuation identical"
    if not original and cell.get('source_token'):
        full = r"(?<![\d.,])" + r"[ \t\u00a0]*".join(map(re.escape, re.sub(r"\s+", "", cell['source_token']))) + r"(?![\d.,])"
        original = re.search(full, source_region)
        if original:
            resolution = ("superscript footnote marker '%s' excluded by glyph size and baseline; "
                          "complete source token matched" % cell.get('footnote_marker', ''))
    seen = original is not None and not unit_conflict and raw_year_present
    status = ('resolved_spacing' if resolution else 'bound') if seen else 'source_conflict'
    footnote_value = (parse_number(cell['raw'], t.get('number_locale')) != parse_number(value.value_raw, t.get('number_locale'))
                      and bool(cell.get('source_token')))
    binding = {"status":status, "table_id":t['id'], "row_id":r['id'], "column_id":col['id'],
               "footnote_value_corrected": footnote_value,
               "cell_id":cell['id'], "document":t['document'], "page":t['page'],
               "table_digest":_digest(t), "row_label":r['label'], "cell_raw":cell['raw'],
               "cell_bbox":cell['bbox'], "year":col['year'], "year_bbox":col.get('year_source_bbox',col['bbox']),
               "source_year_header":col.get('source_header',str(col['year'])),
               "year_footnote_bbox":col.get('footnote_bbox'),
               "column_boundary":col['boundary'], "unit":r['unit'],
               "accounting_basis": basis_link['basis'] if basis_link else "location" if re.search(r"location[ -]based",r["label"],re.I) else "market" if re.search(r"market[ -]based",r["label"],re.I) else "unspecified",
               "basis_link":basis_link,"label_parent":r.get('label_parent'),"label_bbox":r.get('label_bbox'),
               "resolved_field":r['field'],
               "reporting_perimeter":perimeter,"perimeter_header":col.get('perimeter_header'),
               "perimeter_bbox":col.get('perimeter_bbox'),
               "perimeter_header_alignment":perimeter_header_alignment,
               "coverage_boundary":r.get('coverage_boundary','reported') if r.get('coverage_boundary','reported')!='reported' else col['boundary'],
               "source_sha256":hashlib.sha256(raw_text.encode()).hexdigest(),
               "organization_boundary":(r.get('organization_boundary') if r.get('organization_boundary') in ('group_and_operational_control','individual_entity') else col.get('organization_boundary') if col.get('organization_boundary','reported')!='reported' else r.get('organization_boundary','reported')),
               "organization_heading":t.get('source_heading'),
               "organization_column_header":col.get('organization_header'),
               "organization_column_bbox":col.get('organization_bbox'),
               "unit_inherited":inherit_unit, "unit_id":r['unit'].get('id'), "parent_id":r.get('parent_id'),
               "number_locale":t.get('number_locale'),
               "parent_observation":r.get('parent_observation'),
               "parent_category":r['parent_category'], "parent_label":r['parent_label'],
               "parent_scope":r.get('parent_scope'),
               "parent_bbox":r['parent_bbox'], "value_kind":r['value_kind'],
               "quantity_type":r.get('quantity_type','inventory'),
               "source_span": [source_offset+original.start(),source_offset+original.end()] if original else None,
               "raw_row_alignment":raw_row_alignment,
               "source_fragment": original.group(0) if original else None,
               "source_unit_check":{'scope':'same raw row' if row_units else 'page unit inventory',
                                    'units':raw_units,'conflict':unit_conflict},
               "source_year_present":raw_year_present,
               "source_check": ("raw unit multiplier contradicts bound source unit" if unit_conflict else
                                "bound year absent from raw source text" if not raw_year_present else
                                resolution or ("exact complete numeric token in raw PDF text" if seen else
                                "coordinate cell disagrees with linear source text; review required"))}
    return binding, seen


def validate_binding(value, tables, raw_text, document_pages=None):
    old = value.evidence.binding
    if not old:
        return False, "missing structured evidence binding"
    source=next((t for t in tables if t['id']==old.get('table_id')),None)
    if source and source.get('method','').startswith('geometry-year-grid') and source['method']!=VERSION:
        return False, 'source reconstruction version changed; reparse and rebind before verification'
    new, ok = bind_value(value, tables, raw_text, old, document_pages=document_pages)
    if not ok:
        return False, (new or {}).get('reason', (new or {}).get('source_check', 'binding rejected'))
    for key in ('table_digest','document','page','cell_id','row_label','year','unit','unit_id','parent_id','parent_observation','parent_category','parent_scope','value_kind','cell_raw','cell_bbox','year_bbox','parent_bbox','column_boundary','organization_boundary','organization_heading','organization_column_header','organization_column_bbox','reporting_perimeter','perimeter_header','perimeter_bbox','perimeter_header_alignment','source_year_header','year_footnote_bbox','coverage_boundary','accounting_basis','basis_link','resolved_field','label_parent','label_bbox','source_sha256','raw_row_alignment','quantity_type'):
        if old.get(key) != new.get(key):
            return False, f"binding changed or was tampered with: {key}"
    if value.value_kind != new['value_kind']:
        return False, "source hierarchy contradicts extracted value kind"
    if value.field=='scope3_total' and value.value_kind=='category_total' and new['coverage_boundary'] not in ('reported',None):
        return False, 'restricted coverage cannot be exported as an unqualified Scope 3 aggregate'
    if value.value_tco2e != normalize_emission_to_tco2e(parse_number(new['cell_raw'], new.get('number_locale')),new['unit']['text'])[0]:
        return False, "normalized amount does not equal bound cell times bound unit"
    return True, "table-row-year-unit-parent binding independently revalidated"
