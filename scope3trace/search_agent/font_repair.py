# -*- coding: utf-8 -*-
"""Recover characters a font's ToUnicode table maps to U+FFFD.

Some report fonts (e.g. Munich Re's tabular figures ``comma.tab``,
``period.tab``, ``space.tab``) carry a ToUnicode CMap that maps those glyphs
to U+FFFD, so every number reads ``64�353``. The font's own /Encoding
still names the glyph; that name is an exact, deterministic source for the
character, so it is used only when ToUnicode gives U+FFFD. Nothing is
guessed from the surrounding digits.
"""
from __future__ import annotations

import re
from typing import List, Optional

REPLACEMENT = "�"
_DIGIT_FFFD = re.compile(r"\d�\d")
_installed = False


def install() -> bool:
    """Patch pdfminer (used by pdfplumber) once. Returns True when active."""
    global _installed
    if _installed:
        return True
    try:
        from pdfminer.pdffont import PDFSimpleFont
    except Exception:
        return False
    original = PDFSimpleFont.to_unichr

    def to_unichr(self, cid):
        text = original(self, cid)
        if text == REPLACEMENT:
            named = getattr(self, "cid2unicode", {}).get(cid)
            if named and named != REPLACEMENT:
                return named
        return text

    to_unichr._s3t_font_repair = True  # type: ignore[attr-defined]
    PDFSimpleFont.to_unichr = to_unichr
    _installed = True
    return True


def broken_number_pages(texts: List[Optional[str]]) -> List[int]:
    """1-based pages whose text has a digit-U+FFFD-digit sequence."""
    return [i for i, t in enumerate(texts, 1) if t and _DIGIT_FFFD.search(t)]


def repair_pages(texts: List[Optional[str]], pdf_path: str) -> List[int]:
    """Replace, in place, page texts with broken numbers by the pdfplumber
    text read through the repaired fonts, when that text has fewer broken
    numbers. Returns the replaced page numbers."""
    pages = broken_number_pages(texts)
    if not pages or not install():
        return []
    try:
        import pdfplumber
    except Exception:
        return []
    fixed = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for n in pages:
                page = pdf.pages[n - 1]
                try:
                    alt = page.extract_text() or ""
                finally:
                    if hasattr(page, "close"):
                        try:
                            page.close()
                        except Exception:
                            pass
                if len(_DIGIT_FFFD.findall(alt)) < len(_DIGIT_FFFD.findall(texts[n - 1] or "")):
                    texts[n - 1] = alt
                    fixed.append(n)
    except Exception:
        return fixed
    return fixed


install()
