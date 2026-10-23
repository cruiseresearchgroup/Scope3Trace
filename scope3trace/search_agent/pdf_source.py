"""Consistent physical PDF access without modifying the source document.

A reader-repaired in-memory copy is used only when the geometry parser has a
page-tree failure. Page count and original/derived digests are retained. Visible
bounds and duplicate overprint glyphs are applied before table reconstruction.
"""
from contextlib import contextmanager
import hashlib
import io


def visible_page(page):
    box = page.bbox
    crop = page.cropbox
    if crop and page.rotation == 0:
        # PDF crop boxes use bottom-up coordinates; pdfplumber uses top-down.
        media = page.mediabox
        cb = (crop[0], media[3]-crop[3], crop[2], media[3]-crop[1])
        box = (max(box[0],cb[0]),max(box[1],cb[1]),min(box[2],cb[2]),min(box[3],cb[3]))
    bounded = page.within_bbox(box)
    return bounded.dedupe_chars(tolerance=0.5)


@contextmanager
def open_geometry(path, expected_pages=None):
    import pdfplumber
    with open(path,'rb') as source:
        data = source.read()
    original_hash = hashlib.sha256(data).hexdigest()
    pdf = pdfplumber.open(io.BytesIO(data))
    n = len(pdf.pages)
    audit = dict(original_sha256=original_hash,original_geometry_pages=n,
                 backend='pdfplumber',normalization='none',derived_sha256=None)
    if not n or expected_pages is not None and n != expected_pages:
        pdf.close()
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(io.BytesIO(data)); writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        stream = io.BytesIO(); writer.write(stream); repaired = stream.getvalue()
        pdf = pdfplumber.open(io.BytesIO(repaired))
        audit.update(normalization='pypdf page-tree rewrite in memory',
                     derived_sha256=hashlib.sha256(repaired).hexdigest())
        n = len(pdf.pages)
        if n != len(reader.pages) or expected_pages is not None and n != expected_pages:
            pdf.close()
            raise ValueError('PDF parser page-count disagreement after page-tree repair')
    audit['geometry_pages'] = n
    try:
        yield pdf,audit
    finally:
        pdf.close()
