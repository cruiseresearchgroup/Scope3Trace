"""Source-backed continuation/footnote links; no amounts or annotations inferred.

Only a referenced physical footnote in the same numbered table (or its next-page
continuation) may establish the basis of a generic Scope 2 total. Every note line
must also occur in independent linear PDF text. Verify re-derives the link.
"""
import hashlib
import json
import re
from collections import defaultdict
from statistics import median


def _norm(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def attach_page_context(page, words, tables):
    from scope3trace.scope_agent.table_structure import _lines, _text, _bbox
    lines=_lines(words)
    # Footnote lists have repeated marker margins. Row amounts are not notes.
    margins=defaultdict(list)
    for line in lines:
        for i,w in enumerate(line[:-1]):
            if (re.fullmatch(r'[1-9]\d?',w['text']) and w['top']>page.height*.45
                    and 3 < line[i+1]['x0']-w['x1'] < 22
                    and re.match(r'[A-Za-z0-9]',line[i+1]['text'])):
                margins[round(w['x0']/4)*4].append(w)
    margins={x:ws for x,ws in margins.items() if len({w['text'] for w in ws})>=2}
    notes=[]
    for x,markers in sorted(margins.items()):
        right=min((v-3 for v in margins if v>x+20),default=page.width)
        markers=sorted(markers,key=lambda w:w['top'])
        for i,marker in enumerate(markers):
            bottom=markers[i+1]['top']-1 if i+1<len(markers) else page.height-20
            own=[w for w in words if marker['x1']+2 <= w['x0'] and w['x1']<right
                 and marker['top']-1 <= w['top'] < bottom]
            own_lines=_lines(own)
            if not own_lines:
                continue
            notes.append(dict(marker=marker['text'],marker_bbox=_bbox([marker]),
                              bbox=_bbox([marker]+own),lines=[_text(ls) for ls in own_lines],
                              text=_text(own)))
    captions=[]
    for line in lines:
        text=_text(line)
        match=re.search(r'\bTABLE\s+(\d+)\s+(.+)',text,re.I)
        if match:
            captions.append(dict(number=str(int(match.group(1))),text=match.group(0),
                                 title=_norm(re.sub(r'\bCONTINUED\b','',match.group(2),flags=re.I)),
                                 continued=bool(re.search(r'\bCONTINUED\b',match.group(2),re.I)),bbox=_bbox(line)))
    for table in tables:
        header=table['bbox'][1]
        eligible=[c for c in captions if 0 <= header-c['bbox'][1] < 90]
        table['caption']=max(eligible,key=lambda c:c['bbox'][1]) if eligible else None
        table['page_footnotes']=notes
        for row in table['rows']:
            box=row.get('label_bbox') or row['bbox']
            refs=[]
            for w in words:
                if not (box[0]-.5 <= w['x0'] and w['x1'] <= box[2]+.5
                        and box[1]-.5 <= w['top'] < box[3]):
                    continue
                hit=re.fullmatch(r'(.+?)([1-9]\d?)',w['text'])
                if not hit or not re.search('[A-Za-z]',hit.group(1)):
                    continue
                chars=sorted([c for c in page.chars if w['x0']-.1 <= c['x0'] < w['x1']+.1
                              and w['top']-.1 <= c['top'] <= w['bottom']+.1],key=lambda c:c['x0'])
                if ''.join(c['text'] for c in chars)!=w['text']:
                    continue
                base,foot=chars[:-len(hit.group(2))],chars[-len(hit.group(2)):]
                if base and all(c.get('size',c['bottom']-c['top']) < .85*median(b.get('size',b['bottom']-b['top']) for b in base)
                                and c['bottom'] < median(b['bottom'] for b in base)-1 for c in foot):
                    refs.append(dict(marker=hit.group(2),word_text=w['text'],bbox=_bbox(foot)))
            row['footnote_refs']=refs


def _raw_spans(note,raw):
    """Keep line spans across interleaved PDF columns; no digit corrections."""
    if _norm(note['text'])!=_norm(' '.join(note['lines'])):
        return None
    spans=[]
    for i,line in enumerate(note['lines']):
        pattern=r'\s+'.join(re.escape(w) for w in line.split())
        if i==0:
            pattern=r'(?<!\d)'+re.escape(note['marker'])+r'\s+'+pattern
        hits=list(re.finditer(pattern,raw))
        if len(hits)!=1:
            return None
        spans.append([hits[0].start(),hits[0].end()])
    return spans if spans==sorted(spans) else None


def _caption_present(caption,raw):
    match=re.fullmatch(r'TABLE\s+(\d+)\s+(.+)',caption['text'],re.I)
    if (not match or str(int(match.group(1)))!=caption['number']
            or _norm(re.sub(r'\bCONTINUED\b','',match.group(2),flags=re.I))!=caption['title']
            or bool(re.search(r'\bCONTINUED\b',match.group(2),re.I))!=caption['continued']):
        return False
    pattern=r'\s+'.join(re.escape(w) for w in caption['text'].split())
    return len(list(re.finditer(pattern,raw,re.I)))==1


def resolve_scope2_basis(row,table,pages):
    """A narrow explicit exclusion/inclusion statement can qualify a full total.

    Location-only electricity does not determine a full location-based total.
    Nothing is summed, and an electricity/heating child cannot be promoted.
    """
    if (row.get('source_field',row['field'])!='scope2' or not row.get('scope_aggregate')
            or row['value_kind']!='category_total' or row.get('parent_scope')!='scope2'):
        return None
    caption=table.get('caption')
    current=next((p for p in pages if p.doc_id==table['document'] and p.page==table['page']),None)
    if not caption or current is None or not _caption_present(caption,current.raw_text):
        return None
    children=[r for r in table['rows'] if r.get('parent_id')==row.get('parent_id')
              and r['value_kind']=='component' and r['field']=='scope2_location'
              and re.search(r'\belectricity\b',r['label'],re.I)]
    if len(children)!=1:
        return None
    child=children[0]
    if len(child.get('footnote_refs',[]))!=1:
        return None
    ref=child['footnote_refs'][0]
    if not ref['word_text'].endswith(ref['marker']):
        return None
    # The marker must still occur on the referenced original source row.
    from scope3trace.scope_agent.table_structure import _physical_raw_row
    aligned=_physical_raw_row(child,current.table_structures,current.raw_text)
    if not aligned or ref['word_text'] not in current.raw_text[slice(*aligned['line_span'])]:
        return None
    candidates=[]
    for page in pages:
        if page.doc_id!=table['document'] or page.page not in (table['page'],table['page']+1):
            continue
        continuations=[t for t in page.table_structures if t.get('caption')
                       and t['caption']['number']==caption['number']
                       and t['caption']['title']==caption['title']
                       and (page.page==table['page'] or t['caption']['continued'])
                       and _caption_present(t['caption'],page.raw_text)]
        if len(continuations)!=1:
            continue
        linked=continuations[0]
        for note in linked.get('page_footnotes',[]):
            if note['marker']!=ref['marker']:
                continue
            spans=_raw_spans(note,page.raw_text)
            text=_norm(note['text'])
            excluded=bool(re.search(r'location[ -]based emissions from electricity are not included in (?:the )?total emissions accounting\.',text))
            included=bool(re.search(r'\bwe include the (?:market|marked)[ -]based emissions\.',text))
            contradictory=bool(re.search(r'(?:market|marked)[ -]based.{0,60}\bnot included|location[ -]based.{0,60}\bare included',text))
            # Keep all referenced notes, including unsupported/contradictory
            # ones; duplicate marker definitions cannot select a favorable one.
            candidates.append((page,linked,note,spans,excluded and included and not contradictory))
    if len(candidates)!=1 or not candidates[0][3] or not candidates[0][4]:
        return None
    page,linked,note,spans,_=candidates[0]
    return dict(method='referenced physical footnote in same numbered table continuation',basis='market',
                document=table['document'],table_id=table['id'],caption=caption,
                aggregate_row_id=row['id'],reference_row_id=child['id'],reference_row_label=child['label'],
                parent_id=row['parent_id'],marker=ref,footnote_page=page.page,
                continuation_table_id=linked['id'],continuation_caption=linked['caption'],
                footnote=note,footnote_raw_spans=spans,
                footnote_source_sha256=hashlib.sha256(page.raw_text.encode()).hexdigest(),
                scope='printed generic Scope 2 aggregate only; no child sum or inferred location total')


def link_document_basis(pages):
    """Refresh context after initial parsing and after every source rebuild."""
    changed=set()
    for page in pages:
        for table in page.table_structures:
            for row in table['rows']:
                old=json.dumps([row['field'],row.get('basis_link')],sort_keys=True)
                if 'source_field' in row:
                    row['field']=row['source_field']
                row.pop('basis_link',None)
                link=resolve_scope2_basis(row,table,pages)
                if link:
                    row['source_field']='scope2'
                    row['field']='scope2_market'
                    row['basis_link']=link
                if json.dumps([row['field'],row.get('basis_link')],sort_keys=True)!=old:
                    changed.add(page.page)
    return changed
