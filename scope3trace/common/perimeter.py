"""Physical reporting-perimeter columns and conservative duplicate selection."""
import re


def column_perimeters(lines,header_y,cut,columns):
    """Use a dedicated header line, never a nearby body-row qualifier."""
    from scope3trace.scope_agent.table_structure import _text,_bbox
    pattern=re.compile(r'\b(ESRS perimeter|Operated perimeter|Operated domain)\b',re.I)
    headings=[]
    for line in lines:
        y=min(w['top'] for w in line)
        if not header_y-36 <= y < header_y-2:
            continue
        words=[w for w in line if w['x0']>=cut-15]
        text=_text(words)
        if not text or pattern.sub('',text).strip():
            continue
        offset=0
        for hit in pattern.finditer(text):
            selected=[];offset=0
            for w in words:
                end=offset+len(w['text'])
                if end>hit.start() and offset<hit.end():selected.append(w)
                offset=end+1
            if selected:
                headings.append(dict(kind='esrs' if hit.group(1).lower().startswith('esrs') else 'operated',
                                     text=hit.group(1),line=text,bbox=_bbox(selected),top=y))
    if not headings:
        return
    latest=max(h['top'] for h in headings)
    headings=[h for h in headings if abs(h['top']-latest)<3]
    if len({h['kind'] for h in headings})!=len(headings):
        return
    for col in columns:
        chosen=min(headings,key=lambda h:abs(col['anchor']-(h['bbox'][0]+h['bbox'][2])/2))
        col.update(reporting_perimeter=chosen['kind'],perimeter_header=chosen['text'],
                   perimeter_header_line=chosen['line'],perimeter_bbox=chosen['bbox'])


def finalize_perimeter_headers(tables,words):
    """Record ordered full-page physical headers, including unparsed tables."""
    from scope3trace.scope_agent.table_structure import _lines,_text
    lines=_lines(words)
    for table in tables:
        for col in table['columns']:
            if not col.get('reporting_perimeter'):continue
            expected=' '.join(col['perimeter_header_line'].lower().split())
            matching=[ls for ls in lines if ' '.join(_text(ls).lower().split())==expected]
            matching.sort(key=lambda ls:min(w['top'] for w in ls))
            chosen=[i for i,ls in enumerate(matching) if abs(min(w['top'] for w in ls)-col['perimeter_bbox'][1])<3]
            col['perimeter_header_occurrences']=len(matching)
            col['perimeter_header_index']=chosen[0] if len(chosen)==1 else None


def bind_perimeter_header(col,raw_text):
    line=col['perimeter_header_line'];index=col.get('perimeter_header_index')
    pattern=r'(?im)^[ \t\u00a0]*'+r'[ \t\u00a0]+'.join(re.escape(w) for w in line.split())+r'[ \t\u00a0]*$'
    hits=list(re.finditer(pattern,raw_text))
    if index is None or len(hits)!=col.get('perimeter_header_occurrences') or index>=len(hits):
        return None
    hit=hits[index]
    return dict(method='ordered full-page physical and linear perimeter header lines',
                index=index,occurrences=len(hits),line=line,source_span=[hit.start(),hit.end()])


def reported_quantum(value):
    """Resolution in tCO2e of a printed cell, not model float formatting."""
    b=value.evidence.binding or {};unit=b.get('unit') or {}
    factor=unit.get('multiplier') if unit.get('multiplier') is not None else b.get('unit_multiplier')
    raw=b.get('cell_raw') if b.get('cell_raw') is not None else (value.value_raw if b.get('origin')=='text_row' else None)
    if factor is None or raw is None:
        return None
    text=re.sub(r"[\s'()−-]",'',str(raw))
    if ',' in text and '.' in text:
        text=text.replace('.','').replace(',','.') if text.rfind(',')>text.rfind('.') else text.replace(',','')
    elif ',' in text:
        text=text.replace(',','') if len(text.split(',')[-1])==3 else text.replace(',','.')
    if not re.fullmatch(r'\d+(?:\.\d+)?',text):
        return None
    decimals=len(text.split('.')[1]) if '.' in text else 0
    return factor*10**(-decimals)


def select_perimeter(disclosure,field,candidates):
    """Apply declared operational-control scope; conflicting totals stay empty.

    Two source numbers can differ solely through reported rounding. A finer
    printed value wins only when its rounding interval overlaps all coarser
    alternatives. ESRS and operated amounts never enter that comparison.
    """
    if field not in ('scope1','scope2','scope2_market','scope2_location'):
        return candidates,None
    known={(v.evidence.binding or {}).get('reporting_perimeter') for v in candidates}
    known.discard(None)
    if not known:
        return candidates,None
    declared=(disclosure.reporting_boundary or '').lower()
    if 'operational control' in declared:
        target='operated'
    elif len(known)==1:
        target=next(iter(known))
    else:
        return [],'multiple reporting perimeters without a declared selection basis'
    selected=[v for v in candidates if (v.evidence.binding or {}).get('reporting_perimeter')==target]
    if not selected:
        return [],f'no verified {target} reporting-perimeter total for declared boundary'
    for i,a in enumerate(selected):
        for b in selected[i+1:]:
            if a.value_tco2e==b.value_tco2e:
                continue
            qa,qb=reported_quantum(a),reported_quantum(b)
            if qa is None or qb is None or abs(a.value_tco2e-b.value_tco2e) > max(qa,qb)/2+1e-8:
                return [],'conflicting source totals within the same reporting perimeter; rounding cannot reconcile them'
    return selected,None
