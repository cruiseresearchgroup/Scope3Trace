"""Bounded, auditable Verify -> Search/Scope -> Verify coordination.

No reference annotations, answer values or free-form agent conversations enter
this loop. Verify identifies source cells and failure reasons; only Scope creates
fresh candidates. Identical rejected evidence cannot be retried into acceptance.
"""
from collections import defaultdict
from dataclasses import asdict
import copy
import json
import time

from scope3trace.common.rules import parse_number


def _boundary(row, col):
    rb = row.get('organization_boundary','reported')
    cb = col.get('organization_boundary','reported')
    return rb if rb in ('individual_entity','group_and_operational_control') else cb if cb!='reported' else rb


def observations(state, doc_id):
    """Inventory physical cells for completeness checks, not extracted answers."""
    for page in state.parsed_pages.get(doc_id,[]):
        for table in page.table_structures:
            cols = {c['id']:c for c in table['columns']}
            for row in table['rows']:
                if not row['field'] or not row['unit'] or row['unit'].get('evidence_status')=='unresolved':
                    continue
                for cell in row['cells']:
                    col = cols[cell['column_id']]
                    if col['role']!='reported' or parse_number(cell['raw']) is None:
                        continue
                    yield dict(document=doc_id,page=page.page,field=row['field'],year=col['year'],
                               table_id=table['id'],row_id=row['id'],column_id=col['id'],
                               cell_id=cell['id'],label=row['label'],kind=row['value_kind'],
                               boundary=_boundary(row,col))


def preferred_boundaries(state, doc_id):
    preferred = {}
    for ob in observations(state,doc_id):
        if ob['kind']=='category_total' and ob['boundary']=='consolidated_group':
            preferred[(ob['field'],ob['year'])]='consolidated_group'
    return preferred


def boundary_check(value, doc_id, state):
    b = value.evidence.binding or {}
    expected = preferred_boundaries(state,doc_id).get((value.field,value.report_year))
    actual = b.get('organization_boundary','reported')
    if (value.value_kind=='category_total' and expected and
            actual in ('individual_entity','additional_operational_control','group_and_operational_control')):
        return False, f'organization boundary conflict: source is {actual}; a {expected} disclosure exists for this field/year'
    return True, 'organization boundary compatible with source inventory'


def candidate_signature(value):
    b=value.evidence.binding or {}
    if b.get('table_id'):
        # Formatting and synonyms do not create new evidence for a previously
        # rejected physical cell. Changed field/role/source still can repair it.
        return json.dumps([value.field,value.report_year,value.value_tco2e,value.value_kind,
                           value.evidence.page,b.get('table_id'),b.get('cell_id'),b.get('table_digest'),
                           b.get('source_sha256'),b.get('organization_boundary'),b.get('coverage_boundary')],sort_keys=True)
    return json.dumps([value.field,value.report_year,value.value_raw,value.unit_raw,value.value_tco2e,
                       value.label_raw,value.value_kind,value.evidence.page,b],sort_keys=True,ensure_ascii=False)


def collect_issues(state, doc_id):
    """Find rejected candidates AND missing source cells; never assume 15 totals."""
    disclosures=[d for d in state.org_disclosures if d.source_document==doc_id]
    values=[v for d in disclosures for v in d.values+d.category_components]
    preferred=preferred_boundaries(state,doc_id)
    verified_cells={(v.evidence.page,(v.evidence.binding or {}).get('cell_id'),
                     (v.evidence.binding or {}).get('table_id')) for v in values if v.evidence.verified is True}
    issues=[]
    for disclosure in disclosures:
        for field in sorted({v.field for v in disclosure.values}):
            reason=disclosure.selection_issue(field)
            if reason:
                for v in disclosure.values:
                    if v.field==field and v.evidence.verified is True and v.evidence.page:
                        issues.append(dict(document=doc_id,page=v.evidence.page,field=field,year=v.report_year,
                                           code='unresolved_disclosure_selection',reason=reason,
                                           label=v.label_raw,kind=v.value_kind))
    for ob in observations(state,doc_id):
        if (ob['page'],ob['cell_id'],ob['table_id']) in verified_cells:
            continue
        if preferred.get((ob['field'],ob['year'])) and ob['boundary'] in (
                'individual_entity','additional_operational_control','group_and_operational_control'):
            continue
        issues.append(dict(ob,code='missing_or_rejected_source_cell',
                           reason='Source has a numeric cell without a verified candidate; inspect its row, year, unit and role.'))
    for v in values:
        if v.evidence.verified is not False or not v.evidence.page:
            continue
        # Intentional exclusions must stay in the audit; do not demand a
        # consolidated answer from the same standalone-company cell.
        if not boundary_check(v,doc_id,state)[0]:
            continue
        if any(w.evidence.verified is True and w.field==v.field and w.report_year==v.report_year
               and w.evidence.page==v.evidence.page and w.label_raw==v.label_raw for w in values):
            continue
        issues.append(dict(document=doc_id,page=v.evidence.page,field=v.field,year=v.report_year,
                           code='rejected_candidate',reason=v.evidence.verification_note,
                           label=v.label_raw,kind=v.value_kind))
    unique={json.dumps(i,sort_keys=True,ensure_ascii=False):i for i in issues}
    return sorted(unique.values(),key=lambda i:(i['page'],i.get('year') or 0,i['field'],i['code']))


def _verify_document(state, verify, doc_id):
    for d in state.org_disclosures:
        if d.source_document==doc_id:
            verify.evidence_check(d)
            verify.consistency_checks(d)
            # A single category cannot exceed the same disclosure's aggregate.
            # Do not enforce equality of sums: partial disclosure and overlaps
            # are legitimate, and Verify must never manufacture a replacement.
            total=d.get('scope3_total',verified_only=True)
            if total and total.value_tco2e is not None and total.value_tco2e>=0:
                oversized=[v for v in d.scope3_categories(verified_only=True)
                           if v.value_tco2e is not None and v.value_tco2e > total.value_tco2e*1.01]
                if oversized:
                    # Consistency checks are diagnostics (A.4): they flag the
                    # conflict in the Evaluation Report but cannot reject a
                    # value that is independently bound to its source cell.
                    from scope3trace.common.state import DiagnosticResult
                    state.diagnostics.append(DiagnosticResult(
                        'consistency:category_le_scope3_total','warn',
                        f"{doc_id}@{d.reporting_year}: categories "
                        f"{[v.field for v in oversized]} exceed scope3_total={total.value_tco2e}",'org-level'))


def _merge(state, incoming):
    added=0
    for fresh in incoming:
        old=next((d for d in state.org_disclosures if d.source_document==fresh.source_document
                  and d.reporting_year==fresh.reporting_year),None)
        if old is None:
            state.org_disclosures.append(fresh);added+=len(fresh.values)+len(fresh.category_components)
            continue
        known={candidate_signature(v):v for v in old.values+old.category_components}
        for key in ('values','category_components'):
            for v in getattr(fresh,key):
                sig=candidate_signature(v)
                if sig in known:
                    # Keep the old verdict, including negative semantic checks.
                    # A changed source binding or role yields a new signature.
                    continue
                getattr(old,key).append(v);known[sig]=v;added+=1
    return added


def run_verification_with_repairs(state, search, scope, verify, max_rounds=2, max_pages=8):
    """Initial verification plus at most N targeted repair rounds per document."""
    if not 0 <= max_rounds <= 3 or max_pages < 1:
        raise ValueError('repair rounds must be 0..3 and page budget must be positive')
    state.unresolved_issues=[]
    for doc in state.documents:
        _verify_document(state,verify,doc.doc_id)
        issues=collect_issues(state,doc.doc_id) if verify.evidence_grounding else []
        attempts=defaultdict(int)
        exhausted=set()
        for round_no in range(1,max_rounds+1):
            if not issues:
                break
            # Prefer pages with missing physical cells, then rejected-only pages.
            ranked=sorted({i['page'] for i in issues if i['page'] not in exhausted},key=lambda p:(attempts[p],
                          not any(i['page']==p and i['code']=='missing_or_rejected_source_cell' for i in issues),p))
            if not ranked:
                break
            selected=ranked[:max_pages]
            feedback={p:[i for i in issues if i['page']==p] for p in selected}
            before=copy.deepcopy([asdict(p) for p in state.parsed_pages.get(doc.doc_id,[]) if p.page in selected])
            candidates_before=[asdict(v) for d in state.org_disclosures if d.source_document==doc.doc_id
                               for v in d.values+d.category_components if v.evidence.page in selected]
            event=dict(document=doc.doc_id,round=round_no,pages=selected,issues_before=issues,
                       source_before=before,candidates_before=candidates_before,status='started')
            state.repair_history.append(event)
            started=time.time()
            try:
                evidence=search.refresh_evidence(doc,selected)
                fresh=scope.extract_disclosures(doc,evidence,feedback_by_page=feedback,
                                                extraction_round=round_no,append=False)
                event['fresh_candidates']=[asdict(v) for d in fresh for v in d.values+d.category_components]
                added=_merge(state,fresh)
                _verify_document(state,verify,doc.doc_id)
                remaining=collect_issues(state,doc.doc_id)
                after=[asdict(p) for p in state.parsed_pages.get(doc.doc_id,[]) if p.page in selected]
                issue_key=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False)
                before_keys={issue_key(i) for i in issues};after_keys={issue_key(i) for i in remaining}
                resolved=[i for i in issues if issue_key(i) not in after_keys]
                introduced=[i for i in remaining if issue_key(i) not in before_keys]
                event.update(resolved_issues=resolved,new_issues=introduced,
                             verified_new_candidates=sum(v.evidence.verified is True and v.evidence.extraction_round==round_no for d in state.org_disclosures if d.source_document==doc.doc_id for v in d.values+d.category_components))
                event.update(status='completed',new_candidates=added,source_after=after,issues_after=remaining,
                             elapsed_s=time.time()-started)
                for p in selected:attempts[p]+=1
                same_source=all(a['raw_text']==b['raw_text'] and a['table_structures']==b['table_structures']
                                for a,b in zip(before,after)) and len(before)==len(after)
                for p in selected:
                    oldp=next((x for x in before if x['page']==p),None)
                    newp=next((x for x in after if x['page']==p),None)
                    same=oldp is not None and newp is not None and oldp['raw_text']==newp['raw_text'] and oldp['table_structures']==newp['table_structures']
                    if same and not any(i['page']==p for i in resolved):
                        exhausted.add(p)
                event['exhausted_pages']=sorted(exhausted)
                # Stop unchanged requests instead of sampling until a pass.
                if same_source and (added==0 or not resolved) and all(i['page'] in selected for i in remaining):
                    event['stop_reason']='no new candidate or source evidence';issues=remaining;break
                issues=remaining
            except Exception as exc:
                event.update(status='failed',error=type(exc).__name__+': '+str(exc),elapsed_s=time.time()-started,issues_after=issues)
                state.unresolved_issues.extend(issues)
                raise
            finally:
                state.log_phase('orchestrator','repair',f'{doc.doc_id}: round {round_no}, pages {selected}, {event["status"]}')
        state.unresolved_issues.extend(issues)
    return state.repair_history
