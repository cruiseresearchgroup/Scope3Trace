# -*- coding: utf-8 -*-
"""Report search for the Search Agent ("Data Acquisition", paper §3.3.2).

Finds an organization's public sustainability / GHG-inventory report PDFs,
downloads them, checks that each file is that organization's report for the
year, and records a manifest so the run stays reproducible from a fixed
document list.

Backends (``SCOPE3TRACE_SEARCH_BACKEND``, default ``openrouter``):
  openrouter - an LLM with OpenRouter's web-search plugin issues the search;
               only URLs returned by the search tool (``url_citation``
               annotations) are used, never URLs written by the model.
               Key: OPENROUTER_API_KEY; model: SCOPE3TRACE_SEARCH_MODEL
               (default openai/gpt-4o-mini); engine: SCOPE3TRACE_SEARCH_ENGINE (default exa).
  google     - Google Custom Search JSON API. Keys: GOOGLE_API_KEY, GOOGLE_CSE_ID.

Ranking of candidates and the identity check of downloaded files are
deterministic (file type, organization and year in title / URL / first pages);
nothing is extracted here.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"
QUERY_TEMPLATES = (
    "{org} {year} sustainability report",
    "{org} {year} ESG report",
    "{org} {year} annual report greenhouse gas emissions scope 3",
    "{org} {year} GHG emissions inventory report",
)
_REPORT_WORDS = re.compile(r"(?i)sustainab|esg|climate|ghg|greenhouse|emission|carbon|annual[\s_-]*report|"
                           r"integrated[\s_-]*report|non[\s_-]*financial|csrd|esrs|responsib|impact")
_STOP = {"the", "and", "of", "inc", "ltd", "plc", "ag", "sa", "se", "nv", "group", "co", "corp",
         "corporation", "company", "limited", "holding", "holdings"}


class SearchError(RuntimeError):
    """The search API refused or failed the request (message carries no key)."""


@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    query: str = ""
    rank: int = 0
    mime: str = ""
    score: float = 0.0


@dataclass
class DownloadedReport:
    org: str
    year: Optional[int]
    url: str
    local_path: str
    title: str = ""
    query: str = ""
    score: float = 0.0
    sha256: str = ""
    bytes: int = 0
    notes: List[str] = field(default_factory=list)


class GoogleCustomSearch:
    """Minimal client of the Custom Search JSON API (fileType=pdf)."""

    def __init__(self, api_key: Optional[str] = None, cse_id: Optional[str] = None,
                 session=None, timeout: int = 30, pause_s: float = 1.0) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.cse_id = cse_id or os.environ.get("GOOGLE_CSE_ID", "")
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.timeout = timeout
        self.pause_s = pause_s
        self.calls = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.cse_id)

    def search(self, query: str, num: int = 10) -> List[SearchResult]:
        if not self.available:
            raise SearchError("GOOGLE_API_KEY and GOOGLE_CSE_ID are required for report search")
        params = {"key": self.api_key, "cx": self.cse_id, "q": query,
                  "num": max(1, min(int(num), 10)), "fileType": "pdf"}
        if self.calls and self.pause_s:
            time.sleep(self.pause_s)
        self.calls += 1
        resp = self.session.get(GOOGLE_CSE_URL, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            body = (getattr(resp, "text", "") or "").replace(self.api_key, "[REDACTED]")[:300]
            raise SearchError(f"Custom Search API HTTP {resp.status_code}: {body}")
        items = (resp.json() or {}).get("items") or []
        return [SearchResult(url=it.get("link", ""), title=it.get("title", ""), snippet=it.get("snippet", ""),
                             query=query, rank=i + 1, mime=it.get("mime", "") or it.get("fileFormat", ""))
                for i, it in enumerate(items) if it.get("link")]


class OpenRouterWebSearch:
    """Search through an LLM with OpenRouter's web-search plugin.

    The model decides how to search; the result list is the plugin's
    ``url_citation`` annotations (URL, title, page content), so a URL the
    model writes without the search tool returning it is never used."""

    SYSTEM = ("You find corporate report documents on the web. Use web search. Prefer the company's own "
              "website and direct PDF links. Answer with the report titles and their URLs.")
    USER = ("Find the PDF of the report in which {query}. List the matching documents "
            "(sustainability / ESG / climate / annual or integrated report) with their URLs.")

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 engine: Optional[str] = None, base_url: Optional[str] = None,
                 session=None, timeout: int = 120, pause_s: float = 0.0) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("SCOPE3TRACE_SEARCH_MODEL", "openai/gpt-4o-mini")
        self.engine = engine or os.environ.get("SCOPE3TRACE_SEARCH_ENGINE", "exa")
        base = base_url or os.environ.get("OPENROUTER_API_URL") or "https://openrouter.ai/api/v1"
        self.base_url = re.sub(r"/chat/completions/?$", "", base.strip().strip("\"'").rstrip("/"))
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.timeout = timeout
        self.pause_s = pause_s
        self.calls = 0
        self.tokens_in = self.tokens_out = 0
        self.last_answer = ""

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def default_queries(org: str, year: Optional[int]) -> List[str]:
        y = f" for {year}" if year else ""
        return [f"{org} reports its greenhouse gas emissions{y} (sustainability or ESG report)",
                f"{org} publishes its annual or integrated report{y}"]

    def search(self, query: str, num: int = 10) -> List[SearchResult]:
        if not self.available:
            raise SearchError("OPENROUTER_API_KEY is required for the openrouter search backend")
        if self.calls and self.pause_s:
            time.sleep(self.pause_s)
        self.calls += 1
        payload = {"model": self.model, "temperature": 0, "max_tokens": 800,
                   "messages": [{"role": "system", "content": self.SYSTEM},
                                {"role": "user", "content": self.USER.format(query=query)}],
                   "plugins": [{"id": "web", "engine": self.engine, "max_results": max(1, min(int(num), 10))}]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for k, h in (("OPENROUTER_HTTP_REFERER", "HTTP-Referer"), ("OPENROUTER_APP_TITLE", "X-Title")):
            if os.environ.get(k):
                headers[h] = os.environ[k]
        resp = self.session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers,
                                 timeout=self.timeout)
        if resp.status_code != 200:
            body = (getattr(resp, "text", "") or "").replace(self.api_key, "[REDACTED]")[:300]
            raise SearchError(f"OpenRouter web search HTTP {resp.status_code}: {body}")
        data = resp.json() or {}
        usage = data.get("usage") or {}
        self.tokens_in += int(usage.get("prompt_tokens") or 0)
        self.tokens_out += int(usage.get("completion_tokens") or 0)
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        self.last_answer = msg.get("content") or ""
        out: List[SearchResult] = []
        for ann in msg.get("annotations") or []:
            c = ann.get("url_citation") if isinstance(ann, dict) else None
            if not c or not c.get("url"):
                continue
            out.append(SearchResult(url=c["url"], title=c.get("title", "") or "",
                                    snippet=(c.get("content", "") or "")[:500], query=query, rank=len(out) + 1))
        return out


def make_searcher(backend: Optional[str] = None):
    backend = (backend or os.environ.get("SCOPE3TRACE_SEARCH_BACKEND") or "openrouter").lower()
    if backend == "google":
        return GoogleCustomSearch()
    if backend == "openrouter":
        return OpenRouterWebSearch()
    raise SearchError(f"unknown search backend {backend!r} (openrouter | google)")


def build_queries(org: str, year: Optional[int]) -> List[str]:
    y = str(year) if year else ""
    return [re.sub(r"\s+", " ", t.format(org=org, year=y)).strip() for t in QUERY_TEMPLATES]


# entity words that make a different legal entity ("Heineken Holding N.V." is
# not "Heineken N.V."); they count only when the searched name lacks them
_ENTITY_WORDS = ("holding", "holdings", "foundation", "stichting", "trust", "pension", "fund", "foundations")


def _other_entity(text: str, org: str) -> Optional[str]:
    """An entity word printed right after the organization's name (also inside
    a domain, "heinekenholding.com") that the searched name does not contain."""
    low = text.lower()
    org_low = org.lower()
    for tok in _org_tokens(org):
        for w in _ENTITY_WORDS:
            if w in org_low:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(tok)}[\s_.-]*{w}(?![a-z])", low):
                return f"{tok} {w}"
    return None


def _org_tokens(org: str) -> List[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", org.lower()) if t not in _STOP and len(t) > 1]
    return toks or re.findall(r"[a-z0-9]+", org.lower())


def score_result(r: SearchResult, org: str, year: Optional[int]) -> float:
    """Deterministic relevance of one search hit to (org, year)."""
    text = f"{r.title} {r.url} {r.snippet}".lower()
    host_path = (urlparse(r.url).netloc + urlparse(r.url).path).lower()
    toks = _org_tokens(org)
    s = 0.0
    if r.url.lower().split("?")[0].endswith(".pdf") or "pdf" in (r.mime or "").lower():
        s += 2.0
    s += 2.0 * sum(1 for t in toks if t in text) / max(len(toks), 1)
    if any(t in host_path for t in toks):
        s += 1.0  # the organization's own domain or file name
    if year:
        if str(year) in text:
            s += 2.0
        elif re.search(rf"{year - 1}\s?[/_-]\s?(?:{year}|{str(year)[2:]})", text):
            s += 1.5  # fiscal year "2023/24"
        if re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text) and str(year) not in text and str(year - 1) not in text:
            s -= 1.0  # another year's report
    if _REPORT_WORDS.search(text):
        s += 1.0
    if _other_entity(f"{r.title} {r.url}", org):
        s -= 10.0  # another legal entity's report: below any download threshold
    s -= 0.05 * (r.rank - 1)
    return round(s, 3)


def rank_results(results: Iterable[SearchResult], org: str, year: Optional[int]) -> List[SearchResult]:
    best: Dict[str, SearchResult] = {}
    for r in results:
        key = r.url.split("#")[0]
        r.score = score_result(r, org, year)
        if key not in best or r.score > best[key].score:
            best[key] = r
    return sorted(best.values(), key=lambda r: (-r.score, r.rank, r.url))


def download_pdf(url: str, dest_dir: Path, session=None, timeout: int = 60,
                 max_bytes: int = 150 * 2 ** 20) -> Optional[Path]:
    """Download one PDF; keeps it only when the bytes start with %PDF."""
    if session is None:
        import requests
        session = requests.Session()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = session.get(url, timeout=timeout, stream=True,
                           headers={"User-Agent": "Mozilla/5.0 (Scope3Trace report acquisition)"})
    except Exception:
        return None
    if getattr(resp, "status_code", 0) != 200:
        return None
    chunks, size = [], 0
    for chunk in resp.iter_content(chunk_size=1 << 16):
        if not chunk:
            continue
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            return None
    data = b"".join(chunks)
    if not data.startswith(b"%PDF"):
        return None
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(urlparse(url).path).name or "report.pdf")[-80:]
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    path = dest_dir / f"{hashlib.sha256(data).hexdigest()[:12]}_{name}"
    path.write_bytes(data)
    return path


_HREF = re.compile(r"""<a\b[^>]*?href\s*=\s*["']([^"'#]+)["'][^>]*>(.*?)</a>""", re.I | re.S)


def pdf_links(html: str, base_url: str) -> List[SearchResult]:
    """PDF links of an HTML page (a company's report / investor page)."""
    from urllib.parse import urljoin
    out, seen = [], set()
    for href, text in _HREF.findall(html or ""):
        url = urljoin(base_url, href.strip())
        if not re.search(r"(?i)\.pdf(?:$|[?#])", url) or url in seen:
            continue
        seen.add(url)
        label = re.sub(r"<[^>]+>|\s+", " ", text).strip()
        out.append(SearchResult(url=url, title=label, rank=len(out) + 1, mime="application/pdf"))
    return out


def fetch_report(url: str, dest_dir: Path, org: str, year: Optional[int], session=None,
                 max_links: int = 3, timeout: int = 60) -> Optional[Path]:
    """Download a PDF hit; for an HTML hit, follow its best-ranked report PDF links."""
    if session is None:
        import requests
        session = requests.Session()
    path = download_pdf(url, dest_dir, session=session, timeout=timeout)
    if path is not None or re.search(r"(?i)\.pdf(?:$|[?#])", url):
        return path
    try:
        resp = session.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (Scope3Trace report acquisition)"})
        html = resp.text if getattr(resp, "status_code", 0) == 200 else ""
    except Exception:
        html = ""
    for link in rank_results(pdf_links(html, url), org, year)[:max_links]:
        path = download_pdf(link.url, dest_dir, session=session, timeout=timeout)
        if path is not None:
            return path
    return None


def check_report_identity(path: Path, org: str, year: Optional[int], pages: int = 8) -> Tuple[bool, str]:
    """The first pages name the organization and the reporting year."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        page_texts = [(reader.pages[i].extract_text() or "") for i in range(min(pages, len(reader.pages)))]
        meta_title = str((reader.metadata or {}).get("/Title") or "")
    except Exception as e:
        return False, f"unreadable PDF ({type(e).__name__})"
    text = " ".join(page_texts)
    # another entity of the same name ("Heineken Holding N.V." for "Heineken N.V."):
    # its name in the PDF title, or dominating the first pages
    other = _other_entity(meta_title, org)
    if other:
        return False, f"PDF title names another entity ({other!r}): {meta_title[:80]}"
    head = re.sub(r"\s+", " ", " ".join(page_texts[:3]).lower())
    for tok in _org_tokens(org):
        for w in _ENTITY_WORDS:
            if w in org.lower():
                continue
            n_other = len(re.findall(rf"(?<![a-z0-9]){re.escape(tok)}[\s_.-]*{w}(?![a-z])", head))
            n_all = len(re.findall(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", head))
            if n_other >= 3 and n_other >= .5 * (n_all - n_other):
                return False, f"first pages are about another entity ({tok} {w}: {n_other} of {n_all} mentions)"
    if len(text.strip()) < 200:
        return True, "no text layer on the first pages; identity not checked"
    low = re.sub(r"\s+", " ", text.lower())
    toks = _org_tokens(org)
    found = [t for t in toks if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low)]
    if len(found) < max(1, (len(toks) + 1) // 2):
        return False, f"organization name not on the first {pages} pages"
    if year and not (re.search(rf"(?<!\d){year}(?!\d)", low)
                     or re.search(rf"{year - 1}\s?[/-]\s?(?:{year}|{str(year)[2:]})", low)):
        return False, f"year {year} not on the first {pages} pages"
    return True, "organization and year found on the first pages"


MANIFEST_FIELDS = ["org", "report_year", "pdf_url", "local_path", "title", "query", "score", "sha256", "bytes"]


def write_manifest(reports: List[DownloadedReport], path: Path) -> None:
    """A manifest readable by SearchAgent.acquire_from_manifest (appends)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        for r in reports:
            d = asdict(r)
            w.writerow({"org": d["org"], "report_year": d["year"] or "", "pdf_url": d["url"],
                        "local_path": d["local_path"], "title": d["title"], "query": d["query"],
                        "score": d["score"], "sha256": d["sha256"], "bytes": d["bytes"]})


def search_reports(org: str, year: Optional[int], download_dir: Path, max_reports: int = 1,
                   searcher=None,
                   downloader: Optional[Callable[[str, Path], Optional[Path]]] = None,
                   results_per_query: int = 10, min_score: float = 2.0,
                   log: Optional[Callable[[str], None]] = None,
                   identity_check: Optional[Callable[[Path, str, Optional[int]], Tuple[bool, str]]] = None,
                   max_downloads: int = 8) -> List[DownloadedReport]:
    """Search, rank, download and identity-check up to ``max_reports`` PDFs for one organization-year."""
    searcher = searcher or make_searcher()
    downloader = downloader or (lambda url, d: fetch_report(url, d, org, year))
    identity_check = identity_check or check_report_identity
    log = log or (lambda msg: None)
    hits: List[SearchResult] = []
    queries = searcher.default_queries(org, year) if hasattr(searcher, "default_queries") else build_queries(org, year)
    for q in queries:
        try:
            hits.extend(searcher.search(q, num=results_per_query))
        except SearchError as e:
            log(f"search failed for {q!r}: {e}")
            if not hits:
                raise
            break
    ranked = [r for r in rank_results(hits, org, year) if r.score >= min_score]
    log(f"{org} {year or ''}: {len(hits)} hits, {len(ranked)} candidate(s) above score {min_score}")
    out: List[DownloadedReport] = []
    seen_sha = set()
    tried = 0
    for r in ranked:
        if len(out) >= max_reports or tried >= max_downloads:
            break
        tried += 1
        path = downloader(r.url, Path(download_dir))
        if path is None:
            log(f"no PDF at {r.url}")
            continue
        data = Path(path).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        ok, note = identity_check(Path(path), org, year)
        log(f"{'accepted' if ok else 'rejected'} {r.url}: {note}")
        if not ok:
            continue
        out.append(DownloadedReport(org=org, year=year, url=r.url, local_path=str(path), title=r.title,
                                    query=r.query, score=r.score, sha256=sha, bytes=len(data), notes=[note]))
    return out
