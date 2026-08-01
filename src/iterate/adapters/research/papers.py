"""OpenAlex + arXiv clients, disk-cached and failure-tolerant."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote_plus

import httpx

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

log = logging.getLogger(__name__)

# OpenAlex asks callers to identify themselves for its faster "polite pool". It is
# courtesy, not authentication — no account exists to create. We send the project
# URL rather than a user's address so nothing personal leaves the machine.
_POLITE_ID = "https://github.com/anthonyrodrigues443/Autonomous-ML-And-LLM-Iteration-Agent"
# arXiv's terms ask for no more than one request every three seconds.
_ARXIV_MIN_INTERVAL = 3.0
_ABSTRACT_CAP = 1200
_DEFAULT_TIMEOUT = 20.0


@dataclass(frozen=True)
class Paper:
    """One retrieved work. ``identifier`` is the verifiable citation — a DOI or an
    arXiv ID that the SOURCE returned, never a string a model produced."""

    title: str
    identifier: str
    abstract: str
    year: int | None = None
    cited_by: int = 0
    source: str = ""

    def brief(self) -> str:
        """One compact line for the Researcher's prompt."""
        bits = [self.title.strip()]
        if self.year:
            bits.append(f"({self.year})")
        if self.cited_by:
            bits.append(f"[{self.cited_by} citations]")
        bits.append(f"<{self.identifier}>")
        return " ".join(bits)


class PaperSource(Protocol):
    name: str

    def search(self, query: str, *, limit: int = 5) -> list[Paper]: ...


class _Cache:
    """Query-keyed JSON cache on disk. Survives across runs, because the same data
    profile asks the same questions and re-fetching costs seconds for nothing."""

    def __init__(self, directory: Path | None) -> None:
        self._dir = directory

    def _path(self, source: str, query: str, limit: int) -> Path | None:
        if self._dir is None:
            return None
        key = hashlib.sha256(f"{source}|{limit}|{query}".encode()).hexdigest()[:20]
        return self._dir / f"{source}-{key}.json"

    def get(self, source: str, query: str, limit: int) -> list[Paper] | None:
        path = self._path(source, query, limit)
        if path is None or not path.exists():
            return None
        try:
            rows = json.loads(path.read_text())
            return [Paper(**row) for row in rows]
        except Exception:  # a corrupt or stale-schema entry is a cache miss
            return None

    def put(self, source: str, query: str, limit: int, papers: Sequence[Paper]) -> None:
        path = self._path(source, query, limit)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([p.__dict__ for p in papers]))
        except OSError:  # a read-only or full disk must not sink a run
            log.debug("research cache write failed for %s", path)


def _clip(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= _ABSTRACT_CAP else text[:_ABSTRACT_CAP] + "…"


class OpenAlexClient:
    """~320M works, keyless. Returns DOIs and citation counts."""

    name = "openalex"

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        client: Any | None = None,
    ) -> None:
        self._cache = _Cache(cache_dir)
        self._timeout = timeout
        self._client = client

    def search(self, query: str, *, limit: int = 5) -> list[Paper]:
        cached = self._cache.get(self.name, query, limit)
        if cached is not None:
            return cached
        url = (
            "https://api.openalex.org/works"
            f"?search={quote_plus(query)}&per-page={limit}"
            f"&select=title,doi,publication_year,cited_by_count,abstract_inverted_index"
            f"&mailto={quote_plus(_POLITE_ID)}"
        )
        try:
            payload = _get_json(url, timeout=self._timeout, client=self._client)
            papers = [p for p in (self._parse(w) for w in payload.get("results", [])) if p]
        except Exception as exc:
            log.debug("openalex search failed (%s): %s", type(exc).__name__, exc)
            return []
        self._cache.put(self.name, query, limit, papers)
        return papers

    def _parse(self, work: dict[str, Any]) -> Paper | None:
        title = (work.get("title") or "").strip()
        doi = work.get("doi") or ""
        if not title or not doi:
            # No verifiable identifier means it cannot be cited, so it is not useful.
            return None
        return Paper(
            title=title,
            identifier=doi.replace("https://doi.org/", "doi:"),
            abstract=_clip(_deinvert(work.get("abstract_inverted_index"))),
            year=work.get("publication_year"),
            cited_by=int(work.get("cited_by_count") or 0),
            source=self.name,
        )


def _deinvert(index: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as an inverted index (word -> positions) for
    copyright reasons. Rebuild the text by placing each word at its positions."""
    if not index:
        return ""
    positions: list[tuple[int, str]] = [
        (pos, word) for word, spots in index.items() for pos in spots
    ]
    positions.sort()
    return " ".join(word for _, word in positions)


class ArxivClient:
    """Preprints, keyless. Where recent ML technique work appears first."""

    name = "arxiv"
    _last_request: float = 0.0

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        client: Any | None = None,
        min_interval: float = _ARXIV_MIN_INTERVAL,
    ) -> None:
        self._cache = _Cache(cache_dir)
        self._timeout = timeout
        self._client = client
        self._min_interval = min_interval

    def search(self, query: str, *, limit: int = 5) -> list[Paper]:
        cached = self._cache.get(self.name, query, limit)
        if cached is not None:
            return cached
        self._be_polite()
        url = (
            # https directly: the http endpoint 301-redirects, costing a round trip
            # on every uncached query.
            "https://export.arxiv.org/api/query"
            f"?search_query=all:{quote_plus(query)}"
            f"&start=0&max_results={limit}&sortBy=relevance"
        )
        try:
            text = _get_text(url, timeout=self._timeout, client=self._client)
            papers = [p for p in (self._parse(e) for e in _entries(text)) if p]
        except Exception as exc:
            log.debug("arxiv search failed (%s): %s", type(exc).__name__, exc)
            return []
        self._cache.put(self.name, query, limit, papers)
        return papers

    def _be_polite(self) -> None:
        """arXiv's terms ask for at most one request every three seconds. Cached
        queries never reach here, so a re-run pays nothing."""
        elapsed = time.monotonic() - ArxivClient._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        ArxivClient._last_request = time.monotonic()

    def _parse(self, entry: ET.Element) -> Paper | None:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        title_el = entry.find("a:title", ns)
        id_el = entry.find("a:id", ns)
        if title_el is None or id_el is None or not (title_el.text or "").strip():
            return None
        raw_id = (id_el.text or "").strip()
        published = entry.find("a:published", ns)
        summary = entry.find("a:summary", ns)
        year: int | None = None
        if published is not None and published.text:
            with_suppress = published.text[:4]
            year = int(with_suppress) if with_suppress.isdigit() else None
        return Paper(
            title=" ".join((title_el.text or "").split()),
            identifier="arXiv:" + raw_id.rsplit("/abs/", 1)[-1],
            abstract=_clip((summary.text or "") if summary is not None else ""),
            year=year,
            source=self.name,
        )


def _entries(xml_text: str) -> list[ET.Element]:
    root = ET.fromstring(xml_text)
    return root.findall("{http://www.w3.org/2005/Atom}entry")


def _get_json(url: str, *, timeout: float, client: Any | None) -> dict[str, Any]:
    if client is not None:
        return dict(client.get(url, timeout=timeout).json())
    with httpx.Client(timeout=timeout, follow_redirects=True) as http:
        response = http.get(url)
        response.raise_for_status()
        return dict(response.json())


def _get_text(url: str, *, timeout: float, client: Any | None) -> str:
    if client is not None:
        return str(client.get(url, timeout=timeout).text)
    with httpx.Client(timeout=timeout, follow_redirects=True) as http:
        response = http.get(url)
        response.raise_for_status()
        return response.text


def search_all(
    sources: Sequence[PaperSource], query: str, *, limit: int = 5
) -> list[Paper]:
    """Every source, deduped on identifier, best-cited first.

    Ordering puts well-cited work above novelty on purpose: the Researcher is
    picking a technique to spend a real experiment on, and a 1500-citation method
    is a better bet for a floor model than last week's preprint. arXiv results
    carry no citation count, so they sort under OpenAlex's and act as the recency
    tail rather than the headline.
    """
    seen: set[str] = set()
    found: list[Paper] = []
    for source in sources:
        # Guarded per source, not per call: the built-in clients already swallow
        # their own network errors, but this is the aggregation boundary and the
        # never-raises contract has to hold for ANY source, including ones added
        # later. One dead source degrades to the others' results, not to a dead run.
        try:
            results = source.search(query, limit=limit)
        except Exception as exc:
            log.info(
                "research source %r failed (%s: %s)", getattr(source, "name", source), type(exc).__name__, exc
            )
            continue
        for paper in results:
            if paper.identifier in seen:
                continue
            seen.add(paper.identifier)
            found.append(paper)
    return sorted(found, key=lambda p: (-p.cited_by, -(p.year or 0)))
