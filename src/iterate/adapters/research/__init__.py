"""Literature sources for the Researcher: OpenAlex and arXiv, both keyless.

Two rules hold for everything here. Never raise: any failure returns an empty
list. Identifiers come from the source, never from a model: the Researcher may
only choose among fetched papers.
"""

from __future__ import annotations

from iterate.adapters.research.papers import (
    ArxivClient,
    OpenAlexClient,
    Paper,
    PaperSource,
    search_all,
)

__all__ = ["ArxivClient", "OpenAlexClient", "Paper", "PaperSource", "search_all"]
