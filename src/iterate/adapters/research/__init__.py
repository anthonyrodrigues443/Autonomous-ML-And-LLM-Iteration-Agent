"""Literature sources for the Researcher specialist.

Two keyless, free APIs. Nothing here needs an account, so `pip install iterate-ai`
stays the only command a user types:

* **OpenAlex** — ~320M works across journals and conferences, returns DOIs and
  citation counts (a ranking signal the Researcher uses).
* **arXiv** — preprints, which is where recent ML technique work lands first.

Two rules hold for everything in this package:

1. **Never raise.** No network, a slow host, a changed schema, a rate limit — all
   of it returns an empty list. A run without literature grounding is a fine run;
   a run that crashed because arxiv.org was down is not. Same degradation contract
   as the Summarizer's digest.
2. **Identifiers come from the source, never from a model.** A `Paper.identifier`
   is a DOI or arXiv ID the API returned. The Researcher's LLM may only choose
   among fetched papers; it can never author a citation string. A fabricated
   citation in a tool that advertises "literature-aware" is the worst bug this
   project could ship, so the guarantee is structural rather than prompted.
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
