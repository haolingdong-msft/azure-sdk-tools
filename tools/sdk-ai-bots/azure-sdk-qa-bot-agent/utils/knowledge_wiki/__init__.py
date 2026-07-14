"""Wiki-tree retrieval package (lightweight PageIndex + WeKnora alternative).

Public surface:

* :class:`WikiTreeService` / :func:`get_wiki_service` — the warm singleton the
  backend ``/wiki/query`` endpoint drives.
"""

from __future__ import annotations

from utils.knowledge_wiki.service import WikiTreeService, get_wiki_service

__all__ = ["WikiTreeService", "get_wiki_service"]
