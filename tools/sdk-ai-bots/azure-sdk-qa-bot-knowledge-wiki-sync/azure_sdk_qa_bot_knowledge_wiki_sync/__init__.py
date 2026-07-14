"""Azure SDK QA Bot — Knowledge Wiki Sync.

Builds a lightweight **wiki-tree** knowledge index over the markdown corpus the
``azure-sdk-qa-bot-knowledge-sync`` project maintains, as a lighter alternative
to the GraphRAG entity-graph build. The index fuses two ideas:

* **PageIndex** — a hierarchical table-of-contents *tree* the agent navigates by
  reasoning (structure the markdown already has, so no entity extraction).
* **WeKnora Wiki Mode** — LLM-distilled, interlinked wiki *pages* rolled up the
  tree, plus lightweight ``related`` cross-links (in place of a heavy
  entity-cooccurrence graph).

Every node stays anchored to its source document so retrieval hits resolve the
same links as the KB path.
"""

from __future__ import annotations

__version__ = "1.0.0"
