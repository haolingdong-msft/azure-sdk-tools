# Azure SDK QA Bot — Knowledge Wiki Sync

Builds a lightweight **wiki-tree** knowledge index over the markdown corpus the
[`azure-sdk-qa-bot-knowledge-sync`](../azure-sdk-qa-bot-knowledge-sync) project
maintains in blob storage, as a **lighter alternative to the GraphRAG
entity-graph build** ([`azure-sdk-qa-bot-knowledge-graph-sync`](../azure-sdk-qa-bot-knowledge-graph-sync)).

The bot's backend loads the published snapshot into a warm `WikiTreeService` and
serves it through the `search_wiki` agent tool, side-by-side with the existing
`search_knowledge_base` (KB) path.

## Why a wiki tree

GraphRAG rebuilds a global entity-cooccurrence graph over the corpus every run:
LLM entity + relationship extraction over every chunk, Leiden community
detection, and an LLM community report per community. That is expensive, slow,
and — because the corpus is already clean markdown with a header hierarchy —
largely re-derives structure the documents already have.

The wiki tree fuses two lighter ideas:

- **[PageIndex](https://github.com/VectifyAI/PageIndex)** — a hierarchical
  table-of-contents *tree* the agent navigates, built directly from the markdown
  `#` headers (no entity extraction, no chunking).
- **[WeKnora Wiki Mode](https://github.com/Tencent/WeKnora)** — LLM-distilled,
  interlinked wiki *pages* rolled up the tree, plus a lightweight `related`
  cross-link graph (in place of a heavy entity graph).

Every node stays **anchored to its source document** (`source` + `source_path` +
`header_path`) so each retrieval hit resolves the same link as the KB path.

| | GraphRAG (entity graph) | Wiki tree (this project) |
|---|---|---|
| Structure | fixed-token chunks; headers dropped then regex-recovered | native markdown ToC tree |
| Build LLM | entity+relationship extraction over every chunk + Leiden + per-community reports | per-node summary + per-topic roll-up page (bounded by node count) |
| Graph | entity co-occurrence (thousands of nodes) | page/section link graph (`related` edges) |
| Topic clustering / synthesis | Leiden + community reports | the tree hierarchy (free) + bottom-up roll-up (free, hierarchical) |
| Query runtime | load parquets + LocalSearch context builder + preload community embeddings | load `tree.json` + `embeddings.npy`; one query embedding + one matmul |

## Index model

A snapshot is three artefacts (immutable, timestamped, activated by a manifest
flip — the same pattern as the GraphRAG snapshot):

- `tree.json` — the wiki tree: `root → folder → doc → section` nodes, each with a
  navigation `summary`, a rolled-up `page`, source `section_text`, `related`
  cross-links, and a `source`/`rel_title` anchor.
- `embeddings.npy` — `float32 [N, dim]` node-embedding matrix.
- `embedding_ids.json` — node ids aligned to the matrix rows.
- `latest.json` — manifest pointing at the current `build_id`.

## Retrieval

One traversal, three moves (see [design](../docs/wiki_tree_retrieval_design.md)):

1. **Entry** — embed the query, rank nodes by cosine (one matmul), scoped to the
   tenant's source folders.
2. **Expansion** — from the top entry nodes take the section evidence and follow
   1 hop of `related` cross-links.
3. **Synthesis** — surface the most relevant ancestor-document overview `page`
   (the role GraphRAG's community report plays, pre-computed).

Results are returned relevance-ranked and capped, in the **same `Reference`
shape** as `search_knowledge_base` so the chat agent fuses both uniformly.

## Usage

### Local build (development / tests / offline eval)

```bash
pip install -e ".[dev,llm]"
python -m azure_sdk_qa_bot_knowledge_wiki_sync.main \
    --input ../azure-sdk-qa-bot-knowledge-sync/knowledge \
    --output .wiki-out --synth-mode extractive --embed-mode hashing
```

`--synth-mode`/`--embed-mode` `auto` uses Azure OpenAI when
`AZURE_OPENAI_ENDPOINT` is set, else the deterministic
Extractive/Hashing backends (no Azure, fully offline).

### Blob build + publish (production)

```bash
export STORAGE_BLOB_ENDPOINT="https://<account>.blob.core.windows.net"
export STORAGE_KNOWLEDGE_CONTAINER="knowledge"
export STORAGE_WIKI_OUTPUT_CONTAINER="wiki"
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export WIKI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
python -m azure_sdk_qa_bot_knowledge_wiki_sync.main --blob --synth-mode extractive --embed-mode llm
```

Reads the shared knowledge container, builds the tree, embeds nodes, and
publishes an immutable snapshot + `latest.json` to the wiki container. The
backend picks it up on its next manifest poll.

## Backend / agent integration

- Backend `WikiTreeService` (`azure-sdk-qa-bot-agent/utils/knowledge_wiki/`)
  warm-loads the snapshot and serves `POST /wiki/query`.
- Agent tool `search_wiki` (`azure-sdk-qa-bot-agent/tools/wiki_knowledge_tools.py`)
  posts to that endpoint, tenant-scoped, and fails soft.

Required App Configuration keys: `STORAGE_WIKI_OUTPUT_CONTAINER`,
`WIKI_EMBEDDING_DEPLOYMENT`, `WIKI_QUERY_URL`, `WIKI_QUERY_AUDIENCE` (+ optional
`WIKI_RELOAD_POLL_SECONDS`, `WIKI_EMBEDDING_ENDPOINT`).

## Tests

```bash
pip install -e ".[dev]"
PYTHONPATH=. python -m pytest tests -q
```

The tests are Azure-free (deterministic backends), covering ToC parsing, fence
handling, link/anchor encoding, scoped retrieval, and snapshot round-trip.
