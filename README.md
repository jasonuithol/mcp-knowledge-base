# mcp-knowledge-base

Shared FastMCP + ChromaDB scaffolding for RAG-backed knowledge MCP services.

Extracted from three legacy sibling projects (`claude-sandbox`,
`claude-pygame`, `claude-dosre` — all archived) which had each grown a
near-identical `mcp-knowledge` service. ~80% of `mcp-service.py` and the
tag/upsert helpers in `chunker.py` were mechanical duplication. This
package owns the shared parts so each domain (now `mcp-pygame`,
`mcp-valheim`, `mcp-dosre`) only carries genuinely domain-specific code:
its chunker, its tag taxonomy, and any bespoke MCP tools.

## What's in here

| Module | Provides |
|---|---|
| `service.py` | `KnowledgeService`, `ServiceConfig` — assembles ChromaDB + FastMCP + Starlette `/ingest` into one runnable unit |
| `chunks.py` | `tag_key`, `tag_flags`, `upsert_chunks`, `sanitize_for_id`, `now_iso` |
| `format.py` | `format_query_results`, `format_get_results` — header-key-driven, with a 1500-char doc display cap |
| `ingest.py` | `IngestRouter` ABC + `make_ingest_endpoint` factory |

Default MCP tools registered via `KnowledgeService.register_default_tools()`:

- `ask(question)` — top-5 semantic search
- `ask_tagged(question, tags)` — semantic search filtered by `tag_<name>` boolean keys
- `list_sources()` — every indexed source with chunk count
- `forget(source)` — prefix-match deletion (`source` or `source/...`)
- `stats()` — totals + breakdowns by source, type/kind, and tag

`retag_all` is registered separately via
`register_retag_all(pattern_tags, detect_tags)` because both arguments are
domain-specific.

## What's not in here

- `chunk_docs` and `seed_docs` — markdown chunking is generic, but the
  metadata schema isn't (each domain has different mandatory fields:
  `module`/`class_name`/`func_name` for pygame, `class_name`/`method_name`
  for valheim, `kind`/`md5`/`offset`/`length` for dosre). Keep these
  in your domain repo.
- Domain-specific chunkers (Python AST, C# decompilation, DOS
  disassembly).
- Domain-specific MCP tools (`ask_module`, `ask_class`, `ask_file`,
  `ask_offset`, `ask_project`, `seed_*`, `forget_md5`, ...).

## Install

From git, pinned to a tag or sha (recommended for Dockerfiles):

```bash
pip install "mcp-knowledge-base @ git+https://github.com/jasonuithol/mcp-knowledge-base.git@v0.1.0"
```

For local dev, editable install from a sibling clone:

```bash
pip install -e ~/Projects/mcp-knowledge-base
```

## Minimal example

```python
from mcp_knowledge_base import KnowledgeService, ServiceConfig
from .ingest_router import MyDomainRouter
from .chunker import chunk_my_source
from .extractors import PATTERN_TAGS, detect_tags

svc = KnowledgeService(ServiceConfig(
    name="my-knowledge",
    collection_name="my_knowledge",
    port=5174,
    header_keys=["project", "module", "class_name"],
))
svc.register_default_tools()
svc.register_retag_all(PATTERN_TAGS, detect_tags)
svc.set_ingest_router(MyDomainRouter(svc.collection))

@svc.tool()
def ask_module(module: str) -> str:
    """Domain-specific tool — registered after the defaults."""
    results = svc.collection.query(
        query_texts=[module], n_results=10, where={"module": module},
    )
    return svc.format_query(results)

if __name__ == "__main__":
    svc.run()
```

## Tests

The tests cover the pure-Python primitives (`tag_key`, `tag_flags`,
`sanitize_for_id`, `now_iso`, both formatters). They deliberately skip
ChromaDB and FastMCP — those are heavyweight and belong in integration
tests inside each domain service.

```bash
pip install -e ".[test]"
pytest
```

## Versioning

Semantic. Breaking changes to the public API in `__init__.py` bump the
major; new tools or config fields bump the minor; bug fixes the patch.
Pin from Dockerfiles to a tag, not `main`.
