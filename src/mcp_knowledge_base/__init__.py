"""mcp-knowledge-base: shared FastMCP + ChromaDB scaffolding for RAG MCP services.

Public API:

    KnowledgeService, ServiceConfig
        The main entry points. A ServiceConfig captures the per-domain
        wiring; a KnowledgeService assembles ChromaDB, FastMCP and a
        Starlette /ingest endpoint into one runnable unit.

    IngestRouter, make_ingest_endpoint
        Plug points for the domain's /ingest payload handling.

    KnowledgeReporter
        Client-side fire-and-forget POSTer for sending tool-call summaries
        from a sibling build/control MCP into /ingest.

    tag_key, tag_flags, upsert_chunks, sanitize_for_id, now_iso
        Chunk-handling primitives shared across every knowledge domain.

    format_query_results, format_get_results
        Standalone formatters for ChromaDB query/get outputs. The methods
        on KnowledgeService delegate to these.
"""

# Eager imports: light-weight modules used by both server and client sides.
# Heavy server-only modules (service, ingest) are lazy-loaded below so that
# reporter-only consumers don't have to install chromadb/fastmcp/starlette.
from .chunks import (
    now_iso,
    sanitize_for_id,
    tag_flags,
    tag_key,
    upsert_chunks,
)
from .format import format_get_results, format_query_results
from .reporter import KnowledgeReporter

__version__ = "0.2.1"


def __getattr__(name: str):
    # PEP 562 lazy attribute access — defers chromadb/starlette imports until
    # someone actually reaches for the server-side classes.
    if name in ("KnowledgeService", "ServiceConfig"):
        from .service import KnowledgeService, ServiceConfig

        return {"KnowledgeService": KnowledgeService, "ServiceConfig": ServiceConfig}[name]
    if name in ("IngestRouter", "make_ingest_endpoint"):
        from .ingest import IngestRouter, make_ingest_endpoint

        return {"IngestRouter": IngestRouter, "make_ingest_endpoint": make_ingest_endpoint}[name]
    raise AttributeError(f"module 'mcp_knowledge_base' has no attribute {name!r}")

__all__ = [
    "KnowledgeService",
    "ServiceConfig",
    "KnowledgeReporter",
    "IngestRouter",
    "make_ingest_endpoint",
    "tag_key",
    "tag_flags",
    "upsert_chunks",
    "sanitize_for_id",
    "now_iso",
    "format_query_results",
    "format_get_results",
]
