"""mcp-knowledge-base: shared FastMCP + ChromaDB scaffolding for RAG MCP services.

Public API:

    KnowledgeService, ServiceConfig
        The main entry points. A ServiceConfig captures the per-domain
        wiring; a KnowledgeService assembles ChromaDB, FastMCP and a
        Starlette /ingest endpoint into one runnable unit.

    IngestRouter, make_ingest_endpoint
        Plug points for the domain's /ingest payload handling.

    tag_key, tag_flags, upsert_chunks, sanitize_for_id, now_iso
        Chunk-handling primitives shared across every knowledge domain.

    format_query_results, format_get_results
        Standalone formatters for ChromaDB query/get outputs. The methods
        on KnowledgeService delegate to these.
"""

from .chunks import (
    now_iso,
    sanitize_for_id,
    tag_flags,
    tag_key,
    upsert_chunks,
)
from .format import format_get_results, format_query_results
from .ingest import IngestRouter, make_ingest_endpoint
from .service import KnowledgeService, ServiceConfig

__version__ = "0.1.0"

__all__ = [
    "KnowledgeService",
    "ServiceConfig",
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
