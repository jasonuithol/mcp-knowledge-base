"""KnowledgeService: the assembly point.

A KnowledgeService wraps:
  - a ChromaDB persistent collection
  - a FastMCP server (with optional default tools)
  - a Starlette ASGI app that mounts the FastMCP HTTP transport at /mcp
    and an optional /ingest endpoint backed by an IngestRouter

Domain-side usage (sketch)::

    from mcp_knowledge_base import KnowledgeService, ServiceConfig
    from .ingest import MyDomainRouter

    svc = KnowledgeService(ServiceConfig(
        name="pygame-knowledge",
        collection_name="pygame_knowledge",
        port=5174,
        header_keys=["project", "module", "class_name", "func_name"],
    ))
    svc.register_default_tools()
    svc.register_retag_all(pattern_tags=PATTERN_TAGS, detect_tags=detect_tags)
    svc.set_ingest_router(MyDomainRouter(svc.collection))

    @svc.tool()
    def ask_module(module: str) -> str:
        results = svc.collection.query(
            query_texts=[module], n_results=10, where={"module": module},
        )
        return svc.format_query(results)

    if __name__ == "__main__":
        svc.run()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import chromadb
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from .chunks import tag_flags, tag_key, upsert_chunks as _upsert_chunks
from .format import format_get_results, format_query_results
from .ingest import IngestRouter, make_ingest_endpoint


@dataclass
class ServiceConfig:
    """Configuration for a :class:`KnowledgeService`.

    Attributes:
        name: FastMCP server name (e.g. ``"pygame-knowledge"``). Surfaced
            to MCP clients.
        collection_name: ChromaDB collection name. Persistent across runs.
        port: HTTP listen port.
        header_keys: Metadata keys surfaced in result-formatter headers,
            e.g. ``["project", "module", "class_name", "func_name"]``.
            Empty/missing values are skipped at format time.
        knowledge_dir: ChromaDB persistence directory. If ``None``, falls
            back to the ``KNOWLEDGE_DIR`` env var, then ``/opt/knowledge``.
        instructions: Optional MCP server instructions string. Some clients
            display this when listing the server.
        chroma_metadata: Extra metadata to pass to
            ``get_or_create_collection``. The default sets the HNSW space
            to cosine, matching every existing service.
    """

    name: str
    collection_name: str
    port: int = 5174
    header_keys: list[str] = field(default_factory=list)
    knowledge_dir: str | None = None
    instructions: str | None = None
    chroma_metadata: dict[str, Any] = field(
        default_factory=lambda: {"hnsw:space": "cosine"}
    )


class KnowledgeService:
    """ChromaDB + FastMCP + /ingest, packaged as one runnable unit."""

    #: Names of the tools registered by :meth:`register_default_tools` when
    #: no explicit include/exclude is given.
    DEFAULT_TOOLS: tuple[str, ...] = (
        "ask",
        "ask_tagged",
        "list_sources",
        "forget",
        "stats",
    )

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = logging.getLogger(config.name)

        knowledge_dir = config.knowledge_dir or os.environ.get(
            "KNOWLEDGE_DIR", "/opt/knowledge"
        )
        self.client = chromadb.PersistentClient(path=knowledge_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata=config.chroma_metadata,
        )

        mcp_kwargs: dict[str, Any] = {}
        if config.instructions:
            mcp_kwargs["instructions"] = config.instructions
        self.mcp = FastMCP(config.name, **mcp_kwargs)

        self._router: IngestRouter | None = None

    # ------------------------------------------------------------------
    # Public helpers exposed to domain code
    # ------------------------------------------------------------------

    def tool(self, *args, **kwargs):
        """Decorator alias for ``self.mcp.tool``."""
        return self.mcp.tool(*args, **kwargs)

    def format_query(self, results: dict) -> str:
        """Format a ``collection.query()`` result using the configured header keys."""
        return format_query_results(results, self.config.header_keys)

    def format_get(self, results: dict) -> str:
        """Format a ``collection.get()`` result using the configured header keys."""
        return format_get_results(results, self.config.header_keys)

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """Upsert chunks into the collection (auto-expanding tag flags)."""
        _upsert_chunks(self.collection, chunks)

    def set_ingest_router(self, router: IngestRouter) -> None:
        """Attach an :class:`IngestRouter`. /ingest is only mounted if set."""
        self._router = router

    # ------------------------------------------------------------------
    # Default tool registration
    # ------------------------------------------------------------------

    def register_default_tools(
        self,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> None:
        """Register the generic MCP query/maintenance tools.

        Tools registered (when included):
          - ``ask(question)`` — top-5 semantic search
          - ``ask_tagged(question, tags)`` — semantic search filtered by tags
          - ``list_sources()`` — all indexed sources with chunk counts
          - ``forget(source)`` — delete chunks where ``source`` equals or
            prefixes the given value (e.g. ``forget("py-source/MyProj")``
            removes everything under that prefix)
          - ``stats()`` — totals, source breakdown, type/kind breakdown,
            top tags

        :param include: If given, register only these tools.
        :param exclude: If given, skip these tools. Useful when a domain
            wants to provide its own variant of one of the defaults.
        """
        wanted = set(include) if include else set(self.DEFAULT_TOOLS)
        if exclude:
            wanted -= set(exclude)

        if "ask" in wanted:
            self._register_ask()
        if "ask_tagged" in wanted:
            self._register_ask_tagged()
        if "list_sources" in wanted:
            self._register_list_sources()
        if "forget" in wanted:
            self._register_forget()
        if "stats" in wanted:
            self._register_stats()

    def register_retag_all(
        self,
        pattern_tags: list[tuple[Any, str]],
        detect_tags: Callable[[str], list[str]],
    ) -> None:
        """Register the ``retag_all`` MCP tool.

        Walks every chunk, drops content-derived tags (those whose names
        appear in *pattern_tags*), and re-detects them by running
        *detect_tags* over the chunk's document body. Provenance tags
        (project names, ``mod-source``, ``test-failure`` etc.) are
        preserved.

        Kept separate from :meth:`register_default_tools` because the two
        callbacks are domain-specific and have no sensible default.
        """
        collection = self.collection

        @self.tool()
        def retag_all() -> str:
            """Re-run tag auto-detection across every chunk's document.

            Use after tightening or extending the tag-detection regexes.
            Content-derived tags are re-detected; provenance tags are
            preserved.
            """
            content_tag_set = {name for _, name in pattern_tags}
            content_tag_keys = {tag_key(t) for t in content_tag_set}

            existing = collection.get(include=["metadatas", "documents"])
            ids = existing["ids"]
            if not ids:
                return "Collection is empty."

            changed = 0
            BATCH = 5000
            for i in range(0, len(ids), BATCH):
                batch_ids = ids[i : i + BATCH]
                batch_metas = existing["metadatas"][i : i + BATCH]
                batch_docs = existing["documents"][i : i + BATCH]

                new_metas = []
                for meta, doc in zip(batch_metas, batch_docs):
                    old_tags_str = meta.get("tags", "")
                    old_tags = [t.strip() for t in old_tags_str.split(",") if t.strip()]

                    provenance = [t for t in old_tags if t not in content_tag_set]
                    redetected = detect_tags(doc or "")
                    new_tags = provenance + [t for t in redetected if t not in provenance]

                    new_meta = {k: v for k, v in meta.items() if k not in content_tag_keys}
                    new_meta["tags"] = ",".join(new_tags)
                    new_meta.update(tag_flags(new_tags))

                    new_metas.append(new_meta)
                    if new_tags != old_tags:
                        changed += 1

                collection.upsert(
                    ids=batch_ids, documents=batch_docs, metadatas=new_metas
                )

            return f"Retagged {len(ids)} chunks; {changed} had tag changes."

    # ------------------------------------------------------------------
    # Default tool implementations (closures over self.collection)
    # ------------------------------------------------------------------

    def _register_ask(self) -> None:
        collection = self.collection
        format_query = self.format_query

        @self.tool()
        def ask(question: str) -> str:
            """Semantic search — returns the top 5 most relevant knowledge chunks."""
            results = collection.query(query_texts=[question], n_results=5)
            return format_query(results)

    def _register_ask_tagged(self) -> None:
        collection = self.collection
        format_query = self.format_query

        @self.tool()
        def ask_tagged(question: str, tags: list[str]) -> str:
            """Filtered semantic search — restrict to chunks carrying every given tag.

            Tags are stored as boolean ``tag_<name>: True`` metadata keys
            because ChromaDB's ``where`` filter has no substring operator.
            """
            keys = [tag_key(t) for t in tags if t]
            if not keys:
                where = None
            elif len(keys) == 1:
                where = {keys[0]: True}
            else:
                where = {"$and": [{k: True} for k in keys]}
            results = collection.query(
                query_texts=[question], n_results=5, where=where
            )
            return format_query(results)

    def _register_list_sources(self) -> None:
        collection = self.collection

        @self.tool()
        def list_sources() -> str:
            """List all indexed sources with chunk counts."""
            all_meta = collection.get(include=["metadatas"])
            sources: dict[str, int] = {}
            for meta in all_meta["metadatas"]:
                src = meta.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            if not sources:
                return "No sources indexed yet."
            lines = [
                f"  {src}: {count} chunks" for src, count in sorted(sources.items())
            ]
            return f"Indexed sources ({len(sources)}):\n" + "\n".join(lines)

    def _register_forget(self) -> None:
        collection = self.collection

        @self.tool()
        def forget(source: str) -> str:
            """Remove chunks whose ``source`` metadata equals or prefixes *source*.

            Prefix-matching uses ``source + "/"`` as the boundary, so
            ``forget("py-source/Foo")`` will delete ``py-source/Foo`` and
            ``py-source/Foo/bar`` but not ``py-source/Foobar``.
            """
            results = collection.get(include=["metadatas"])
            ids = [
                id_
                for id_, meta in zip(results["ids"], results["metadatas"])
                if meta.get("source", "") == source
                or meta.get("source", "").startswith(source + "/")
            ]
            if not ids:
                return f"No chunks found matching source: {source}"
            collection.delete(ids=ids)
            return f"Deleted {len(ids)} chunks matching source: {source}"

    def _register_stats(self) -> None:
        collection = self.collection

        @self.tool()
        def stats() -> str:
            """Collection size + breakdown by source, type/kind, and tag.

            Both ``type`` and ``kind`` metadata fields are checked to
            accommodate domains that use one or the other.
            """
            count = collection.count()
            if count == 0:
                return "Knowledge base is empty."

            all_meta = collection.get(include=["metadatas"])
            sources: dict[str, int] = {}
            types: dict[str, int] = {}
            tags_count: dict[str, int] = {}

            for meta in all_meta["metadatas"]:
                src = meta.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

                t = meta.get("type") or meta.get("kind") or "unknown"
                types[t] = types.get(t, 0) + 1

                for tag in meta.get("tags", "").split(","):
                    tag = tag.strip()
                    if tag:
                        tags_count[tag] = tags_count.get(tag, 0) + 1

            lines = [f"Total chunks: {count}", "", f"Top sources ({len(sources)}):"]
            for src, c in sorted(sources.items(), key=lambda x: -x[1])[:20]:
                lines.append(f"  {src}: {c}")
            lines.append("\nTypes:")
            for t, c in sorted(types.items(), key=lambda x: -x[1]):
                lines.append(f"  {t}: {c}")
            lines.append("\nTop tags:")
            for tag, c in sorted(tags_count.items(), key=lambda x: -x[1])[:20]:
                lines.append(f"  {tag}: {c}")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # ASGI app build & launch
    # ------------------------------------------------------------------

    def build_app(self) -> Starlette:
        """Build the Starlette ASGI app combining the FastMCP /mcp transport
        with /ingest (if a router has been attached).
        """
        mcp_app = self.mcp.http_app("/mcp")
        routes: list[Any] = []
        if self._router is not None:
            routes.append(
                Route(
                    "/ingest",
                    make_ingest_endpoint(self._router, self.logger),
                    methods=["POST"],
                )
            )
        routes.append(Mount("/", app=mcp_app))
        return Starlette(routes=routes, lifespan=mcp_app.lifespan)

    def run(self, host: str = "0.0.0.0") -> None:
        """Launch the service via uvicorn (blocking)."""
        import uvicorn

        app = self.build_app()
        self.logger.info(
            "Starting %s on port %d (collection=%s)",
            self.config.name,
            self.config.port,
            self.config.collection_name,
        )
        uvicorn.run(app, host=host, port=self.config.port, log_level="info")
