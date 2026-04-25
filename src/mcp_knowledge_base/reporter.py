"""Fire-and-forget ingest reporter.

Used by *clients* of a knowledge service (typically a sibling build/control
MCP) to POST tool-call summaries to ``/ingest``. Mirrors the payload shape
the :class:`IngestRouter` accepts.

Every previous incarnation across the sibling repos was a copy-paste of the
same ``_report`` function with subtle drift (timeout 2 vs 5s, naive vs
tz-aware UTC, varying logger handling). This is the canonical version.

Usage::

    from mcp_knowledge_base import KnowledgeReporter

    reporter = KnowledgeReporter(service="mcp-build")  # URL from $KNOWLEDGE_URL

    @mcp.tool()
    def my_tool(x: int) -> str:
        result = do_work(x)
        reporter.report("my_tool", {"x": x}, result, success=True)
        return result

The ``url`` resolution order is:
    1. ``url=`` constructor argument, if given
    2. ``$KNOWLEDGE_URL`` env var
    3. ``http://localhost:5174/ingest`` (legacy default — override in production)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:5174/ingest"


class KnowledgeReporter:
    """Posts tool-call reports to a knowledge service's ``/ingest`` endpoint.

    Never raises — network errors are swallowed deliberately so that
    instrumentation cannot break the tool it instruments.

    :param service: The reporting service's name (e.g. ``"mcp-build"``).
        Stored verbatim in the payload's ``service`` field; downstream
        consumers use it to disambiguate sources.
    :param url: Full URL to POST to. If ``None``, falls back to
        ``$KNOWLEDGE_URL`` and then :data:`DEFAULT_URL`.
    :param timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        service: str,
        url: str | None = None,
        timeout: float = 5.0,
    ):
        self.service = service
        self.url = url or os.environ.get("KNOWLEDGE_URL", DEFAULT_URL)
        self.timeout = timeout

    def report(
        self,
        tool: str,
        args: dict[str, Any],
        result: str,
        success: bool,
    ) -> None:
        """Fire-and-forget POST to ``/ingest``. Never raises."""
        try:
            httpx.post(
                self.url,
                json={
                    "tool": tool,
                    "args": args,
                    "result": result,
                    "success": success,
                    "timestamp": datetime.now(timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "service": self.service,
                },
                timeout=self.timeout,
            )
        except Exception:
            pass
