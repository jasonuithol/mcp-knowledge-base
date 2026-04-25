"""HTTP /ingest endpoint scaffolding.

Each domain implements an :class:`IngestRouter` subclass that knows how to
turn an inbound JSON payload from its build/control sibling into chunks.
The KnowledgeService mounts a Starlette POST handler at ``/ingest`` that
delegates to this router.

Payload shape (the contract every router must accept)::

    {
        "tool": "<tool-name>",     # required, validated before route() is called
        ...                        # arbitrary tool-specific fields
    }

Router return shape::

    {
        "action": "<short-string>",   # what was done (e.g. "indexed", "skipped")
        "chunks": <int>,              # number of chunks upserted (0 if skipped)
        ...                           # router-specific extras are allowed
    }
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from starlette.requests import Request
from starlette.responses import JSONResponse


class IngestRouter(ABC):
    """Routes inbound ``/ingest`` payloads to chunk-producing handlers.

    Subclasses typically take the Chroma collection in ``__init__`` and
    dispatch on ``payload["tool"]`` inside :meth:`route`.
    """

    @abstractmethod
    def route(self, payload: dict) -> dict:
        """Process one /ingest payload.

        Returns ``{"action": str, "chunks": int, ...}``. The caller has
        already verified that ``payload["tool"]`` is non-empty.
        """


def make_ingest_endpoint(router: IngestRouter, logger: logging.Logger):
    """Return a Starlette async endpoint backed by *router*.

    The endpoint:
      - 400s on invalid JSON or missing ``tool`` field
      - 500s on uncaught router exceptions (with the message in the body)
      - 200s with the router's dict on success, also logging the action
    """

    async def ingest_endpoint(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        tool = payload.get("tool")
        if not tool:
            return JSONResponse({"error": "missing 'tool' field"}, status_code=400)

        try:
            result = router.route(payload)
            logger.info(
                "Ingested %s -> %s (%d chunks)",
                tool,
                result.get("action", "?"),
                result.get("chunks", 0),
            )
            return JSONResponse(result, status_code=200)
        except Exception as e:
            logger.exception("Ingest error for tool=%s", tool)
            return JSONResponse({"error": str(e)}, status_code=500)

    return ingest_endpoint
