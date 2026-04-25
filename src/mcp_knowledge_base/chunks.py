"""Chunk-handling primitives shared across knowledge services.

Domain-specific chunkers (Python AST, C# decompilation, DOS binary
disassembly, ...) live in their respective service repos. This module
provides only the helpers every domain needs:

    - tag_key / tag_flags     — boolean tag-key normalisation for Chroma
                                metadata `where` clauses
    - upsert_chunks           — Chroma upsert that auto-expands the
                                comma-joined ``tags`` field into individual
                                ``tag_<name>: True`` boolean keys
    - sanitize_for_id         — make a string safe to use in a Chroma id
    - now_iso                 — UTC timestamp in the format already used
                                across existing services
"""

from __future__ import annotations

import re
from datetime import datetime, timezone


_TAG_KEY_RE = re.compile(r"[^a-z0-9_]")


def now_iso() -> str:
    """UTC timestamp formatted ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tag_key(tag: str) -> str:
    """Normalise a tag into a Chroma metadata key.

    Example::

        tag_key("status-effect") == "tag_status_effect"

    ChromaDB's ``where`` filter has no ``$contains`` operator for metadata,
    so the only reliable way to filter by tag is to store each tag as its
    own boolean key.
    """
    return "tag_" + _TAG_KEY_RE.sub("_", tag.lower())


def tag_flags(tags: list[str]) -> dict:
    """Return ``{tag_<name>: True}`` for each non-empty tag in *tags*."""
    return {tag_key(t): True for t in tags if t}


def sanitize_for_id(s: str) -> str:
    """Make a string safe to use as a fragment of a ChromaDB id."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def upsert_chunks(collection, chunks: list[dict]) -> None:
    """Upsert *chunks* into *collection*, expanding tag flags first.

    Each chunk is a dict with keys ``id``, ``document``, ``metadata``. The
    comma-joined ``tags`` in metadata is expanded into individual
    ``tag_<name>: True`` boolean keys via :func:`tag_flags` so they can be
    filtered via ChromaDB's metadata ``where`` clause.
    """
    if not chunks:
        return
    for c in chunks:
        tag_str = c["metadata"].get("tags", "")
        tag_list = [t.strip() for t in tag_str.split(",") if t.strip()]
        c["metadata"].update(tag_flags(tag_list))
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
