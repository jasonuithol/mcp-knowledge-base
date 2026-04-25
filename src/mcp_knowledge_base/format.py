"""Result formatting for ChromaDB ``query()`` and ``get()`` outputs.

These two formatters produce the same broad shape used by every existing
knowledge service::

    [n] <source> | key1=val1 | key2=val2 | similarity=0.42 | tags=[a,b,c]
    <document, capped at 1500 chars>
    <blank line>

What goes between the source/id and the trailing ``similarity``/``tags`` is
controlled by ``header_keys`` — a list of metadata keys the caller wants
surfaced in the header. Empty/missing values are skipped.
"""

from __future__ import annotations


DOC_DISPLAY_CAP = 1500


def _build_header(prefix: str, meta: dict, header_keys: list[str], extras: list[str]) -> str:
    parts = [prefix]
    for key in header_keys:
        val = meta.get(key, "")
        if val:
            parts.append(f"{key}={val}")
    parts.extend(extras)
    tags = meta.get("tags", "")
    if tags:
        parts.append(f"tags=[{tags}]")
    return " | ".join(parts)


def format_query_results(results: dict, header_keys: list[str]) -> str:
    """Format ``collection.query(...)`` output.

    Each hit's header includes:
      - ``[n] <source>`` — index and source metadata
      - one ``key=value`` per non-empty value in ``header_keys``
      - ``similarity=<x.xx>`` — derived from cosine distance (1 - dist)
      - ``tags=[...]`` — if a non-empty tags string is present
    """
    if not results.get("ids") or not results["ids"][0]:
        return "No results found."

    lines: list[str] = []
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        source = meta.get("source", "unknown")
        similarity = 1 - dist
        prefix = f"[{i + 1}] {source}"
        header = _build_header(
            prefix, meta, header_keys, [f"similarity={similarity:.2f}"]
        )
        lines.append(header)
        lines.append((doc or "")[:DOC_DISPLAY_CAP])
        lines.append("")
    return "\n".join(lines)


def format_get_results(results: dict, header_keys: list[str]) -> str:
    """Format ``collection.get(...)`` output (no distances).

    Each result's header is ``[<id>]`` followed by the configured header
    keys and a trailing tags string if present.
    """
    ids = results.get("ids") or []
    if not ids:
        return "No results found."

    lines: list[str] = [f"{len(ids)} results:", ""]
    for id_, doc, meta in zip(ids, results["documents"], results["metadatas"]):
        prefix = f"[{id_}]"
        header = _build_header(prefix, meta, header_keys, [])
        lines.append(header)
        lines.append((doc or "")[:DOC_DISPLAY_CAP])
        lines.append("")
    return "\n".join(lines)
