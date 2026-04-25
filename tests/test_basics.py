"""Smoke tests covering the pure-Python primitives.

Deliberately avoids ChromaDB and FastMCP — those have heavyweight setup
(persistence, ONNX models). The KnowledgeService end-to-end belongs in an
integration test suite that lives outside this package.
"""

from __future__ import annotations

from mcp_knowledge_base import (
    format_get_results,
    format_query_results,
    now_iso,
    sanitize_for_id,
    tag_flags,
    tag_key,
)


# -- tag_key / tag_flags --------------------------------------------------


def test_tag_key_lowercases_and_replaces_non_alnum():
    assert tag_key("Status-Effect") == "tag_status_effect"
    assert tag_key("RPC") == "tag_rpc"
    assert tag_key("foo bar") == "tag_foo_bar"
    assert tag_key("alreadyok_1") == "tag_alreadyok_1"


def test_tag_key_collapses_punctuation_to_underscore():
    # The regex replaces every non-[a-z0-9_] char individually, so
    # "a--b" -> "a__b" rather than "a_b".  This matches the existing
    # behaviour in all three knowledge services.
    assert tag_key("a--b") == "tag_a__b"


def test_tag_flags_returns_dict_of_true():
    assert tag_flags(["rpc", "ZDO"]) == {"tag_rpc": True, "tag_zdo": True}


def test_tag_flags_skips_empty_entries():
    # Real callers always pre-strip via
    # `[t.strip() for t in s.split(",") if t.strip()]`, so the only
    # falsy value we ever see in practice is the empty string.
    assert tag_flags(["", "ok"]) == {"tag_ok": True}


# -- sanitize_for_id ------------------------------------------------------


def test_sanitize_for_id_keeps_safe_chars():
    assert sanitize_for_id("Foo.Bar-Baz_1") == "Foo.Bar-Baz_1"


def test_sanitize_for_id_replaces_unsafe_chars():
    assert sanitize_for_id("a/b c:d") == "a_b_c_d"


# -- now_iso --------------------------------------------------------------


def test_now_iso_format():
    s = now_iso()
    # YYYY-MM-DDTHH:MM:SSZ
    assert len(s) == 20
    assert s[4] == "-" and s[7] == "-"
    assert s[10] == "T"
    assert s[13] == ":" and s[16] == ":"
    assert s.endswith("Z")


# -- format_query_results -------------------------------------------------


def _query_payload(hits):
    """Build a ChromaDB-shaped .query() result from a list of (doc, meta, dist) tuples."""
    return {
        "ids": [[f"id{i}" for i, _ in enumerate(hits)]],
        "documents": [[h[0] for h in hits]],
        "metadatas": [[h[1] for h in hits]],
        "distances": [[h[2] for h in hits]],
    }


def test_format_query_results_empty():
    assert format_query_results({"ids": [[]]}, header_keys=[]) == "No results found."
    assert format_query_results({}, header_keys=[]) == "No results found."


def test_format_query_results_uses_header_keys_in_order():
    hits = [
        (
            "the document body",
            {"source": "py-source/Foo", "module": "bar", "class_name": "Baz", "tags": "x,y"},
            0.25,
        )
    ]
    out = format_query_results(_query_payload(hits), header_keys=["module", "class_name"])
    assert "[1] py-source/Foo" in out
    assert "module=bar" in out
    assert "class_name=Baz" in out
    assert "similarity=0.75" in out
    assert "tags=[x,y]" in out
    assert "the document body" in out


def test_format_query_results_skips_empty_header_values():
    hits = [
        (
            "doc",
            {"source": "s", "module": "", "class_name": "C", "tags": ""},
            0.0,
        )
    ]
    out = format_query_results(_query_payload(hits), header_keys=["module", "class_name"])
    assert "module=" not in out  # missing value should be skipped entirely
    assert "class_name=C" in out
    assert "tags=" not in out


def test_format_query_results_caps_long_documents():
    long_doc = "x" * 5000
    hits = [(long_doc, {"source": "s"}, 0.0)]
    out = format_query_results(_query_payload(hits), header_keys=[])
    # Display cap is 1500 chars from the document
    assert "x" * 1500 in out
    assert "x" * 1501 not in out


# -- format_get_results ---------------------------------------------------


def _get_payload(items):
    return {
        "ids": [item[0] for item in items],
        "documents": [item[1] for item in items],
        "metadatas": [item[2] for item in items],
    }


def test_format_get_results_empty():
    assert format_get_results({}, header_keys=[]) == "No results found."
    assert format_get_results({"ids": []}, header_keys=[]) == "No results found."


def test_format_get_results_renders_ids_and_header_keys():
    items = [
        ("id-A", "body A", {"kind": "note", "md5": "abc", "tags": "t1"}),
        ("id-B", "body B", {"kind": "disasm", "md5": "def", "tags": ""}),
    ]
    out = format_get_results(_get_payload(items), header_keys=["kind", "md5"])
    assert "2 results:" in out
    assert "[id-A]" in out
    assert "[id-B]" in out
    assert "kind=note" in out
    assert "md5=abc" in out
    assert "tags=[t1]" in out
    assert "body A" in out and "body B" in out
