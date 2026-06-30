"""Unit tests for ai-agent's core logic — the deterministic transforms the request
path depends on: RRF fusion, tool-call parsing, think-tag stripping, source rendering,
and the verified-citation pipeline. No vLLM / Qdrant / network involved."""
import main


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def test_rrf_merge_ranks_shared_hit_first():
    # point 2 appears in both lists, so its fused score must beat the singletons.
    vector = [(0.9, 1, {"content": "a"}), (0.8, 2, {"content": "b"})]
    lexical = [{"point_id": "2", "content": "b"}, {"point_id": "3", "content": "c", "url": "u"}]
    merged = main._rrf_merge(vector, lexical)
    assert [p["content"] for _, p in merged] == ["b", "a", "c"]
    assert len(merged) == 3


def test_rrf_merge_skips_lexical_without_point_id():
    assert main._rrf_merge([], [{"content": "x"}]) == []


def test_rrf_merge_keeps_lexical_only_payload():
    merged = main._rrf_merge([], [{"point_id": "7", "url": "u", "title": "t", "content": "c"}])
    assert len(merged) == 1
    assert merged[0][1]["url"] == "u" and merged[0][1]["chunk_index"] == 0


# ── think-tag stripping ───────────────────────────────────────────────────────
def test_strip_think_inline_block():
    assert main.strip_think_tags("<think>reasoning</think>the answer") == "the answer"


def test_strip_think_closing_tag_only():
    assert main.strip_think_tags("leaked reasoning</think>final") == "final"


def test_strip_think_multiline():
    assert main.strip_think_tags("<think>a\nb</think>  done ") == "done"


def test_strip_think_passthrough_when_absent():
    assert main.strip_think_tags("plain answer") == "plain answer"


# ── tool-call parsing (three accepted formats) ────────────────────────────────
def test_extract_tool_call_xml_tag():
    c = '<tool_call>{"name": "rag_search", "arguments": {"query": "x"}}</tool_call>'
    assert main.extract_tool_calls(c) == [{"name": "rag_search", "arguments": {"query": "x"}}]


def test_extract_tool_call_bracket_array():
    c = '[TOOL_CALLS] [{"name": "web_search", "arguments": {"query": "y"}}]'
    assert main.extract_tool_calls(c)[0]["name"] == "web_search"


def test_extract_tool_call_bare_json():
    c = 'sure, calling {"name": "rag_search", "arguments": {"query": "z"}} now'
    assert main.extract_tool_calls(c)[0]["arguments"]["query"] == "z"


def test_extract_tool_call_none_found():
    assert main.extract_tool_calls("no tool calls here") == []


# ── source rendering ──────────────────────────────────────────────────────────
def test_format_sources_links_web_url():
    out = main.format_sources([{"vendor": "NV", "title": "Page", "url": "https://x.com/a", "doc_id": "d1"}])
    assert "[Page](https://x.com/a)" in out


def test_format_sources_links_uploaded_file(monkeypatch):
    monkeypatch.setattr(main, "INGESTION_PUBLIC_URL", "http://host:8002")
    out = main.format_sources([{"vendor": "Dell", "title": "g.pdf", "url": "g.pdf", "doc_id": "abc", "page": 3}])
    assert "[g.pdf](http://host:8002/documents/abc/file)" in out
    assert "p.3" in out


def test_format_sources_plain_when_no_public_url(monkeypatch):
    monkeypatch.setattr(main, "INGESTION_PUBLIC_URL", "")
    out = main.format_sources([{"vendor": "Dell", "title": "g.pdf", "url": "g.pdf", "doc_id": "abc"}])
    assert "g.pdf" in out and "](http" not in out


def test_format_sources_dedups_by_doc_and_page():
    sources = [
        {"doc_id": "d1", "page": 1, "vendor": "V", "title": "T", "url": "u1"},
        {"doc_id": "d1", "page": 1, "vendor": "V", "title": "T", "url": "u1"},  # duplicate
        {"doc_id": "d1", "page": 2, "vendor": "V", "title": "T", "url": "u1"},
    ]
    out = main.format_sources(sources)
    assert out.count("- [V] T") == 2   # page 1 + page 2, the dup collapsed
    assert "p.1" in out and "p.2" in out


# ── verified-citation pipeline ────────────────────────────────────────────────
def test_extract_citations_splits_answer_and_quotes():
    answer, quotes = main.extract_citations('The answer is 42.\n[[CITATIONS]]\n"quote one" "quote two"')
    assert answer == "The answer is 42."
    assert quotes == ["quote one", "quote two"]


def test_extract_citations_without_block():
    answer, quotes = main.extract_citations("just an answer")
    assert answer == "just an answer" and quotes == []


def test_normalize_unifies_dashes_quotes_and_whitespace():
    assert main._normalize("Foo—“Bar”\n\n  baz") == 'foo-"bar" baz'


def test_verify_citations_matches_normalized_substring():
    res = main.verify_citations(["quick brown fox"], [{"content": "The Quick Brown Fox jumps", "page": 3}])
    assert res[0]["verified"] is True and res[0]["page"] == 3


def test_verify_citations_flags_unmatched():
    res = main.verify_citations(["not present"], [{"content": "abc", "page": 1}])
    assert res[0]["verified"] is False and res[0]["page"] is None


def test_format_citations_marks_verified_and_unverified():
    out = main.format_citations([
        {"quote": "a", "verified": True, "page": 2},
        {"quote": "b", "verified": False, "page": None},
    ])
    assert "✓" in out and "p.2" in out
    assert "⚠" in out and "not found" in out


def test_format_citations_empty_is_blank():
    assert main.format_citations([]) == ""
