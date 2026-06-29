"""Unit tests for ingestion's core logic — the deterministic transforms behind the
ingest endpoints: text chunking, whitespace cleaning, multi-format document
extraction, and file storage. No Qdrant / embedding service involved.

Async helpers are driven with asyncio.run() so no asyncio pytest plugin is needed.
(Importing `main` pulls the full ingestion stack; run where its requirements are installed.)"""
import asyncio
import io

import pytest

import main


# ── chunk_text ────────────────────────────────────────────────────────────────
def test_chunk_text_short_text_is_one_chunk():
    assert main.chunk_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_chunk_text_splits_long_text_into_many():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = main.chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)            # no empty/whitespace-only chunks
    assert any("word499" in c for c in chunks)       # the tail of the input is covered


def test_chunk_text_whitespace_only_yields_nothing():
    assert main.chunk_text("   \n\n  ", chunk_size=50, overlap=5) == []


# ── _clean ────────────────────────────────────────────────────────────────────
def test_clean_collapses_runs_of_spaces_and_blank_lines():
    assert main._clean("a    b\n\n\n\nc  ") == "a b\n\nc"


# ── extract_document ──────────────────────────────────────────────────────────
def test_extract_document_txt_cleans_text():
    title, segs = asyncio.run(main.extract_document("notes.txt", b"line one\n\n\n\nline two   "))
    assert title == "notes.txt"
    assert len(segs) == 1 and segs[0]["page"] is None and segs[0]["has_image"] is False
    assert segs[0]["text"] == "line one\n\nline two"


def test_extract_document_md_decodes_utf8():
    _, segs = asyncio.run(main.extract_document("r.md", "café résumé".encode("utf-8")))
    assert "café" in segs[0]["text"] and "résumé" in segs[0]["text"]


def test_extract_document_rejects_unsupported_extension():
    with pytest.raises(ValueError):
        asyncio.run(main.extract_document("malware.exe", b"x"))


def test_extract_document_docx():
    from docx import Document
    d = Document()
    d.add_paragraph("Hello world")
    d.add_paragraph("Second paragraph")
    buf = io.BytesIO()
    d.save(buf)
    _, segs = asyncio.run(main.extract_document("x.docx", buf.getvalue()))
    assert "Hello world" in segs[0]["text"] and "Second paragraph" in segs[0]["text"]


def test_extract_document_pptx():
    from pptx import Presentation
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])   # "Title Only" layout
    slide.shapes.title.text = "Slide Heading"
    buf = io.BytesIO()
    prs.save(buf)
    _, segs = asyncio.run(main.extract_document("x.pptx", buf.getvalue()))
    assert "Slide Heading" in segs[0]["text"]


# ── save_file ─────────────────────────────────────────────────────────────────
def test_save_file_writes_named_by_doc_id(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "FILES_DIR", str(tmp_path))
    path = main.save_file("doc123", "report.PDF", b"%PDF-bytes")
    assert path.endswith("doc123.pdf")               # extension lower-cased
    with open(path, "rb") as f:
        assert f.read() == b"%PDF-bytes"


def test_save_file_defaults_extension_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "FILES_DIR", str(tmp_path))
    path = main.save_file("doc999", "noext", b"data")
    assert path.endswith("doc999.bin")
