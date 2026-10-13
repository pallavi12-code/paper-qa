from sciqa.index import VectorStore
from sciqa.pdf_loader import chunk_text, extract_pages, load_pdf
from sciqa.qa import NO_CONTEXT, answer, build_prompt
from tests.conftest import fake_embed


def test_chunk_text_overlap_and_empty():
    assert chunk_text("   ") == []
    chunks = chunk_text("a" * 2500, size=1000, overlap=200)
    assert len(chunks) == 3
    assert all(len(c) <= 1000 for c in chunks)


def test_extract_pages_keeps_page_numbers(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert [p for p, _ in pages] == [1, 2]
    assert "Graphene" in pages[1][1]


def test_search_returns_correct_page(sample_pdf):
    store = VectorStore(fake_embed)
    store.add(load_pdf(sample_pdf, "paper.pdf"))
    chunk, _ = store.search("graphene conductivity", k=1)[0]
    assert chunk.page == 2 and chunk.source == "paper.pdf"


def test_answer_builds_cited_prompt(sample_pdf):
    store = VectorStore(fake_embed)
    store.add(load_pdf(sample_pdf, "paper.pdf"))
    seen = {}

    def fake_llm(system, prompt):
        seen["prompt"] = prompt
        return "ok"

    text, hits = answer("What about graphene?", store, fake_llm, k=2)
    assert text == "ok" and hits
    assert "[paper.pdf, p.2]" in seen["prompt"]
    assert "What about graphene?" in build_prompt("What about graphene?", hits)


def test_answer_with_empty_store():
    text, hits = answer("anything", VectorStore(fake_embed), lambda s, p: "x")
    assert text == NO_CONTEXT and hits == []
