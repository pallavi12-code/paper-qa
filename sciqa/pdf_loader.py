from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Chunk:
    text: str
    page: int  # 1-based
    source: str  # file name


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with 1-based page numbers."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping character chunks."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while True:
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def load_pdf(pdf_bytes: bytes, source: str) -> list[Chunk]:
    """PDF bytes -> chunks that remember their page number."""
    chunks = []
    for page, text in extract_pages(pdf_bytes):
        for piece in chunk_text(text):
            chunks.append(Chunk(text=piece, page=page, source=source))
    return chunks
