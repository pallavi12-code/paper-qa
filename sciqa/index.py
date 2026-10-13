import faiss
import numpy as np

from sciqa.pdf_loader import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"


def sentence_transformer_embedder(model_name: str = MODEL_NAME):
    """Return embed(texts) -> normalized float32 array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    return embed


class VectorStore:
    """FAISS inner-product index over normalized embeddings (= cosine similarity)."""

    def __init__(self, embed_fn):
        self.embed = embed_fn
        self.chunks: list[Chunk] = []
        self.index = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self.embed([c.text for c in chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        qvec = self.embed([query])
        scores, ids = self.index.search(qvec, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
