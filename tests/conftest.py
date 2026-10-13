import re
import zlib

import fitz
import numpy as np
import pytest


def fake_embed(texts, dim=256):
    """Deterministic bag-of-words embedder so tests need no model download."""
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % dim] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


@pytest.fixture
def sample_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Introduction. Transformers use attention.")
    doc.new_page().insert_text((72, 72), "Results. Graphene conductivity improved tenfold.")
    data = doc.tobytes()
    doc.close()
    return data
