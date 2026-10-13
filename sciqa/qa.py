from sciqa.index import VectorStore

SYSTEM = (
    "You answer questions about scientific papers using ONLY the provided excerpts. "
    "Cite every claim with the excerpt's label, e.g. [paper.pdf, p.3]. "
    "If the excerpts don't contain the answer, say you couldn't find it in the "
    "uploaded papers. Do not use outside knowledge."
)

NO_CONTEXT = "Upload a PDF first, then ask a question."


def build_prompt(question: str, hits) -> str:
    blocks = [f"[{c.source}, p.{c.page}]\n{c.text}" for c, _ in hits]
    context = "\n\n---\n\n".join(blocks)
    return f"Excerpts:\n\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"


def answer(question: str, store: VectorStore, llm_fn, k: int = 5):
    """Return (answer_text, hits). llm_fn(system, prompt) -> str."""
    hits = store.search(question, k=k)
    if not hits:
        return NO_CONTEXT, []
    return llm_fn(SYSTEM, build_prompt(question, hits)), hits
