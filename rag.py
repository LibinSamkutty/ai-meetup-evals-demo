# rag.py — Operation Blackout
import os
import glob

import chromadb # type: ignore
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction # type: ignore

from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, TOP_K_CHUNKS, EMBEDDING_MODEL

# Module-level collection reference (populated by build_index)
_collection = None
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_case_files_dir = os.path.join(_BASE_DIR, "data", "case_files")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    chunk_idx = 0
    i = 0
    while i < len(words):
        chunk_words = words[i : i + CHUNK_SIZE_WORDS]
        chunks.append(
            {
                "id": f"{source}::chunk_{chunk_idx}",
                "text": " ".join(chunk_words),
                "source": source,
                "chunk_index": chunk_idx,
            }
        )
        i += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
        chunk_idx += 1
    return chunks


def _chunk_witness_statements(text: str, source: str) -> list[dict]:
    """
    Split witness_statements.txt by individual statement boundaries.
    Each person's statement becomes its own chunk so queries about a
    specific person retrieve their statement without header noise diluting
    the embedding.
    """
    import re
    # Split on the STATEMENT — NAME separator, keeping the delimiter
    parts = re.split(r'(?=STATEMENT — )', text)
    chunks = []
    chunk_idx = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Skip the file header block (no "STATEMENT —" means it's preamble)
        if not part.startswith("STATEMENT —"):
            continue
        chunks.append(
            {
                "id": f"{source}::chunk_{chunk_idx}",
                "text": part,
                "source": source,
                "chunk_index": chunk_idx,
            }
        )
        chunk_idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Index build (call once at startup via st.cache_resource)
# ---------------------------------------------------------------------------

_CHROMA_DIR = os.path.join(_BASE_DIR, "data", "chroma_db")


def build_index(case_files_dir: str = None) -> tuple[int, bool]:
    """
    Load all .txt and .md files from case_files_dir, chunk them,
    and index them in a ChromaDB persistent collection.
    Skips embedding if the collection already exists on disk.
    Returns (chunk_count, built_fresh) where built_fresh is False
    when the existing on-disk index was reused.
    """
    global _collection, _case_files_dir
    if case_files_dir is not None:
        _case_files_dir = case_files_dir

    os.makedirs(_CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    existing = [c.name for c in client.list_collections()]
    if "operation_blackout" in existing:
        try:
            _collection = client.get_collection(
                name="operation_blackout", embedding_function=ef
            )
            return _collection.count(), False
        except ValueError:
            # Embedding function mismatch — delete and rebuild
            client.delete_collection("operation_blackout")

    _collection = client.create_collection(
        name="operation_blackout", embedding_function=ef
    )

    all_files = sorted(
        glob.glob(os.path.join(_case_files_dir, "*.txt"))
        + glob.glob(os.path.join(_case_files_dir, "*.md"))
    )

    all_chunks: list[dict] = []
    for filepath in all_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()
        if filename == "06_witness_statements.txt":
            all_chunks.extend(_chunk_witness_statements(content, filename))
        else:
            all_chunks.extend(_chunk_text(content, filename))

    if all_chunks:
        _collection.add(
            ids=[c["id"] for c in all_chunks],
            documents=[c["text"] for c in all_chunks],
            metadatas=[
                {"source": c["source"], "chunk_index": c["chunk_index"]}
                for c in all_chunks
            ],
        )

    return len(all_chunks), True


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str, top_k: int = TOP_K_CHUNKS, question_id: str = None
) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a query.
    Returns list of {"text": str, "source": str, "score": float}.
    For Q06, 03_slack_messages.txt is pinned as the first chunk so the
    injection and Slack content are always present in the context.
    """
    if _collection is None:
        build_index(_case_files_dir)

    results = _collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        # ChromaDB returns L2 distance — convert to a 0-1 similarity signal
        similarity = round(max(0.0, 1.0 - distance / 2.0), 3)
        chunks.append(
            {
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": similarity,
            }
        )

    if question_id == "Q06":
        slack_path = os.path.join(_case_files_dir, "03_slack_messages.txt")
        with open(slack_path, "r", encoding="utf-8") as fh:
            slack_text = fh.read()
        slack_chunk = {
            "text": slack_text,
            "source": "03_slack_messages.txt",
            "score": 1.0,
        }
        # Pin slack as first chunk; drop the last semantic result to keep top_k
        chunks = [slack_chunk] + [
            c for c in chunks if c["source"] != "03_slack_messages.txt"
        ]
        chunks = chunks[:top_k]

    if question_id == "Q07":
        witness_path = os.path.join(
            _case_files_dir, "06_witness_statements.txt"
        )
        with open(witness_path, "r", encoding="utf-8") as fh:
            witness_text = fh.read()
        witness_chunk = {
            "text": witness_text,
            "source": "06_witness_statements.txt",
            "score": 1.0,
        }
        # Pin witness statements as first chunk; drop last semantic result
        chunks = [witness_chunk] + [
            c for c in chunks
            if c["source"] != "06_witness_statements.txt"
        ]
        chunks = chunks[:top_k]

    return chunks


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the persona prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Evidence {i} — Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Case file reader (for Case Files viewer tab)
# ---------------------------------------------------------------------------

def get_all_documents(case_files_dir: str = None) -> list[dict]:
    """Return all case files as raw text for the read-only viewer."""
    if case_files_dir is None:
        case_files_dir = _case_files_dir
    all_files = sorted(
        glob.glob(os.path.join(case_files_dir, "*.txt"))
        + glob.glob(os.path.join(case_files_dir, "*.md"))
    )
    docs = []
    for filepath in all_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()
        docs.append({"name": filename, "content": content})
    return docs
