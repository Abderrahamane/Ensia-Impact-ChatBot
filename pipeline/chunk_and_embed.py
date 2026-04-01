"""
Phase 2: Chunk, embed, and store messages into ChromaDB.

Run AFTER parse_json.py and extract_files.py.

Install deps (once):
    pip install sentence-transformers chromadb

What this script does:
  1. Load data/processed/messages.json
  2. Split long texts into overlapping chunks (≤ 400 tokens)
  3. Embed every chunk with a multilingual sentence-transformers model
     (works for Arabic, French, and English ; all 3 languages in your data)
  4. Store chunks + embeddings in a local ChromaDB vector store
  5. Run a quick sanity-check query at the end
"""

import json
import re
import os
from pathlib import Path


# PATHS

BASE_DIR      = Path(__file__).resolve().parent.parent
MESSAGES_FILE = BASE_DIR / "data" / "processed" / "messages.json"
VECTORSTORE   = BASE_DIR / "data" / "vectorstore"
VECTORSTORE.mkdir(parents=True, exist_ok=True)


# CONFIG

EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Best free model for mixed Arabic/French/English text
# 118 MB download, runs on CPU in < 1s per batch

CHUNK_SIZE    = 400   # approximate characters per chunk (not tokens)
CHUNK_OVERLAP = 80    # overlap to preserve context across chunks
COLLECTION    = "ensia_impact"


# 1. CHUNKING

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by character count.
    Short texts (< size) are returned as a single chunk.
    We split on sentence/newline boundaries where possible.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    # Prefer splitting on newlines or sentence endings
    separators = re.split(r'(?<=[.!?؟\n])\s+', text)

    chunks = []
    current = ""

    for sentence in separators:
        if len(current) + len(sentence) <= size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            # Start new chunk with overlap from previous
            if chunks:
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = overlap_text + " " + sentence
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_chunks(messages: list[dict]) -> list[dict]:
    """
    Turn each message into one or more chunk records.
    Each chunk carries its metadata for later retrieval.
    """
    all_chunks = []

    for msg in messages:
        raw = msg.get("raw_content", "").strip()
        if not raw:
            continue

        parts = chunk_text(raw)

        for i, chunk_text_part in enumerate(parts):
            chunk = {
                #  vector store needs string IDs
                "id":           f"msg{msg['id']}_chunk{i}",
                "text":         chunk_text_part,
                #  metadata stored alongside the vector
                "metadata": {
                    "message_id":   str(msg["id"]),
                    "date":         msg.get("date", ""),
                    "from":         msg.get("from", ""),
                    "content_type": msg.get("content_type", ""),
                    "file_name":    msg.get("file_name", ""),
                    "links":        " | ".join(msg.get("links", [])),
                    "reactions":    str(msg.get("reactions", 0)),
                    "chunk_index":  str(i),
                    "total_chunks": str(len(parts)),
                },
            }
            all_chunks.append(chunk)

    return all_chunks



# 2. EMBED + STORE

def embed_and_store(chunks: list[dict], vectorstore_path: str):
    from sentence_transformers import SentenceTransformer
    import chromadb

    print(f"  Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"  Opening ChromaDB at: {vectorstore_path}")
    client = chromadb.PersistentClient(path=str(vectorstore_path))

    # Delete old collection if re-running
    try:
        client.delete_collection(COLLECTION)
        print(f"  Deleted old collection '{COLLECTION}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for sentence embeddings
    )

    texts     = [c["text"]     for c in chunks]
    ids       = [c["id"]       for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Embed in batches of 64
    BATCH = 64
    all_embeddings = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        vecs  = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(vecs)
        print(f"  Embedded {min(start + BATCH, len(texts))}/{len(texts)} chunks …")

    print("  Storing in ChromaDB …")
    collection.add(
        ids        = ids,
        embeddings = all_embeddings,
        documents  = texts,
        metadatas  = metadatas,
    )

    print(f"  Stored {collection.count()} chunks in collection '{COLLECTION}'")
    return collection, model



# 3. SANITY-CHECK QUERY

def test_query(collection, model, query: str, top_k: int = 3):
    print(f"\n  Test query: \"{query}\"")
    vec = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings = vec,
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = round(1 - dist, 3)   # cosine distance → similarity
        print(f"\n  [{i+1}] score={score}  date={meta['date']}  from={meta['from']}")
        print(f"       {doc[:200]}")



# ENTRY POINT

def main():
    print(f"  Loading messages from {MESSAGES_FILE} …")
    with open(MESSAGES_FILE, encoding="utf-8") as f:
        messages = json.load(f)
    print(f"    {len(messages)} messages loaded")

    print("\n   Chunking …")
    chunks = build_chunks(messages)
    print(f"    {len(chunks)} chunks created from {len(messages)} messages")

    # Save chunks for inspection (optional)
    chunks_file = MESSAGES_FILE.parent / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"    Chunks saved → {chunks_file}")

    print("\n  Embedding & storing …")
    collection, model = embed_and_store(chunks, VECTORSTORE)

    #  Sanity-check with 3 test queries
    test_query(collection, model, "internship companies AI Algeria")
    test_query(collection, model, "FYP final year project")
    test_query(collection, model, "فرصة عمل تدريب")   # Arabic: "job opportunity internship"

    print("\n  Phase 2 complete! Vector store ready at:", VECTORSTORE)


if __name__ == "__main__":
    main()