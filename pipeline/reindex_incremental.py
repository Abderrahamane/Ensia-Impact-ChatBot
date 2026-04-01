"""Incremental re-indexer for ENSIA IMPACT vectorstore.

This script avoids full re-embedding when only part of Telegram export changed.

Run:
    python pipeline/reindex_incremental.py
    python pipeline/reindex_incremental.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AUTO_REBUILD_STRUCTURED_ON_REINDEX, COLLECTION_NAME, EMBEDDING_MODEL, PROCESSED_DIR, VECTORSTORE_DIR
from pipeline.chunk_and_embed import build_chunks
from pipeline.parse_json import JSON_FILE, parse_messages

MANIFEST_PATH = PROCESSED_DIR / "reindex_manifest.json"
MESSAGES_PATH = PROCESSED_DIR / "messages.json"


def message_signature(msg: dict) -> str:
    tracked = {
        "id": msg.get("id"),
        "date": msg.get("date"),
        "from": msg.get("from"),
        "raw_content": msg.get("raw_content"),
        "reply_to": msg.get("reply_to"),
        "file_name": msg.get("file_name"),
        "links": msg.get("links", []),
        "reactions": msg.get("reactions", 0),
    }
    payload = json.dumps(tracked, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("signatures", {})


def save_manifest(signatures: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"signatures": signatures}
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_chunk_ids_for_message(collection, message_id: str) -> list[str]:
    result = collection.get(where={"message_id": message_id}, include=[])
    return result.get("ids", [])


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(model.encode(batch, show_progress_bar=False).tolist())
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental re-index for ENSIA IMPACT")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing to vectorstore")
    args = parser.parse_args()

    print(f"Loading messages from {JSON_FILE} ...")
    messages, skipped = parse_messages(JSON_FILE)

    current_signatures = {str(m["id"]): message_signature(m) for m in messages}
    old_signatures = load_manifest()

    changed_or_new_ids = [
        msg_id
        for msg_id, sig in current_signatures.items()
        if old_signatures.get(msg_id) != sig
    ]
    removed_ids = [msg_id for msg_id in old_signatures.keys() if msg_id not in current_signatures]

    print(f"Messages parsed: {len(messages)} (skipped {skipped})")
    print(f"Changed/new messages: {len(changed_or_new_ids)}")
    print(f"Removed messages: {len(removed_ids)}")

    if args.dry_run:
        print("Dry-run mode: no writes applied.")
        return

    # Keep processed messages in sync with latest export.
    MESSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MESSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    if not changed_or_new_ids and not removed_ids:
        print("No changes detected; manifest remains unchanged.")
        save_manifest(current_signatures)
        return

    print("Loading embedding model and opening ChromaDB ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # Delete stale chunks for removed or changed messages.
    for message_id in removed_ids + changed_or_new_ids:
        ids = get_chunk_ids_for_message(collection, str(message_id))
        if ids:
            collection.delete(ids=ids)

    changed_messages = [m for m in messages if str(m["id"]) in set(changed_or_new_ids)]
    new_chunks = build_chunks(changed_messages)
    if new_chunks:
        ids = [c["id"] for c in new_chunks]
        docs = [c["text"] for c in new_chunks]
        metas = [c["metadata"] for c in new_chunks]
        vecs = embed_texts(model, docs)
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)

    save_manifest(current_signatures)

    print("Incremental re-index complete.")
    print(f"Chunks upserted: {len(new_chunks)}")
    print(f"Current collection count: {collection.count()}")

      if AUTO_REBUILD_STRUCTURED_ON_REINDEX:
        try:
          subprocess.Popen([sys.executable, str(ROOT_DIR / "pipeline" / "build_structured_tables.py")], cwd=str(ROOT_DIR))
          print("Triggered background structured-table rebuild.")
        except Exception as err:
          print(f"Warning: could not trigger background structured rebuild: {err}")


if __name__ == "__main__":
    main()

