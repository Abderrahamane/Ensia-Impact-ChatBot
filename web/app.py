"""Web chat API + frontend for ENSIA IMPACT assistant."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import EMBEDDING_MODEL, GENERATION_BACKEND, LOCAL_BASE_URL, RAG_TOP_K
from pipeline.rag_query import RAGEngine

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm up heavy retrieval pieces once so first visitor response is faster.
    try:
        get_engine().warmup()
    except Exception:
        pass
    yield


app = FastAPI(title="ENSIA IMPACT Web API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_engine: RAGEngine | None = None


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None


class ChatResponse(BaseModel):
    ok: bool
    answer: str
    mode: str
    sources: list[dict[str, Any]]



def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


def _local_backend_reachable() -> bool:
    backend = (GENERATION_BACKEND or "").strip().lower()
    if backend not in {"local_model_1", "local_model_2", "local", "ollama"}:
        return True
    try:
        from urllib import request

        health_url = LOCAL_BASE_URL.rstrip("/") + "/api/tags"
        with request.urlopen(health_url, timeout=2) as resp:
            return 200 <= int(resp.status) < 500
    except Exception:
        return False


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    reachable = _local_backend_reachable()
    return {
        "ok": reachable,
        "backend": GENERATION_BACKEND,
        "message": "ready" if reachable else "server is down please try later",
    }


@app.get("/api/info")
def info() -> dict[str, Any]:
    return {
        "name": "ENSIA IMPACT Assistant",
        "status": "online" if _local_backend_reachable() else "degraded",
        "backend": GENERATION_BACKEND,
        "embedding_model": EMBEDDING_MODEL,
        "default_top_k": RAG_TOP_K,
        "features": [
            "RAG over ENSIA indexed data",
            "Source-aware answers",
            "Multilingual retrieval (AR/FR/EN)",
        ],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not _local_backend_reachable():
        return ChatResponse(
            ok=False,
            answer="Server is down, please try later.",
            mode="server-down",
            sources=[],
        )

    question = payload.message.strip()
    if not question:
        return ChatResponse(ok=False, answer="Please enter a message.", mode="invalid", sources=[])

    result = get_engine().answer_query(question, top_k=payload.top_k)
    return ChatResponse(
        ok=True,
        answer=str(result.get("answer") or ""),
        mode=str(result.get("mode") or ""),
        sources=result.get("sources") or [],
    )


if __name__ == "__main__":
    host = os.getenv("ENSIA_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("ENSIA_WEB_PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)

