"""Centralized runtime settings for the ENSIA IMPACT chatbot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# In this project, we want the checked-in/local .env to win over stale shell/user env vars.
load_dotenv(BASE_DIR / ".env", override=True)

# Paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
LOG_DIR = BASE_DIR / "logs"
BACKUP_DIR = BASE_DIR / "backups"

# Retrieval config
COLLECTION_NAME = os.getenv("ENSIA_COLLECTION", "ensia_impact")
EMBEDDING_MODEL = os.getenv(
	"ENSIA_EMBED_MODEL",
	"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
RAG_TOP_K = int(os.getenv("ENSIA_RAG_TOP_K", "5"))
RAG_MIN_SIMILARITY = float(os.getenv("ENSIA_RAG_MIN_SIM", "0.35"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("ENSIA_RAG_MAX_CONTEXT_CHARS", "3500"))

# Generation config
GENERATION_BACKEND = os.getenv("ENSIA_GENERATION_BACKEND", "extractive")
GEMINI_MODEL = os.getenv("ENSIA_GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODELS = os.getenv(
        "ENSIA_GEMINI_FALLBACK_MODELS",
        "gemini-2.0-flash-001,gemini-2.0-flash-lite-001,gemini-2.0-flash-lite,gemini-1.5-flash",
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
	GROQ_API_KEY = GROQ_API_KEY.strip()

GROQ_MODEL = os.getenv("ENSIA_GROQ_MODEL", "llama-3.3-70b-versatile").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", GOOGLE_API_KEY)
if GEMINI_API_KEY:
	GEMINI_API_KEY = GEMINI_API_KEY.strip()

GENERATION_TIMEOUT_S = int(os.getenv("ENSIA_GENERATION_TIMEOUT_S", "35"))
GENERATION_RETRY_COUNT = int(os.getenv("ENSIA_GENERATION_RETRY_COUNT", "2"))
GENERATION_RETRY_BACKOFF_S = float(os.getenv("ENSIA_GENERATION_RETRY_BACKOFF_S", "1.5"))
HF_BASE_MODEL = os.getenv("ENSIA_HF_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_LORA_ADAPTER_DIR = os.getenv("ENSIA_HF_LORA_ADAPTER_DIR", "")
HF_MAX_NEW_TOKENS = int(os.getenv("ENSIA_HF_MAX_NEW_TOKENS", "220"))
HF_TEMPERATURE = float(os.getenv("ENSIA_HF_TEMPERATURE", "0.2"))
ALLOW_EXTRACTIVE_FALLBACK = os.getenv("ENSIA_ALLOW_EXTRACTIVE_FALLBACK", "1") == "1"
GENERATION_MIN_TOP_SCORE = float(os.getenv("ENSIA_GENERATION_MIN_TOP_SCORE", "0.42"))
GENERATION_MIN_AVG_SCORE = float(os.getenv("ENSIA_GENERATION_MIN_AVG_SCORE", "0.35"))

# Local/OpenAI-compatible endpoint backend (for Ollama/vLLM/local gateways)
LOCAL_BASE_URL = os.getenv("ENSIA_LOCAL_BASE_URL", "http://127.0.0.1:11434")
LOCAL_API_KEY = os.getenv("ENSIA_LOCAL_API_KEY", "")
LOCAL_MODEL_1 = os.getenv("ENSIA_LOCAL_MODEL_1", "qwen2.5:1.5b-instruct")
LOCAL_MODEL_2 = os.getenv("ENSIA_LOCAL_MODEL_2", "phi3:mini")

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BOT_COOLDOWN_SECONDS = float(os.getenv("ENSIA_BOT_COOLDOWN_SECONDS", "3"))
BOT_RATE_LIMIT_PER_MINUTE = int(os.getenv("ENSIA_BOT_RATE_LIMIT_PER_MINUTE", "8"))
BOT_TIMEOUT_SECONDS = int(os.getenv("ENSIA_BOT_TIMEOUT_SECONDS", "45"))
BOT_RETRY_COUNT = int(os.getenv("ENSIA_BOT_RETRY_COUNT", "2"))
SOURCES_DEFAULT_ON = os.getenv("ENSIA_SOURCES_DEFAULT_ON", "1") == "1"
ADMIN_USER_IDS = {
	int(x.strip())
	for x in os.getenv("ENSIA_ADMIN_USER_IDS", "").split(",")
	if x.strip().isdigit()
}
USER_PREFS_FILE = PROCESSED_DIR / "user_prefs.json"
FEEDBACK_PATH = PROCESSED_DIR / "feedback.jsonl"

# Ops / observability
LOG_LEVEL = os.getenv("ENSIA_LOG_LEVEL", "INFO")
BACKUP_VECTORSTORE_DIR = BACKUP_DIR / "vectorstore"
BACKUP_RETENTION_DAYS = int(os.getenv("ENSIA_BACKUP_RETENTION_DAYS", "7"))
