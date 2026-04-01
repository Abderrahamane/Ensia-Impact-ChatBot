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
STRUCTURED_PARTNERSHIPS_PATH = PROCESSED_DIR / "partnerships_table.json"
STRUCTURED_EVENTS_PATH = PROCESSED_DIR / "events_timeline.json"
INTENT_MODEL_PATH = BASE_DIR / "assets" / "intent_classifier_seed.json"

# Retrieval config
COLLECTION_NAME = os.getenv("ENSIA_COLLECTION", "ensia_impact")
EMBEDDING_MODEL = os.getenv(
	"ENSIA_EMBED_MODEL",
	"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
RAG_TOP_K = int(os.getenv("ENSIA_RAG_TOP_K", "5"))
RAG_MIN_SIMILARITY = float(os.getenv("ENSIA_RAG_MIN_SIM", "0.35"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("ENSIA_RAG_MAX_CONTEXT_CHARS", "3500"))
QUERY_REWRITE_ENABLED = os.getenv("ENSIA_QUERY_REWRITE_ENABLED", "1") == "1"
HYBRID_RETRIEVAL_ENABLED = os.getenv("ENSIA_HYBRID_RETRIEVAL_ENABLED", "1") == "1"
HYBRID_DENSE_WEIGHT = float(os.getenv("ENSIA_HYBRID_DENSE_WEIGHT", "0.65"))
HYBRID_BM25_WEIGHT = float(os.getenv("ENSIA_HYBRID_BM25_WEIGHT", "0.35"))
BM25_TOP_N = int(os.getenv("ENSIA_BM25_TOP_N", "40"))
CONTEXT_DIVERSITY_ENABLED = os.getenv("ENSIA_CONTEXT_DIVERSITY_ENABLED", "1") == "1"
CONTEXT_DIVERSITY_MAX_SIM = float(os.getenv("ENSIA_CONTEXT_DIVERSITY_MAX_SIM", "0.72"))
RERANKER_ENABLED = os.getenv("ENSIA_RERANKER_ENABLED", "1") == "1"
RERANKER_MODEL = os.getenv("ENSIA_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_MULTILINGUAL_MODEL = os.getenv("ENSIA_RERANKER_MULTILINGUAL_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_CANDIDATES = int(os.getenv("ENSIA_RERANK_CANDIDATES", "24"))

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
INTENT_MIN_SCORE = float(os.getenv("ENSIA_INTENT_MIN_SCORE", "0.35"))
INTENT_CLASSIFIER_ENABLED = os.getenv("ENSIA_INTENT_CLASSIFIER_ENABLED", "1") == "1"
INTENT_CLASSIFIER_MIN_CONFIDENCE = float(os.getenv("ENSIA_INTENT_CLASSIFIER_MIN_CONFIDENCE", "0.42"))
CONF_PARTNERSHIP_TOP = float(os.getenv("ENSIA_CONF_PARTNERSHIP_TOP", "0.58"))
CONF_PARTNERSHIP_AVG = float(os.getenv("ENSIA_CONF_PARTNERSHIP_AVG", "0.48"))
CONF_EVENT_TOP = float(os.getenv("ENSIA_CONF_EVENT_TOP", "0.52"))
CONF_EVENT_AVG = float(os.getenv("ENSIA_CONF_EVENT_AVG", "0.40"))
ALLOW_GENERAL_FALLBACK = os.getenv("ENSIA_ALLOW_GENERAL_FALLBACK", "0") == "1"
GENERAL_FALLBACK_SCOPE = os.getenv("ENSIA_GENERAL_FALLBACK_SCOPE", "low_or_empty").strip().lower()

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
BOT_CACHE_ENABLED = os.getenv("ENSIA_BOT_CACHE_ENABLED", "1") == "1"
BOT_CACHE_TTL_SECONDS = int(os.getenv("ENSIA_BOT_CACHE_TTL_SECONDS", "180"))
BOT_CACHE_MAX_ITEMS = int(os.getenv("ENSIA_BOT_CACHE_MAX_ITEMS", "200"))
BOT_WARMUP_ON_START = os.getenv("ENSIA_BOT_WARMUP_ON_START", "1") == "1"
FEEDBACK_ADAPTATION_ENABLED = os.getenv("ENSIA_FEEDBACK_ADAPTATION_ENABLED", "1") == "1"
FEEDBACK_WRONG_THRESHOLD = int(os.getenv("ENSIA_FEEDBACK_WRONG_THRESHOLD", "3"))
FEEDBACK_CORRECT_THRESHOLD = int(os.getenv("ENSIA_FEEDBACK_CORRECT_THRESHOLD", "3"))
AUTO_REBUILD_STRUCTURED_ON_REINDEX = os.getenv("ENSIA_AUTO_REBUILD_STRUCTURED_ON_REINDEX", "1") == "1"
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
