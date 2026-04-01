"""Telegram bot interface for the ENSIA IMPACT RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from config import (
	ADMIN_USER_IDS,
	BOT_COOLDOWN_SECONDS,
	BOT_RATE_LIMIT_PER_MINUTE,
	BOT_RETRY_COUNT,
	BOT_TIMEOUT_SECONDS,
	GENERATION_BACKEND,
	FEEDBACK_PATH,
	GEMINI_API_KEY,
	LOG_DIR,
	LOG_LEVEL,
	PROCESSED_DIR,
	GROQ_API_KEY,
	SOURCES_DEFAULT_ON,
	TELEGRAM_BOT_TOKEN,
	USER_PREFS_FILE,
	VECTORSTORE_DIR,
)
from ops.backup_data import run_backup
from pipeline.rag_query import RAGEngine


class JsonFormatter(logging.Formatter):
	def format(self, record: logging.LogRecord) -> str:
		payload = {
			"time": datetime.now(UTC).isoformat(),
			"level": record.levelname,
			"logger": record.name,
			"message": record.getMessage(),
		}
		if hasattr(record, "event"):
			payload["event"] = getattr(record, "event")
		if hasattr(record, "meta"):
			payload["meta"] = getattr(record, "meta")
		return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
	LOG_DIR.mkdir(parents=True, exist_ok=True)
	level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
	root = logging.getLogger()
	root.setLevel(level)
	root.handlers.clear()

	stream = logging.StreamHandler()
	stream.setFormatter(JsonFormatter())
	root.addHandler(stream)

	file_handler = logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8")
	file_handler.setFormatter(JsonFormatter())
	root.addHandler(file_handler)


def _acquire_process_lock() -> None:
	lock_file = LOG_DIR / "bot.pid"
	LOG_DIR.mkdir(parents=True, exist_ok=True)
	if lock_file.exists():
		try:
			existing_pid = int(lock_file.read_text(encoding="utf-8").strip())
			os.kill(existing_pid, 0)
		except Exception:
			pass
		else:
			raise RuntimeError(
				f"Another bot instance appears to be running (pid={existing_pid}). Stop it before starting a new one."
			)
	lock_file.write_text(str(os.getpid()), encoding="utf-8")


setup_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)

engine: RAGEngine | None = None
request_timestamps: dict[int, deque[float]] = defaultdict(deque)
last_request_ts: dict[int, float] = {}
user_sources_pref: dict[str, bool] = {}
bot_started_at = time.time()
runtime_metrics: dict[str, Any] = {
	"queries_total": 0,
	"queries_failed": 0,
	"last_query_at": None,
	"last_error": None,
}

SMALLTALK_KEYWORDS = {
	"en": {"hi", "hello", "hey", "how are you", "who are you", "what are you", "thanks", "thank you", "bye", "good morning", "good evening"},
	"fr": {"salut", "bonjour", "bonsoir", "ca va", "comment ca va", "qui es tu", "merci", "au revoir", "coucou"},
	"ar": {"سلام", "مرحبا", "اهلا", "أهلا", "كيف حالك", "شكرا", "شكراً", "مع السلامة", "من انت", "من أنت"},
}

ENSIA_KEYWORDS = {
	"ensia", "impact", "incubator", "incubateur", "cde", "fyp", "pfe", "partnership", "partenariat",
	"startup", "startups", "internship", "stage", "opportunit", "job", "emploi", "resource", "ressource",
	"companies", "entreprise", "events", "hackathon", "scientific", "research", "consulting", "freelance",
	"براءة", "شراكة", "شركة", "شركات", "تربص", "تدريب", "فرص", "حاضنة", "مشروع", "تخرج",
}


def load_user_prefs() -> None:
	global user_sources_pref
	if USER_PREFS_FILE.exists():
		try:
			user_sources_pref = json.loads(USER_PREFS_FILE.read_text(encoding="utf-8"))
		except Exception:
			user_sources_pref = {}


def save_user_prefs() -> None:
	USER_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
	USER_PREFS_FILE.write_text(
		json.dumps(user_sources_pref, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)


def get_engine() -> RAGEngine:
	global engine
	if engine is None:
		engine = RAGEngine()
	return engine


def get_sources_pref(user_id: int) -> bool:
	return user_sources_pref.get(str(user_id), SOURCES_DEFAULT_ON)


def set_sources_pref(user_id: int, enabled: bool) -> None:
	user_sources_pref[str(user_id)] = enabled
	save_user_prefs()


def write_feedback(update: Any, text: str) -> None:
	FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
	record = {
		"timestamp": datetime.now(UTC).isoformat(),
		"user_id": update.effective_user.id if update.effective_user else None,
		"username": update.effective_user.username if update.effective_user else None,
		"chat_id": update.effective_chat.id if update.effective_chat else None,
		"text": text,
	}
	with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
		f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_rate_limit(user_id: int) -> tuple[bool, str | None]:
	now = time.time()
	last_ts = last_request_ts.get(user_id)
	if last_ts is not None and (now - last_ts) < BOT_COOLDOWN_SECONDS:
		wait_s = round(BOT_COOLDOWN_SECONDS - (now - last_ts), 1)
		return False, f"Please wait {wait_s}s before sending another question."

	window = request_timestamps[user_id]
	while window and (now - window[0]) > 60:
		window.popleft()

	if len(window) >= BOT_RATE_LIMIT_PER_MINUTE:
		return False, "Rate limit reached (per minute). Please try again shortly."

	window.append(now)
	last_request_ts[user_id] = now
	return True, None


async def query_with_retry(question: str) -> dict[str, Any]:
	last_error: Exception | None = None
	for attempt in range(1, BOT_RETRY_COUNT + 2):
		try:
			return await asyncio.wait_for(
				asyncio.to_thread(get_engine().answer_query, question),
				timeout=BOT_TIMEOUT_SECONDS,
			)
		except Exception as err:
			last_error = err
			if attempt <= BOT_RETRY_COUNT:
				await asyncio.sleep(min(4, attempt))
				continue

	raise RuntimeError(f"Generation failed after retries: {last_error}")


async def start(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	await update.message.reply_text(
		"Salam! I am the ENSIA IMPACT assistant. "
		"Ask me about opportunities, resources, projects, internships, or events."
	)


async def help_command(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	await update.message.reply_text(
		"Send any question in Arabic, French, or English.\n"
		"Example: 'Where can I find internship opportunities in AI?'\n\n"
		"Commands:\n"
		"/sources on|off - show or hide sources\n"
		"/feedback <text> - send feedback\n"
		"/admins - check your admin status\n"
		"/health - admin health check\n"
		"/stats - admin runtime stats\n"
		"/backup_now [dry] - admin backup trigger"
	)


async def admins_command(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not update.message or not update.effective_user:
		return

	user_id = update.effective_user.id
	is_admin = user_id in ADMIN_USER_IDS
	status = "YES" if is_admin else "NO"

	await update.message.reply_text(
		"Admin check\n"
		f"- your_user_id: {user_id}\n"
		f"- is_admin: {status}\n"
		f"- configured_admin_count: {len(ADMIN_USER_IDS)}"
	)


def _is_admin(update: Any) -> bool:
	if not update.effective_user:
		return False
	if not ADMIN_USER_IDS:
		return False
	return update.effective_user.id in ADMIN_USER_IDS


async def _require_admin(update: Any) -> bool:
	if _is_admin(update):
		return True
	if update.message:
		await update.message.reply_text("This command is admin-only.")
	return False


def _format_uptime() -> str:
	seconds = int(time.time() - bot_started_at)
	h = seconds // 3600
	m = (seconds % 3600) // 60
	s = seconds % 60
	return f"{h:02d}:{m:02d}:{s:02d}"


async def health_command(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not await _require_admin(update):
		return
	engine_ready = engine is not None
	vector_ok = VECTORSTORE_DIR.exists()
	status = (
		"✅ bot: running\n"
		f"- uptime: {_format_uptime()}\n"
		f"- engine_loaded: {engine_ready}\n"
		f"- vectorstore_exists: {vector_ok}\n"
		f"- admins_configured: {len(ADMIN_USER_IDS)}"
	)
	await update.message.reply_text(status)


async def stats_command(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not await _require_admin(update):
		return

	stats_file = PROCESSED_DIR / "stats.json"
	data_stats = {}
	if stats_file.exists():
		try:
			data_stats = json.loads(stats_file.read_text(encoding="utf-8"))
		except Exception:
			data_stats = {}

	msg = (
		"📊 Runtime stats\n"
		f"- uptime: {_format_uptime()}\n"
		f"- queries_total: {runtime_metrics['queries_total']}\n"
		f"- queries_failed: {runtime_metrics['queries_failed']}\n"
		f"- last_query_at: {runtime_metrics['last_query_at']}\n"
		f"- last_error: {runtime_metrics['last_error']}\n\n"
		"📁 Data stats\n"
		f"- kept_messages: {data_stats.get('total_kept', 'n/a')}\n"
		f"- skipped_messages: {data_stats.get('total_skipped', 'n/a')}"
	)
	await update.message.reply_text(msg)


async def backup_now_command(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not await _require_admin(update):
		return

	dry_run = bool(context.args and context.args[0].strip().lower() == "dry")
	await update.message.reply_text("Starting backup..." if not dry_run else "Running backup dry-run...")

	try:
		result = await asyncio.to_thread(run_backup, dry_run)
		if dry_run:
			await update.message.reply_text(
				f"Dry-run ok. Planned archive: {result['planned_archive_path']} | old backups to remove: {result['removed']}"
			)
		else:
			await update.message.reply_text(
				f"Backup done: {result['archive_path']} | old backups removed: {result['removed']}"
			)
	except Exception as err:
		await update.message.reply_text(f"Backup failed: {err}")


async def sources_command(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not update.message or not update.effective_user:
		return
	if not context.args:
		state = "on" if get_sources_pref(update.effective_user.id) else "off"
		await update.message.reply_text(f"Sources are currently {state}. Use /sources on or /sources off")
		return

	value = context.args[0].strip().lower()
	if value not in {"on", "off"}:
		await update.message.reply_text("Usage: /sources on|off")
		return

	set_sources_pref(update.effective_user.id, value == "on")
	await update.message.reply_text(f"Sources display set to {value}.")


async def feedback_command(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not update.message:
		return
	text = " ".join(context.args).strip() if context.args else ""
	if not text:
		await update.message.reply_text("Usage: /feedback <your feedback>")
		return
	write_feedback(update, text)
	await update.message.reply_text("Thanks! Your feedback was saved.")


def _normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize_for_intent(text: str) -> set[str]:
	return {tok for tok in re.findall(r"\w+", text.lower()) if tok}


def _media_source_label(src: dict[str, Any]) -> str | None:
	content_type = (src.get("content_type") or "").strip().lower()
	file_name = (src.get("file_name") or "").strip().lower()
	if content_type == "photo":
		return "image"
	if content_type == "file" and file_name.endswith(".pdf"):
		return "pdf"
	return None


def _select_media_sources(sources: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
	selected: list[tuple[dict[str, Any], str]] = []
	for src in sources:
		label = _media_source_label(src)
		if label:
			selected.append((src, label))
	return selected


def is_ensia_query(text: str) -> bool:
	norm = _normalize_text(text)
	tokens = _tokenize_for_intent(norm)
	for keyword in ENSIA_KEYWORDS:
		key = keyword.lower().strip()
		if not key:
			continue
		if " " in key:
			if key in norm:
				return True
			continue
		if key in tokens:
			return True
		# Keep prefix support for stems like "opportunit" used in keyword set.
		if any(tok.startswith(key) for tok in tokens):
			return True
	return False


def _is_clear_smalltalk(text: str) -> bool:
	norm = _normalize_text(text)
	words = _tokenize_for_intent(norm)
	if len(words) > 6:
		return False

	smalltalk_terms = {term.lower().strip() for term in set().union(*SMALLTALK_KEYWORDS.values()) if term.strip()}
	if norm in smalltalk_terms:
		return True

	# Allow short greeting with punctuation or tiny suffix, but avoid substring matches like "hi" inside "this".
	for term in smalltalk_terms:
		if " " in term:
			continue
		if re.fullmatch(rf"{re.escape(term)}[!?.,]*", norm):
			return True
	return False


def _looks_conversational(text: str) -> bool:
	# Route common capability/identity chat prompts away from RAG retrieval.
	patterns = [
		r"\bwhat can you do\b",
		r"\bwho are you\b",
		r"\bwhat are you\b",
		r"\bhow can you help\b",
		r"\bqui es[- ]?tu\b",
		r"\bcomment tu peux aider\b",
		r"\bمن انت\b",
		r"\bمن أنت\b",
	]
	return any(re.search(p, text) for p in patterns)


def detect_intent(text: str) -> str:
	norm = _normalize_text(text)
	if is_ensia_query(norm):
		return "ensia_query"

	if _is_clear_smalltalk(norm) or _looks_conversational(norm):
		return "smalltalk"

	# Route non-greeting messages to retrieval so they are handled by generation backend.
	return "ensia_query"


def build_smalltalk_reply(text: str) -> str:
	norm = _normalize_text(text)
	if any(k in norm for k in ["how are you", "ca va", "comment ca va", "كيف حالك"]):
		return (
			"I am doing great, thanks for asking. "
			"I can also help you with ENSIA IMPACT topics like internships, partnerships, incubator teams, or events."
		)
	if any(k in norm for k in ["who are you", "what are you", "qui es tu", "من انت", "من أنت"]):
		return (
			"I am your ENSIA IMPACT assistant. "
			"I can chat briefly, and for ENSIA questions I answer with grounded sources from the indexed data."
		)
	if any(k in norm for k in ["thanks", "thank you", "merci", "شكرا", "شكراً"]):
		return "You are welcome. If you want, ask me an ENSIA question and I will include relevant sources."
	if any(k in norm for k in ["bye", "au revoir", "مع السلامة"]):
		return "Good luck. I am here whenever you need help with ENSIA IMPACT information."
	if any(k in norm for k in ["hi", "hello", "hey", "salut", "bonjour", "bonsoir", "سلام", "مرحبا", "اهلا", "أهلا", "coucou"]):
		return (
			"Salam! Nice to meet you. "
			"You can chat with me, or ask ENSIA IMPACT questions (internships, partnerships, incubator, FYP, events)."
		)
	return (
		"I can chat normally, and I am specialized in ENSIA IMPACT knowledge. "
		"Ask me about opportunities, partnerships, incubator teams, FYP formats, or resources."
	)


async def handle_message(update: Any, context: ContextTypes.DEFAULT_TYPE) -> None:
	if not update.message or not update.message.text:
		return
	if not update.effective_user:
		return

	user_id = update.effective_user.id
	allowed, reason = check_rate_limit(user_id)
	if not allowed:
		await update.message.reply_text(reason)
		return

	question = update.message.text.strip()
	intent = detect_intent(question)

	if intent == "smalltalk":
		reply = build_smalltalk_reply(question)
		logging.info(
			"bot_smalltalk_ok",
			extra={
				"event": "bot_smalltalk_ok",
				"meta": {"user_id": user_id, "intent": intent, "text_len": len(question)},
			},
		)
		await update.message.reply_text(reply)
		return

	runtime_metrics["queries_total"] += 1
	runtime_metrics["last_query_at"] = datetime.now(UTC).isoformat()
	started = time.perf_counter()
	try:
		result = await query_with_retry(question)
	except Exception as err:
		runtime_metrics["queries_failed"] += 1
		runtime_metrics["last_error"] = str(err)
		logging.error(
			"bot_query_failed",
			extra={"event": "bot_query_failed", "meta": {"user_id": user_id, "error": str(err)}},
		)
		await update.message.reply_text("Sorry, I could not process your question right now. Please try again.")
		return

	answer = result["answer"]
	latency_ms = round((time.perf_counter() - started) * 1000, 2)

	show_sources = get_sources_pref(user_id)
	mode = str(result.get("mode") or "")
	if show_sources and result["sources"] and "general" not in mode:
		media_sources = _select_media_sources(result["sources"])
		if media_sources:
			top_sources = media_sources[:2]
			lines = ["\n\nSources:"]
			for src, label in top_sources:
				file_name = src.get("file_name", "")
				detail = f" | {label}"
				if file_name:
					detail += f" ({file_name})"
				lines.append(f"- {src['date']} | {src['from']} | msg {src['message_id']}{detail}")
			answer += "\n" + "\n".join(lines)

	logging.info(
		"bot_query_ok",
		extra={
			"event": "bot_query_ok",
			"meta": {
				"user_id": user_id,
				"mode": result.get("mode"),
				"generation_error": result.get("generation_error"),
				"latency_ms": latency_ms,
				"sources": len(result.get("sources", [])),
			},
		},
	)

	await update.message.reply_text(answer)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
	logging.error(f"Exception while handling an update: {context.error}")


def main() -> None:
	if not TELEGRAM_BOT_TOKEN:
		raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment variables.")
	_acquire_process_lock()

	application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
	application.add_error_handler(error_handler)
	load_user_prefs()
	application.add_handler(CommandHandler("start", start))
	application.add_handler(CommandHandler("help", help_command))
	application.add_handler(CommandHandler("admins", admins_command))
	application.add_handler(CommandHandler("health", health_command))
	application.add_handler(CommandHandler("stats", stats_command))
	application.add_handler(CommandHandler("backup_now", backup_now_command))
	application.add_handler(CommandHandler("sources", sources_command))
	application.add_handler(CommandHandler("feedback", feedback_command))
	application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
	logging.info(
		"bot_started",
		extra={
			"event": "bot_started",
			"meta": {
				"pid": os.getpid(),
				"sources_default": SOURCES_DEFAULT_ON,
				"configured_backend": GENERATION_BACKEND,
				"groq_key_present": bool(GROQ_API_KEY),
				"gemini_key_present": bool(GEMINI_API_KEY),
			},
		},
	)
	application.run_polling()


if __name__ == "__main__":
	main()
