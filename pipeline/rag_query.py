"""Phase 3: Retrieve relevant chunks and generate grounded answers."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import re
import sys
import time
from urllib import error, request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from config import (
	ALLOW_GENERAL_FALLBACK,
	ALLOW_EXTRACTIVE_FALLBACK,
	COLLECTION_NAME,
	EMBEDDING_MODEL,
	GENERATION_BACKEND,
	GENERATION_MIN_AVG_SCORE,
	GENERATION_MIN_TOP_SCORE,
	GEMINI_API_KEY,
	GEMINI_FALLBACK_MODELS,
	GEMINI_MODEL,
	GENERATION_RETRY_BACKOFF_S,
	GENERATION_RETRY_COUNT,
	GENERATION_TIMEOUT_S,
	GROQ_API_KEY,
	GROQ_MODEL,
	GENERAL_FALLBACK_SCOPE,
	HF_BASE_MODEL,
	HF_LORA_ADAPTER_DIR,
	HF_MAX_NEW_TOKENS,
	HF_TEMPERATURE,
	LOCAL_API_KEY,
	LOCAL_BASE_URL,
	LOCAL_MODEL_1,
	LOCAL_MODEL_2,
	RAG_MAX_CONTEXT_CHARS,
	RAG_MIN_SIMILARITY,
	RERANK_CANDIDATES,
	RERANKER_ENABLED,
	RERANKER_MODEL,
	RAG_TOP_K,
	VECTORSTORE_DIR,
)


logger = logging.getLogger(__name__)

QUERY_TYPO_ALIASES = {
	"parternship": "partnership",
	"parternships": "partnerships",
	"partership": "partnership",
	"intership": "internship",
	"interships": "internships",
}


@dataclass
class RetrievedChunk:
	text: str
	score: float
	metadata: dict[str, Any]


class RAGEngine:
	def __init__(
		self,
		vectorstore_path=VECTORSTORE_DIR,
		collection_name: str = COLLECTION_NAME,
		embedding_model_name: str = EMBEDDING_MODEL,
		top_k: int = RAG_TOP_K,
		min_similarity: float = RAG_MIN_SIMILARITY,
		max_context_chars: int = RAG_MAX_CONTEXT_CHARS,
	) -> None:
		self.top_k = top_k
		self.min_similarity = min_similarity
		self.max_context_chars = max_context_chars

		self.model = SentenceTransformer(embedding_model_name)
		self.client = chromadb.PersistentClient(path=str(vectorstore_path))
		self.collection = self.client.get_collection(collection_name)
		self._hf_tokenizer: Any | None = None
		self._hf_model: Any | None = None
		self._cross_encoder: Any | None = None
		self._cross_encoder_disabled = not RERANKER_ENABLED

	def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
		k = top_k or self.top_k
		pool_k = max(k, min(RERANK_CANDIDATES, k * 4))
		normalized_query = self._normalize_query_typos(query)
		expanded_query = self._expand_query(normalized_query)
		query_vec = self.model.encode([expanded_query]).tolist()
		result = self.collection.query(
			query_embeddings=query_vec,
			n_results=pool_k,
			include=["documents", "metadatas", "distances"],
		)

		docs = result.get("documents", [[]])[0]
		metas = result.get("metadatas", [[]])[0]
		distances = result.get("distances", [[]])[0]

		retrieved: list[RetrievedChunk] = []
		for doc, meta, dist in zip(docs, metas, distances):
			score = 1 - float(dist)
			if score >= self.min_similarity:
				meta_obj = dict(meta or {})
				meta_obj.setdefault("dense_score", score)
				retrieved.append(RetrievedChunk(text=doc, score=score, metadata=meta_obj))

		if not retrieved:
			# Keep top candidates if strict threshold filters everything.
			retrieved = [
				RetrievedChunk(
					text=doc,
					score=1 - float(dist),
					metadata={**(meta or {}), "dense_score": 1 - float(dist)},
				)
				for doc, meta, dist in zip(docs[:pool_k], metas[:pool_k], distances[:pool_k])
			]

		keyword_reranked = self._rerank_by_keyword_overlap(query, retrieved)
		cross_reranked = self._rerank_with_cross_encoder(query, keyword_reranked)
		deduped = self._dedupe_by_message_id(cross_reranked)
		return deduped[:k]

	def _normalize_query_typos(self, query: str) -> str:
		parts = re.findall(r"\w+|\W+", query)
		normalized: list[str] = []
		for part in parts:
			if re.fullmatch(r"\w+", part):
				normalized.append(QUERY_TYPO_ALIASES.get(part.lower(), part))
			else:
				normalized.append(part)
		return "".join(normalized)

	def _dedupe_by_message_id(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
		seen: set[str] = set()
		deduped: list[RetrievedChunk] = []
		for chunk in chunks:
			msg_id = str(chunk.metadata.get("message_id", "")).strip()
			if msg_id:
				if msg_id in seen:
					continue
				seen.add(msg_id)
			deduped.append(chunk)
		return deduped

	def _get_cross_encoder(self) -> Any | None:
		if self._cross_encoder_disabled:
			return None
		if self._cross_encoder is not None:
			return self._cross_encoder
		try:
			from sentence_transformers import CrossEncoder  # type: ignore

			self._cross_encoder = CrossEncoder(RERANKER_MODEL)
			return self._cross_encoder
		except Exception as err:
			self._cross_encoder_disabled = True
			logger.warning("Cross-encoder reranker disabled: %s", err)
			return None

	def _rerank_with_cross_encoder(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
		reranker = self._get_cross_encoder()
		if reranker is None or not chunks:
			return chunks
		pairs = [[query, c.text] for c in chunks]
		try:
			scores = reranker.predict(pairs)
		except Exception as err:
			logger.warning("Cross-encoder prediction failed; using keyword rerank: %s", err)
			return chunks

		reweighted: list[RetrievedChunk] = []
		for chunk, ce_score in zip(chunks, scores):
			ce_norm = 1.0 / (1.0 + math.exp(-float(ce_score)))
			# Keep vector score dominant; cross-encoder acts as reranking signal.
			blended = (0.75 * chunk.score) + (0.25 * ce_norm)
			reweighted.append(RetrievedChunk(text=chunk.text, score=blended, metadata=chunk.metadata))
		return sorted(reweighted, key=lambda c: c.score, reverse=True)

	def _tokenize(self, text: str) -> set[str]:
		return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}

	def _expand_query(self, query: str) -> str:
		q = query.lower()
		hints: list[str] = []

		if any(t in q for t in ["stage", "intern", "تدريب", "تربص", "opportunit", "job", "emploi"]):
			hints.append("internship stage job opportunities platform internships.ensia.edu.dz")
		if any(t in q for t in ["partner", "partnership", "partenariat", "شراكة", "companies"]):
			hints.append("school partnerships companies institutions formalized mobilis djezzy bomare inapi asal sonatrach")
		if any(t in q for t in ["incubator", "incubated", "startup", "diploma", "accelerator", "cde"]):
			hints.append("school incubator startup decree 1275 diploma startup patent advantages teams")
		if any(t in q for t in ["company", "entreprise", "companies", "شركات"]):
			hints.append("companies institutions AI data science")
		if any(t in q for t in ["pfe", "fyp", "final year", "مشروع", "تخرج"]):
			hints.append("fyp final year project pfe")

		return query if not hints else f"{query} {' '.join(hints)}"

	def _rerank_by_keyword_overlap(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
		query_tokens = self._tokenize(query)
		if not query_tokens:
			return chunks
		intent_keywords = self._intent_keywords(query)
		needs_recency = any(t in query.lower() for t in ["current", "latest", "now", "aujourd", "actuel"])

		def blended_score(chunk: RetrievedChunk) -> float:
			doc_tokens = self._tokenize(chunk.text)
			overlap = len(query_tokens.intersection(doc_tokens)) / max(1, len(query_tokens))
			intent_overlap = 0.0
			if intent_keywords:
				intent_overlap = len(intent_keywords.intersection(doc_tokens)) / max(1, len(intent_keywords))

			recency_bonus = 0.0
			if needs_recency:
				date = str(chunk.metadata.get("date", ""))
				if date >= "2025-01-01":
					recency_bonus = 0.05

			return (0.7 * chunk.score) + (0.2 * overlap) + (0.15 * intent_overlap) + recency_bonus

		return sorted(chunks, key=blended_score, reverse=True)

	def _intent_keywords(self, query: str) -> set[str]:
		q = query.lower()
		if any(t in q for t in ["partner", "partnership", "partenariat", "شراكة"]):
			return {
				"partnership", "partners", "formalized", "companies", "institutions", "mobilis",
				"djezzy", "bomare", "sonatrach", "sonelgaz", "asal", "inapi", "huawei"
			}
		if any(t in q for t in ["incubator", "incubated", "startup", "1275"]):
			return {"incubator", "startup", "decree", "1275", "diploma", "patent", "advantages", "teams"}
		if any(t in q for t in ["fyp", "pfe", "final year", "graduation"]):
			return {"fyp", "pfe", "final", "year", "project", "state", "art"}
		return set()

	def _build_context(self, chunks: list[RetrievedChunk]) -> str:
		context_parts: list[str] = []
		used_chars = 0

		for i, chunk in enumerate(chunks, start=1):
			src = (
				f"[Source {i}] date={chunk.metadata.get('date', '')} "
				f"from={chunk.metadata.get('from', '')} "
				f"message_id={chunk.metadata.get('message_id', '')}\n"
			)
			block = src + chunk.text.strip() + "\n"
			if used_chars + len(block) > self.max_context_chars:
				break
			context_parts.append(block)
			used_chars += len(block)

		return "\n".join(context_parts)

	def _generate_with_gemini(self, query: str, chunks: list[RetrievedChunk], strict_grounding: bool = True) -> str:
		if not GEMINI_API_KEY:
			raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing.")

		context = self._build_context(chunks)
		if strict_grounding:
			prompt = (
				"You are a helpful assistant for ENSIA IMPACT community. "
				"Answer ONLY using the provided context. "
				"Do not invent ENSIA-specific facts not in context. "
				"If the context is insufficient, say so clearly. "
				"When you use information, cite sources like [Source 1].\n\n"
				f"Question:\n{query}\n\n"
				f"Context:\n{context}\n\n"
				"Provide a concise answer in the same language as the question."
			)
		else:
			prompt = (
				"You are a helpful assistant. "
				"Answer with your general knowledge in the same language as the question. "
				"If the user asks for ENSIA-specific facts and you are unsure, say that server context is insufficient.\n\n"
				f"Question:\n{query}\n"
			)

		model_candidates = [GEMINI_MODEL]
		for m in GEMINI_FALLBACK_MODELS.split(","):
			model_name = m.strip()
			if model_name and model_name not in model_candidates:
				model_candidates.append(model_name)

		# Try both bare and prefixed model names because SDK/backends may expect either.
		normalized_candidates: list[str] = []
		for model_name in model_candidates:
			bare = model_name.replace("models/", "")
			for candidate in (bare, f"models/{bare}"):
				if candidate not in normalized_candidates:
					normalized_candidates.append(candidate)

		def run_with_client(client, model_name: str) -> str | None:
			for attempt in range(1, GENERATION_RETRY_COUNT + 2):
				try:
					response = client.models.generate_content(
						model=model_name,
						contents=prompt,
					)
					text = getattr(response, "text", "")
					return text.strip() if text else None
				except Exception as err:
					err_str = str(err).upper()
					transient = (
						"RESOURCE_EXHAUSTED" in err_str
						or "429" in err_str
						or "TIMEOUT" in err_str
					)
					if transient and attempt <= GENERATION_RETRY_COUNT:
						time.sleep(GENERATION_RETRY_BACKOFF_S * attempt)
						continue
					raise

		# Prefer the new SDK: google.genai.
		try:
			genai = importlib.import_module("google.genai")
			types_mod = importlib.import_module("google.genai.types")
			last_error = None
			not_found_errors: list[str] = []

			# Some accounts/models work only on one API version.
			for api_version in ("v1", "v1beta"):
				client = genai.Client(
					api_key=GEMINI_API_KEY,
					http_options=types_mod.HttpOptions(api_version=api_version),
				)
				for model_name in normalized_candidates:
					try:
						text = run_with_client(client, model_name)
						if text:
							return text
					except Exception as err:
						last_error = err
						err_str = str(err).upper()
						if "NOT_FOUND" in err_str or "404" in err_str:
							not_found_errors.append(f"{api_version}:{model_name}")
							continue
						raise

			if last_error is not None:
				not_found_preview = ", ".join(not_found_errors[:6])
				raise RuntimeError(
					"Gemini model(s) unavailable across v1/v1beta. "
					f"Tried {len(normalized_candidates)} model names. "
					f"Examples: {not_found_preview}"
				) from last_error
			return "I could not generate an answer from the available context."
		except ModuleNotFoundError:
			# Backward-compatible fallback for older environments.
			legacy_genai = importlib.import_module("google.generativeai")
			legacy_genai.configure(api_key=GEMINI_API_KEY)
			model = legacy_genai.GenerativeModel(GEMINI_MODEL)
			response = model.generate_content(prompt)
			text = getattr(response, "text", "")
			return text.strip() if text else "I could not generate an answer from the available context."

	def _generate_with_groq(self, query: str, chunks: list[RetrievedChunk], strict_grounding: bool = True) -> str:
		if not GROQ_API_KEY:
			raise RuntimeError("GROQ_API_KEY is missing.")

		try:
			from groq import Groq
		except ModuleNotFoundError as err:
			raise RuntimeError("groq is required for the groq backend. Please install it with 'pip install groq'.") from err

		context = self._build_context(chunks)
		if strict_grounding:
			system_prompt = (
				"You are a helpful assistant for ENSIA IMPACT community. "
				"Answer ONLY using the provided context. "
				"Do not invent ENSIA-specific facts not in context. "
				"If the context is insufficient, say so clearly. "
				"When you use information, cite sources like [Source 1]."
			)
			user_content = f"Question:\n{query}\n\nContext:\n{context}"
		else:
			system_prompt = (
				"You are a helpful assistant. "
				"Answer with your general knowledge in the same language as the user. "
				"If the user asks ENSIA-specific facts and you are unsure, clearly say context is insufficient."
			)
			user_content = query
		
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_content},
		]

		client = Groq(api_key=GROQ_API_KEY)
		
		for attempt in range(1, GENERATION_RETRY_COUNT + 2):
			try:
				completion = client.chat.completions.create(
					model=GROQ_MODEL,
					messages=messages,
					temperature=HF_TEMPERATURE,
					max_tokens=None,
				)
				return completion.choices[0].message.content or "I could not generate an answer from the available context."
			except Exception as err:
				err_str = str(err).lower()
				transient = (
					"rate_limit" in err_str
					or "429" in err_str
					or "timeout" in err_str
					or "connection" in err_str
					or "apiconnectionerror" in err_str
				)
				if transient and attempt <= GENERATION_RETRY_COUNT:
					time.sleep(GENERATION_RETRY_BACKOFF_S * attempt)
					continue
				raise
				
		return "I could not generate an answer from the available context."

	def _load_hf_lora_runtime(self) -> tuple[Any, Any]:
		if self._hf_tokenizer is not None and self._hf_model is not None:
			return self._hf_tokenizer, self._hf_model

		try:
			import torch  # type: ignore
			from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
		except ModuleNotFoundError as err:
			raise RuntimeError(
				"Missing Hugging Face runtime dependencies for hf_lora backend. "
				"Install requirements-train.txt."
			) from err

		tokenizer_source = HF_LORA_ADAPTER_DIR or HF_BASE_MODEL
		if HF_LORA_ADAPTER_DIR and Path(HF_LORA_ADAPTER_DIR).is_dir() and not (Path(HF_LORA_ADAPTER_DIR) / "tokenizer.json").exists() and not (Path(HF_LORA_ADAPTER_DIR) / "tokenizer_config.json").exists():
			print(f"Warning: LoRA directory '{HF_LORA_ADAPTER_DIR}' exists but lacks tokenizer files. Yielding to base model '{HF_BASE_MODEL}'.")
			tokenizer_source = HF_BASE_MODEL

		try:
			tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
		except Exception as e:
			tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False, trust_remote_code=True)
			print(f"Warning: Fast tokenizer load failed: {e}. Falling back to slow tokenizer.")

		if tokenizer.pad_token is None:
			tokenizer.pad_token = tokenizer.eos_token

		model_kwargs: dict[str, Any] = {"trust_remote_code": True}
		if torch.cuda.is_available():
			model_kwargs["device_map"] = "auto"
			model_kwargs["torch_dtype"] = torch.float16

		model = AutoModelForCausalLM.from_pretrained(HF_BASE_MODEL, **model_kwargs)

		if HF_LORA_ADAPTER_DIR and Path(HF_LORA_ADAPTER_DIR).is_dir() and (Path(HF_LORA_ADAPTER_DIR) / "adapter_config.json").exists():
			try:
				from peft import PeftModel  # type: ignore

				model = PeftModel.from_pretrained(model, HF_LORA_ADAPTER_DIR)
			except ModuleNotFoundError as err:
				raise RuntimeError(
					"peft is required to load ENSIA_HF_LORA_ADAPTER_DIR. Install requirements-train.txt."
				) from err
			except Exception as err:
				raise RuntimeError(f"Failed to load LoRA adapter '{HF_LORA_ADAPTER_DIR}': {err}") from err
		elif HF_LORA_ADAPTER_DIR:
			print(f"Warning: LoRA adapter directory '{HF_LORA_ADAPTER_DIR}' lacks adapter_config.json. Proceeding with base model only.")

		model.eval()
		self._hf_tokenizer = tokenizer
		self._hf_model = model
		return tokenizer, model

	def _generate_with_hf_lora(self, query: str, chunks: list[RetrievedChunk], strict_grounding: bool = True) -> str:
		tokenizer, model = self._load_hf_lora_runtime()

		context = self._build_context(chunks)
		if strict_grounding:
			system_prompt = (
				"You are a helpful assistant for ENSIA IMPACT community. "
				"Answer only from provided context. If context is insufficient, say that clearly. "
				"Do not invent ENSIA-specific facts not in context. "
				"Cite sources like [Source 1] when relevant."
			)
			user_content = f"Question:\n{query}\n\nContext:\n{context}"
		else:
			system_prompt = (
				"You are a helpful assistant. "
				"Answer with general knowledge in the same language as the user. "
				"If ENSIA-specific facts are requested and you are unsure, say context is insufficient."
			)
			user_content = query
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_content},
		]

		try:
			if hasattr(tokenizer, "apply_chat_template"):
				prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
			else:
				raise ValueError("no_chat_template")
		except Exception:
			prompt = f"System:\n{system_prompt}\n\nUser:\nQuestion: {query}\n\nContext:\n{context}\n\nAssistant:\n"

		try:
			import torch  # type: ignore
		except ModuleNotFoundError as err:
			raise RuntimeError("torch is required for hf_lora generation.") from err

		inputs = tokenizer(
			prompt,
			return_tensors="pt",
			truncation=True,
			max_length=4096,
		)

		if hasattr(model, "device"):
			inputs = {k: v.to(model.device) for k, v in inputs.items()}

		do_sample = HF_TEMPERATURE > 0
		with torch.no_grad():
			output_ids = model.generate(
				**inputs,
				max_new_tokens=HF_MAX_NEW_TOKENS,
				do_sample=do_sample,
				temperature=HF_TEMPERATURE if do_sample else 1.0,
				pad_token_id=tokenizer.pad_token_id,
				eos_token_id=tokenizer.eos_token_id,
			)

		prompt_len = inputs["input_ids"].shape[1]
		gen_ids = output_ids[0][prompt_len:]
		text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
		return text or "I could not generate an answer from the available context."

	def _generate_with_local_endpoint(self, query: str, chunks: list[RetrievedChunk], model_name: str, strict_grounding: bool = True) -> str:
		context = self._build_context(chunks)
		if strict_grounding:
			system_prompt = (
				"You are a helpful assistant for ENSIA IMPACT community. "
				"Answer ONLY using the provided context. "
				"Do not invent ENSIA-specific facts not in context. "
				"If context is insufficient, clearly say that. "
				"Cite sources like [Source 1]."
			)
			user_content = f"Question:\n{query}\n\nContext:\n{context}"
		else:
			system_prompt = (
				"You are a helpful assistant. "
				"Answer with general knowledge in the same language as the user. "
				"If ENSIA-specific facts are requested and you are unsure, say context is insufficient."
			)
			user_content = query
		payload = {
			"model": model_name,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_content},
			],
			"stream": False,
			"options": {"temperature": HF_TEMPERATURE},
		}

		endpoint = LOCAL_BASE_URL.rstrip("/") + "/api/chat"
		body = json.dumps(payload).encode("utf-8")
		headers = {"Content-Type": "application/json"}
		if LOCAL_API_KEY:
			headers["Authorization"] = f"Bearer {LOCAL_API_KEY}"

		for attempt in range(1, GENERATION_RETRY_COUNT + 2):
			try:
				req = request.Request(endpoint, data=body, headers=headers, method="POST")
				with request.urlopen(req, timeout=GENERATION_TIMEOUT_S) as resp:
					raw = resp.read().decode("utf-8")
				data = json.loads(raw)
				text = ((data.get("message") or {}).get("content") or "").strip()
				if text:
					return text
				raise RuntimeError(f"Empty response from local endpoint model '{model_name}'.")
			except error.HTTPError as err:
				err_text = err.read().decode("utf-8", errors="ignore") if hasattr(err, "read") else ""
				msg = f"HTTP {err.code}: {err_text}".strip()
				transient = err.code in (408, 409, 429, 500, 502, 503, 504)
				if transient and attempt <= GENERATION_RETRY_COUNT:
					time.sleep(GENERATION_RETRY_BACKOFF_S * attempt)
					continue
				raise RuntimeError(msg) from err
			except Exception as err:
				err_str = str(err).lower()
				transient = "timeout" in err_str or "connection" in err_str
				if transient and attempt <= GENERATION_RETRY_COUNT:
					time.sleep(GENERATION_RETRY_BACKOFF_S * attempt)
					continue
				raise

		return "I could not generate an answer from the available context."

	def _generate_extractive(self, query: str, chunks: list[RetrievedChunk]) -> str:
		q = query.lower()

		if any(t in q for t in ["partnership", "partner", "partenariat", "شراكة"]):
			companies: list[str] = []
			for chunk in chunks:
				for line in chunk.text.splitlines():
					clean = line.strip().lstrip("- ").strip()
					if not clean:
						continue
					if len(clean) < 2 or len(clean) > 60:
						continue
					if any(ch.isdigit() for ch in clean) or ":" in clean:
						continue
					if clean.startswith("📢") or clean.startswith("📍") or clean.startswith("🕘"):
						continue
					if len(clean.split()) > 5:
						continue
					if any(x in clean.lower() for x in ["partnership", "formalized", "progress", "school"]):
						continue
					if clean not in companies:
						companies.append(clean)

			if companies:
				listed = "\n".join(f"- {c}" for c in companies[:15])
				return (
					"From recent ENSIA IMPACT messages, these school partnerships/related entities were mentioned:\n"
					f"{listed}\n\n"
					"Note: this list reflects retrieved chat content and may not be exhaustive."
				)

		if any(t in q for t in ["incubator", "incubated", "startup", "advantages"]):
			benefits: list[str] = []
			for chunk in chunks:
				for line in chunk.text.splitlines():
					clean = line.strip().lstrip("- ").strip()
					if len(clean) < 4 or len(clean) > 100:
						continue
					if any(x in clean.lower() for x in ["support", "mentoring", "startup", "advantage", "funding", "incubator", "label", "patent"]):
						if clean not in benefits:
							benefits.append(clean)
			if benefits:
				listed = "\n".join(f"- {b}" for b in benefits[:8])
				return (
					"From retrieved ENSIA incubator messages, these advantages/support points were mentioned:\n"
					f"{listed}\n\n"
					"If you want, ask specifically for 'incubated teams list' and I will search that separately."
				)

		lines = [
			"I found relevant messages in ENSIA IMPACT. Here are the closest sources:",
		]
		for i, chunk in enumerate(chunks[:3], start=1):
			date = chunk.metadata.get("date", "")
			author = chunk.metadata.get("from", "")
			preview = chunk.text.replace("\n", " ").strip()
			preview = preview[:280] + ("..." if len(preview) > 280 else "")
			lines.append(f"{i}. ({date} - {author}) {preview}")

		lines.append("\nAsk a more specific question if you need a precise answer.")
		return "\n".join(lines)

	def _resolve_generation_backend(self) -> str:
		raw_backend = os.getenv("ENSIA_GENERATION_BACKEND", GENERATION_BACKEND)
		backend = (raw_backend or "extractive").strip().lower()
		aliases = {
			"local_model_1": "local_model_1",
			"local_model_2": "local_model_2",
			"local": "local_model_1",
			"ollama": "local_model_1",
		}
		backend = aliases.get(backend, backend)
		allowed = {"extractive", "gemini", "groq", "hf_lora", "local_model_1", "local_model_2"}
		if backend not in allowed:
			logger.warning("Unknown ENSIA_GENERATION_BACKEND='%s'. Falling back to extractive.", raw_backend)
			return "extractive"
		return backend

	def _has_confident_context(self, chunks: list[RetrievedChunk]) -> tuple[bool, str | None]:
		if not chunks:
			return False, "no_chunks"
		ranked_scores = sorted(
			max(float(c.score), float(c.metadata.get("dense_score", c.score))) for c in chunks
		)
		ranked_scores.reverse()
		top_score = ranked_scores[0]
		effective_n = min(3, len(ranked_scores))
		avg_score = sum(ranked_scores[:effective_n]) / effective_n
		if top_score < GENERATION_MIN_TOP_SCORE:
			return False, f"top_score={top_score:.3f} < min_top={GENERATION_MIN_TOP_SCORE:.3f}"
		if avg_score < GENERATION_MIN_AVG_SCORE:
			return False, f"top{effective_n}_avg={avg_score:.3f} < min_avg={GENERATION_MIN_AVG_SCORE:.3f}"
		return True, None

	def _sanitize_error(self, err: Exception) -> str:
		msg = str(err).replace("\n", " ").strip()
		if GEMINI_API_KEY and GEMINI_API_KEY in msg:
			msg = msg.replace(GEMINI_API_KEY, "***")
		if GROQ_API_KEY and GROQ_API_KEY in msg:
			msg = msg.replace(GROQ_API_KEY, "***")
		return f"{err.__class__.__name__}: {msg}" if msg else err.__class__.__name__

	def _general_fallback_enabled(self, has_chunks: bool, confident: bool) -> bool:
		if not ALLOW_GENERAL_FALLBACK:
			return False
		scope = (GENERAL_FALLBACK_SCOPE or "low_or_empty").strip().lower()
		if scope not in {"empty_only", "low_or_empty"}:
			scope = "low_or_empty"
		if not has_chunks:
			return True
		if scope == "low_or_empty" and not confident:
			return True
		return False

	def _generate_general(self, backend: str, query: str) -> tuple[str, str, str | None]:
		try:
			if backend == "gemini":
				return self._generate_with_gemini(query, [], strict_grounding=False), "gemini-general", None
			if backend == "groq":
				return self._generate_with_groq(query, [], strict_grounding=False), "groq-general", None
			if backend == "hf_lora":
				return self._generate_with_hf_lora(query, [], strict_grounding=False), "hf_lora-general", None
			if backend == "local_model_1":
				return self._generate_with_local_endpoint(query, [], LOCAL_MODEL_1, strict_grounding=False), "local_model_1-general", None
			if backend == "local_model_2":
				return self._generate_with_local_endpoint(query, [], LOCAL_MODEL_2, strict_grounding=False), "local_model_2-general", None
		except Exception as err:
			return (
				"General generation fallback failed. Please retry or switch generation backend.",
				"generation-error",
				self._sanitize_error(err),
			)
		return (
			"General generation fallback is unavailable for the selected backend.",
			"generation-error",
			"unsupported_general_backend",
		)

	def generate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, str, str | None]:
		backend = self._resolve_generation_backend()
		if not chunks:
			if self._general_fallback_enabled(has_chunks=False, confident=False):
				return self._generate_general(backend, query)
			return (
				"I could not find relevant information in the indexed ENSIA IMPACT data. "
				"Try rephrasing your question with keywords (topic, date, person, or resource type).",
				"none",
				None,
			)

		confident, confidence_reason = self._has_confident_context(chunks)
		if not confident:
			if self._general_fallback_enabled(has_chunks=True, confident=False):
				return self._generate_general(backend, query)
			return (
				"I found low-confidence context for this question, so I am not generating an answer to avoid misinformation. "
				"Please rephrase with more specific keywords (person, date, event name, or exact phrase).",
				"low-confidence-context",
				confidence_reason,
			)

		def finalize_failure(err: Exception) -> tuple[str, str, str | None]:
			safe_err = self._sanitize_error(err)
			if ALLOW_EXTRACTIVE_FALLBACK:
				return self._generate_extractive(query, chunks), "extractive-fallback", safe_err
			return (
				"Generation failed for the selected backend and extractive fallback is disabled. "
				"Please retry, switch backend, or check provider credentials/connectivity.",
				"generation-error",
				safe_err,
			)

		if backend == "gemini":
			try:
				return self._generate_with_gemini(query, chunks, strict_grounding=True), "gemini", None
			except Exception as err:
				return finalize_failure(err)

		if backend == "groq":
			try:
				return self._generate_with_groq(query, chunks, strict_grounding=True), "groq", None
			except Exception as err:
				return finalize_failure(err)

		if backend == "hf_lora":
			try:
				return self._generate_with_hf_lora(query, chunks, strict_grounding=True), "hf_lora", None
			except Exception as err:
				return finalize_failure(err)

		if backend == "local_model_1":
			try:
				return self._generate_with_local_endpoint(query, chunks, LOCAL_MODEL_1, strict_grounding=True), "local_model_1", None
			except Exception as err:
				return finalize_failure(err)

		if backend == "local_model_2":
			try:
				return self._generate_with_local_endpoint(query, chunks, LOCAL_MODEL_2, strict_grounding=True), "local_model_2", None
			except Exception as err:
				return finalize_failure(err)

		return self._generate_extractive(query, chunks), "extractive", None

	def answer_query(self, query: str, top_k: int | None = None) -> dict[str, Any]:
		chunks = self.retrieve(query, top_k=top_k)
		answer, mode, generation_error = self.generate(query, chunks)
		return {
			"query": query,
			"answer": answer,
			"mode": mode,
			"generation_error": generation_error,
			"sources": [
				{
					"score": round(c.score, 4),
					"date": c.metadata.get("date", ""),
					"from": c.metadata.get("from", ""),
					"message_id": c.metadata.get("message_id", ""),
					"content_type": c.metadata.get("content_type", ""),
					"file_name": c.metadata.get("file_name", ""),
					"text_preview": c.text[:220],
				}
				for c in chunks
			],
		}


def run_cli() -> None:
	parser = argparse.ArgumentParser(description="Query ENSIA IMPACT vector store")
	parser.add_argument("--query", type=str, help="Question to ask")
	parser.add_argument("--top-k", type=int, default=RAG_TOP_K, help="Number of chunks to retrieve")
	args = parser.parse_args()

	engine = RAGEngine()

	if args.query:
		result = engine.answer_query(args.query, top_k=args.top_k)
		print(f"\nMode: {result['mode']}")
		if result.get("generation_error"):
			print(f"Generation fallback reason: {result['generation_error']}")
		print(f"\nAnswer:\n{result['answer']}")
		print("\nSources:")
		for i, src in enumerate(result["sources"], start=1):
			detail = ""
			if src.get("content_type"):
				detail += f" type={src['content_type']}"
			if src.get("file_name"):
				detail += f" file={src['file_name']}"
			print(
				f"  [{i}] score={src['score']} date={src['date']} from={src['from']} "
				f"msg={src['message_id']}{detail}"
			)
		return

	print("Interactive mode. Type 'exit' to quit.")
	while True:
		question = input("\nYou: ").strip()
		if not question or question.lower() in {"exit", "quit"}:
			break
		result = engine.answer_query(question, top_k=args.top_k)
		print(f"\nBot ({result['mode']}): {result['answer']}")


if __name__ == "__main__":
	run_cli()

