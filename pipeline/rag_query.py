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
from datetime import datetime
from collections import Counter
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
	CONF_EVENT_AVG,
	CONF_EVENT_TOP,
	CONF_PARTNERSHIP_AVG,
	CONF_PARTNERSHIP_TOP,
	CONTEXT_DIVERSITY_ENABLED,
	CONTEXT_DIVERSITY_MAX_SIM,
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
	BM25_TOP_N,
	HYBRID_BM25_WEIGHT,
	HYBRID_DENSE_WEIGHT,
	HYBRID_RETRIEVAL_ENABLED,
	QUERY_REWRITE_ENABLED,
	RAG_MAX_CONTEXT_CHARS,
	RAG_MIN_SIMILARITY,
	RERANK_CANDIDATES,
	RERANKER_ENABLED,
	RERANKER_MULTILINGUAL_MODEL,
	RERANKER_MODEL,
	RAG_TOP_K,
	STRUCTURED_EVENTS_PATH,
	STRUCTURED_PARTNERSHIPS_PATH,
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
		self._cross_encoder_multilingual: Any | None = None
		self._cross_encoder_disabled = not RERANKER_ENABLED
		self._bm25_docs: list[str] = []
		self._bm25_metas: list[dict[str, Any]] = []
		self._bm25_tokens: list[list[str]] = []
		self._bm25_df: Counter[str] = Counter()
		self._bm25_avgdl = 0.0
		self._bm25_ready = False
		self._structured_partners: list[dict[str, Any]] = []
		self._structured_events: list[dict[str, Any]] = []
		self._load_structured_tables()

	def warmup(self) -> None:
		"""Preload heavy retrieval components to reduce first-query latency."""
		try:
			self._ensure_bm25_corpus()
		except Exception as err:
			logger.warning("BM25 warm-up failed: %s", err)
		try:
			_ = self._get_cross_encoder()
		except Exception as err:
			logger.warning("Reranker warm-up failed: %s", err)

	def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
		k = top_k or self.top_k
		pool_k = max(k, min(RERANK_CANDIDATES, k * 4))
		rewritten_query = self._rewrite_query_for_retrieval(query)
		expanded_query = self._expand_query(rewritten_query)
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

		if HYBRID_RETRIEVAL_ENABLED:
			retrieved = self._fuse_with_bm25(rewritten_query, retrieved, top_n=max(pool_k, BM25_TOP_N))

		keyword_reranked = self._rerank_by_keyword_overlap(rewritten_query, retrieved)
		cross_reranked = self._rerank_with_cross_encoder(rewritten_query, keyword_reranked)
		entities = self._extract_entities(rewritten_query)
		boosted = self._apply_entity_boosts(cross_reranked, entities)
		deduped = self._dedupe_by_message_id(boosted)
		deduped = self._inject_structured_chunks(rewritten_query, deduped)
		diverse = self._select_diverse_chunks(deduped)
		entity_first = self._force_entity_first(rewritten_query, diverse, min_entity_chunks=2)
		return entity_first[:k]

	def _force_entity_first(self, query: str, chunks: list[RetrievedChunk], min_entity_chunks: int = 2) -> list[RetrievedChunk]:
		entities = self._extract_entities(query)
		if not any(entities.values()):
			return chunks
		entity_hits: list[RetrievedChunk] = []
		others: list[RetrievedChunk] = []
		for c in chunks:
			text_l = c.text.lower()
			matched = any(e in text_l for e in entities["company"]) or any(e in text_l for e in entities["event"]) or any(e in text_l for e in entities["date"])
			if matched:
				entity_hits.append(c)
			else:
				others.append(c)
		if not entity_hits:
			return chunks
		# Ensure 1-2 entity-matching chunks are always kept at the top.
		forced = entity_hits[:min_entity_chunks]
		remaining = [c for c in entity_hits[min_entity_chunks:] + others if c not in forced]
		return forced + remaining

	def _rewrite_query_for_retrieval(self, query: str) -> str:
		rewritten = self._normalize_query_typos(query)
		if not QUERY_REWRITE_ENABLED:
			return rewritten
		clean = re.sub(r"\b(with|our|have|has|about|please|tell me|search|find)\b", " ", rewritten, flags=re.IGNORECASE)
		clean = re.sub(r"\s+", " ", clean).strip()
		q_lower = clean.lower()
		if "ensia" in q_lower and any(t in q_lower for t in ["partner", "partnership", "partnerships", "partenariat", "شراكة"]):
			clean = f"{clean} partnerships with ENSIA school companies institutions"
		return clean or rewritten

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

	def _select_diverse_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
		if not CONTEXT_DIVERSITY_ENABLED or len(chunks) <= 2:
			return chunks
		selected: list[RetrievedChunk] = []
		for chunk in chunks:
			if not selected:
				selected.append(chunk)
				continue
			candidate_tokens = self._tokenize(chunk.text)
			max_sim = 0.0
			for s in selected:
				s_tokens = self._tokenize(s.text)
				inter = len(candidate_tokens.intersection(s_tokens))
				union = max(1, len(candidate_tokens.union(s_tokens)))
				sim = inter / union
				if sim > max_sim:
					max_sim = sim
			if max_sim <= CONTEXT_DIVERSITY_MAX_SIM or len(selected) < 2:
				selected.append(chunk)
		return selected

	def _ensure_bm25_corpus(self) -> None:
		if self._bm25_ready:
			return
		self._bm25_ready = True
		try:
			total = self.collection.count()
			if total <= 0:
				return
			batch = 500
			all_docs: list[str] = []
			all_metas: list[dict[str, Any]] = []
			for offset in range(0, total, batch):
				res = self.collection.get(offset=offset, limit=batch, include=["documents", "metadatas"])
				docs = res.get("documents", []) or []
				metas = res.get("metadatas", []) or []
				for doc, meta in zip(docs, metas):
					if not isinstance(doc, str) or not doc.strip():
						continue
					all_docs.append(doc)
					all_metas.append(meta or {})
			self._bm25_docs = all_docs
			self._bm25_metas = all_metas
			self._bm25_tokens = [list(self._tokenize(d)) for d in self._bm25_docs]
			lengths = [len(t) for t in self._bm25_tokens if t]
			self._bm25_avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0
			for tokens in self._bm25_tokens:
				for term in set(tokens):
					self._bm25_df[term] += 1
		except Exception as err:
			logger.warning("BM25 corpus init failed: %s", err)

	def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> float:
		if not query_tokens or not doc_tokens or not self._bm25_docs:
			return 0.0
		tf = Counter(doc_tokens)
		dl = len(doc_tokens)
		avgdl = self._bm25_avgdl or 1.0
		N = len(self._bm25_docs)
		score = 0.0
		for term in query_tokens:
			if term not in tf:
				continue
			df = self._bm25_df.get(term, 0)
			idf = math.log(1 + ((N - df + 0.5) / (df + 0.5)))
			num = tf[term] * (k1 + 1)
			den = tf[term] + k1 * (1 - b + b * (dl / avgdl))
			score += idf * (num / max(1e-9, den))
		return score

	def _fuse_with_bm25(self, query: str, dense_chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
		self._ensure_bm25_corpus()
		if not self._bm25_docs:
			return dense_chunks
		q_tokens = list(self._tokenize(query))
		if not q_tokens:
			return dense_chunks

		bm25_scored: list[tuple[int, float]] = []
		for idx, doc_tokens in enumerate(self._bm25_tokens):
			s = self._bm25_score(q_tokens, doc_tokens)
			if s > 0:
				bm25_scored.append((idx, s))
		bm25_scored.sort(key=lambda x: x[1], reverse=True)
		bm25_scored = bm25_scored[:top_n]

		dense_map: dict[str, RetrievedChunk] = {}
		for c in dense_chunks:
			key = f"{c.metadata.get('message_id','')}::{hash(c.text)}"
			dense_map[key] = c

		combined: dict[str, RetrievedChunk] = {}
		dense_scores = [c.score for c in dense_chunks] or [0.0]
		dense_min, dense_max = min(dense_scores), max(dense_scores)
		bm25_vals = [s for _, s in bm25_scored] or [0.0]
		bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)

		def norm(v: float, lo: float, hi: float) -> float:
			return 0.0 if hi <= lo else (v - lo) / (hi - lo)

		for c in dense_chunks:
			key = f"{c.metadata.get('message_id','')}::{hash(c.text)}"
			d = norm(c.score, dense_min, dense_max)
			meta = dict(c.metadata)
			meta.setdefault("dense_score", c.score)
			fused = (HYBRID_DENSE_WEIGHT * d)
			combined[key] = RetrievedChunk(text=c.text, score=fused, metadata=meta)

		for idx, bm in bm25_scored:
			doc = self._bm25_docs[idx]
			meta = dict(self._bm25_metas[idx])
			key = f"{meta.get('message_id','')}::{hash(doc)}"
			b = norm(bm, bm25_min, bm25_max)
			if key in combined:
				curr = combined[key]
				combined[key] = RetrievedChunk(text=curr.text, score=curr.score + (HYBRID_BM25_WEIGHT * b), metadata=curr.metadata)
			else:
				combined[key] = RetrievedChunk(text=doc, score=(HYBRID_BM25_WEIGHT * b), metadata=meta)

		fused_chunks = sorted(combined.values(), key=lambda c: c.score, reverse=True)
		return fused_chunks

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
			logger.warning("Primary reranker unavailable, trying multilingual fallback: %s", err)
			try:
				from sentence_transformers import CrossEncoder  # type: ignore

				if self._cross_encoder_multilingual is None:
					self._cross_encoder_multilingual = CrossEncoder(RERANKER_MULTILINGUAL_MODEL)
				return self._cross_encoder_multilingual
			except Exception as err2:
				self._cross_encoder_disabled = True
				logger.warning("Cross-encoder reranker disabled: %s", err2)
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

	def _load_structured_tables(self) -> None:
		try:
			if STRUCTURED_PARTNERSHIPS_PATH.exists():
				data = json.loads(STRUCTURED_PARTNERSHIPS_PATH.read_text(encoding="utf-8"))
				self._structured_partners = data.get("partners", [])
		except Exception as err:
			logger.warning("Failed loading partnerships table: %s", err)
		try:
			if STRUCTURED_EVENTS_PATH.exists():
				data = json.loads(STRUCTURED_EVENTS_PATH.read_text(encoding="utf-8"))
				self._structured_events = data.get("events", [])
		except Exception as err:
			logger.warning("Failed loading events table: %s", err)

	def _extract_entities(self, query: str) -> dict[str, set[str]]:
		q = query.lower()
		entities: dict[str, set[str]] = {"company": set(), "date": set(), "event": set()}
		for partner in self._structured_partners:
			name = str(partner.get("name", "")).lower().strip()
			if name and name in q:
				entities["company"].add(name)
		for m in re.findall(r"\b20\d{2}\b", query):
			entities["date"].add(m)
		for hint in ["summit", "conference", "hackathon", "event", "forum", "expo", "tech"]:
			if hint in q:
				entities["event"].add(hint)
		return entities

	def _apply_entity_boosts(self, chunks: list[RetrievedChunk], entities: dict[str, set[str]]) -> list[RetrievedChunk]:
		if not any(entities.values()):
			return chunks
		boosted: list[RetrievedChunk] = []
		for c in chunks:
			text_l = c.text.lower()
			boost = 0.0
			if any(e in text_l for e in entities["company"]):
				boost += 0.08
			if any(e in text_l for e in entities["date"]):
				boost += 0.06
			if any(e in text_l for e in entities["event"]):
				boost += 0.05
			boosted.append(RetrievedChunk(text=c.text, score=c.score + boost, metadata=c.metadata))
		return sorted(boosted, key=lambda x: x.score, reverse=True)

	def _inject_structured_chunks(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
		q = query.lower()
		injected: list[RetrievedChunk] = list(chunks)
		if any(t in q for t in ["partner", "partnership", "partenariat", "شراكة"]) and self._structured_partners:
			lines = [f"- {p.get('name','')} (first_seen={p.get('first_seen','')})" for p in self._structured_partners[:25]]
			injected.insert(
				0,
				RetrievedChunk(
					text="Structured partnerships table:\n" + "\n".join(lines),
					score=0.99,
					metadata={"message_id": "structured_partnerships", "content_type": "table", "date": "", "from": "system"},
				),
			)
		if any(t in q for t in ["event", "conference", "summit", "hackathon"]) and self._structured_events:
			lines = [f"- {e.get('date','')} | {e.get('preview','')}" for e in self._structured_events[:20]]
			injected.insert(
				0,
				RetrievedChunk(
					text="Structured events timeline:\n" + "\n".join(lines),
					score=0.96,
					metadata={"message_id": "structured_events", "content_type": "table", "date": "", "from": "system"},
				),
			)
		return injected

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

	def _answer_format_policy(self) -> str:
		return (
			"Format your response exactly with these sections:\n"
			"Direct answer: <1-2 sentences>\n"
			"Key points:\n- <bullet 1>\n- <bullet 2>\n"
			"If unsure: <state missing context briefly>"
		)

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
				"Provide a concise answer in the same language as the question.\n"
				f"{self._answer_format_policy()}"
			)
		else:
			prompt = (
				"You are a helpful assistant. "
				"Answer with your general knowledge in the same language as the question. "
				"If the user asks for ENSIA-specific facts and you are unsure, say that server context is insufficient.\n\n"
				f"Question:\n{query}\n\n"
				f"{self._answer_format_policy()}"
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
				"When you use information, cite sources like [Source 1]. "
				f"{self._answer_format_policy()}"
			)
			user_content = f"Question:\n{query}\n\nContext:\n{context}"
		else:
			system_prompt = (
				"You are a helpful assistant. "
				"Answer with your general knowledge in the same language as the user. "
				"If the user asks ENSIA-specific facts and you are unsure, clearly say context is insufficient. "
				f"{self._answer_format_policy()}"
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
				"Cite sources like [Source 1] when relevant. "
				f"{self._answer_format_policy()}"
			)
			user_content = f"Question:\n{query}\n\nContext:\n{context}"
		else:
			system_prompt = (
				"You are a helpful assistant. "
				"Answer with general knowledge in the same language as the user. "
				"If ENSIA-specific facts are requested and you are unsure, say context is insufficient. "
				f"{self._answer_format_policy()}"
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
				"Cite sources like [Source 1]. "
				f"{self._answer_format_policy()}"
			)
			user_content = f"Question:\n{query}\n\nContext:\n{context}"
		else:
			system_prompt = (
				"You are a helpful assistant. "
				"Answer with general knowledge in the same language as the user. "
				"If ENSIA-specific facts are requested and you are unsure, say context is insufficient. "
				f"{self._answer_format_policy()}"
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

	def _query_intent_type(self, query: str) -> str:
		q = query.lower()
		if any(t in q for t in ["partner", "partnership", "partenariat", "شراكة", "company", "companies"]):
			return "partnership"
		if any(t in q for t in ["event", "conference", "summit", "hackathon", "forum", "expo"]):
			return "event"
		return "general"

	def _has_confident_context(self, query: str, chunks: list[RetrievedChunk]) -> tuple[bool, str | None]:
		if not chunks:
			return False, "no_chunks"
		intent_type = self._query_intent_type(query)
		min_top = GENERATION_MIN_TOP_SCORE
		min_avg = GENERATION_MIN_AVG_SCORE
		if intent_type == "partnership":
			min_top = max(min_top, CONF_PARTNERSHIP_TOP)
			min_avg = max(min_avg, CONF_PARTNERSHIP_AVG)
		elif intent_type == "event":
			min_top = min(min_top, CONF_EVENT_TOP)
			min_avg = min(min_avg, CONF_EVENT_AVG)

		ranked_scores = sorted(
			max(float(c.score), float(c.metadata.get("dense_score", c.score))) for c in chunks
		)
		ranked_scores.reverse()
		top_score = ranked_scores[0]
		effective_n = min(3, len(ranked_scores))
		avg_score = sum(ranked_scores[:effective_n]) / effective_n
		if top_score < min_top:
			return False, f"top_score={top_score:.3f} < min_top={min_top:.3f} ({intent_type})"
		if avg_score < min_avg:
			return False, f"top{effective_n}_avg={avg_score:.3f} < min_avg={min_avg:.3f} ({intent_type})"
		return True, None

	def _sanitize_error(self, err: Exception) -> str:
		msg = str(err).replace("\n", " ").strip()
		if GEMINI_API_KEY and GEMINI_API_KEY in msg:
			msg = msg.replace(GEMINI_API_KEY, "***")
		if GROQ_API_KEY and GROQ_API_KEY in msg:
			msg = msg.replace(GROQ_API_KEY, "***")
		return f"{err.__class__.__name__}: {msg}" if msg else err.__class__.__name__

	def _enforce_answer_structure(self, text: str) -> str:
		clean = (text or "").strip()
		if not clean:
			return (
				"Direct answer: I could not generate a complete response.\n\n"
				"Key points:\n- No answer content was produced.\n\n"
				"If unsure: Please retry with a more specific ENSIA question."
			)
		if "direct answer:" not in clean.lower():
			clean = f"Direct answer: {clean}"
		if "key points:" not in clean.lower():
			clean += "\n\nKey points:\n- See direct answer above."
		if "if unsure:" not in clean.lower():
			clean += "\n\nIf unsure: Please provide a more specific ENSIA question (company, date, event, or keyword)."
		return clean

	def _dedupe_answer_facts(self, text: str) -> str:
		lines = [ln.rstrip() for ln in text.splitlines()]
		seen: set[str] = set()
		out: list[str] = []
		for ln in lines:
			norm = re.sub(r"\W+", " ", ln.lower()).strip()
			if ln.strip().startswith("-"):
				if norm in seen:
					continue
				seen.add(norm)
			out.append(ln)
		return "\n".join(out).strip()

	def _format_answer_by_intent(self, query: str, answer: str, chunks: list[RetrievedChunk]) -> str:
		intent = self._query_intent_type(query)
		if intent == "partnership":
			partners: list[str] = []
			if self._structured_partners:
				partners = [str(p.get("name", "")).strip() for p in self._structured_partners if str(p.get("name", "")).strip()]
				partners = list(dict.fromkeys(partners))
			else:
				for c in chunks:
					for p in re.findall(r"\b[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,})*\b", c.text):
						if p.lower() in {"source", "direct", "key", "if", "ensia", "impact", "structured"}:
							continue
						if p not in partners:
							partners.append(p)
			if partners:
				rows = "\n".join(f"- {name}" for name in partners[:12])
				return (
					"Direct answer: Here is a table-style list of partnerships found in ENSIA context.\n\n"
					f"Key points:\n{rows}\n\n"
					"If unsure: Ask for a specific company or year to narrow results."
				)
		if intent == "event":
			timeline: list[str] = []
			for c in chunks:
				date = str(c.metadata.get("date", "")).strip()
				preview = re.sub(r"\s+", " ", c.text).strip()[:95]
				if date and preview:
					line = f"- {date}: {preview}"
					if line not in timeline:
						timeline.append(line)
			if timeline:
				return (
					"Direct answer: Here is a timeline of matching ENSIA events.\n\n"
					f"Key points:\n{'\n'.join(timeline[:8])}\n\n"
					"If unsure: Ask by year, event name, or organizer for more precise results."
				)
		if any(t in query.lower() for t in ["resource", "resources", "datacamp", "link", "materials"]):
			links: list[str] = []
			for c in chunks:
				for m in re.findall(r"https?://\S+", c.text):
					if m not in links:
						links.append(m)
			if links:
				return (
					"Direct answer: Here are the relevant resources found in ENSIA context.\n\n"
					f"Key points:\n" + "\n".join(f"- {u}" for u in links[:8]) +
					"\n\nIf unsure: Ask for a specific platform or topic (e.g., DataCamp, internship portal)."
				)
		return answer

	def _finalize_answer(self, query: str, chunks: list[RetrievedChunk], answer: str, mode: str, generation_error: str | None) -> tuple[str, str, str | None]:
		intent_shaped = self._format_answer_by_intent(query, answer, chunks)
		structured = self._enforce_answer_structure(intent_shaped)
		deduped = self._dedupe_answer_facts(structured)
		return deduped, mode, generation_error

	def _clarification_prompt(self, query: str) -> str:
		intent = self._query_intent_type(query)
		if intent == "partnership":
			return "I need a quick clarification: do you mean school-company partnerships, incubator/startup partnerships, or research collaborations?"
		if intent == "event":
			return "I need a quick clarification: do you want ENSIA events by year, by topic, or registration links only?"
		return "I need a quick clarification to improve accuracy: can you specify company, event name, year, or resource type?"

	def _source_trust_label(self, query: str, chunk: RetrievedChunk) -> str:
		score = float(chunk.score)
		date = str(chunk.metadata.get("date", ""))
		recency_bonus = 0.0
		if date:
			try:
				dt = datetime.strptime(date, "%Y-%m-%d")
				if dt.year >= 2025:
					recency_bonus = 0.06
			except Exception:
				pass
		entities = self._extract_entities(query)
		text_l = chunk.text.lower()
		entity_bonus = 0.08 if (
			any(e in text_l for e in entities["company"]) or any(e in text_l for e in entities["event"]) or any(e in text_l for e in entities["date"])
		) else 0.0
		trust = score + recency_bonus + entity_bonus
		return "high" if trust >= 0.62 else "medium"

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
				return self._finalize_answer(query, [], self._generate_with_gemini(query, [], strict_grounding=False), "gemini-general", None)
			if backend == "groq":
				return self._finalize_answer(query, [], self._generate_with_groq(query, [], strict_grounding=False), "groq-general", None)
			if backend == "hf_lora":
				return self._finalize_answer(query, [], self._generate_with_hf_lora(query, [], strict_grounding=False), "hf_lora-general", None)
			if backend == "local_model_1":
				return self._finalize_answer(query, [], self._generate_with_local_endpoint(query, [], LOCAL_MODEL_1, strict_grounding=False), "local_model_1-general", None)
			if backend == "local_model_2":
				return self._finalize_answer(query, [], self._generate_with_local_endpoint(query, [], LOCAL_MODEL_2, strict_grounding=False), "local_model_2-general", None)
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
		intent_type = self._query_intent_type(query)
		query_terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
		if intent_type in {"partnership", "event"} and len(query_terms) <= 2:
			return self._clarification_prompt(query), "needs-clarification", "underspecified_query"
		if not chunks:
			if self._general_fallback_enabled(has_chunks=False, confident=False):
				return self._generate_general(backend, query)
			return (
				"I could not find relevant information in the indexed ENSIA IMPACT data. "
				"Try rephrasing your question with keywords (topic, date, person, or resource type).",
				"none",
				None,
			)

		confident, confidence_reason = self._has_confident_context(query, chunks)
		if not confident:
			if intent_type in {"partnership", "event"}:
				return self._clarification_prompt(query), "needs-clarification", confidence_reason
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
				return self._finalize_answer(query, chunks, self._generate_extractive(query, chunks), "extractive-fallback", safe_err)
			return (
				"Generation failed for the selected backend and extractive fallback is disabled. "
				"Please retry, switch backend, or check provider credentials/connectivity.",
				"generation-error",
				safe_err,
			)

		if backend == "gemini":
			try:
				return self._finalize_answer(query, chunks, self._generate_with_gemini(query, chunks, strict_grounding=True), "gemini", None)
			except Exception as err:
				return finalize_failure(err)

		if backend == "groq":
			try:
				return self._finalize_answer(query, chunks, self._generate_with_groq(query, chunks, strict_grounding=True), "groq", None)
			except Exception as err:
				return finalize_failure(err)

		if backend == "hf_lora":
			try:
				return self._finalize_answer(query, chunks, self._generate_with_hf_lora(query, chunks, strict_grounding=True), "hf_lora", None)
			except Exception as err:
				return finalize_failure(err)

		if backend == "local_model_1":
			try:
				return self._finalize_answer(query, chunks, self._generate_with_local_endpoint(query, chunks, LOCAL_MODEL_1, strict_grounding=True), "local_model_1", None)
			except Exception as err:
				return finalize_failure(err)

		if backend == "local_model_2":
			try:
				return self._finalize_answer(query, chunks, self._generate_with_local_endpoint(query, chunks, LOCAL_MODEL_2, strict_grounding=True), "local_model_2", None)
			except Exception as err:
				return finalize_failure(err)

		return self._finalize_answer(query, chunks, self._generate_extractive(query, chunks), "extractive", None)

	def answer_query(self, query: str, top_k: int | None = None) -> dict[str, Any]:
		chunks = self.retrieve(query, top_k=top_k)
		answer, mode, generation_error = self.generate(query, chunks)
		intent_type = self._query_intent_type(query)
		needs_clarification = mode == "needs-clarification"
		entities = self._extract_entities(query)
		top_entities = sorted(set().union(*entities.values()))[:8]
		top_source = chunks[0] if chunks else None
		return {
			"query": query,
			"answer": answer,
			"mode": mode,
			"intent_type": intent_type,
			"needs_clarification": needs_clarification,
			"top_entities": top_entities,
			"top_source": {
				"date": top_source.metadata.get("date", "") if top_source else "",
				"from": top_source.metadata.get("from", "") if top_source else "",
				"message_id": top_source.metadata.get("message_id", "") if top_source else "",
				"trust": self._source_trust_label(query, top_source) if top_source else "",
			} if top_source else {},
			"generation_error": generation_error,
			"sources": [
				{
					"score": round(c.score, 4),
					"trust": self._source_trust_label(query, c),
					"date": c.metadata.get("date", ""),
					"from": c.metadata.get("from", ""),
					"message_id": c.metadata.get("message_id", ""),
					"content_type": c.metadata.get("content_type", ""),
					"file_name": c.metadata.get("file_name", ""),
					"links": c.metadata.get("links", ""),
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

