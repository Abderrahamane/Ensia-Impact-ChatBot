"""P0 quality gate for ENSIA IMPACT RAG.

Tracks:
- Retrieval hit@k (k=1,3,5) using weak labels (expected keywords)
- Groundedness (answer tokens supported by retrieved source previews)
- Citation usefulness (presence + metadata quality + relevance)

Run:
    python tests/eval_rag.py
    python tests/eval_rag.py --top-k 5 --limit 10 --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.rag_query import RAGEngine

CASES_PATH = ROOT_DIR / "tests" / "eval_cases.json"
REPORT_PATH = ROOT_DIR / "tests" / "eval_rag_report.json"

STOPWORDS = {
    # EN
    "the", "a", "an", "and", "or", "to", "for", "in", "on", "of", "is", "are", "be", "it",
    "this", "that", "with", "from", "as", "at", "by", "about", "any", "can", "i", "we", "you",
    # FR
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "pour", "dans", "sur", "est",
    "sont", "avec", "par", "au", "aux", "ce", "ces", "comment", "quoi", "ou", "puis", "il", "elle",
    # AR (small practical set)
    "في", "من", "على", "الى", "إلى", "هل", "ما", "ماذا", "كيف", "وين", "كاين", "وش", "عن", "مع",
}


@dataclass
class CaseResult:
    case_id: str
    language: str
    query: str
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    groundedness: float
    citation_usefulness: float
    intent_accuracy: float
    mode: str
    top_sources: list[dict[str, Any]]


def load_cases(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def source_blob(sources: list[dict[str, Any]], k: int) -> str:
    return " ".join((s.get("text_preview") or "") for s in sources[:k]).lower()


def keyword_hit(sources: list[dict[str, Any]], expected_any: list[str], k: int) -> int:
    blob = source_blob(sources, k)
    if not expected_any:
        return int(bool(blob.strip()))
    return int(any(kw.lower() in blob for kw in expected_any))


def score_groundedness(answer: str, sources: list[dict[str, Any]]) -> float:
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0

    source_tokens = set(tokenize(source_blob(sources, k=len(sources))))
    if not source_tokens:
        return 0.0

    supported = answer_tokens.intersection(source_tokens)
    return round(len(supported) / max(1, len(answer_tokens)), 4)


def score_citation_usefulness(
    answer: str,
    sources: list[dict[str, Any]],
    expected_any: list[str],
) -> float:
    if not sources:
        return 0.0

    cited = sources[:2]

    # 1) Presence/availability of citations
    presence = 1.0 if cited else 0.0

    # 2) Metadata quality (date/from/message_id should be available)
    quality_parts = []
    for src in cited:
        filled = sum(
            int(bool((src.get(field) or "").strip()))
            for field in ("date", "from", "message_id")
        )
        quality_parts.append(filled / 3.0)
    metadata_quality = statistics.mean(quality_parts) if quality_parts else 0.0

    # 3) Relevance: expected keywords appear in top cited sources
    cited_blob = source_blob(cited, k=len(cited))
    if expected_any:
        relevance = 1.0 if any(kw.lower() in cited_blob for kw in expected_any) else 0.0
    else:
        relevance = 1.0 if cited_blob else 0.0

    # Small bonus if answer explicitly references source markers
    marker_bonus = 0.1 if ("source" in answer.lower() or "[" in answer) else 0.0

    score = 0.4 * presence + 0.3 * metadata_quality + 0.3 * relevance + marker_bonus
    return round(min(1.0, score), 4)


def evaluate_case(engine: RAGEngine, case: dict[str, Any], top_k: int) -> CaseResult:
    result = engine.answer_query(case["query"], top_k=top_k)
    sources = result.get("sources", [])
    expected_any = case.get("expected_any", [])
    expected_intent = case.get("expected_intent", "ensia_query")
    predicted_intent = result.get("intent_type", "general")
    expected_norm = str(expected_intent).lower()
    predicted_norm = str(predicted_intent).lower()
    if expected_norm == "ensia_query":
        intent_ok = 1.0 if predicted_norm in {"ensia_query", "general", "partnership", "event"} else 0.0
    else:
        intent_ok = 1.0 if predicted_norm == expected_norm else 0.0

    return CaseResult(
        case_id=case["id"],
        language=case.get("language", "unknown"),
        query=case["query"],
        hit_at_1=keyword_hit(sources, expected_any, 1),
        hit_at_3=keyword_hit(sources, expected_any, 3),
        hit_at_5=keyword_hit(sources, expected_any, 5),
        groundedness=score_groundedness(result.get("answer", ""), sources),
        citation_usefulness=score_citation_usefulness(result.get("answer", ""), sources, expected_any),
        intent_accuracy=intent_ok,
        mode=result.get("mode", "unknown"),
        top_sources=sources[:3],
    )


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    if not results:
        return {}

    def avg(vals: list[float]) -> float:
        return round(sum(vals) / max(1, len(vals)), 4)

    overall = {
        "cases": len(results),
        "intent_accuracy": avg([r.intent_accuracy for r in results]),
        "hit@1": avg([r.hit_at_1 for r in results]),
        "hit@3": avg([r.hit_at_3 for r in results]),
        "hit@5": avg([r.hit_at_5 for r in results]),
        "groundedness": avg([r.groundedness for r in results]),
        "citation_usefulness": avg([r.citation_usefulness for r in results]),
    }

    by_language: dict[str, dict[str, Any]] = {}
    langs = sorted({r.language for r in results})
    for lang in langs:
        subset = [r for r in results if r.language == lang]
        by_language[lang] = {
            "cases": len(subset),
            "intent_accuracy": avg([r.intent_accuracy for r in subset]),
            "hit@1": avg([r.hit_at_1 for r in subset]),
            "hit@3": avg([r.hit_at_3 for r in subset]),
            "hit@5": avg([r.hit_at_5 for r in subset]),
            "groundedness": avg([r.groundedness for r in subset]),
            "citation_usefulness": avg([r.citation_usefulness for r in subset]),
        }

    return {"overall": overall, "by_language": by_language}


def pass_fail(summary: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    thresholds = {
        "intent_accuracy": 0.8,
        "hit@5": 0.7,
        "groundedness": 0.2,
        "citation_usefulness": 0.6,
    }
    overall = summary.get("overall", {})
    passed = all(overall.get(metric, 0.0) >= val for metric, val in thresholds.items())
    return passed, thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ENSIA IMPACT RAG quality gate")
    parser.add_argument("--cases", default=str(CASES_PATH), help="Path to eval cases JSON")
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top-k")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test cases")
    parser.add_argument("--verbose", action="store_true", help="Print per-case details")
    parser.add_argument("--report", default=str(REPORT_PATH), help="Output report JSON path")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    print(f"Loaded {len(cases)} evaluation cases from {args.cases}")
    print("Building RAG engine once and running quality gate...")

    engine = RAGEngine(top_k=args.top_k)
    results: list[CaseResult] = []

    for i, case in enumerate(cases, start=1):
        case_result = evaluate_case(engine, case, top_k=args.top_k)
        results.append(case_result)

        if args.verbose:
            print(
                f"[{i:02d}/{len(cases)}] {case_result.case_id} "
                f"hit@5={case_result.hit_at_5} "
                f"grounded={case_result.groundedness:.2f} "
                f"citation={case_result.citation_usefulness:.2f}"
            )

    summary = aggregate(results)
    passed, thresholds = pass_fail(summary)

    report_payload = {
        "thresholds": thresholds,
        "passed": passed,
        "summary": summary,
        "results": [
            {
                "case_id": r.case_id,
                "language": r.language,
                "query": r.query,
                "hit@1": r.hit_at_1,
                "hit@3": r.hit_at_3,
                "hit@5": r.hit_at_5,
                "groundedness": r.groundedness,
                "citation_usefulness": r.citation_usefulness,
                "intent_accuracy": r.intent_accuracy,
                "mode": r.mode,
                "top_sources": r.top_sources,
            }
            for r in results
        ],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    overall = summary.get("overall", {})
    print("\n=== P0 Quality Gate Summary ===")
    print(f"cases               : {overall.get('cases', 0)}")
    print(f"intent_accuracy     : {overall.get('intent_accuracy', 0):.2f}")
    print(f"hit@1 / hit@3 / hit@5: {overall.get('hit@1', 0):.2f} / {overall.get('hit@3', 0):.2f} / {overall.get('hit@5', 0):.2f}")
    print(f"groundedness        : {overall.get('groundedness', 0):.2f}")
    print(f"citation_usefulness : {overall.get('citation_usefulness', 0):.2f}")
    print(f"gate passed         : {passed}")
    print(f"report saved        : {report_path}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

