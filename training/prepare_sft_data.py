"""Build supervised fine-tuning (SFT) chat data from ENSIA eval queries.

This script converts retrieval-grounded Q/A examples into JSONL format for LoRA/SFT.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_SYSTEM_PROMPT = (
    "You are ENSIA IMPACT assistant. Answer in the user's language. "
    "Use only the given context and add source citations like [Source 1]. "
    "If context is insufficient, say it clearly."
)


@dataclass
class SFTExample:
    example_id: str
    language: str
    query: str
    context: str
    answer: str

    def to_json(self, system_prompt: str) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "language": self.language,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Question: {self.query}\n\nContext:\n{self.context}",
                },
                {"role": "assistant", "content": self.answer},
            ],
        }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_context(sources: list[dict[str, Any]], max_sources: int) -> str:
    if not sources:
        return "[No source chunks were retrieved.]"

    lines: list[str] = []
    for idx, src in enumerate(sources[:max_sources], start=1):
        preview = (src.get("text_preview") or "").replace("\n", " ").strip()
        meta = (
            f"date={src.get('date', '')} "
            f"from={src.get('from', '')} "
            f"message_id={src.get('message_id', '')}"
        )
        lines.append(f"[Source {idx}] {meta}\n{preview}")

    return "\n\n".join(lines)


def _ensure_citations(answer: str, source_count: int) -> str:
    if source_count <= 0:
        return answer

    if "[Source" in answer:
        return answer

    top_refs = " ".join(f"[Source {i}]" for i in range(1, min(source_count, 2) + 1))
    return f"{answer}\n\nCitations: {top_refs}".strip()


def build_examples(
    eval_cases: list[dict[str, Any]],
    top_k: int,
    max_sources: int,
) -> list[SFTExample]:
    # Lazy import so --help and lightweight checks do not pull model/DB dependencies.
    from pipeline.rag_query import RAGEngine

    engine = RAGEngine(top_k=top_k)
    examples: list[SFTExample] = []

    for case in eval_cases:
        query = str(case.get("query", "")).strip()
        if not query:
            continue

        result = engine.answer_query(query, top_k=top_k)
        sources = result.get("sources", [])
        context = _build_context(sources, max_sources=max_sources)
        answer = _ensure_citations(str(result.get("answer", "")).strip(), len(sources))

        examples.append(
            SFTExample(
                example_id=str(case.get("id", f"case_{len(examples)+1}")),
                language=str(case.get("language", "unknown")),
                query=query,
                context=context,
                answer=answer,
            )
        )

    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT JSONL data from ENSIA eval cases")
    parser.add_argument(
        "--eval-cases",
        type=Path,
        default=ROOT_DIR / "tests" / "eval_cases.json",
        help="Path to eval_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data" / "processed" / "sft_train.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved chunks per query")
    parser.add_argument("--max-sources", type=int, default=3, help="Max context sources per example")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit number of produced examples (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt included in every sample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.eval_cases.exists():
        raise FileNotFoundError(f"Eval cases not found: {args.eval_cases}")

    eval_cases = _load_json(args.eval_cases)
    if not isinstance(eval_cases, list):
        raise ValueError("Eval cases must be a JSON list.")

    random.seed(args.seed)
    random.shuffle(eval_cases)

    if args.max_examples > 0:
        eval_cases = eval_cases[: args.max_examples]

    examples = build_examples(eval_cases=eval_cases, top_k=args.top_k, max_sources=args.max_sources)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_file:
        for ex in examples:
            out_file.write(json.dumps(ex.to_json(args.system_prompt), ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()

