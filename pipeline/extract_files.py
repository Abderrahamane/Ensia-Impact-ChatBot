"""Phase 1.5: Extract text from attached PDFs/photos and enrich messages.json.

Run AFTER parse_json.py and BEFORE chunk_and_embed.py.

Features:
- PDF text extraction via PyMuPDF (fitz)
- Image OCR via pytesseract (if installed)
- Keeps message linkage via existing message_id/file_name metadata
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from config import PROCESSED_DIR
from pipeline.parse_json import build_raw_content


MESSAGES_FILE = PROCESSED_DIR / "messages.json"
MEDIA_STATS_FILE = PROCESSED_DIR / "media_stats.json"


def _clean_text(text: str) -> str:
	text = re.sub(r"\s+", " ", text or "").strip()
	return text


def extract_pdf_text(pdf_path: Path, max_pages: int = 30) -> str:
	import fitz

	if not pdf_path.exists():
		return ""

	doc = fitz.open(pdf_path)
	parts: list[str] = []
	try:
		for i, page in enumerate(doc):
			if i >= max_pages:
				break
			parts.append(page.get_text("text") or "")
	finally:
		doc.close()

	return _clean_text("\n".join(parts))


def extract_image_text(image_path: Path) -> str:
	from PIL import Image
	import pytesseract

	# Make OCR work on Windows even when Tesseract is installed but not on PATH.
	tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
	if tesseract_cmd:
		pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
	else:
		default_win = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
		if default_win.exists():
			pytesseract.pytesseract.tesseract_cmd = str(default_win)

	if not image_path.exists():
		return ""

	image = Image.open(image_path)
	try:
		text = pytesseract.image_to_string(image)
	finally:
		image.close()
	return _clean_text(text)


def extract_media_text_for_message(msg: dict[str, Any]) -> tuple[str, str]:
	"""Return (media_text, media_source) for one message."""
	content_type = (msg.get("content_type") or "").lower()
	file_path = msg.get("file_path")
	photo_path = msg.get("photo_path")

	try:
		if content_type == "file" and file_path:
			path = Path(file_path)
			suffix = path.suffix.lower()
			if suffix == ".pdf":
				return extract_pdf_text(path), "pdf"
			if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
				return extract_image_text(path), "image_file"
			return "", "unsupported_file"

		if content_type == "photo" and photo_path:
			return extract_image_text(Path(photo_path)), "photo"
	except ModuleNotFoundError as err:
		# Optional dependencies missing (e.g. pytesseract or fitz)
		return "", f"missing_dependency:{err.name}"
	except Exception as err:  # noqa: BLE001
		if err.__class__.__name__ == "TesseractNotFoundError":
			return "", "missing_tesseract"
		return "", f"error:{err.__class__.__name__}"

	return "", "none"


def build_enriched_raw_content(msg: dict[str, Any], media_text: str, media_source: str) -> str:
	base = build_raw_content(
		text=msg.get("text", "") or "",
		links=msg.get("links", []) or [],
		file_name=msg.get("file_name", "") or "",
		photo_path=msg.get("photo_path"),
	)
	if not media_text:
		return base

	if media_source == "pdf":
		return f"{base}\n\n[Extracted PDF text]\n{media_text}".strip()
	return f"{base}\n\n[Extracted image text]\n{media_text}".strip()


def enrich_messages(messages: list[dict[str, Any]], limit: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	stats = {
		"total_messages": len(messages),
		"processed_media_messages": 0,
		"pdf_extracted": 0,
		"image_extracted": 0,
		"missing_tesseract": 0,
		"missing_dependency": 0,
		"unsupported_or_empty": 0,
		"errors": 0,
	}

	for i, msg in enumerate(messages):
		if limit and i >= limit:
			break

		if msg.get("content_type") not in {"file", "photo"}:
			continue

		stats["processed_media_messages"] += 1
		media_text, media_source = extract_media_text_for_message(msg)

		msg["media_text"] = media_text
		msg["media_source"] = media_source
		msg["raw_content"] = build_enriched_raw_content(msg, media_text, media_source)

		if media_text and media_source == "pdf":
			stats["pdf_extracted"] += 1
		elif media_text and media_source in {"photo", "image_file"}:
			stats["image_extracted"] += 1
		elif media_source == "missing_tesseract":
			stats["missing_tesseract"] += 1
		elif media_source.startswith("missing_dependency"):
			stats["missing_dependency"] += 1
		elif media_source.startswith("error"):
			stats["errors"] += 1
		else:
			stats["unsupported_or_empty"] += 1

	return messages, stats


def main() -> None:
	parser = argparse.ArgumentParser(description="Extract media text and enrich messages.json")
	parser.add_argument("--limit", type=int, default=0, help="Process only first N messages (debug)")
	args = parser.parse_args()

	if not MESSAGES_FILE.exists():
		raise RuntimeError(f"messages.json not found: {MESSAGES_FILE}. Run parse_json.py first.")

	with open(MESSAGES_FILE, encoding="utf-8") as f:
		messages = json.load(f)

	enriched, stats = enrich_messages(messages, limit=args.limit)

	with open(MESSAGES_FILE, "w", encoding="utf-8") as f:
		json.dump(enriched, f, ensure_ascii=False, indent=2)

	with open(MEDIA_STATS_FILE, "w", encoding="utf-8") as f:
		json.dump(stats, f, ensure_ascii=False, indent=2)

	print("Media extraction complete.")
	print(stats)
	if stats.get("missing_tesseract"):
		print("Note: Install Tesseract OCR engine to extract text from photos/images.")


if __name__ == "__main__":
	main()

