# ENSIA IMPACT Group ChatBot

RAG-powered chatbot over Telegram-exported ENSIA IMPACT content.

## Current status

- Phase 1: `pipeline/parse_json.py` (done)
- Phase 2: `pipeline/chunk_and_embed.py` (done)
- Phase 3: `pipeline/rag_query.py` (done)
- Telegram interface: `bot/telegram_bot.py` (ready)

## What the bot can do today

- Answer ENSIA IMPACT questions using RAG over indexed Telegram exports
- Work in Arabic, French, and English with multilingual embeddings
- Retrieve from:
  - text messages
  - attached PDFs (extracted text)
  - photos/images (OCR text)
- Show grounded sources with message references (`date`, `from`, `message_id`)
- In bot replies, source lines can include `content_type` and `file_name` when relevant
- Route message intent in `bot/telegram_bot.py`:
  - `smalltalk` (greetings/general chat) -> natural conversational response (no retrieval)
  - `ensia_query` (ENSIA-related) -> full RAG answer + optional sources

## Important current behavior

- The bot now has an intent router before retrieval.
- If your message is conversational (example: "hi", "how are you", "who are you"), the bot replies normally without querying ENSIA index.
- If your message looks ENSIA-related (internships, partnerships, incubator, FYP, events, etc.), the bot runs RAG and can attach sources depending on your `/sources` preference.
- This avoids irrelevant retrieval answers for pure chat while keeping ENSIA answers grounded.

## Project layout

- `config.py`: environment-driven settings (RAG, bot, logging, retries)
- `data/raw/result.json`: Telegram export
- `data/processed/messages.json`: cleaned message records
- `data/processed/chunks.json`: chunked records
- `data/processed/user_prefs.json`: per-user source visibility prefs
- `data/processed/feedback.jsonl`: user feedback logs
- `data/vectorstore/`: Chroma persistent index
- `pipeline/parse_json.py`: JSON cleaning/parsing
- `pipeline/extract_files.py`: PDF/OCR extraction into message content
- `pipeline/chunk_and_embed.py`: chunking + embeddings + index build
- `pipeline/reindex_incremental.py`: partial updates without full rebuild
- `pipeline/rag_query.py`: retrieval + generation backends
- `bot/telegram_bot.py`: Telegram runtime + intent routing + admin commands
- `ops/backup_data.py`: backup automation for processed data/vectorstore
- `training/prepare_sft_data.py`: build SFT JSONL from retrieval-grounded eval queries
- `training/train_sft_lora.py`: LoRA SFT training entrypoint
- `requirements-train.txt`: optional dependencies for model fine-tuning

## Quick start

```bash
python -m pip install -r requirements.txt
python pipeline/parse_json.py
python pipeline/extract_files.py
python pipeline/chunk_and_embed.py
python pipeline/rag_query.py --query "Where can I find internship opportunities in AI?"
```

Media extraction notes:

- `pipeline/extract_files.py` enriches `data/processed/messages.json` by extracting text from:
  - PDFs in attachments (`content_type=file`)
  - Photos/images (`content_type=photo`) via OCR
- Ensure Tesseract OCR engine is installed on your OS for image extraction.
- Extracted text is appended to each message `raw_content`, so answers can be traced back to the original `message_id` source.

Windows (optional OCR setup):

```bash
winget install UB-Mannheim.TesseractOCR
```

Then rerun extraction and embedding:

```bash
python pipeline/extract_files.py
python pipeline/chunk_and_embed.py
```

## Optional: Gemini generation backend

By default, Phase 3 uses `extractive` mode (no paid API needed).

To enable Gemini answers:

```bash
set "ENSIA_GENERATION_BACKEND=gemini"
set "GOOGLE_API_KEY=your_key_here"
set "ENSIA_GEMINI_MODEL=gemini-2.0-flash"
set "ENSIA_GEMINI_FALLBACK_MODELS=gemini-2.0-flash-lite,gemini-1.5-flash"
```

Then run:

```bash
python pipeline/rag_query.py --query "Donne moi des opportunites de stage"
```

If you hit Gemini quota errors (`429 RESOURCE_EXHAUSTED`), switch to Groq backend instead of retrying Gemini:

```cmd
set "ENSIA_GENERATION_BACKEND=groq"
set "GROQ_API_KEY=your_groq_key_here"
set "ENSIA_GROQ_MODEL=llama-3.3-70b-versatile"
python -m pip install -r requirements.txt
python pipeline/rag_query.py --query "Where can I find internship opportunities in AI?"
```

## Optional: Groq generation backend

```cmd
set "ENSIA_GENERATION_BACKEND=groq"
set "GROQ_API_KEY=your_groq_key_here"
set "ENSIA_GROQ_MODEL=llama-3.3-70b-versatile"
python -m pip install -r requirements.txt
python pipeline/rag_query.py --query "What are current ENSIA partnerships?"
```

If you still get connection errors:

- Ensure only one network/VPN policy is active
- Retry after 30-60 seconds (transient provider/network failures are retried)
- Verify your key by running the same command in a fresh shell
- Restart the bot process after editing `.env` (config is loaded at process start)

## Quota-free development mode (no API quota)

If you want to avoid provider quotas during development, use a local CPU model via Ollama.

```cmd
set "ENSIA_GENERATION_BACKEND=local_model_1"
set "ENSIA_LOCAL_BASE_URL=http://127.0.0.1:11434"
set "ENSIA_LOCAL_MODEL_1=qwen2.5:1.5b-instruct"
set "ENSIA_LOCAL_MODEL_2=phi3:mini"
set "ENSIA_ALLOW_EXTRACTIVE_FALLBACK=0"
python pipeline/rag_query.py --query "search about Global Africa Tech 2026 in this server"
```

Switch backend instantly by changing one variable:

- `ENSIA_GENERATION_BACKEND=groq`
- `ENSIA_GENERATION_BACKEND=local_model_1`
- `ENSIA_GENERATION_BACKEND=local_model_2`
- `ENSIA_GENERATION_BACKEND=hf_lora`

Strict mode behavior:

- Set `ENSIA_ALLOW_EXTRACTIVE_FALLBACK=0` to disable extractive fallback entirely.
- Use confidence gates before generation:
  - `ENSIA_GENERATION_MIN_TOP_SCORE`
  - `ENSIA_GENERATION_MIN_AVG_SCORE`
- If confidence is low, the bot refuses generation and asks for a more specific query.

Intent and reranking controls:

- `ENSIA_INTENT_MIN_SCORE` (default `0.35`): hard ENSIA intent threshold before retrieval.
- `ENSIA_RERANKER_ENABLED` (default `1`): enable cross-encoder reranking.
- `ENSIA_RERANKER_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- `ENSIA_RERANK_CANDIDATES` (default `24`): max dense candidates passed to reranker.

Retrieval quality safeguards:

- Retrieved chunks are deduplicated by `message_id` before building context.
- Grounded prompts include a hard guardrail: `Do not invent ENSIA-specific facts not in context.`

## Optional: Hugging Face LoRA generation backend

After training and saving an adapter, you can run the bot with local HF generation:

```cmd
set "ENSIA_GENERATION_BACKEND=hf_lora"
set "ENSIA_HF_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct"
set "ENSIA_HF_LORA_ADAPTER_DIR=artifacts\lora-ensia-assistant"
set "ENSIA_HF_MAX_NEW_TOKENS=220"
set "ENSIA_HF_TEMPERATURE=0.2"
python pipeline/rag_query.py --query "What are current company partnerships with ENSIA school?"
```

Notes:

- If `ENSIA_HF_LORA_ADAPTER_DIR` is empty, the backend uses the base model only.
- Install training/runtime deps first: `python -m pip install -r requirements-train.txt`
- If tokenizer conversion fails, ensure `sentencepiece` and `tiktoken` are installed from `requirements-train.txt`.
- If HF loading/generation fails, the system gracefully falls back to extractive mode.

## Run Telegram bot

```bash
set "ENSIA_ADMIN_USER_IDS=123456789"
set "TELEGRAM_BOT_TOKEN=your_bot_token_here"
python bot/telegram_bot.py
```

## P0 Quality Gate (multilingual eval)

This project includes 24 multilingual checks in `tests/eval_cases.json`
(`en`, `fr`, `ar`, and mixed).

Run the evaluator:

```bash
python tests/eval_rag.py --top-k 5 --verbose
```

The evaluator tracks:

- Retrieval: `hit@1`, `hit@3`, `hit@5`
- Answer groundedness (token support from retrieved source previews)
- Citation usefulness (citation presence, metadata quality, and relevance)

It writes a report to `tests/eval_rag_report.json` and exits with code `1`
if thresholds are not met.

## Project report (snapshot)

Snapshot based on latest local pipeline runs (March 30, 2026):

- Dataset records in `data/processed/messages.json`: `409`
- Media-bearing messages processed by `pipeline/extract_files.py`: `147`
- PDF texts extracted: `10`
- Photo/image OCR texts extracted: `131`
- Indexed chunks after latest embed pass: `1448`
- Vector store: `data/vectorstore/` (Chroma)

Implemented system components:

- Data parsing/cleaning: `pipeline/parse_json.py`
- Media extraction (PDF + OCR): `pipeline/extract_files.py`
- Chunking/indexing: `pipeline/chunk_and_embed.py`
- Retrieval + generation: `pipeline/rag_query.py`
- Telegram runtime and admin ops: `bot/telegram_bot.py`
- Incremental updates: `pipeline/reindex_incremental.py`
- Backup/ops: `ops/backup_data.py`

Evaluation and observability artifacts:

- Eval cases: `tests/eval_cases.json`
- Eval report: `tests/eval_rag_report.json`
- Parse stats: `data/processed/stats.json`
- Media extraction stats: `data/processed/media_stats.json`
- Bot logs: `logs/bot.log`

## Current limitations

- Intent routing is keyword/rule-based (not a classifier), so some edge cases can still be misrouted
- OCR quality depends on image quality and language in image text
- Retrieval can still miss narrow intent variants without query expansion/reranking tuning
- Extractive mode is grounded but can be less natural than LLM synthesis
- Gemini usage depends on project quota and model availability

## How to improve (recommended order)

1. Replace rule-based intent routing with a lightweight multilingual intent classifier
2. Extend evaluation set with small-talk and intent-switch regression checks
3. Improve retrieval with hybrid search (dense + keyword/BM25)
4. Add reranker model for better top-k relevance
5. Tune chunking/metadata filters for partnership/incubator/FYP intents
6. Keep ENSIA answers source-grounded; hide sources for small-talk by default
7. After retrieval quality is stable, fine-tune an answer-style model if needed

## Fine-tuning an LLM from Hugging Face (recommended path)

Fine-tuning can help style/fluency, but it will not fix missing facts if retrieval quality is weak.
For this project, best practice is:

1. Keep RAG as truth source (messages/files/OCR)
2. Fine-tune only for answer style, instruction-following, and multilingual tone
3. Continue to show sources from retrieval

Recommended setup: LoRA SFT on a 7B instruct model.

Suggested base models (choose one that fits your GPU budget):

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-2-9b-it` (if license/compute fit)

### Step 1: Prepare SFT data

Create `data/processed/sft_train.jsonl` in instruction format:

```json
{"messages":[{"role":"system","content":"You are ENSIA IMPACT assistant. Use provided context only and cite [Source n]."},{"role":"user","content":"Question: ...\nContext: [Source 1] ..."},{"role":"assistant","content":"Grounded answer ... [Source 1]"}]}
```

Use your existing evaluation cases and high-quality historical Q/A pairs to build 300-2000 examples.

### Step 2: Install training dependencies

```cmd
python -m pip install -r requirements-train.txt
```

### Step 3: Build SFT dataset

```cmd
python training/prepare_sft_data.py --top-k 5 --max-sources 3
```

Output file (default): `data/processed/sft_train.jsonl`

### Step 4: Run LoRA SFT training

```cmd
python training/train_sft_lora.py --train-file data/processed/sft_train.jsonl --base-model Qwen/Qwen2.5-7B-Instruct --output-dir artifacts/lora-ensia-assistant --epochs 1 --batch-size 1 --grad-accum 8 --max-seq-len 1024
```

Notes:

- Uses 4-bit automatically when CUDA is available (disable with `--no-4bit`)
- Saves adapter + tokenizer + `train_meta.json` to output dir
- Script supports multiple TRL versions (old/new constructor signatures)

### Step 5: Evaluate before integration

Run your current gate and compare baseline vs tuned model:

```cmd
python tests/eval_rag.py --top-k 5 --verbose
```

Track deltas on:

- `hit@k` (should stay stable; this is retrieval-side)
- groundedness (should improve or remain stable)
- citation usefulness (should improve formatting/compliance)

### Step 6: Integrate tuned model safely

Keep a switchable backend in `pipeline/rag_query.py` (baseline vs tuned).
Only promote tuned model if eval improves and hallucination risk does not increase.

## Exact setup: Gemini API key

1. Open Google AI Studio: `https://aistudio.google.com/`
2. Sign in with your Google account.
3. Create API key (Get API key -> Create API key in new/existing project).
4. In `cmd.exe`, set environment variables:

```bash
set "ENSIA_GENERATION_BACKEND=gemini"
set "GOOGLE_API_KEY=PASTE_YOUR_REAL_KEY_HERE"
set "ENSIA_GEMINI_MODEL=gemini-2.0-flash"
set "ENSIA_GEMINI_FALLBACK_MODELS=gemini-2.0-flash-lite,gemini-1.5-flash"
```

5. Test quickly:

```bash
python pipeline/rag_query.py --query "Ou trouver des opportunites de stage?"
```

If you get a model `NOT_FOUND` error, list available models for your key:

```bash
python -c "from google import genai; import os; c=genai.Client(api_key=os.environ['GOOGLE_API_KEY']); print('\n'.join([m.name for m in c.models.list() if 'generateContent' in getattr(m, 'supported_actions', [])]))"
```

## Exact setup: Telegram bot token

1. Open Telegram and chat with `@BotFather`.
2. Send `/newbot`.
3. Give bot name (example: `ENSIA Impact Assistant`).
4. Give bot username ending with `bot` (example: `ensia_impact_assistant_bot`).
5. Copy the bot token returned by BotFather.
6. In `cmd.exe`, set token:

```bash
set "TELEGRAM_BOT_TOKEN=PASTE_YOUR_BOT_TOKEN_HERE"
```

7. Start the bot:

```bash
python bot/telegram_bot.py
```

The bot uses long polling, so it keeps running and does not exit on its own.
Use `Ctrl+C` to stop it.

## P1 Bot Hardening

Implemented in `bot/telegram_bot.py`:

- Per-user cooldown + per-minute rate limit
- Retry + timeout around query generation
- `/sources on|off` toggle (persisted in `data/processed/user_prefs.json`)
- `/feedback <text>` command (stored in `data/processed/feedback.jsonl`)
- `/admins` to verify your Telegram user ID admin status
- Admin-only `/health`, `/stats`, `/backup_now [dry]`
- Structured JSON logs in `logs/bot.log`

Set admin IDs (comma-separated Telegram user IDs):

```bash
set "ENSIA_ADMIN_USER_IDS=123456789,987654321"
```

## P1 Data Freshness (incremental re-index)

Use incremental indexing when new Telegram export messages arrive:

```bash
python pipeline/extract_files.py
python pipeline/reindex_incremental.py --dry-run
python pipeline/reindex_incremental.py
```

This script updates only changed/new message chunks and removes deleted-message chunks,
without rebuilding the full vector index.

## P2 Deployment

`Dockerfile` and `.env.example` are included.

Build and run locally:

```bash
docker build -t ensia-impact-bot .
docker run --env-file .env -v %cd%\data:/app/data -v %cd%\logs:/app/logs ensia-impact-bot
```

Deployment notes:

- Render: use `render.yaml` or create a Worker service from this repo (Docker runtime).
- Railway: deploy repo with Dockerfile, set start command `python bot/telegram_bot.py`.
- VPS (systemd): use `deploy/ensia-bot.service` and set `Restart=always`.
- Keep `data/vectorstore` on persistent storage/volume.

Linux VPS auto-restart setup:

```bash
sudo cp deploy/ensia-bot.service /etc/systemd/system/ensia-bot.service
sudo systemctl daemon-reload
sudo systemctl enable ensia-bot
sudo systemctl restart ensia-bot
sudo systemctl status ensia-bot
```

## P2 Observability and backups

Daily backup script:

```bash
python ops/backup_data.py
```

Dry-run mode:

```bash
python ops/backup_data.py --dry-run
```

Windows Task Scheduler example action:

```bash
python C:\path\to\project\ops\backup_data.py
```

Linux cron daily backup example:

```bash
0 3 * * * /opt/ensia-impact-bot/.venv/bin/python /opt/ensia-impact-bot/ops/backup_data.py
```

## Notes

- Embedding model is multilingual (`Arabic/French/English`).
- Retrieval is grounded in your local Chroma index (`data/vectorstore`).
- If no reliable context is found, the bot asks users to rephrase instead of hallucinating.

