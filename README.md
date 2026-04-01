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
- In bot replies, media source lines now include a human-readable topic preview (from file name or OCR/text preview) instead of only raw message IDs
- Source lines include trust labels (`high`/`medium`) and first available deep-link URL when present
- Route message intent in `bot/telegram_bot.py`:
  - `smalltalk` (greetings/general chat) -> natural conversational response (no retrieval)
  - `ensia_query` (ENSIA-related) -> full RAG answer + optional sources
- UX transparency commands:
  - `/why` explains why last answer was chosen (intent, top entities, top source)
  - `/mode` shows current backend, reranker models, and confidence thresholds
  - `/commands` lists all available commands with brief descriptions
  - `/feedback_buttons on|off` toggles wrong/correct feedback buttons (admin: global, normal user: per-account)
  - `/quality` (admin) shows recent wrong/correct ratio, top failed queries, and last eval gate status
- Feedback controls in bot answers:
  - `Wrong answer` button stores full query/mode/sources snapshot
  - `Correct answer` button stores validated snapshots and reinforces successful patterns
  - admin command `/feedback_stats` shows live per-query wrong/correct counters

## Important current behavior

- The bot now has an intent router before retrieval.
- If your message is conversational (example: "hi", "how are you", "who are you"), the bot replies normally without querying ENSIA index.
- If your message looks ENSIA-related (internships, partnerships, incubator, FYP, events, etc.), the bot runs RAG and can attach sources depending on your `/sources` preference.
- This avoids irrelevant retrieval answers for pure chat while keeping ENSIA answers grounded.
- Bot replies are polished to match user language style (Arabic/French/English headers and smalltalk tone).
- Source deep-linking is included when URLs are available in source metadata.

## Production hardening (implemented)

- Response caching for repeated questions:
  - `ENSIA_BOT_CACHE_ENABLED`
  - `ENSIA_BOT_CACHE_TTL_SECONDS`
  - `ENSIA_BOT_CACHE_MAX_ITEMS`
- Warm-up on startup (intent router + RAG reranker/BM25):
  - `ENSIA_BOT_WARMUP_ON_START`
- Feedback-adaptive behavior:
  - repeated wrong votes push stricter answers/clarification
  - repeated correct votes reinforce current response path
  - controls: `ENSIA_FEEDBACK_ADAPTATION_ENABLED`, `ENSIA_FEEDBACK_WRONG_THRESHOLD`, `ENSIA_FEEDBACK_CORRECT_THRESHOLD`
- Background structured-table rebuild after incremental reindex:
  - `ENSIA_AUTO_REBUILD_STRUCTURED_ON_REINDEX=1`

Secrets hygiene (required):

- Rotate exposed tokens immediately (Telegram/Groq keys).
- Keep `.env` out of git and replace shared credentials with placeholders.

## Phase 1 quick wins (implemented)

- Query rewriting before retrieval:
  - typo normalization (example: `parternships` -> `partnerships`)
  - lightweight grammar cleanup and ENSIA intent expansion terms
- Hybrid retrieval:
  - dense vector retrieval + lexical BM25-style scoring fusion
  - weighted fusion controlled by env vars
- Context packing improvements:
  - deduplicate by `message_id`
  - diversity-aware selection to reduce near-duplicate chunks in context
- Answer policy:
  - responses are instructed to follow `Direct answer`, `Key points`, and `If unsure` sections

## Phase 2 quality upgrades (implemented)

- Multilingual reranker fallback:
  - primary: `ENSIA_RERANKER_MODEL`
  - fallback: `ENSIA_RERANKER_MULTILINGUAL_MODEL` for FR/AR/EN robustness
- Tiny trained intent classifier:
  - centroid classifier from `assets/intent_classifier_seed.json`
  - intents: `smalltalk`, `ensia_query`, `admin_op`
- Dynamic confidence calibration by intent type:
  - partnerships use stricter thresholds
  - events use slightly lower thresholds
- Entity-aware retrieval:
  - detects company names, years/dates, and event cues from query
  - boosts chunks containing matched entities
- Dedicated structured extractors:
  - `pipeline/build_structured_tables.py` builds
    - `data/processed/partnerships_table.json`
    - `data/processed/events_timeline.json`
  - these tables are injected into retrieval context for exact partnership/event answers

Additional high-impact upgrades (implemented):

- Clarification loop on low confidence:
  - for partnership/event intents, bot asks a follow-up question instead of giving a loose answer
- Structured answer mode by intent:
  - partnerships -> table-style partner list
  - events -> timeline-style bullets
  - resources -> link list when URLs are present
- Hybrid retriever v2 (entity-first):
  - forces 1-2 entity-matching chunks into final context when company/date/event entities are detected
- Source trust scoring:
  - each source now includes a trust label (`high` or `medium`) based on score, recency, and entity match
- Answer-level fact dedup:
  - repeated bullet facts are removed in final post-processing

Build/update Phase 2 artifacts:

```cmd
python training/train_intent_classifier.py
python pipeline/build_structured_tables.py
```

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

## Deployment (current laptop-hosted model)

You now have two user channels:

1. Telegram bot (`bot/telegram_bot.py`)
2. Website chat (`web/app.py`)

Install dependencies:

```cmd
python -m pip install -r requirements.txt
```

Run both services locally (single command):

```cmd
python ops/run_local_deploy.py
```

Or run separately:

```cmd
python web/app.py
python bot/telegram_bot.py
```

Website URL:

- `http://127.0.0.1:8000`

Frontend notes:

- ChatGPT-style layout with sidebar + chat bubbles
- Enter to send, Shift+Enter for newline
- New Chat button clears local browser chat history
- Health-aware input lock when backend is down

Server-down behavior:

- If the local model endpoint is down while web app is running, the website shows `Server is off, try later` and disables sending.
- If your laptop is fully off, no local service can be reached (Telegram/web hosted on same laptop are unavailable).

## Daily freshness automation

The project includes a daily pipeline runner:

```cmd
python ops/run_daily_freshness.py
```

Dry-run:

```cmd
python ops/run_daily_freshness.py --dry-run
```

Steps:

1. `pipeline/parse_json.py`
2. `pipeline/extract_files.py`
3. `pipeline/reindex_incremental.py`
4. `pipeline/build_structured_tables.py`

GitHub schedule file:

- `.github/workflows/daily-freshness.yml`

## Datacenter migration (later, minimal changes)

To move inference from laptop to datacenter later, keep the same app and only change runtime env/config:

- Keep Telegram + web code unchanged.
- Point generation backend to datacenter endpoint/model settings.
- Keep freshness and quality scripts unchanged.

Recommended migration checklist:

1. Deploy repository on datacenter VM/server.
2. Install dependencies and model runtime there.
3. Set production `.env` on server (no local secrets reuse).
4. Run `python ops/run_local_deploy.py` (or separate process manager services).
5. Update DNS/reverse proxy for website URL.

## Windows auto-start on boot (production script)

Files:

- `deploy/windows/start_ensia_stack.bat`
- `deploy/windows/ensia_stack_on_boot.xml`
- `deploy/windows/install_task.bat`
- `deploy/windows/uninstall_task.bat`
- `deploy/windows/ensia_health_tray.ps1`
- `deploy/windows/start_tray_monitor.bat`

Install scheduled task (run as administrator CMD):

```cmd
cd /d C:\Users\MICROSOFT\PycharmProjects\ENSIA_IMPACT_Group_ChatBot
deploy\windows\install_task.bat
```

Manual test:

```cmd
schtasks /Run /TN "ENSIA_IMPACT_Stack_OnBoot"
```

Uninstall startup task:

```cmd
deploy\windows\uninstall_task.bat
```

Run tray health monitor (shows web/bot/api status):

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File deploy\windows\ensia_health_tray.ps1
```

One double-click launcher:

```cmd
deploy\windows\start_tray_monitor.bat
```

Quick one-shot health check (no tray):

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -File deploy\windows\ensia_health_tray.ps1 -Once
```

Logs are written to:

- `logs/web_stdout.log`, `logs/web_stderr.log`
- `logs/bot_stdout.log`, `logs/bot_stderr.log`

## Website deployment prep

`render.yaml` is prepared with two services:

- `ensia-impact-web` (FastAPI website)
- `ensia-impact-bot` (Telegram worker)

Note: while model inference stays local on your laptop, remote web/worker deployment cannot access local inference unless you later expose/migrate backend inference endpoint (recommended in datacenter phase).

Frontend-only deploy notes:

- You can deploy only `web/static` to any static host (Vercel/Netlify/GitHub Pages).
- A ready Vercel config is provided at `deploy/frontend/vercel.json`.
- In deployed frontend, use the sidebar field `API base URL` and click `Save endpoint`.
- Set it to your reachable backend URL (for now local or tunneled endpoint, later datacenter API).

Examples:

```text
http://127.0.0.1:8000
https://your-backend-domain.example
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

Query rewrite / hybrid retrieval controls:

- `ENSIA_QUERY_REWRITE_ENABLED` (default `1`)
- `ENSIA_HYBRID_RETRIEVAL_ENABLED` (default `1`)
- `ENSIA_HYBRID_DENSE_WEIGHT` (default `0.65`)
- `ENSIA_HYBRID_BM25_WEIGHT` (default `0.35`)
- `ENSIA_BM25_TOP_N` (default `40`)
- `ENSIA_CONTEXT_DIVERSITY_ENABLED` (default `1`)
- `ENSIA_CONTEXT_DIVERSITY_MAX_SIM` (default `0.72`)

Phase 2 intent/reranker controls:

- `ENSIA_RERANKER_MULTILINGUAL_MODEL` (default `BAAI/bge-reranker-v2-m3`)
- `ENSIA_INTENT_CLASSIFIER_ENABLED` (default `1`)
- `ENSIA_INTENT_CLASSIFIER_MIN_CONFIDENCE` (default `0.42`)
- `ENSIA_CONF_PARTNERSHIP_TOP` / `ENSIA_CONF_PARTNERSHIP_AVG`
- `ENSIA_CONF_EVENT_TOP` / `ENSIA_CONF_EVENT_AVG`

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

## Critical quality + evaluation pipeline

Build a failure benchmark set from logs and wrong-answer feedback:

```cmd
python tests/build_failure_benchmark.py --max-cases 100
```

Run the full quality pipeline (auto-eval + regression gate):

```cmd
python ops/run_quality_gate.py
```

This pipeline checks:

- `intent_accuracy`
- retrieval `hit@1`, `hit@3`, `hit@5`
- `groundedness`
- `citation_usefulness`

Regression gate:

- Baseline file: `tests/eval_baseline.json`
- Current report: `tests/eval_rag_report.json`
- Gate script: `tests/eval_regression_gate.py`
- Fails with exit code `1` when metrics drop beyond tolerance

CI automation:

- GitHub Actions workflow runs on each push/PR: `.github/workflows/quality-gate.yml`

## Daily data freshness automation

Run full refresh pipeline manually:

```cmd
python ops/run_daily_freshness.py
```

Dry-run (show planned steps only):

```cmd
python ops/run_daily_freshness.py --dry-run
```

Pipeline order:

1. `pipeline/parse_json.py`
2. `pipeline/extract_files.py`
3. `pipeline/reindex_incremental.py`
4. `pipeline/build_structured_tables.py`

Scheduled automation:

- GitHub Actions daily cron workflow: `.github/workflows/daily-freshness.yml`

Wrong-answer feedback button:

- Bot replies include a `Wrong answer` button.
- On click, snapshot is stored in `data/processed/feedback.jsonl` with:
  - original query
  - mode / intent type
  - top sources
  - generation error (if any)

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

