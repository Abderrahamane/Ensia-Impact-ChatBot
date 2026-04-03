# Portfolio + Free Cloud Deployment Guide

This guide gives you a practical path to publish this chatbot on GitHub, add it to your portfolio, and keep it online without your laptop.

## 1) Make your repository public-ready

1. Confirm sensitive data is ignored:
   - `.env`
   - `data/raw/`
   - `data/processed/`
   - `data/vectorstore/`
2. Create a clean `.env` from `.env.example`.
3. Keep only sample/demo-safe assets in GitHub.
4. Commit legal/security docs (`LICENSE`, `SECURITY.md`).

## 2) Create your GitHub repository

```cmd
git init
git add .
git commit -m "Prepare ENSIA chatbot for portfolio deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/ensia-impact-chatbot.git
git push -u origin main
```

## 3) Pick free cloud architecture (recommended)

### Option A (recommended for always-on): Oracle Cloud Always Free VM

- Pros: best chance of true 24/7 free hosting, full control.
- Cons: setup takes longer than one-click platforms.

Run web + bot with process manager (`systemd`) or Docker Compose on the VM.
Persist `data/` as a mounted volume.

### Option B (faster MVP): Render + static frontend

- Pros: easy deploy UX.
- Cons: free tiers can sleep or limit uptime by provider policy.

Use this option for demos, then migrate to Oracle VM for stronger always-on behavior.

## 4) Deploy backend API and Telegram worker

### VM-based deployment checklist

1. Install Python 3.11+, git, and build tools.
2. Clone repository.
3. Create `.env` on server with production keys.
4. Install dependencies:

```cmd
python -m pip install -r requirements.txt
```

5. Build data index (if needed):

```cmd
python pipeline/parse_json.py
python pipeline/extract_files.py
python pipeline/chunk_and_embed.py
python pipeline/build_structured_tables.py
```

6. Start services:
   - Web: `uvicorn web.app:app --host 0.0.0.0 --port 8000`
   - Bot: `python bot/telegram_bot.py`
7. Add restart policy (`systemd` or Docker restart always).
8. Add reverse proxy + HTTPS (Nginx/Caddy).

## 5) Deploy your portfolio website

The UI is already in `web/static/`.

Two ways:

1. Serve directly from FastAPI (`/` route in `web/app.py`).
2. Or deploy static site separately (Vercel/Netlify) and set API URL in sidebar.

## 6) Portfolio content to show

Include these on your portfolio page:

- Problem solved: multilingual Q/A over ENSIA group knowledge.
- Architecture: Telegram export -> parsing -> embeddings -> Chroma -> RAG -> web/Telegram.
- Features: source-grounded answers, trust labels, partnership/event extraction.
- Engineering: retries, confidence gates, feedback logging, backup scripts.
- Demo links: web app URL + GitHub repo.

## 7) Reliability checklist (must-have)

- Add health check monitor on `/api/health`.
- Run daily backup (`ops/backup_data.py`).
- Rotate API keys every time you accidentally expose them.
- Keep logs (`logs/`) and watch crash loops.
- Test restart after VM reboot.

## 8) Cost and uptime reality

- "Free forever + always-on" is hardest on one-click PaaS.
- For strongest free always-on outcome, use Oracle Always Free VM.
- Keep a fallback static page message if backend is temporarily unavailable.

