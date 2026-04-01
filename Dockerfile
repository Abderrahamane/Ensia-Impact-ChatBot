FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime dirs for logs/backups/data
RUN mkdir -p /app/logs /app/backups /app/data

CMD ["python", "bot/telegram_bot.py"]

