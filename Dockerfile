# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HuggingFace Spaces runs as a non-root user (uid 1000)
RUN useradd -m -u 1000 hfuser

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY . .
RUN chown -R hfuser:hfuser /app

USER hfuser

# ── Runtime ───────────────────────────────────────────────────────────────────
# HF Spaces exposes port 7860.
# Override PORT env-var for local dev (default 8000).
ENV PORT=7860
ENV WORLD_SIZE=200
ENV NUM_OBSTACLES=15

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
