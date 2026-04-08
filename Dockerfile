# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HuggingFace Spaces runs as a non-root user (uid 1000)
RUN useradd -m -u 1000 hfuser

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
# Use --chown to avoid a separate chown layer (HF Spaces best practice)
COPY --chown=hfuser:hfuser . .

USER hfuser

# ── Runtime ───────────────────────────────────────────────────────────────────
# HF Spaces exposes port 7860.
ENV HOME=/home/hfuser
ENV PORT=7860
ENV WORLD_SIZE=200
ENV NUM_OBSTACLES=15

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
