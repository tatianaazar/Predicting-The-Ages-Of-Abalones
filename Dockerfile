# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small while matching a stable TF-compatible version
FROM python:3.11-slim

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
# (only re-runs if requirements.txt changes, not on every code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code and models ───────────────────────────────────────────────
COPY src/ ./src/
COPY models/ ./models/

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the FastAPI server ──────────────────────────────────────────────────
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
