# SellerPolicy Assistant - production image
#
# Build:  docker compose build        Run:  docker compose up -d
# The app is served by Streamlit on port 8501 inside the container; Traefik (or Caddy)
# in front of it provides the domain name and HTTPS.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# 1) Dependencies first, so Docker can cache this slow layer between code changes.
#    CPU-only PyTorch keeps the image ~2 GB smaller (no CUDA libraries on a VPS).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Application code (see .dockerignore for what is left out, e.g. .env and .git).
COPY . .

# 3) Build the BM25 + vector indexes at build time. This also downloads the embedding
#    model once into the image, so the container starts fast and works offline.
ARG EMBEDDING_PROVIDER=sentence-transformers
ENV EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER}
RUN python -m scripts.build_index

# 4) Run as a normal user, not root (safer if the app is ever compromised).
#    The user owns /app so the "Rebuild index" button can still write to storage/.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Docker marks the container "unhealthy" if Streamlit stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
