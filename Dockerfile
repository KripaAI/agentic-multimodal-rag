# Linux runner for the mmrag command line, used for ingestion on Windows machines where Smart App
# Control blocks native libraries such as tiktoken (Phase 8). The project folder is mounted at
# /app at run time (see the `ingest` service in docker-compose.yml): only dependencies live here.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
WORKDIR /app
# Chromium: kaleido renders chart PNGs with a headless browser
RUN apt-get update && apt-get install -y --no-install-recommends chromium && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt pytest
# Phase 10: LOCAL_EMBED=1 adds a local embedding model's packages (CPU PyTorch) for config.gemma.yaml.
ARG LOCAL_EMBED=0
COPY requirements-local-embed.txt /tmp/requirements-local-embed.txt
RUN if [ "$LOCAL_EMBED" = "1" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
      pip install -r /tmp/requirements-local-embed.txt; \
    fi

ENTRYPOINT ["python", "-m", "mmrag.cli"]
