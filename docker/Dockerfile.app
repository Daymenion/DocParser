FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-app.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

ENV PYTHONPATH=/app

# App config
ENV APP_PORT=7860 \
    VLLM_HOST=doc-parser-vllm \
    VLLM_PORT=8000

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
