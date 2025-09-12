# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl ca-certificates \
    build-essential cmake ninja-build \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 libx11-6 \
    locales \
    libreoffice \
    fonts-dejavu fonts-noto fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
    locale-gen en_US.UTF-8

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio && \
    pip install -r /app/requirements.txt && \
    pip install fastapi uvicorn[standard]

COPY . /app

# Default to API. The Gradio demo can be run separately if needed.
EXPOSE 7860

# Provide an entrypoint to run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
