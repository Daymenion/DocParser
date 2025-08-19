# Tek serviste hem vLLM hem FastAPI: resmî vLLM imajını baz al
FROM vllm/vllm-openai:latest

# Sağlık kontrolü ve init için ufak araçlar
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl tini \
    && rm -rf /var/lib/apt/lists/*

# Uygulama bağımlılıkları
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Uygulama kodları
COPY . /app

# Giriş betiği
COPY docker/start-app.sh /opt/entrypoint/start-app.sh
RUN chmod +x /opt/entrypoint/start-app.sh

# Varsayılanlar (compose ile override edebilirsin)
ENV APP_PORT=7860 \
    VLLM_PORT=9998 \
    HF_MODEL_PATH=/workspace/weights/DotsOCR

EXPOSE 7860 9998

# Tini ile düzgün sinyal aktarımı / process reaping
ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/entrypoint/start-app.sh"]
