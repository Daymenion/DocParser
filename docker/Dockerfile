# Tek serviste hem vLLM hem FastAPI
FROM vllm/vllm-openai:latest

# Küçük yardımcılar
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt proje KÖKÜNDE; build context'i kök yaptığımız için direkt kopyalanır
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Uygulama kodlarını kökten kopyala (api/, app.py vs. dahil)
COPY . /app

# Giriş script'i docker klasöründe
COPY docker/start-app.sh /opt/entrypoint/start-app.sh
RUN chmod +x /opt/entrypoint/start-app.sh

# Varsayılan ortam
ENV APP_PORT=7860 \
    VLLM_PORT=9998 \
    HF_MODEL_PATH=/workspace/weights/DotsOCR

EXPOSE 7860 9998

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/entrypoint/start-app.sh"]
