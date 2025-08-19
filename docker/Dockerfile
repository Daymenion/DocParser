FROM vllm/vllm-openai:latest

# Küçük yardımcılar
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Proje bağımlılıkları — CUDA Torch’u EZME!
# 1) requirements.txt’yi kopyala
COPY requirements.txt /app/requirements.txt
# 2) torch ve triton satırlarını ayıkla, geri kalanı kur
RUN awk '!/^[[:space:]]*torch([<=>[:space:]]|$)/ && !/^[[:space:]]*triton([<=>[:space:]]|$)/ {print}' \
      /app/requirements.txt > /app/requirements.nogpu.txt && \
    pip install --no-cache-dir -r /app/requirements.nogpu.txt

# (Gerekiyorsa transformer sürümü)
RUN pip install --no-cache-dir --upgrade "transformers>=4.53.0"

# Uygulama kodu
COPY . /app

# Start script
COPY docker/start-app.sh /opt/entrypoint/start-app.sh
RUN chmod +x /opt/entrypoint/start-app.sh

ENV APP_PORT=7860 \
    VLLM_PORT=6013 \
    HF_MODEL_PATH=/workspace/weights/DotsOCR

EXPOSE 7860 6013
ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/entrypoint/start-app.sh"]
