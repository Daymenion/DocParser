FROM vllm/vllm-openai:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt proje kökünde
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# transformers uyumu (layer_type_validation hatasını önler)
RUN pip install --no-cache-dir --upgrade "transformers>=4.53.0"

# Uygulama kodu
COPY . /app

# Start script'i image içine
COPY docker/start-app.sh /opt/entrypoint/start-app.sh
RUN chmod +x /opt/entrypoint/start-app.sh

ENV APP_PORT=7860 \
    VLLM_PORT=6013 \
    HF_MODEL_PATH=/workspace/weights/DotsOCR

EXPOSE 7860 6013

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/entrypoint/start-app.sh"]
