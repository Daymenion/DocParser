FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# requirements.txt kökte
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# app kodu
COPY . /workspace

# start script
COPY docker/start-app.sh /opt/entrypoint/start-app.sh
RUN chmod +x /opt/entrypoint/start-app.sh

ENV APP_PORT=7860
EXPOSE 7860

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/entrypoint/start-app.sh"]
