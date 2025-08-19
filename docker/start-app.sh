#!/usr/bin/env bash
set -Eeuo pipefail

: "${APP_PORT:=7860}"
: "${VLLM_HOST:=host.docker.internal}"   # hosttaki vLLM
: "${VLLM_PORT:=6013}"

echo "APP_PORT   = ${APP_PORT}"
echo "VLLM_HOST  = ${VLLM_HOST}"
echo "VLLM_PORT  = ${VLLM_PORT}"

# vLLM health (bekle ama bloklama yok – 20 sn, geçilemezse uyar ve yine de app'i başlat)
for _ in {1..20}; do
  if curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM reachable."
    break
  fi
  sleep 1
done || true

# FastAPI (api/server.py içinde app objesi var demiştin)
exec uvicorn api.server:app --host 0.0.0.0 --port "${APP_PORT}"
