#!/usr/bin/env bash
set -Eeuo pipefail

# ---- Parametreler / Defaults ----
: "${HF_MODEL_PATH:=/workspace/weights/DotsOCR}"  # klasör adı noktasız olmalı (DotsOCR)
: "${APP_PORT:=7860}"
: "${VLLM_PORT:=9998}"
: "${CUDA_VISIBLE_DEVICES:=${NVIDIA_VISIBLE_DEVICES:-all}}"

# PYTHONPATH: model klasörünün parent'ı
export PYTHONPATH="$(dirname "$HF_MODEL_PATH"):${PYTHONPATH:-}"

echo "HF_MODEL_PATH = ${HF_MODEL_PATH}"
echo "PYTHONPATH    = ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

# ---- Model klasörü kontrolü ----
if [[ ! -d "${HF_MODEL_PATH}" ]] || [[ -z "$(ls -A "${HF_MODEL_PATH}" 2>/dev/null || true)" ]]; then
  echo "ERROR: '${HF_MODEL_PATH}' yok veya boş. Host'tan doğru mount ettiğinden emin ol."
  exit 1
fi

# ---- vLLM entrypoint patch (idempotent) ----
VLLM_BIN="$(command -v vllm)"
if ! grep -q 'from DotsOCR import modeling_dots_ocr_vllm' "${VLLM_BIN}"; then
  # Upstream önerisi: 'main' import'unun altına ekle
  sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' "${VLLM_BIN}"
fi

# ---- vLLM'i başlat ----
echo "Starting vLLM on :${VLLM_PORT} ..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  vllm serve "${HF_MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.95}" \
    --chat-template-content-format string \
    --served-model-name model \
    --trust-remote-code &
VLLM_PID=$!

# ---- vLLM health bekle ----
for i in {1..90}; do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM is up."
    break
  fi
  sleep 1
done

# ---- FastAPI'yi başlat ----
# Uygulaman 'api/server.py' içinde ve 'app' değişkeni var dedin:
exec uvicorn api.server:app --host 0.0.0.0 --port "${APP_PORT}"
