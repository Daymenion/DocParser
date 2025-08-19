#!/usr/bin/env bash
set -Eeuo pipefail

: "${HF_MODEL_PATH:=/workspace/weights/DotsOCR}"
: "${APP_PORT:=7860}"
: "${VLLM_PORT:=6013}"

# Model klasörünün parent'ını PYTHONPATH'e ekle
export PYTHONPATH="$(dirname "$HF_MODEL_PATH"):${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "HF_MODEL_PATH = ${HF_MODEL_PATH}"
echo "PYTHONPATH    = ${PYTHONPATH}"
echo "NVIDIA_VISIBLE_DEVICES = ${NVIDIA_VISIBLE_DEVICES:-<unset>}"

# Klasör kontrolü
if [[ ! -d "${HF_MODEL_PATH}" ]] || [[ -z "$(ls -A "${HF_MODEL_PATH}" 2>/dev/null || true)" ]]; then
  echo "ERROR: '${HF_MODEL_PATH}' yok veya boş."
  exit 1
fi

# vLLM entrypoint patch (idempotent)
VLLM_BIN="$(command -v vllm)"
if ! grep -q 'from DotsOCR import modeling_dots_ocr_vllm' "${VLLM_BIN}"; then
  sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' "${VLLM_BIN}"
fi

echo "Starting vLLM on :${VLLM_PORT} ..."
# CUDA_VISIBLE_DEVICES ayarlamıyoruz; MIG seçimini NVIDIA_VISIBLE_DEVICES belirleyecek
vllm serve "${HF_MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.65}" \
  --chat-template-content-format string \
  --served-model-name model \
  --trust-remote-code &

VLLM_PID=$!

# Health check
for _ in {1..90}; do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM is up."
    break
  fi
  sleep 1
done

# FastAPI
exec uvicorn api.server:app --host 0.0.0.0 --port "${APP_PORT}"
