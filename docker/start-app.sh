#!/usr/bin/env bash
set -Eeuo pipefail

# ---- Parametreler / Defaults ----
: "${HF_MODEL_PATH:=/workspace/weights/DotsOCR}"  # klasör adı noktasız olmalı (DotsOCR)
: "${APP_PORT:=7860}"
: "${VLLM_PORT:=6006}"

# PYTHONPATH: model klasörünün parent'ı
export PYTHONPATH="$(dirname "$HF_MODEL_PATH"):${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # cihaz sırası için iyi pratik

echo "HF_MODEL_PATH = ${HF_MODEL_PATH}"
echo "PYTHONPATH    = ${PYTHONPATH}"

# ---- Model klasörü kontrolü ----
if [[ ! -d "${HF_MODEL_PATH}" ]] || [[ -z "$(ls -A "${HF_MODEL_PATH}" 2>/dev/null || true)" ]]; then
  echo "ERROR: '${HF_MODEL_PATH}' yok veya boş. Host'tan doğru mount ettiğinden emin ol."
  exit 1
fi

# ---- vLLM entrypoint patch (idempotent) ----
VLLM_BIN="$(command -v vllm)"
if ! grep -q 'from DotsOCR import modeling_dots_ocr_vllm' "${VLLM_BIN}"; then
  sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' "${VLLM_BIN}"
fi

# ---- vLLM'i başlat ----
echo "Starting vLLM on :${VLLM_PORT} ..."
vllm serve "${HF_MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.55}" \
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
exec uvicorn api.server:app --host 0.0.0.0 --port "${APP_PORT}"
