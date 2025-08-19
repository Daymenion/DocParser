#!/usr/bin/env bash
set -euo pipefail

echo '--- Starting dots.ocr vLLM server on :9998 (GPU 1) ---'

# Yol/ortam
hf_model_path="/workspace/weights/DotsOCR"
export PYTHONPATH="/workspace/weights:${PYTHONPATH:-}"

echo "HF model path: ${hf_model_path}"
echo "PYTHONPATH: ${PYTHONPATH}"

# Model klasörü var mı, boş mu?
if [[ ! -d "${hf_model_path}" ]] || [[ -z "$(ls -A "${hf_model_path}" 2>/dev/null || true)" ]]; then
  echo "ERROR: Model path '${hf_model_path}' is missing or empty."
  echo "Mount your weights to /workspace/weights/DotsOCR (e.g., ../weights/DotsOCR)."
  exit 1
fi

# vLLM entrypoint'i DotsOCR custom modeling'i import edecek şekilde patchle
echo "Patching vllm entrypoint to import DotsOCR custom modeling..."
vllm_bin="$(command -v vllm)"
# Sadece bir kere ekle (idempotent)
grep -q 'from DotsOCR import modeling_dots_ocr_vllm' "${vllm_bin}" \
  || sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a from DotsOCR import modeling_dots_ocr_vllm' "${vllm_bin}"

echo "Patched snippet:"
grep -A 1 'from vllm.entrypoints.cli.main import main' "${vllm_bin}" || true

echo "Launching vllm serve..."
CUDA_VISIBLE_DEVICES=1 exec vllm serve "${hf_model_path}" \
  --host 0.0.0.0 \
  --port 9998 \
  --device cuda \
  --dtype auto \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.65 \
  --chat-template-content-format string \
  --served-model-name model \
  --trust-remote-code
