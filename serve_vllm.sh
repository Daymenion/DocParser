#!/usr/bin/env bash
set -Eeuo pipefail

# --- AYARLAR ---
: "${HF_MODEL_PATH:=$HOME/Projects/DocParser/weights/DotsOCR}"  # DotsOCR klasörü (nokta yok!)
: "${PORT:=6013}"                                               # vLLM portu
: "${PY:=python3}"                                              # sistem python
: "${VENV_DIR:=$HOME/Projects/DocParser/.venv_vllm}"            # venv konumu

# MIG/GPÜ seçimi (İSTEĞE BAĞLI) — MIG kullanıyorsan UUID'i burada ver:
# export CUDA_VISIBLE_DEVICES="MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
export CUDA_DEVICE_ORDER=PCI_BUS_ID


echo "HF_MODEL_PATH = $HF_MODEL_PATH"
echo "PORT          = $PORT"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"

[[ -d "$HF_MODEL_PATH" ]] || { echo "ERR: $HF_MODEL_PATH yok."; exit 1; }
[[ -n "$(ls -A "$HF_MODEL_PATH" 2>/dev/null)" ]] || { echo "ERR: $HF_MODEL_PATH boş."; exit 1; }

# --- VENV + KURULUM ---
if [[ ! -d "$VENV_DIR" ]]; then
  $PY -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -U pip

# PyTorch CUDA (sürücüne göre cu128/cu126/cu121; resmi "Get Started" sayfasına göre)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# vLLM + transformers
pip install "vllm>=0.8.3,<0.11" "transformers>=4.53.0"

# --- PYTHONPATH (DotsOCR parent dizinini ekle) ---
export PYTHONPATH="$(dirname "$HF_MODEL_PATH"):${PYTHONPATH:-}"

# --- vLLM entrypoint patch (DotsOCR yönergesi) ---
VLLM_BIN="$(command -v vllm)"
if ! grep -q 'from DotsOCR import modeling_dots_ocr_vllm' "$VLLM_BIN"; then
  sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' "$VLLM_BIN"
fi

# --- Hızlı CUDA testi ---
python - <<'PY'
import os, torch
print("torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
PY

# --- vLLM SERVE ---
echo ">>> vLLM starting on 0.0.0.0:${PORT}"
exec vllm serve "${HF_MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device cuda \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.65 \
  --chat-template-content-format string \
  --served-model-name model \
  --trust-remote-code
