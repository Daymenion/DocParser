$env:APP_PORT = 7860
$env:VLLM_HOST = "127.0.0.1"
$env:VLLM_PORT = 9998
python -m uvicorn api.server:app --host 0.0.0.0 --port $env:APP_PORT
