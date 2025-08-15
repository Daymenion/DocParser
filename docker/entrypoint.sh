#!/bin/bash
"""
Docker entrypoint script for DocParser
"""

set -e

echo "🐳 Starting DocParser Docker Container"
echo "=================================="

# Function to wait for vLLM server
wait_for_vllm() {
    local host=${1:-localhost}
    local port=${2:-9998}
    local max_attempts=30
    local attempt=1
    
    echo "🔍 Waiting for vLLM server at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://$host:$port/health" > /dev/null 2>&1; then
            echo "✅ vLLM server is ready!"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts - vLLM server not ready yet..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "❌ vLLM server not available after $max_attempts attempts"
    exit 1
}

# Parse environment variables
VLLM_HOST=${VLLM_SERVER:-localhost}
VLLM_PORT=${VLLM_PORT:-9998}
APP_MODE=${APP_MODE:-combined}
APP_HOST=${APP_HOST:-0.0.0.0}
APP_PORT=${APP_PORT:-7860}

# If vLLM host contains port, split it
if [[ $VLLM_HOST == *":"* ]]; then
    VLLM_PORT=${VLLM_HOST##*:}
    VLLM_HOST=${VLLM_HOST%:*}
fi

echo "Configuration:"
echo "  vLLM Server: $VLLM_HOST:$VLLM_PORT"
echo "  App Mode: $APP_MODE"
echo "  App Host: $APP_HOST:$APP_PORT"

# Wait for vLLM server if not running locally
if [ "$VLLM_HOST" != "localhost" ] && [ "$VLLM_HOST" != "127.0.0.1" ]; then
    wait_for_vllm "$VLLM_HOST" "$VLLM_PORT"
fi

# Create necessary directories
mkdir -p /app/logs /app/temp

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# Start the application
echo "🚀 Starting DocParser Application..."
exec python app.py \
    --mode "$APP_MODE" \
    --host "$APP_HOST" \
    --port "$APP_PORT" \
    --vllm-ip "$VLLM_HOST" \
    --vllm-port "$VLLM_PORT"
