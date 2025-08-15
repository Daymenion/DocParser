FROM vllm/vllm-openai:v0.9.1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for DotsOCR
RUN pip3 install flash_attn==2.8.0.post2
RUN pip3 install transformers==4.51.3

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --force-reinstall -r requirements.txt

# Copy the entire project
COPY . .

# Download DotsOCR model weights if not present
RUN python tools/download_model.py || echo "Model download failed, but continuing..."

# Copy and setup entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create necessary directories
RUN mkdir -p weights logs temp

# Expose ports
EXPOSE 7860

# Environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV APP_MODE=combined
ENV APP_HOST=0.0.0.0
ENV APP_PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]